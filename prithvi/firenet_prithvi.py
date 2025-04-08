import torch
import torch.nn as nn
import numpy as np
from prithvi_mae import PrithviMAE
from prithvi_dataloader import FireNetDataset
from segmentation_models_pytorch.losses import JaccardLoss
import torchmetrics
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import os
from argparse import Namespace
import wandb


class LinearMappingLayer(nn.Module):
    """
    Linear mapping layer to project input channels to a different number of output channels.
    This is used to map the input channels of the FireNet dataset to the expected input channels of the Prithvi model.
    """
    def __init__(self, input_channels, output_channels):
        """
        Args:
            input_channels (int): Number of input channels (FireNet).
            output_channels (int): Number of output channels (Prithvi).
        """
        super().__init__()
        # Use Conv3d to map 39 input channels to 6 output channels
        self.linear = nn.Sequential(
            nn.Conv3d(39, 5, kernel_size=1),
            nn.BatchNorm3d(5) # normalize across batch and spatial dims
        )
    def forward(self, x):
        """
        Forward pass of the linear mapping layer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, output_channels, T, H, W).\
        """
        # x shape: (B, C, T, H, W)
        return self.linear(x)


class PrithviEncoder(nn.Module):
    def __init__(self, model, device, num_frames=3):
        """
        Encoder for the Prithvi model.
        This class wraps the Prithvi model and extracts features from the last three hidden states.
        Args:
            model (torch.nn.Module): The Prithvi model.
            device (torch.device): The device to run the model on.
            num_frames (int): Number of frames to process at once.
        """
        super().__init__()
        self.model = model
        self.device = device
        self.num_frames = num_frames

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        returns: list of 3 feature maps: [(B, embedding_dim, H, W), ...]
        """
        x = x.to(self.device)

        with torch.no_grad():
            hidden_states = self.model.forward_features(x)  # ← name it hidden_states

        processed_features = []
        for feature in hidden_states[-3:]:
            # Remove CLS token
            feature = feature[:, 1:, :]  # (B, num_patches, dim)
            B, N, C = feature.shape

            H = int(np.sqrt(N // self.num_frames))
            assert H * H * self.num_frames == N, "Patch dimensions don't match expected frame count."

            # Reshape: (B, num_frames, H, H, C) → (B, C, T, H, W)
            feature = feature.view(B, self.num_frames, H, H, C)
            feature = feature[:, -1, :, :, :]  # Only keep last frame → (B, H, H, C)
            feature = feature.permute(0, 3, 1, 2)  # → (B, C, H, H)
            processed_features.append(feature)

        return processed_features



class CNNDecoder(nn.Module):
    def __init__(self, embedding_dim):
        """
        CNN Decoder for the segmentation task.
        This class takes the output of the encoder and decodes it to a segmentation map.
        Args:
            embedding_dim (int): Dimension of the input features.
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Added sigmoid for segmentation
        )

    def forward(self, x):
        """
        x: (B, embedding_dim, T, H, W)
        output: (B, 224, 224) with pixel-wise probabilities
        """
        B, C, T, H, W = x.shape

        # Collapse over time dimension
        x = x.mean(dim=2)        # (B, C, H, W)

        x = self.decoder(x)      # (B, 1, 224, 224)
        x = x.squeeze(1)         # (B, 224, 224)
        return x



class SegformerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        MLP for Segformer decoder head.
        This class takes the input features and projects them to a different dimension.
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: Tensor of shape (B, num_patches, input_dim).
        Returns:
            hidden_states: Tensor of shape (B, num_patches, output_dim).
        """
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)
        hidden_states = self.proj(hidden_states)                  # (B, H*W, decoder_dim)
        return hidden_states


class SegformerDecoderHead(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: a configuration object with attributes:
                - num_encoder_stages: number of hidden states (stages) from the encoder to use.
                - hidden_sizes: list of encoder hidden dimensions for each stage.
                - decoder_hidden_size: target hidden dimension in the decoder.
                - classifier_dropout_prob: dropout probability.
                - num_labels: number of segmentation classes.
                - (optional) decoder_output_size: desired final output spatial size (H_out, W_out).
        """
        super().__init__()
        assert len(config.decoder_hidden_sizes) == config.num_encoder_stages
        self.mlps = nn.ModuleList([
            SegformerMLP(input_dim, output_dim)
            for input_dim, output_dim in zip(config.hidden_sizes, config.decoder_hidden_sizes)
        ])

        total_decoder_channels = sum(config.decoder_hidden_sizes)
        self.fuse_conv = nn.Conv2d(
            in_channels=total_decoder_channels,
            out_channels=config.decoder_hidden_sizes[-1],  # or any target
            kernel_size=1,
            bias=False
        )

        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_sizes[-1])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_sizes[-1], config.num_labels, kernel_size=1)
        self.config = config


    def forward(self, encoder_hidden_states: list) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: List of tensors, each of shape (B, embedding_dim_i, H_i, W_i).
        Returns:
            logits: Segmentation logits of shape (B, num_labels, H_out, W_out)
        """
        batch_size = encoder_hidden_states[0].size(0)
        target_size = encoder_hidden_states[0].shape[2:]  # Highest-res stage

        processed_states = []
        for state, mlp, out_dim in zip(encoder_hidden_states, self.mlps, self.config.decoder_hidden_sizes):
            H, W = state.shape[2], state.shape[3]
            state = mlp(state)  # (B, H*W, decoder_dim)
            state = state.transpose(1, 2).view(batch_size, out_dim, H, W)  # (B, decoder_dim, H, W)
            state = F.interpolate(state, size=target_size, mode="bilinear", align_corners=False)
            processed_states.append(state)

        x = torch.cat(processed_states, dim=1)  # (B, sum(decoder_dims), H, W)
        x = self.fuse_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        if hasattr(self.config, "decoder_output_size"):
            x = F.interpolate(x, size=self.config.decoder_output_size, mode="bilinear", align_corners=False)

        logits = self.classifier(x)
        return logits



class FireNet(nn.Module):
    def __init__(self, linear_map, encoder, decoder):
        """
        FireNet model that combines a linear mapping layer, encoder, and decoder.
        Args:
            linear_map (LinearMappingLayer): Linear mapping layer to project input channels.
            encoder (PrithviEncoder): Encoder for the Prithvi model.
            decoder (SegformerDecoderHead): Decoder head for segmentation.
        """
        super().__init__()
        self.linear_map = linear_map
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """
        Forward pass of the FireNet model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 40, T, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, 1, H_out, W_out).
        """
        x_env = x[:, :39, ...]  
        x_fire = x[:, 39:, ...]
        x_env_mapped = self.linear_map(x_env) # (B, 6, T, H, W)
        x_combined = torch.cat([x_env_mapped, x_fire], dim=1)
        z = self.encoder(x_combined) # (B, embedding_dim, T, H', W')
        out = self.decoder(z)
        return out


def train_firenet(model, train_loader, val_loader, device, num_epochs=10,
                  lr_frozen=1e-3, lr_unfrozen=1e-4,
                  log_file="train_logs/training_log.txt", model_dir="saved_models",
                  patience=5, early_stop_patience=10):

    # Freeze encoder
    encoder_frozen = False
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
        encoder_frozen = True

    criterion = JaccardLoss(mode="binary", from_logits=True)
    train_f1 = torchmetrics.F1Score(task="binary").to(device)
    val_f1 = torchmetrics.F1Score(task="binary").to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen
    )

    model.to(device)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    best_val_loss = float('inf')
    epochs_since_improvement = 0
    early_stop_counter = 0
    scheduler = None
    unfreeze_epoch = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_f1.reset()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = preds.squeeze(1)
            train_f1(preds, y.int())

        avg_train_loss = total_loss / len(train_loader)
        train_f1_score = train_f1.compute().item()
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | F1: {train_f1_score:.4f}")

        avg_val_loss, val_f1_score = validate_firenet(model, val_loader, device, criterion, val_f1)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_f1": train_f1_score,
            "val_loss": avg_val_loss,
            "val_f1": val_f1_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        # Save model only if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            early_stop_counter = 0
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch + 1
            print(f"Model improved. Saved to {best_model_path}")
        else:
            epochs_since_improvement += 1
            early_stop_counter += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s)")

        # Unfreeze encoder and reduce LR on plateau
        if encoder_frozen and epochs_since_improvement >= patience:
            print(f"Unfreezing encoder and lowering LR at epoch {epoch+1}")
            wandb.log({"encoder_unfrozen_epoch": epoch + 1})
            with open(log_file, "a") as f:
                f.write(f"Encoder unfrozen at epoch {epoch+1}\n")

            for param in model.encoder.parameters():
                param.requires_grad = True
            model.encoder.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr_unfrozen)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            encoder_frozen = False
            unfreeze_epoch = epoch + 1
            early_stop_counter = 0

        # Step scheduler only after unfreezing
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Early stopping check
        if not encoder_frozen and early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            wandb.log({"early_stopped_epoch": epoch + 1})
            break

        # Log to file (in addition to wandb)
        with open(log_file, "a") as f:
            f.write(
                f"Epoch {epoch+1}: "
                f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1_score:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1_score:.4f}\n"
            )


def validate_firenet(model, val_loader, device, criterion, val_f1):
    model.eval()
    total_loss = 0.0
    val_f1.reset()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).float()
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
            preds = preds.squeeze(1)
            val_f1(preds, y.int())

    avg_loss = total_loss / len(val_loader)
    val_f1_score = val_f1.compute().item()
    print(f"Validation Loss: {avg_loss:.4f} | F1: {val_f1_score:.4f}")
    model.train()
    return avg_loss, val_f1_score


def main():
    checkpoint = "prithvi/Prithvi_EO_V1_100M.pt"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.\n")


    # Loading data ---------------------------------------------------------------------------------

    with open("prithvi/prithvi.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    dataset = FireNetDataset(**data_config)

    print(f"Dataset loaded from: {data_config['data_dir']}")
    dataset.setup()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    val_loader = dataset.val_dataloader()

    # Create model and load checkpoint -------------------------------------------------------------
    prithvi_config_path = "prithvi/config.json"
    with open(prithvi_config_path, "r") as f:
        prithvi_config = yaml.safe_load(f)['pretrained_cfg']

    bands = prithvi_config['bands']
    num_frames = 3

    prithvi_config.update(
        num_frames=num_frames,
        in_chans=len(bands),
    )

    prithvi = PrithviMAE(**prithvi_config)
    prithvi.to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    # discard fixed pos_embedding weight
    for k in list(state_dict.keys()):
        if 'pos_embed' in k:
            del state_dict[k]
    prithvi.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint}")

    linear_map = LinearMappingLayer(input_channels=39, output_channels=5)
    encoder = PrithviEncoder(model=prithvi, device=device, num_frames=3)
    
    config = Namespace(
        num_encoder_stages=3,
        hidden_sizes=[768, 768, 768],
        decoder_hidden_sizes=[768, 384, 192],           
        classifier_dropout_prob=0.1,
        num_labels=1,
        decoder_output_size=(224, 224)
    )

    decoder = SegformerDecoderHead(config)

    # Full model
    firenet_model = FireNet(linear_map, encoder, decoder)
    firenet_model.to(device)

    total_params = sum(p.numel() for p in firenet_model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")

    train_firenet(firenet_model, train_loader, val_loader, device)

if __name__ == '__main__':
    wandb.init(project="firenet-training", name="fine-tuning")
    main()



