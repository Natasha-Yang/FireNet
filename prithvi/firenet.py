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

class LinearMappingLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LinearMappingLayer, self).__init__()
        # Use Conv3d to map 23 input channels to 6 output channels
        self.linear = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        return self.linear(x)


class PrithviEncoder(nn.Module):
    def __init__(self, model, device, num_frames=3):
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
    """
    Linear Embedding.
    """
    def __init__(self, config, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: (B, C, H, W)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # -> (B, H*W, C)
        hidden_states = self.proj(hidden_states)                  # -> (B, H*W, decoder_hidden_size)
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
        # Create one MLP for each encoder stage we want to use.
        self.mlps = nn.ModuleList([
            SegformerMLP(config, input_dim=hidden_size)
            for hidden_size in config.hidden_sizes[:config.num_encoder_stages]
        ])
        # Fusion layer: Concatenate processed features along the channel dimension.
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_stages,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)
        self.config = config

    def forward(self, encoder_hidden_states: list) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: List of tensors, each of shape (B, embedding_dim_i, H_i, W_i).
        Returns:
            logits: Segmentation logits of shape (B, num_labels, H_out, W_out)
        """
        batch_size = encoder_hidden_states[0].size(0)
        processed_states = []
        # Use the spatial size of the first (usually highest-resolution) stage as target.
        target_size = encoder_hidden_states[0].squeeze(2).shape[2:]  # (H0, W0)
        for state, mlp in zip(encoder_hidden_states, self.mlps):
            x = state  # shape: (B, embedding_dim, H, W)
            H, W = x.shape[2], x.shape[3]
            # Apply the MLP to project to decoder_hidden_size.
            x = mlp(x)  # -> (B, H*W, decoder_hidden_size)
            # Reshape back to (B, decoder_hidden_size, H, W)
            x = x.transpose(1, 2).view(batch_size, self.config.decoder_hidden_size, H, W)
            # Upsample to the target size.
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            processed_states.append(x)
        # Concatenate along the channel dimension.
        x = torch.cat(processed_states, dim=1)
        # Fuse features.
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        # Optionally upsample to a final output size if provided.
        if hasattr(self.config, "decoder_output_size"):
            x = F.interpolate(x, size=self.config.decoder_output_size, mode="bilinear", align_corners=False)
        logits = self.classifier(x)
        return logits




class FireNet(nn.Module):
    def __init__(self, linear_map, encoder, decoder):
        super().__init__()
        self.linear_map = linear_map
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x_env = x[:, :39, ...]  
        x_fire = x[:, 39:, ...]
        x_env_mapped = self.linear_map(x_env) # (B, 6, T, H, W)
        x_combined = torch.cat([x_env_mapped, x_fire], dim=1)
        z = self.encoder(x_combined) # (B, embedding_dim, T, H', W')
        out = self.decoder(z) # (B, 224, 224)
        return out



def train_firenet(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3,
                  log_file="train_logs/training_log.txt", model_dir="saved_models"):
    # Freeze encoder
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()

    # Loss, optimizer, and metrics
    criterion = JaccardLoss(mode="binary")
    train_f1 = torchmetrics.F1Score(task="binary").to(device)
    val_f1 = torchmetrics.F1Score(task="binary").to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    model.to(device)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    best_val_loss = float('inf')

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
            # From shape: (B, 1, H, W) → (B, H, W)
            preds = preds.squeeze(1)
            train_f1(preds, y.int())

        avg_train_loss = total_loss / len(train_loader)
        train_f1_score = train_f1.compute().item()
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | F1: {train_f1_score:.4f}")

        avg_val_loss, val_f1_score = validate_firenet(model, val_loader, device, criterion, val_f1)

        # Save model only if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Model improved. Saved to {best_model_path}")

        # Log to text file
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
            # From shape: (B, 1, H, W) → (B, H, W)
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
        decoder_hidden_size=256,           # you choose this
        classifier_dropout_prob=0.1,       # typical value
        num_labels=1,                      # binary mask output
        decoder_output_size=(224, 224)     # optional; match input resolution if needed
    )

    decoder = SegformerDecoderHead(config)

    # Full model
    firenet_model = FireNet(linear_map, encoder, decoder)
    firenet_model.to(device)

    total_params = sum(p.numel() for p in firenet_model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")

    train_firenet(firenet_model, train_loader, val_loader, device)

if __name__ == '__main__':
    main()



