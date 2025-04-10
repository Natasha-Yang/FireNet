import torch
import torch.nn as nn
import numpy as np
from firenet_prithvi import *
from prithvi_dataloader import FireNetDataset
import torchmetrics
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import os
import wandb



class LinearMappingLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Linear mapping layer to project input channels to a different number of output channels.
        This is used to map the input channels of the FireNet dataset to the expected input channels of the Prithvi model.
        Args:
            input_channels (int): Number of input channels (FireNet).
            output_channels (int): Number of output channels (Prithvi).
        """
         # Initialize the parent class
         # Define a 3D convolutional layer with kernel size 1 to perform linear mapping
        super(LinearMappingLayer, self).__init__()
        self.linear = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the linear mapping layer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, output_channels, T, H, W).
        """
        # x: (B, 40, T, H, W)
        x_env = x[:, :-1, :, :, :]  # (B, 39, T, H, W)
        x_fire = x[:, -1:, :, :, :]  # (B, 1, T, H, W) — use : instead of indexing to preserve 5D
        projected = self.linear(x_env)  # (B, 5, T, H, W)
        combined = torch.cat([projected, x_fire], dim=1)  # (B, 6, T, H, W)
        return combined


class PretrainPrithvi(nn.Module):
    def __init__(self, model, projector):
        """
        Pretraining wrapper for the Prithvi model.
        This class wraps the Prithvi model and a linear mapping layer to project the input channels.
        Args:
            model (torch.nn.Module): The Prithvi model.
            projector (LinearMappingLayer): The linear mapping layer to project input channels.
        """
         # Initialize the parent class
         # Store the model and projector
        super().__init__()
        self.model = model
        self.projector = projector

    def forward(self, x, **kwargs):
        """
        Forward pass of the pretraining model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).
            **kwargs: Additional arguments for the model.
        Returns:
                torch.Tensor: Output tensor from the model.
        """
        x = self.projector(x)
        return self.model(x, **kwargs)


def pretrain_prithvi(
    prithvi_model,
    dataloader,
    device,
    epochs=50,
    mask_ratio=0.75,
    input_channels=40,
    mapped_channels=6,
    lr=1e-4,
    save_path="prithvi_pretrained.pt",
    wandb_project="prithvi-pretraining",
    run_name="prithvi-run",
    early_stop_patience=5
):
    """
    Pretrain the Prithvi model using a linear mapping layer to project input channels.
    Args:
        prithvi_model (torch.nn.Module): The Prithvi model to be pretrained.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        device (torch.device): Device to run the training on (CPU or GPU).
        epochs (int): Number of epochs to train.
        mask_ratio (float): Ratio of input data to mask during training.
        input_channels (int): Number of input channels in the dataset.
        mapped_channels (int): Number of output channels after mapping.
        lr (float): Learning rate for the optimizer.
        save_path (str): Path to save the pretrained model.
        wandb_project (str): WandB project name for logging.
        run_name (str): WandB run name for logging.
        early_stop_patience (int): Number of epochs with no improvement before stopping training.
    """
    # Initialize wandb
    wandb.init(project=wandb_project, name=run_name, config={
        "epochs": epochs,
        "mask_ratio": mask_ratio,
        "input_channels": input_channels,
        "mapped_channels": mapped_channels,
        "learning_rate": lr
    })

    projector = LinearMappingLayer(input_channels - 1, mapped_channels - 1).to(device)
    model = PretrainPrithvi(prithvi_model, projector).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            x, _ = batch
            x = x.to(device)

            latent, pred, mask = model(x, mask_ratio=mask_ratio)
            target = prithvi_model.patchify(model.projector(x))  # ensure target matches pred shape

            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        # Save best model if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": prithvi_model.state_dict(),
                "projector_state_dict": projector.state_dict()
            }, save_path)
            print(f"✅ Model improved. Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

        if patience_counter >= early_stop_patience:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
            break

    wandb.finish()






if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset config and dataset
    with open("prithvi/prithvi.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    dataset = FireNetDataset(**data_config, prithvi=True)
    print(f"Dataset loaded from: {data_config['data_dir']}")

    dataset.setup()
    train_loader = dataset.train_dataloader()

    # Load Prithvi config
    prithvi_config_path = "prithvi/config.json"
    with open(prithvi_config_path, "r") as f:
        prithvi_config = yaml.safe_load(f)['pretrained_cfg']

    # Inject number of bands (from dataset) and number of frames (time dimension)
    bands = prithvi_config['bands']  # This should be a list of 40 bands (input_channels)
    num_frames = data_config['n_leading_observations']

    prithvi_config.update(
        num_frames=num_frames,
        in_chans=6  # After linear mapping: 5 projected + 1 fire channel
    )

    # Initialize model
    prithvi = PrithviMAE(**prithvi_config)
    prithvi.to(device)

    # Load checkpoint (optional)
    checkpoint = "prithvi/Prithvi_EO_V1_100M.pt"
    state_dict = torch.load(checkpoint, map_location=device)

    # Remove incompatible position embeddings
    for k in list(state_dict.keys()):
        if 'pos_embed' in k:
            del state_dict[k]

    prithvi.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint}")

    # === Pretrain model ===
    pretrain_prithvi(
        prithvi,
        train_loader,
        device,
        epochs=50,
        input_channels=40,  # from FireNet input
        mapped_channels=6,  # 5 projected + 1 fire channel
        lr=1e-4,
        save_path="prithvi_pretrained.pt"
    )
