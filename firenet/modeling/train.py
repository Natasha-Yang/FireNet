from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from firenet.config import MODELS_DIR, NDWS_PROCESSED_DATA_DIR

app = typer.Typer()

class CNN_Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout2d() 
        )
    def forward(self, x):  
        return self.encoder(x)


class CNN_Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1)
        )
    
    def forward(self, x):
        return self.decoder(x)
    


class FireMaskLSTM(nn.Module):
    def __init__(self, pretrained_encoder, rnn_type="LSTM", hidden_dim = 64):
        super().__init__()

        # load 5 CNN encoders
        self.encoder1 = pretrained_encoder(1)  # topography
        self.encoder2 = pretrained_encoder(4)  # weather
        self.encoder3 = pretrained_encoder(3)  # moisture
        self.encoder4 = pretrained_encoder(3)  # fuel
        self.encoder5 = pretrained_encoder(1)  # previous fire mask

        self.hidden_dim = hidden_dim

        # Feature Fusion Layer (Concat + Conv)
        self.fusion_layer = nn.Conv2d(5 * 32, 64, kernel_size = 1)

        # RNN for Temporal Modeling
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size = 64, hidden_size = hidden_dim,
                               num_layers = 2, batch_first = True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size = 64, hidden_size = hidden_dim,
                              num_layers = 2, batch_first = True)

        # decoder: convert RNN output to fire mask
        self.decoder = CNN_Decoder(hidden_dim)


    def forward(self, x):
        B, T, C, H, W = x.shape  # (B, T, C, H, W)

        rnn_inputs = []  # encoded features for each timestep
        for t in range(T):
            x_t = x[:, t, :, :, :]  

            x1 = x_t[:, 0, :, :].unsqueeze(1) # topography
            x2 = x_t[:, 1:5, :, :] # weather
            x3 = x_t[:, 5:8, :, :] # moisture
            x4 = x_t[:, 8:11, :, :] # fuel
            x5 = x_t[:, 11, :, :].unsqueeze(1) # previous fire mask

            # encoder forward pass
            z1 = self.encoder1(x1)
            z2 = self.encoder2(x2)
            z3 = self.encoder3(x3)
            z4 = self.encoder4(x4)
            z5 = self.encoder5(x5)

            fused_features = torch.cat([z1, z2, z3, z4, z5], dim=1)  # (B, 5*C, H/32, W/32)
            fused_features = self.fusion_layer(fused_features)  # fuse embeddings

            # flatten 2D feature maps into a sequence for RNN 
            fused_features = fused_features.mean(dim=[2, 3])

            rnn_inputs.append(fused_features)

        # stack features across timesteps: (B, T, C)
        rnn_inputs = torch.stack(rnn_inputs, dim=1)

        # forward pass through RNN
        rnn_out, _ = self.rnn(rnn_inputs)  # (B, T, hidden_dim)

        # reshape
        last_timestep_features = rnn_out[:, -1, :].view(B, self.hidden_dim, 1, 1)
        
        # decode to fire mask
        mask = self.decoder(last_timestep_features)

        return torch.sigmoid(mask) # binary fire mask output


def masked_weighted_BCE(y_pred, y_true):
    # (B, C, 64, 64)
    valid_mask = (y_true != -1).float()
    y_true = y_true * valid_mask
    y_pred = y_pred * valid_mask
    loss = 0.9 * y_true * torch.log(y_pred + 1e-6) + 0.1 * (1 - y_true) * torch.log(1 - y_pred + 1e-6)
    return -torch.sum(loss) / (torch.sum(valid_mask) + 1e-6)


def train(model, optimizer, device, train_loader, val_loader, num_epochs, criterion):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for input, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True):
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input, label in val_loader:
                input, label = input.to(device), label.to(device)
                output = model(input)
                loss = criterion(output, label)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # save the model
    torch.save(model.state_dict(), 'cnn_gru.pth')


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = NDWS_PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = NDWS_PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
