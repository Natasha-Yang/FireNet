from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from firenet.config import MODELS_DIR, NDWS_PROCESSED_DATA_DIR
from firenet.dataset_NDWS import *

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
    

class FireMaskRNN(nn.Module):
    def __init__(self, pretrained_encoder, hidden_dim, rnn_type="LSTM"):
        super().__init__()

        # load 5 CNN encoders
        self.encoder1 = pretrained_encoder(1)  # topography
        self.encoder2 = pretrained_encoder(4)  # weather
        self.encoder3 = pretrained_encoder(3)  # moisture
        self.encoder4 = pretrained_encoder(3)  # fuel
        self.encoder5 = pretrained_encoder(1)  # previous fire mask
        
        # Feature Fusion Layer (Concat + Conv)
        self.fusion_layer = nn.Conv2d(5 * 32, 64, kernel_size = 1)
        self.hidden_dim = hidden_dim

        # RNN for Temporal Modeling
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size = 64, hidden_size = self.hidden_dim,
                               num_layers = 2, batch_first = True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size = 64, hidden_size = self.hidden_dim,
                              num_layers = 2, batch_first = True)

        # decoder: convert RNN output to fire mask
        self.decoder = CNN_Decoder(self.hidden_dim)


    def forward(self, x, ht):
        B, T, C, H, W = x.shape  # (B, 1, C, H, W)
        
        x = x.squeeze(dim=1)

        x1 = x[:, 0, :, :].unsqueeze(1) # topography
        x2 = x[:, 1:5, :, :] # weather
        x3 = x[:, 5:8, :, :] # moisture
        x4 = x[:, 8:11, :, :] # fuel
        x5 = x[:, 11, :, :].unsqueeze(1) # previous fire mask

        # encoder forward pass
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        z3 = self.encoder3(x3)
        z4 = self.encoder4(x4)
        z5 = self.encoder5(x5)

        fused_features = torch.cat([z1, z2, z3, z4, z5], dim=1)  # (B, 5*C, H/32, W/32)
        fused_features = self.fusion_layer(fused_features)  # fuse embeddings

        # flatten 2D feature maps into a sequence for RNN 
        fused_features = fused_features.mean(dim=[2, 3]) # (B, 64, H/32, H/32) -> (B, 64)

        rnn_inputs = fused_features.unsqueeze(dim=1) # (B, 1, 64)

        # forward pass through RNN
        rnn_out, rnn_hidden = self.rnn(rnn_inputs, ht)
        rnn_out = rnn_out.squeeze(dim=1).view(B, self.hidden_dim, 1, 1)

        # decode to fire mask
        mask = self.decoder(rnn_out)

        return torch.sigmoid(mask), rnn_hidden  # binary fire mask output

def masked_weighted_BCE(y_pred, y_true):
    # (B, C, 64, 64)
    valid_mask = (y_true != -1).float()
    y_true = y_true * valid_mask
    y_pred = y_pred * valid_mask
    loss = 0.9 * y_true * torch.log(y_pred + 1e-6) + 0.1 * (1 - y_true) * torch.log(1 - y_pred + 1e-6)
    return -torch.sum(loss) / (torch.sum(valid_mask) + 1e-6)


def tbptt_train(model, optimizer, device, train_loader, val_loader, num_epochs, criterion):
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        train_steps = 0
        for seq, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True):
            seq, labels = seq.to(device), labels.to(device)
            B, T, C, H, W = seq.shape
            
            # initial hidden state
            h0 = torch.zeros(2, seq.shape[0], model.hidden_dim) 
            if isinstance(model.rnn, nn.LSTM):
                h0 = (h0, h0)  # LSTM needs (h, c)
            
            ht = h0

            accumulated_loss = 0
            for t in range(T): # sequence length
                pred, ht = model(seq[:, t, :, :].unsqueeze(1), ht)
                loss = criterion(pred, labels[:, t, :, :].unsqueeze(1))
                accumulated_loss = accumulated_loss + loss
                
                # detach gradients every 10 steps
                if (t + 1) % 10 == 0 or (t + 1) == T:
                    optimizer.zero_grad()
                    accumulated_loss.backward()
                    optimizer.step()

                    epoch_train_loss += accumulated_loss.item()
                    train_steps += 1

                    accumulated_loss = 0

                    if isinstance(ht, tuple):  # LSTM case
                        ht = (ht[0].detach(), ht[1].detach())
                    else:  # GRU case
                        ht = ht.detach()     
        
        avg_train_loss = epoch_train_loss / train_steps if train_steps > 0 else 0.0
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)
                h0 = torch.zeros(2, seq.shape[0], model.hidden_dim) 
                if isinstance(model.rnn, nn.LSTM):
                    h0 = (h0, h0) 
                ht = h0
                for t in range(seq.shape[1]):
                    pred, ht = model(seq[:, t, :, :, :].unsqueeze(1), ht) 
                    loss = criterion(pred, labels[:, t, :, :].unsqueeze(1))
                    val_loss += loss.item()
                    val_steps += 1

        val_loss /= val_steps
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    
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
    #app()
    train_dataset, val_dataset, test_dataset = make_interim_datasets()

    # prepare dataloaders
    train_loader, val_loader, test_loader = get_dataloaders_all_splits(train_dataset,
                                                                       val_dataset,
                                                                       test_dataset,
                                                                       batch_sz = 1,
                                                                       shuffle = False)
    
    model = FireMaskRNN(CNN_Encoder, 64, "GRU")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = masked_weighted_BCE

    torch.autograd.set_detect_anomaly(True)

    online_train(model, optimizer, device, train_loader, val_loader, 3, criterion)
