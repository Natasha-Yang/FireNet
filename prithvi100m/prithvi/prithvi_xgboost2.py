import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import yaml
import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from prithvi_dataloader import FireNetDataset
from prithvi_mae import PrithviViT, PrithviMAE
from sklearn.metrics import mean_squared_error
from inference_wsts1 import visualize_mae_outputs



class LinearMappingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Conv3d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        return self.linear(x)

# Feature extraction and XGBoost training
class XGBoostDecoder:
    def __init__(self):
        self.model = xgb.XGBRegressor()

    def fit(self, features, targets):
        features = features.reshape(features.shape[0], -1)
        print("a")
        #self.model.fit(features, targets)
        features = features[:100]
        targets = targets[:100]
        self.model.fit(features, targets)
        print("b")

    def predict(self, features):
        features = features.reshape(features.shape[0], -1)
        return self.model.predict(features)


def run_model_with_xgboost(model, xgboost_decoder, input_data, mask_ratio, device):
    with torch.no_grad():
        x = input_data.to(device)
        latent_features = model.forward_features(x)
        features = latent_features[-1].detach().cpu().numpy()

        # Use XGBoost for decoding
        predictions = xgboost_decoder.predict(features)
        #print(f"Input data shape: {input_data.shape}")
        #print(f"Target shape: {y.shape}")
        #print(f"Predictions shape before reshaping: {predictions.shape}")
        # Reshape predictions to match input
        predictions = torch.tensor(predictions).view(-1)
        return predictions


def main():
    checkpoint = "prithvi100m/prithvi/Prithvi_EO_V1_100M.pt"
    model_config_path = "prithvi100m/prithvi/config.json"
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)['pretrained_cfg']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.\n")

    # Load dataset
    dataset = FireNetDataset(**yaml.safe_load(open("prithvi100m/prithvi/prithvi.yaml")))
    dataset.setup()
    test_loader = dataset.test_dataloader()
    
    # Initialize PrithviMAE and XGBoost decoder
    #model = PrithviViT(**model_config).to(device)
    model = PrithviMAE(**model_config)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")

    model.to(device)
    #state_dict = torch.load(checkpoint, map_location=device)
    #model.load_state_dict(state_dict, strict=False)
    state_dict = torch.load(checkpoint, map_location=device)
    for k in list(state_dict.keys()):
        if 'pos_embed' in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint}")

    xgboost_decoder = XGBoostDecoder()

    input_dim = 40#len(bands) * img_size * img_size * num_frames  # Calculate input dimension
    output_dim = 6#input_dim  # Define output dimension as needed (for now, keeping the same size)
    linear_mapping = LinearMappingLayer(input_dim, output_dim).to(device)

    # Running model --------------------------------------------------------------------------------

    model.eval()
    linear_mapping.eval()
    # Train XGBoost using extracted features
    for x, y in test_loader:
        x = x.to(device)

        x = linear_mapping(x)
        #y = y[:100]
        latent_features = model.forward_features(x)  # Corrected
        print("made it here")

    # Use the final layer's output
    
        features = latent_features[-1].detach().cpu().numpy()
        y = y.view(-1).numpy()  # Flatten y to 1D
        features = features.reshape(features.shape[0], -1) 
        #features = features[:100]
        print("here too")
        xgboost_decoder.fit(features, y)
        print("i run this successfully at least once!")
        #latent_features, _, _ = model.forward_features(x)
        #features = latent_features[-1].detach().cpu().numpy()
        #xgboost_decoder.fit(features, y.numpy())

    # Run model with XGBoost decoder
    for x, y in test_loader:
        print("stuck?")
        x = x.to(device)
        x = linear_mapping(x)
        #y = y[:100]

        predictions = run_model_with_xgboost(model, xgboost_decoder, x, mask_ratio=0.5, device=device)
        print(f"Predictions shape: {predictions.shape}")
        #predictions = predictions.view(y.shape).cpu().numpy()

        #mse = mean_squared_error(y.flatten().cpu().numpy(), predictions.flatten().cpu().numpy())
        #print(f"Mean Squared Error: {mse}")

        print(f"Reconstructed shape: {predictions.shape}")
        grid_size = int(predictions.numel() ** 0.5)  # Assuming predictions are square (e.g., 10x10)
        predictions = predictions.view(1, 1, grid_size, grid_size)  # Add batch and channel dimensions

# Interpolate to match the shape of y (e.g., [1, 224, 224])
        predictions_resized = torch.nn.functional.interpolate(predictions, size=(224, 224), mode='bilinear', align_corners=False)
        predictions_resized = predictions_resized.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions
        predictions = predictions_resized
        #print(f"Mask shape: {mask_img.shape}")
        # Reshape predictions and ground truth to match the segmentation map dimensions
        #predictions_map = predictions#predictions.view(y.shape).cpu().numpy()  # Reshape predictions
        ground_truth_map = y.cpu().numpy()  # Reshape ground truth
        #visualize_mae_outputs(x, predictions, predictions, bands=[3, 2, 1])
        
        # Plot the predictions and ground truth
        plt.figure(figsize=(10, 5))

        # Ground truth
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth")
        plt.imshow(ground_truth_map[0], cmap='gray')  # Assuming single-channel segmentation map
        plt.axis('off')

        # Predictions
        plt.subplot(1, 2, 2)
        plt.title("Predictions")
        plt.imshow(predictions[0], cmap='gray')  # Assuming single-channel segmentation map
        plt.axis('off')

        plt.show()
        
if __name__ == '__main__':
    main()
