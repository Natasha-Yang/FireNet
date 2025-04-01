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

def save_predictions_and_ground_truth(predictions, ground_truth, pred_filename="predictions.npy", gt_filename="ground_truth.npy"):
    # Save predictions and ground truth separately
    np.save(pred_filename, predictions)
    np.save(gt_filename, ground_truth)
    print(f"Predictions saved to {pred_filename}")
    print(f"Ground truth saved to {gt_filename}")

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
    model = PrithviMAE(**model_config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")
    model.to(device)

    # Load state dict
    state_dict = torch.load(checkpoint, map_location=device)
    for k in list(state_dict.keys()):
        if 'pos_embed' in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint}")

    xgboost_decoder = XGBoostDecoder()

    input_dim = 40
    output_dim = 6
    linear_mapping = LinearMappingLayer(input_dim, output_dim).to(device)

    # Running model --------------------------------------------------------------------------------
    model.eval()
    linear_mapping.eval()

    # Save predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    for x, y in test_loader:
        x = x.to(device)
        x = linear_mapping(x)

        with torch.no_grad():
            latent_features = model.forward_features(x)
            features = latent_features[-1].detach().cpu().numpy()

            # Flatten the ground truth
            y = y.view(-1).numpy()

            # Train the XGBoost model
            xgboost_decoder.fit(features, y)
            print("XGBoost model trained successfully!")
            break

    for x,y in test_loader:
        x = x.to(device)
        x = linear_mapping(x)
        predictions = run_model_with_xgboost(model, xgboost_decoder, x, mask_ratio=0.5, device=device)

        # Collect the ground truth and predictions
        all_predictions.append(predictions.cpu().numpy())
        all_ground_truth.append(y.cpu().numpy()[:100])
        break

    # Convert to numpy arrays and save them
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Save predictions and ground truth to files
    save_predictions_and_ground_truth(all_predictions, all_ground_truth, "saved_predictions.npy", "saved_ground_truth.npy")

if __name__ == '__main__':
    main()