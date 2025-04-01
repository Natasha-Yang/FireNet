# xgboost_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from prithvi_dataloader import FireNetDataset
import xgboost as xgb
import torch
import numpy as np
from features import get_features
import numpy as np
import yaml
from inference import load_example
from prithvi_mae import PrithviMAE
from inference_wsts1 import LinearMappingLayer


class XGBoostModel:
    def __init__(self, params=None):
        """
        Initializes the XGBoostModel class with optional hyperparameters.
        
        Args:
        - params (dict, optional): Hyperparameters for XGBoost. Defaults to None.
        """
        if params is None:
            # Default parameters for regression; you can modify this as needed
            self.params = {
                'objective': 'reg:squarederror',  # Change to 'binary:logistic' or 'multi:softmax' for classification
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'rmse'  # Adjust the evaluation metric based on your problem
            }
        else:
            self.params = params

        self.model = None

    def train(self, model, input_data, device, labels, num_frames=3, num_boost_round=100):
        """
        Extracts features and trains the XGBoost model.
        
        Args:
        - model (torch.nn.Module): The pretrained MAE model.
        - input_data (torch.Tensor): The input data (batch of images).
        - device (torch.device): The device (GPU/CPU) for running the model.
        - labels (torch.Tensor): The target labels for training.
        - num_frames (int, optional): Number of frames in the input. Defaults to 3.
        - num_boost_round (int, optional): Number of boosting rounds. Defaults to 100.
        
        Returns:
        - self: The trained XGBoost model.
        """
        # Step 1: Get features using the provided model and input data
        features = get_features(model, input_data, device, num_frames)
        
        # Step 2: Convert features and labels to NumPy arrays (XGBoost expects this)
        features_flattened = features.reshape(features.shape[0], -1).cpu().numpy()

        labels_np = labels.cpu().numpy().flatten()
        print(f"Labels shape: {labels_np.shape}")


        # Step 3: Convert to DMatrix format for XGBoost
        dtrain = xgb.DMatrix(features_flattened, label=labels_np)
        
        # Step 4: Train the model
        self.model = xgb.train(self.params, dtrain, num_boost_round=num_boost_round)
        return self

    def predict(self, features):
        """
        Makes predictions using the trained XGBoost model.
        
        Args:
        - features (torch.Tensor): The features for prediction.
        
        Returns:
        - np.ndarray: The predicted values.
        """
        if self.model is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        
        # Convert features to NumPy array
        features_flattened = features.reshape(features.shape[0], -1).cpu().numpy()

        # Convert to DMatrix for prediction
        dtest = xgb.DMatrix(features_flattened, )
        
        # Make predictions
        preds = self.model.predict(dtest)
        
        return preds

    def save_model(self, file_path):
        """
        Save the trained XGBoost model to a file.
        
        Args:
        - file_path (str): The path where the model should be saved.
        """
        if self.model is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        
        self.model.save_model(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        Load an XGBoost model from a file.
        
        Args:
        - file_path (str): The path to the saved model file.
        """
        self.model = xgb.Booster()
        self.model.load_model(file_path)
        print(f"Model loaded from {file_path}")

# main.py

import torch
import yaml
from prithvi_mae import PrithviMAE
from features import get_features
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from torchvision import transforms


def run_model(model, x, mask_ratio, device):
    """
    Simulating a function that runs the MAE model with masked inputs and returns reconstructed images.
    """
    # Assuming `model` is a pretrained MAE model that does the reconstruction
    rec_img, mask_img = model(x)
    return rec_img, mask_img

def visualize_mae_outputs(x, mask_img, rec_img, bands):
    """
    Dummy function for visualization. Replace with your visualization logic.
    """
    # Visualize original, masked, and reconstructed images
    print(f"Visualizing with bands {bands}")
    pass  # Add your visualization logic here.

def main():
    checkpoint = "prithvi100m/prithvi/Prithvi_EO_V1_100M.pt"
    model_config_path = "prithvi100m/prithvi/config.json"
    
    # Load model config
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)['pretrained_cfg']

    # Set training parameters
    batch_size = 1
    bands = model_config['bands']
    num_frames = 3
    mean = [1.95905826e+03,  2.94404070e+03,  1.80315792e+03,  4.18785304e+03,
            2.20147914e+03,  4.75643503e-01]
    std = [1.13697378e+03, 1.63707682e+03, 1.89291266e+03, 2.21105881e+03,
            1.12833037e+03, 2.15286520e+00]
    img_size = model_config['img_size']

    # Select device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.\n")

    # Loading data
    with open("prithvi100m/prithvi/prithvi.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    dataset = FireNetDataset(**data_config)
    print(f"Dataset loaded from: {data_config['data_dir']}")
    dataset.setup()
    test_loader = dataset.test_dataloader()

    # Create model and load checkpoint
    model_config.update(
        num_frames=num_frames,
        in_chans=len(bands),
    )

    model = PrithviMAE(**model_config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")
    model.to(device)

    state_dict = torch.load(checkpoint, map_location=device)
    for k in list(state_dict.keys()):
        if 'pos_embed' in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint}")

    # Create the linear mapping layer
    input_dim = 40  # Update with correct dimensions for input
    output_dim = 6  # Update based on your task, for now keeping it simple
    linear_mapping = LinearMappingLayer(input_dim, output_dim).to(device)

    # Initialize XGBoost model
    xgb_model = XGBoostModel()

    # Model evaluation
    model.eval()
    linear_mapping.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)  # x shape: (1, C, T, H, W)

            # Apply the linear mapping layer before passing to the Prithvi model
            x_mapped = linear_mapping(x)

            # Get features for XGBoost
            features = get_features(model, x_mapped, device, num_frames)
            
            # Reshape or flatten features to match XGBoost input format (num_samples, num_features)
            #features_flattened = features.reshape(features.shape[0], -1).cpu().numpy()
            features_flattened = features.reshape(features.shape[0], -1).cpu().numpy()

            # Train the XGBoost model or make predictions
            xgb_model.train(model, x_mapped, device, y, num_frames=3, num_boost_round=100)

            # Making predictions
            predictions = xgb_model.predict(features_flattened)
            print(f"Predictions: {predictions}")

            # Optionally, visualize outputs from the MAE model
            rec_img, mask_img = run_model(model, x_mapped, mask_ratio=0.5, device=device)
            print(f"Reconstructed shape: {rec_img.shape}")
            print(f"Mask shape: {mask_img.shape}")
            visualize_mae_outputs(x, mask_img, rec_img, bands=[3, 2, 1])

            # Uncomment for one batch test
            break  # Only run on one batch

if __name__ == "__main__":
    main()

