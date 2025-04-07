import torch
import torch.nn as nn
import yaml
from inference import load_example
from einops import rearrange
import numpy as np
from prithvi_mae import PrithviMAE
from prithvi_dataloader import FireNetDataset
from inference_wsts import LinearMappingLayer
import matplotlib.pyplot as plt


def get_features(model, input_data, device, num_frames = 3):
    '''Get embeddings'''

    # input_data dimensions: (B, C, T, H, W)
    with torch.no_grad():
        x = input_data.to(device)
        features = model.forward_features(x)[-1]
    # remove cls token
    features = features[:, 1:, :] 
    B, _, C = features.shape
    # reshape
    H = int(np.sqrt(features.shape[1] / num_frames))
    reshaped_features = features.reshape(B, num_frames, H, H, C)
    reshaped_features = reshaped_features.permute(0, 4, 1, 2, 3) # (B, embedding dimension, T, H, W)
    
    return reshaped_features 




def visualize_embedding_channels(embeddings, time_idx=0, num_channels=6):
    """
    embeddings: (1, 768, T, H, W)
    Visualizes first `num_channels` channels at a given time step
    """
    emb = embeddings.squeeze(0)  # → (768, T, H, W)
    emb_t = emb[:, time_idx]     # → (768, H, W)

    fig, axes = plt.subplots(1, num_channels, figsize=(3 * num_channels, 3))
    for i in range(num_channels):
        axes[i].imshow(emb_t[i].cpu(), cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"embeddings.png")
    plt.show()


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
    #train_loader = dataset.train_dataloader()
    #val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    # Create model and load checkpoint -------------------------------------------------------------
    model_config_path = "prithvi/config.json"
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)['pretrained_cfg']

    bands = model_config['bands']
    num_frames = 3

    model_config.update(
        num_frames=num_frames,
        in_chans=len(bands),
    )

    model = PrithviMAE(**model_config)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")

    model.to(device)

    state_dict = torch.load(checkpoint, map_location=device)
    # discard fixed pos_embedding weight
    for k in list(state_dict.keys()):
        if 'pos_embed' in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint}")

    linear_mapping = LinearMappingLayer(input_channels=40, output_channels=6).to(device)

    # Get features
    features = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)  # x shape: (1, C, T, H, W)
            x_mapped = linear_mapping(x)
            feature = get_features(model, x_mapped, device) # (B, embedding dimension, T, H, W)
            visualize_embedding_channels(feature, num_channels = 20)
            break
    

if __name__ == "__main__":
    main()