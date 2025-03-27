import torch
import torch.nn as nn
import yaml
from inference import load_example
from einops import rearrange
import numpy as np
from prithvi_mae import PrithviMAE


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

def main():
    batch_size = 1
    bands = config['bands']
    num_frames = len(data_files)
    mean = config['mean']
    std = config['std']
    img_size = config['img_size']

    print(
        f"\nTreating {len(data_files)} files as {len(data_files)} time steps from the same location\n"
    )
    if len(data_files) != 3:
        print(
            "The original model was trained for 3 time steps (expecting 3 files). \nResults with different numbers of timesteps may vary"
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.\n")

    # Loading data ---------------------------------------------------------------------------------

    input_data, meta_data = load_example(
        file_paths=data_files, indices=input_indices, mean=mean, std=std
    )

    # Create model and load checkpoint -------------------------------------------------------------

    config.update(
        num_frames=num_frames,
        in_chans=len(bands),
    )

    model = PrithviMAE(**config)

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

    # Running model --------------------------------------------------------------------------------

    model.eval()

    # Reflect pad if not divisible by img_size
    original_h, original_w = input_data.shape[-2:]
    print(original_h, original_w)
    pad_h = img_size - (original_h % img_size)
    pad_w = img_size - (original_w % img_size)
    input_data = np.pad(
        input_data, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
    )

    # Build sliding window
    batch = torch.tensor(input_data, device="cpu")
    windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    h1, w1 = windows.shape[3:5]
    windows = rearrange(
        windows, "b c t h1 w1 h w -> (b h1 w1) c t h w", h=img_size, w=img_size
    )

    # Split into batches if number of windows > batch_size
    num_batches = windows.shape[0] // batch_size if windows.shape[0] > batch_size else 1
    windows = torch.tensor_split(windows, num_batches, dim=0)

    # Get features
    features = []
    for x in windows:
        # (B, embedding dimension, T, H, W)
        feature = get_features(model, x, device) 
        features.append(feature)
    features = torch.concat(features, dim=0)

if __name__ == "__main__":
    config_path = "config.json"
    checkpoint = "Prithvi_EO_V1_100M.pt"
    data_files = ["examples/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
                  "examples/HLS.L30.T13REN.2018029T172738.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
                  "examples/HLS.L30.T13REN.2018061T172724.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif"
                 ]
    input_indices = None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)['pretrained_cfg']
    main()