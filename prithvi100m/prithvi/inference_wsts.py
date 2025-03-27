from prithvi_dataloader import FireNetDataset
from prithvi_mae import PrithviMAE
import torch
import torch.nn as nn
import yaml

class LinearMappingLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LinearMappingLayer, self).__init__()
        # Use Conv3d to map input channels to output channels
        self.linear = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        return self.linear(x)

def run_model(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    mask_ratio: float,
    device: torch.device,
):
    with torch.no_grad():
        x = input_data.to(device)

        _, pred, mask = model(x, mask_ratio=mask_ratio)

    # Create mask and prediction images (un-patchify) → KEEP ON GPU
    mask_img = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]))
    pred_img = model.unpatchify(pred)

    # Mix visible and predicted patches on GPU
    rec_img = input_data.clone().to(device)
    rec_img[mask_img == 1] = pred_img[mask_img == 1]

    # Convert mask to binary visualization format
    mask_img = (~(mask_img.to(torch.bool))).to(torch.float)  # swaps 0s and 1s
    # 1 = keep, 0 = remove

    # Move outputs to CPU for saving or visualization
    return rec_img.detach().cpu(), mask_img.detach().cpu()

import matplotlib.pyplot as plt
import torch

def enhance_for_display(img):
    img = img.clone()
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    return img.clamp(0, 1)

def visualize_mae_outputs(x, mask_img, rec_img, bands=[3, 2, 1]):
    """
    Visualizes original, masked, and reconstructed images from a masked autoencoder.
    mask_img is expected to be 1 where patches were masked, and 0 where visible.
    All tensors: (1, C, T, H, W)
    """
    x = x.squeeze(0).cpu()           # (C, T, H, W)
    mask_img = mask_img.squeeze(0).cpu()
    rec_img = rec_img.squeeze(0).cpu()

    C, T, H, W = x.shape
    fig, axes = plt.subplots(T, 3, figsize=(12, 4 * T))

    for t in range(T):
        # Extract the time slice and RGB bands
        original = enhance_for_display(x[bands, t].permute(1, 2, 0))  # (H, W, 3)
        reconstructed = enhance_for_display(rec_img[bands, t].permute(1, 2, 0))

        # Build a visual mask from mask_img (0=visible, 1=masked)
        visual_mask = mask_img[bands, t]  # (3, H, W)
        visual_mask = visual_mask.permute(1, 2, 0)  # (H, W, 3)

        masked = original * visual_mask

        axes[t, 0].imshow(original)
        axes[t, 0].set_title(f"Original t={t}")
        axes[t, 0].axis("off")

        axes[t, 1].imshow(masked)
        axes[t, 1].set_title(f"Masked input t={t}")
        axes[t, 1].axis("off")

        axes[t, 2].imshow(reconstructed)
        axes[t, 2].set_title(f"Reconstructed t={t}")
        axes[t, 2].axis("off")

    plt.tight_layout()
    plt.savefig("prithvi_wsts.png")
    plt.show()

def main():
    checkpoint = "prithvi/Prithvi_EO_V1_100M.pt"

    model_config_path = "prithvi/config.json"
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)['pretrained_cfg']
    
    batch_size = 1
    bands = model_config['bands']
    num_frames = 3
    img_size = model_config['img_size']

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
    test_loader = dataset.test_dataloader()

    # Create model and load checkpoint -------------------------------------------------------------

    model_config.update(
        num_frames=num_frames,
        in_chans=len(bands),
    )
    mask_ratio = model_config['mask_ratio']

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

    # Running model --------------------------------------------------------------------------------
    
    # Create the linear mapping layer and add it before the model
    linear_mapping = LinearMappingLayer(input_channels=40, output_channels=6).to(device)

    model.eval()
    linear_mapping.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)  # x shape: (1, C, T, H, W)
            x_mapped = linear_mapping(x)
            rec_img, mask_img = run_model(model, x_mapped, mask_ratio=mask_ratio, device=device)
            print(f"Reconstructed shape: {rec_img.shape}")
            print(f"Mask shape: {mask_img.shape}")
            print(f"Mask ratio: {mask_ratio}")
            visualize_mae_outputs(x_mapped, mask_img, rec_img, bands=[3, 2, 1])

            break  # Only run on one batch

if __name__ == '__main__':
    main()
