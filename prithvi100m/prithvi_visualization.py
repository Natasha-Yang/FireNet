import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch
import yaml

# Constants used in the functions
NO_DATA = 0
NO_DATA_FLOAT = -9999.0
PERCENTILES = (2, 98)


def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        img = src.read()
        img = img[:6]  # load first 6 bands
        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    return img

def enhance_raster_for_visualization(raster, ref_img=None):
    if ref_img is None:
        ref_img = raster
    channels = []
    for channel in range(raster.shape[0]):
        valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        normalized_raster = (raster[channel] - mins) / (maxs - mins)
        normalized_raster[~valid_mask] = 0
        clipped = np.clip(normalized_raster, 0, 1)
        channels.append(clipped)
    clipped = np.stack(channels)
    channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
    rgb = channels_last[..., ::-1]
    return rgb


# Load the TIFF file (change crop size or remove it if needed)
original0 = load_raster("../prithvi/output/original_rgb_t0.tiff")
pred0 = load_raster("../prithvi/output/predicted_rgb_t0.tiff")
masked0 = load_raster("../prithvi/output/masked_rgb_t0.tiff")

original1 = load_raster("../prithvi/output/original_rgb_t1.tiff")
pred1 = load_raster("../prithvi/output/predicted_rgb_t1.tiff")
masked1 = load_raster("../prithvi/output/masked_rgb_t1.tiff")

original2 = load_raster("../prithvi/output/original_rgb_t2.tiff")
pred2 = load_raster("../prithvi/output/predicted_rgb_t2.tiff")
masked2 = load_raster("../prithvi/output/masked_rgb_t2.tiff")

# Organize into rows for plotting
images = [
    [original0, masked0, pred0],
    [original1, masked1, pred1],
    [original2, masked2, pred2],
]

titles = [
    ["Original t0", "Masked t0", "Predicted t0"],
    ["Original t1", "Masked t1", "Predicted t1"],
    ["Original t2", "Masked t2", "Predicted t2"],
]

# Plot
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row in range(3):
    for col in range(3):
        img = enhance_raster_for_visualization(images[row][col])
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(titles[row][col])
        ax.axis("off")

plt.tight_layout()
plt.savefig("demo.png")
plt.show()