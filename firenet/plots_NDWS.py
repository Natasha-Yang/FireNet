from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import json

import matplotlib.pyplot as plt
import numpy as np

from firenet.config import FIGURES_DIR, NDWS_PROCESSED_DATA_DIR
from firenet.dataset_NDWS import *
from firenet.constants_NDWS import *

app = typer.Typer()

# data from https://allisonhorst.github.io/palmerpenguins/


def plot_mean_and_std(INPUT_FEATURES, stats, figure_name):
    # Create a figure with 3 rows and 4 columns (12 subplots total)
    fig, axes = plt.subplots(3, 4, figsize=(30, 40))
    axes = axes.flatten()  # Flatten for easier indexing

    # Bar width and x locations
    x = np.arange(3)  # 3 dataset splits (Train, Validation, Test)

    # Colors for bars
    colors = ['royalblue', 'darkorange', 'seagreen']
    labels = ['Train', 'Validation', 'Test']

    # Loop through each feature and create its subplot
    for i, feature in enumerate(INPUT_FEATURES):
        ax = axes[i]

        # Extract means and standard deviations for this feature
        means = [stats[split][i] for split in ['mean_train', 'mean_val', 'mean_test']]
        stds = [stats[split][i] for split in ['std_train', 'std_val', 'std_test']]

        # Compute y-axis limits to ensure scale is larger than standard deviation
        max_value = max(means) + max(stds) * 1.5  # Ensure std is accounted for
        min_value = min(0, min(means) - max(stds) * 1.5)  # Handle negative means

        # Create bars with smaller spacing and add error bars (standard deviation)
        for j in range(3):  # Loop through Train, Validation, Test
            ax.errorbar(x[j], means[j], yerr=stds[j], fmt='o', color=colors[j], capsize=5, markersize=8, label=labels[j])

        # Formatting
        ax.set_title(feature)
        ax.set_xticks(x)
        ax.set_xticklabels(['Train', 'Validation', 'Test'])
        ax.set_ylim(min_value, max_value)  # 🔹 Scale ensures std is visible

    # Adjust layout and add a global title
    fig.suptitle("Feature Means with Standard Deviation by Split", fontsize=16)
    fig.subplots_adjust(hspace=0.3, wspace=0.2)

    plt.savefig(figure_name)
    plt.show()


def plot_fire_masks(dataloader):
    colors = {1: [1, 0, 0],  # Red for active fire
              0: [0.5, 0.5, 0.5],  # Grey for no fire
              -1: [0, 0, 0]}  # Black for uncertain

    inputs, fire_masks = next(iter(dataloader))
    # (B, T, C, H, W)
    prev_fire_masks = inputs[0, :10, -1, :, :]  # Extract previous fire mask (T, H, W)

    assert len(prev_fire_masks.shape) == 3, "Expected (T, H, W) shape for prev_fire_masks"
    
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(25, 10))  # 2 rows, 10 columns
    print(prev_fire_masks.shape)
    print(fire_masks.shape)
    for t in tqdm(range(10), desc="Plotting fire masks"):
        # Extract the t-th fire mask and previous fire mask
        fire_mask_t = fire_masks[0, t, :, :]
        prev_fire_mask_t = prev_fire_masks[t, :, :]

        assert fire_mask_t.shape == (64, 64), "Expected fire_mask_t to have shape (64, 64)"
        assert prev_fire_mask_t.shape == (64, 64), "Expected prev_fire_mask_t to have shape (64, 64)"

        # Convert to RGB images
        rgb_fire_mask = np.zeros((64, 64, 3))
        rgb_prev_fire_mask = np.zeros((64, 64, 3))
        
        for value, color in colors.items():
            rgb_fire_mask[fire_mask_t == value] = color
            rgb_prev_fire_mask[prev_fire_mask_t == value] = color

        # Plot previous fire mask
        axes[0, t].imshow(rgb_prev_fire_mask)
        axes[0, t].set_title(f"Prev. Mask (t={t})")
        axes[0, t].axis("off")

        # Plot current fire mask
        axes[1, t].imshow(rgb_fire_mask)
        axes[1, t].set_title(f"Mask (t={t})")
        axes[1, t].axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig("fire")



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = NDWS_PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    
    # Load normalized channel stats
    with open("channel_stats_transformed.json", "r") as f:
        stats = json.load(f)

    # Feature names
    INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 
                    'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
    
    plot_mean_and_std(INPUT_FEATURES, stats, 'mean_std_normalized.png')


if __name__ == "__main__":
    #app()
    paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*eval_*.tfrecord"))
    dataset = get_dataset(paths, INPUT_FEATURES, OUTPUT_FEATURES, description)
    loader = get_dataloader(dataset, 1, shuffle = False)
    plot_fire_masks(loader)
    

    
