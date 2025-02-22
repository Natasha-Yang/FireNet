from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from transformers import AutoImageProcessor, ViTModel
import torch

from firenet.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")

    # define ViT
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    # change the number of input channels to 12
    model.config.num_channels = 12
    new_conv_layer = torch.nn.Conv2d(in_channels = 12,
                                     out_channels = model.embeddings.patch_embeddings.projection.hidden_size,
                                     kernel_size = model.embeddings.patch_embeddings.projection.patch_size,
                                     stride = model.embeddings.patch_embeddings.projection.stride)
    model.embeddings.patch_embeddings.projection = new_conv_layer


    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
