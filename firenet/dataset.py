from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from transformers import AutoImageProcessor
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader

from firenet.config import NDWS_PROCESSED_DATA_DIR, NDWS_RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = NDWS_RAW_DATA_DIR / "dataset.csv",
    output_path: Path = NDWS_PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # load raw dataset

    #logger.info("Processing dataset...")
    #for i in tqdm(range(10), total=10):
        #processed_inputs = image_processor(i, return_tensors="pt")
    #logger.success("Processing dataset complete.")
    # -----------------------------------------



if __name__ == "__main__":
    app()
    
