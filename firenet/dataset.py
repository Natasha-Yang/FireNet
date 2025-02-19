from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from firenet.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------

import kagglehub


if __name__ == "__main__":
    #app()
    import tensorflow as tf

    file_path = "path/to/your/file.tfrecord"

    # Define the expected structure of TFRecord features
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),  # Image stored as raw bytes
        "label": tf.io.FixedLenFeature([], tf.int64)    # Label stored as integer
    }

    def _parse_function(proto):
        """Parses a single example from TFRecord."""
        parsed = tf.io.parse_single_example(proto, feature_description)
        
        # Decode the image
        image = tf.io.decode_jpeg(parsed["image"])  # Change to decode_png if needed
        label = parsed["label"]

        return image, label

    # Load dataset
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(_parse_function)

