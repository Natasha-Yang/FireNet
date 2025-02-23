from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from transformers import AutoImageProcessor
from tfrecord.torch.dataset import TFRecordDataset
import torch
from torch.utils.data import Dataset, DataLoader
from data.NextDayWildfireSpread.stats import *

from firenet.config import NDWS_DATA_DIR, NDWS_RAW_DATA_DIR, NDWS_PROCESSED_DATA_DIR
import json




app = typer.Typer()


@app.command()

class NDWS_Dataset(Dataset):
    def __init__(self, dataset, input_features, output_features, dataset_size = 1000):
        self.dataset = dataset
        self.input_features = input_features
        self.output_features = output_features
        self.num_channels = len(input_features) # 12
        self.dataset_size = dataset_size # 1000
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index):
        sample = next(iter(self.dataset))
        inputs = []
        for i in self.input_features:
            # convert to tensor and reshape to (1, 64, 64)
            feature = torch.tensor(sample[i], dtype = torch.float32).reshape(1, 64, 64)
            inputs.append(feature)
        
        # concatenate the channels
        inputs = torch.cat(inputs, dim = 0) # (num_channels, 64, 64)

        label = torch.tensor(sample["FireMask"], dtype = torch.float32).reshape(64, 64)

        return inputs, label
        

def normalize(dataloader, num_channels):
    sum_channels = torch.zeros(num_channels)  # Sum per channel
    sum_sq_channels = torch.zeros(num_channels)  # Sum of squares per channel
    num_pixels = 0  # Total number of pixels

    # Iterate through dataset
    for inputs, _ in tqdm(dataloader, desc="Computing Mean & Std"):
        # inputs shape: (batch_size, num_channels, height, width)
        print(inputs.shape)
        batch_pixels = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]  # Batch * H * W
        num_pixels += batch_pixels

        sum_channels += inputs.sum(dim=[0, 2, 3])  # Sum over batch, height, width
        sum_sq_channels += (inputs ** 2).sum(dim=[0, 2, 3])  # Sum of squares

    # Compute mean and std
    mean = sum_channels / num_pixels
    std = torch.sqrt((sum_sq_channels / num_pixels) - (mean ** 2))

    # Print results
    print(f"Mean per channel: {mean.tolist()}")
    print(f"Std per channel: {std.tolist()}")


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
    #app()
    print(mean_per_channel)
    print(std_per_channel)


    INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

    mean = dict()
    std = dict()
    channel_stats = {"mean": mean, "std": std}
    for i in range(len(INPUT_FEATURES)):
        channel_stats["mean"][INPUT_FEATURES[i]] = mean_per_channel[i]
        channel_stats["std"][INPUT_FEATURES[i]] = std_per_channel[i]
    print(channel_stats)

    with open("channel_stats.json", "w") as f:
        json.dump(channel_stats, f, indent=4) 
    print("Saved channel statistics to channel_stats.json")

    '''
    OUTPUT_FEATURES = ['FireMask']


    description = {feature_name: "float" for feature_name in INPUT_FEATURES + OUTPUT_FEATURES}

    file_path = os.path.join(NDWS_RAW_DATA_DIR, 'next_day_wildfire_spread_eval_00.tfrecord')

    dataset = TFRecordDataset(file_path, index_path=None, description = description)
    dataset = NDWS_Dataset(dataset, INPUT_FEATURES, OUTPUT_FEATURES)

    dataloader = DataLoader(dataset, batch_size = 1000, shuffle = False)

    normalize(dataloader, len(INPUT_FEATURES))

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("  Inputs Shape:", inputs.shape) # [8, 768, 64, 1]
        print("  Labels Shape:", labels.shape) # [8, 64, 64, 1]
        break'''