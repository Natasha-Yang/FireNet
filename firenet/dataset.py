from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from transformers import AutoImageProcessor
from tfrecord.torch.dataset import TFRecordDataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from firenet.config import NDWS_RAW_DATA_DIR, NDWS_PROCESSED_DATA_DIR
import os

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
            # convert to tensor and reshape to (64, 64, 1)
            feature = torch.tensor(sample[i], dtype = torch.float32).reshape(64, 64, 1)
            inputs.append(feature)
        
        # concatenate the channels
        inputs = torch.cat(inputs, dim = 0) # (num_channels, 64, 64)

        label = torch.tensor(sample["FireMask"], dtype = torch.float32).reshape(64, 64, 1)

        return inputs, label
        




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


    INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
    OUTPUT_FEATURES = ['FireMask']


    description = {feature_name: "float" for feature_name in INPUT_FEATURES + OUTPUT_FEATURES}

    file_path = os.path.join(NDWS_RAW_DATA_DIR, 'next_day_wildfire_spread_eval_00.tfrecord')

    dataset = TFRecordDataset(file_path, index_path=None, description = description)
    dataset = NDWS_Dataset(dataset, INPUT_FEATURES, OUTPUT_FEATURES)

    data_loader = DataLoader(dataset, batch_size = 8, shuffle = False)

    
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print("  Inputs Shape:", inputs.shape) # [8, 768, 64, 1]
        print("  Labels Shape:", labels.shape) # [8, 64, 64, 1]
        break