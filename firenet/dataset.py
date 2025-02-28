from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from transformers import AutoImageProcessor
from tfrecord.torch.dataset import TFRecordDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ChainDataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from firenet.config import NDWS_DATA_DIR, NDWS_RAW_DATA_DIR, NDWS_PROCESSED_DATA_DIR
import os
import glob
import json


app = typer.Typer()


@app.command()

class NDWS_Dataset(Dataset):
    '''
    Dataset class for the NextDayWildfireSpread dataset
    '''
    def __init__(self, datasets, input_features, output_features, transform = None):
        self.datasets = datasets
        self.input_features = input_features
        self.output_features = output_features
        self.num_channels = len(input_features) # 12
        self.transform = transform
    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, index):
        sequence = self.datasets[index]
        inputs, labels = [], [] # inputs and labels for the entire sequence
        for sample in sequence: # data for a single day within a sequence
            input = torch.stack([
                torch.tensor(sample[feature], dtype=torch.float32).reshape(64, 64)
                for feature in self.input_features # stack all 12 feature maps
            ], dim = 0) # (num_channels, 64, 64)

            if self.transform:
                input = self.transform(input)

            label = torch.tensor(sample["FireMask"], dtype = torch.float32).reshape(64, 64)

            inputs.append(input)
            labels.append(label)

        inputs = torch.stack(inputs, dim = 0)  # (T, num_channels, 64, 64)
        labels = torch.stack(labels, dim = 0) # (T, 64, 64)

        return inputs, labels


def get_dataset(file_paths, input_features, output_features, description, transform = None):
    '''
    Returns a NDWS dataset

    Args:
    file_paths: a list of TFRecord file paths
    input_features: a list of input feature names
    output_features: a list of output feature names
    normalize: a boolean value that indicates whether to normalize data
    '''

    combined_ds = []
    for file in file_paths:
        ds = TFRecordDataset(file, index_path=None, description = description)
        combined_ds.append(ds)
    
    return NDWS_Dataset(combined_ds, input_features, output_features, transform)
    
    


def get_dataloader(dataset, batch_size, shuffle):

    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn)


def collate_fn(batch):
    '''
    Pads sequences so that they are the same length
    '''

    inputs, labels = zip(*batch)  # Unpack batch into lists

    # Pad sequences along time dimension (T)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # (batch_size, max_T, num_channels, 64, 64)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)  # (batch_size, max_T, 1, 64, 64)

    return padded_inputs, padded_labels



def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = NDWS_RAW_DATA_DIR / "dataset.csv",
    output_path: Path = NDWS_PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    # load raw dataset

    #logger.info("Processing dataset...")
    #for i in tqdm(range(10), total=10):
        #processed_inputs = image_processor(i, return_tensors="pt")
    #logger.success("Processing dataset complete.")
    # -----------------------------------------
    return




if __name__ == "__main__":
    #app()

    INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

    OUTPUT_FEATURES = ['FireMask']

    num_features = len(INPUT_FEATURES)

    description = {feature_name: "float" for feature_name in INPUT_FEATURES + OUTPUT_FEATURES}

    # load the mean and standard deviation of data
    with open("channel_stats.json", "r") as f:
            stats = json.load(f)

    # extract train, val, test files
    train_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*train_*.tfrecord"))
    val_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*eval_*.tfrecord"))
    test_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*test_*.tfrecord"))

    # create datasets
    train_transform = transforms.Normalize(stats["mean_train"], stats["std_train"])
    train_dataset = get_dataset(train_file_paths, INPUT_FEATURES, OUTPUT_FEATURES, description, train_transform)

    val_transform = transforms.Normalize(stats["mean_val"], stats["std_val"])
    val_dataset = get_dataset(val_file_paths, INPUT_FEATURES, OUTPUT_FEATURES, description, val_transform)

    test_transform = transforms.Normalize(stats["mean_test"], stats["std_test"])
    test_dataset = get_dataset(test_file_paths, INPUT_FEATURES, OUTPUT_FEATURES, description, True)

    # prepare dataloaders
    train_loader = get_dataloader(train_dataset, batch_size = 1, shuffle = False)
    val_loader = get_dataloader(val_dataset, batch_size = 1, shuffle = False)
    test_loader = get_dataloader(test_dataset, batch_size = 1, shuffle = False)
    
    # from util import check_batch_shape
    # check_batch_shape(test_loader)


    
        
        






    