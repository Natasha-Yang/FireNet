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
        inputs, labels = [], []
        for sample in sequence:
            input = torch.stack([
                torch.tensor(sample[feature], dtype=torch.float32).reshape(64, 64)
                for feature in INPUT_FEATURES
            ], dim = 0)

            if self.transform:
                input = self.transform(input)

            label = torch.tensor(sample["FireMask"], dtype = torch.float32).reshape(64, 64)

            inputs.append(input)
            labels.append(label)

        inputs = torch.stack(inputs, dim = 0)  # (T, num_channels, 64, 64)
        labels = torch.stack(labels, dim = 0)

        return inputs, label # (T, 64, 64)
        

def compute_stats(dataloader, num_channels):
    sum_channels = torch.zeros(num_channels)
    sum_sq_channels = torch.zeros(num_channels)
    num_pixels = 0
    
    for inputs, _ in tqdm(dataloader, desc = "Computing Mean & Std"):
        # inputs shape: (batch_size, num_channels, height, width)
        print(inputs.shape)
        batch_pixels = inputs.shape[0] * inputs.shape[1] * inputs.shape[3] * inputs.shape[4]  # batch_size * H * W
        num_pixels += batch_pixels

        sum_channels += inputs.sum(dim=[0, 1, 3, 4])  # sum over batch, height, width
        sum_sq_channels += (inputs ** 2).sum(dim=[0, 1, 3, 4])

    mean = sum_channels / num_pixels
    std = torch.sqrt((sum_sq_channels / num_pixels) - (mean ** 2))

    
    return mean.tolist(), std.tolist()


def get_data_loader(dataset_path, input_features, output_features, description, mean, 
                    std, batch_size, shuffle):
    
    dataset = TFRecordDataset(dataset_path, index_path=None, description = description)
    normalize = transforms.Normalize(mean, std)

    dataset = NDWS_Dataset(dataset, input_features, output_features, transform = normalize)

    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    
def collate_fn(batch):

    inputs, labels = zip(*batch)  # Unpack batch into lists
    
    # Get original sequence lengths
    sequence_lengths = torch.tensor([seq.shape[0] for seq in inputs], dtype=torch.long)  # (batch_size,)

    # Pad sequences along time dimension (T)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # (batch_size, max_T, num_channels, 64, 64)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)  # (batch_size, max_T, 1, 64, 64)

    return padded_inputs, padded_labels, sequence_lengths



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

    # extract train, val, test files
    train_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*train_*.tfrecord"))
    val_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*eval_*.tfrecord"))
    test_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*test_*.tfrecord"))

    # create datasets
    train_datasets, val_datasets, test_datasets = [], [], []
    for file in train_file_paths:
        data = TFRecordDataset(file, index_path=None, description = description)
        train_datasets.append(data)
    

    for file in val_file_paths:
        data = TFRecordDataset(val_file_paths[0], index_path=None, description = description)
        val_datasets.append(data)
    
    
    for file in test_file_paths:
        test_data = TFRecordDataset(test_file_paths[0], index_path=None, description = description)
        test_datasets.append(test_data)
    


    # load the mean and standard deviation of data
    with open("channel_stats.json", "r") as f:
            stats = json.load(f)

    # prepare dataloaders
    normalize_train = transforms.Normalize(stats["mean_train"], stats["std_train"])
    train_loader = DataLoader(NDWS_Dataset(train_datasets, INPUT_FEATURES, OUTPUT_FEATURES,
                                           normalize_train),
                              batch_size = 1,
                              shuffle=False,
                              collate_fn = collate_fn)
    
    normalize_val = transforms.Normalize(stats["mean_val"], stats["std_val"])
    val_loader = DataLoader(NDWS_Dataset(val_datasets, INPUT_FEATURES, OUTPUT_FEATURES,
                                         normalize_val),
                            batch_size = 1,
                            shuffle=False,
                            collate_fn = collate_fn)
    
    normalize_test = transforms.Normalize(stats["mean_test"], stats["std_test"])
    test_loader = DataLoader(NDWS_Dataset(test_datasets, INPUT_FEATURES, OUTPUT_FEATURES,
                                          normalize_test),
                             batch_size = 1,
                             shuffle=False,
                             collate_fn = collate_fn)
    
    # from util import check_batch_shape
    # check_batch_shape(test_loader)


    
        
        






    