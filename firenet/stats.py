import torch
from dataset import get_dataset, get_dataloader

import torchvision.transforms as transforms

from firenet.config import NDWS_RAW_DATA_DIR
import os
import glob
import json

from tqdm import tqdm

def compute_stats(dataloader, num_channels):
    '''
    Computes the mean and standard deviation of the dataset for each channel

    Args:
    dataloader: dataloader for a TFRecord dataset; assumes input data has shape
    (batch_size, num_channels, H, W)
    num_channels: number of input channels
    '''
    sum_channels = torch.zeros(num_channels)
    sum_sq_channels = torch.zeros(num_channels)
    num_pixels = 0
    
    for inputs, _ in tqdm(dataloader, desc = "Computing Mean & Std"):
        # inputs shape: (batch_size, T, num_channels, H, W)
        # batch_size * T * H * W
        batch_pixels = inputs.shape[0] * inputs.shape[1] * inputs.shape[3] * inputs.shape[4]
        num_pixels += batch_pixels

        sum_channels += inputs.sum(dim=[0, 1, 3, 4]) # sum over batch, time, height, width
        sum_sq_channels += (inputs ** 2).sum(dim=[0, 1, 3, 4])

    mean = sum_channels / num_pixels
    std = torch.sqrt((sum_sq_channels / num_pixels) - (mean ** 2))

    
    return mean.tolist(), std.tolist()



if __name__ == '__main__':

    INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                    'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']


    OUTPUT_FEATURES = ['FireMask']

    num_features = len(INPUT_FEATURES)

    description = {feature_name: "float" for feature_name in INPUT_FEATURES + OUTPUT_FEATURES}

    # load the mean and standard deviation of data
    with open("channel_stats_raw.json", "r") as f:
        stats = json.load(f)

    # extract train, val, test files
    train_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*train_*.tfrecord"))
    val_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*eval_*.tfrecord"))
    test_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*test_*.tfrecord"))

    transform = True
    if transform == True:
        train_transform = transforms.Normalize(stats["mean_train"], stats["std_train"])
        val_transform = transforms.Normalize(stats["mean_val"], stats["std_val"])
        test_transform = transforms.Normalize(stats["mean_test"], stats["std_test"])
    else:
        train_transform, val_transform, test_transform = None, None, None
    
    # create datasets
    train_dataset = get_dataset(train_file_paths, INPUT_FEATURES, OUTPUT_FEATURES, description, train_transform)
    val_dataset = get_dataset(val_file_paths, INPUT_FEATURES, OUTPUT_FEATURES, description, val_transform)
    test_dataset = get_dataset(test_file_paths, INPUT_FEATURES, OUTPUT_FEATURES, description, test_transform)
    
    # prepare dataloaders
    train_loader = get_dataloader(train_dataset, batch_size = 1, shuffle = False)
    val_loader = get_dataloader(val_dataset, batch_size = 1, shuffle = False)
    test_loader = get_dataloader(test_dataset, batch_size = 1, shuffle = False)

    mean_train, std_train = compute_stats(train_loader, num_features)
    mean_val, std_val = compute_stats(val_loader, num_features)
    mean_test, std_test = compute_stats(test_loader, num_features)

    stats = {"mean_train": mean_train, "std_train": std_train,
            "mean_val": mean_val, "std_val": std_val,
            "mean_test": mean_test, "std_test": std_test}

    with open("channel_stats_transformed.json", "w") as f:
        json.dump(stats, f, indent=4) 





