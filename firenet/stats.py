import torch
from dataset import *

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from constants import *

from firenet.config import *
import os
import glob
import json

from tqdm import tqdm

def compute_stats(dataloader, num_channels):
    '''
    Computes the mean and standard deviation for each channel of a dataset

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


def compute_class_proportions(dataloader):
    '''
    Computes the proportions of class labels in a dataset

    Args:
    dataloader: dataloader for a TFRecord dataset; assumes input data has shape
    (batch_size, num_channels, H, W)
    num_channels: number of input channels
    '''
    
    n_active, n_inactive, n_uncertain = 0, 0, 0

    for _, fire_masks in tqdm(dataloader, desc = "Computing Percentage of Pixels with Active Fires"):
        # batch_size * T * num_channels * H * W
        mask_size = fire_masks.shape[2] * fire_masks.shape[3]
        assert mask_size == 64 * 64
        n_active += torch.sum(fire_masks == 1).item()
        n_inactive += torch.sum(fire_masks == 0).item()
        n_uncertain += torch.sum(fire_masks == -1).item()
    
    total = n_active + n_inactive + n_uncertain

    return n_active / total, n_inactive / total, n_uncertain / total


def compute_stats_all_splits(transformed = False):
    '''
    Computes and saves the mean and standard deviation of all three splits
    '''
    
    if transformed: # get transformed datasets
        train_dataset, val_dataset, test_dataset = make_interim_datasets()
        save_filename = "reports/channel_stats_transformed.json"
    else: # get raw datasets
        train_dataset, val_dataset, test_dataset = make_raw_datasets()
        save_filename = "reports/channel_stats_raw.json"
    
    # prepare dataloaders
    train_loader, val_loader, test_loader = get_dataloaders_all_splits(train_dataset,
                                                                       val_dataset,
                                                                       test_dataset,
                                                                       batch_sz = 1,
                                                                       shuffle = False)
    
    # compute mean and standard deviation
    mean_train, std_train = compute_stats(train_loader, num_features)
    mean_val, std_val = compute_stats(val_loader, num_features)
    mean_test, std_test = compute_stats(test_loader, num_features)

    stats = {"mean_train": mean_train, "std_train": std_train,
            "mean_val": mean_val, "std_val": std_val,
            "mean_test": mean_test, "std_test": std_test}

    # save statistics
    with open(save_filename, "w") as f:
        json.dump(stats, f, indent=4) 



def compute_class_proportions_all_splits():
    '''
    Computes the class proportions of all three splits
    '''
    train_dataset, val_dataset, test_dataset = make_raw_datasets()
    train_loader, val_loader, test_loader = get_dataloaders_all_splits(train_dataset,
                                                                       val_dataset,
                                                                       test_dataset,
                                                                       batch_sz = 1,
                                                                       shuffle = False)
    
    print("Train:")
    print(compute_class_proportions(train_loader))
    print("Validation:")
    print(compute_class_proportions(val_loader))
    print("Test:")
    print(compute_class_proportions(test_loader))








if __name__ == '__main__':
    # computes and saves the mean and standard deviation of each channel after normalization
    # compute_stats_all_splits(transformed = True)
    compute_class_proportions_all_splits()
    
    

    
    
        
    
