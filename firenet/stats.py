
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader, ChainDataset
from dataset import NDWS_Dataset, compute_stats, collate_fn

from firenet.config import NDWS_RAW_DATA_DIR
import os
import glob
import json

INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']


OUTPUT_FEATURES = ['FireMask']

num_features = len(INPUT_FEATURES)


description = {feature_name: "float" for feature_name in INPUT_FEATURES + OUTPUT_FEATURES}

# extract train, val, test files
train_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*train_*.tfrecord"))
val_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*eval_*.tfrecord"))
test_file_paths = glob.glob(os.path.join(NDWS_RAW_DATA_DIR, "*test_*.tfrecord"))
print(test_file_paths)

# create datasets
train_datasets, val_datasets, test_datasets = [], [], []
for file in train_file_paths:
    data = TFRecordDataset(file, index_path=None, description = description)
    train_datasets.append(data)
if len(train_datasets) > 1:
    train_data = ChainDataset(train_datasets)

for file in val_file_paths:
    data = TFRecordDataset(val_file_paths[0], index_path=None, description = description)
    val_datasets.append(data)
if len(val_datasets) > 1:
    val_data = ChainDataset(val_datasets)

for file in test_file_paths:
    test_data = TFRecordDataset(test_file_paths[0], index_path=None, description = description)
    test_datasets.append(test_data)
if len(test_datasets) > 1:
    test_data = ChainDataset(test_datasets)


# prepare dataloaders
train_loader = DataLoader(NDWS_Dataset(train_datasets, INPUT_FEATURES, OUTPUT_FEATURES),
                          batch_size = 1,
                          shuffle=False,
                          collate_fn = collate_fn)
val_loader = DataLoader(NDWS_Dataset(val_datasets, INPUT_FEATURES, OUTPUT_FEATURES),
                        batch_size = 1,
                        shuffle=False,
                        collate_fn = collate_fn)
test_loader = DataLoader(NDWS_Dataset(test_datasets, INPUT_FEATURES, OUTPUT_FEATURES),
                         batch_size = 1,
                         shuffle=False,
                         collate_fn = collate_fn)


mean_train, std_train = compute_stats(train_loader, num_features)
mean_val, std_val = compute_stats(val_loader, num_features)
mean_test, std_test = compute_stats(test_loader, num_features)

stats = {"mean_train": mean_train, "std_train": std_train,
            "mean_val": mean_val, "std_val": std_val,
            "mean_test": mean_test, "std_test": std_test}

with open("channel_stats.json", "w") as f:
    json.dump(stats, f, indent=4) 





