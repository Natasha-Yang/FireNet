import os
import h5py
import numpy as np
from tqdm import tqdm

def check_nans_per_channel_from_hdf5(data_dir: str, dataset_key: str = "data"):
    """
    Check for NaNs in x (T-1, C, H, W) and y (H, W) in all HDF5 files in a directory.
    Print which channels in x have NaNs.
    """
    h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5") or f.endswith(".hdf5")]

    for filename in tqdm(h5_files, desc="Checking HDF5 files"):
        file_path = os.path.join(data_dir, filename)

        with h5py.File(file_path, "r") as f:
            if dataset_key not in f:
                print(f"Warning: '{dataset_key}' not found in {filename}")
                continue

            imgs = f[dataset_key][()]  # shape: (T, C, H, W)
            if imgs.ndim != 4:
                print(f"Unexpected shape in {filename}: {imgs.shape}")
                continue

            x = imgs[:-1]         # (T-1, C, H, W)
            y = imgs[-1, -1, ...] # (H, W)

            T_minus_1, C, H, W = x.shape
            x = x.transpose(1, 0, 2, 3).reshape(C, -1)  # (C, T-1*H*W)

            x_nan_flags = np.isnan(x).any(axis=1)
            y_has_nan = np.isnan(y).any()

            if np.any(x_nan_flags) or y_has_nan:
                print(f"\n⚠️ NaNs found in file: {filename}")
                for i, has_nan in enumerate(x_nan_flags):
                    if has_nan:
                        print(f"  - x Channel {i}: has NaNs")
                if y_has_nan:
                    print("  - y: has NaNs")



def accumulate_stats(data_dir, dataset_key="data"):
    sum_per_channel = None
    sq_sum_per_channel = None
    count_per_channel = None

    h5_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".h5") or f.endswith(".hdf5"))

    for filename in tqdm(h5_files, desc=f"Processing {os.path.basename(data_dir)}"):
        file_path = os.path.join(data_dir, filename)

        with h5py.File(file_path, "r") as f:
            if dataset_key not in f:
                print(f"'{dataset_key}' not in {filename}, skipping.")
                continue

            data = f[dataset_key][()]  # Shape: (T, C, H, W)
            if data.ndim != 4:
                print(f"Unexpected shape in {filename}: {data.shape}, skipping.")
                continue

            x = data[:-1]  # Shape: (T-1, C, H, W)
            T, C, H, W = x.shape
            x = x.transpose(1, 0, 2, 3).reshape(C, -1)  # Shape: (C, T*H*W)

            if sum_per_channel is None:
                sum_per_channel = np.zeros(C, dtype=np.float64)
                sq_sum_per_channel = np.zeros(C, dtype=np.float64)
                count_per_channel = np.zeros(C, dtype=np.int64)

            valid_mask = ~np.isnan(x)
            sum_per_channel += np.where(valid_mask, x, 0).sum(axis=1)
            sq_sum_per_channel += np.where(valid_mask, x**2, 0).sum(axis=1)
            count_per_channel += valid_mask.sum(axis=1)

    return sum_per_channel, sq_sum_per_channel, count_per_channel

def compute_combined_mean_std(*dirs, dataset_key="data"):
    total_sum = None
    total_sq_sum = None
    total_count = None

    for d in dirs:
        s, s2, n = accumulate_stats(d, dataset_key)
        if total_sum is None:
            total_sum = s
            total_sq_sum = s2
            total_count = n
        else:
            total_sum += s
            total_sq_sum += s2
            total_count += n

    mean = total_sum / total_count
    std = np.sqrt(total_sq_sum / total_count - mean**2)
    return mean.astype(np.float32), std.astype(np.float32)

# === Run on train1 + train2 ===
train1_dir = "data/WildfireSpreadTS/processed/train1"
train2_dir = "data/WildfireSpreadTS/processed/train2"

combined_mean, combined_std = compute_combined_mean_std(train1_dir, train2_dir)

print("Combined Stats for train1 + train2")
print("Means:\n", combined_mean)
print("Stds:\n", combined_std)