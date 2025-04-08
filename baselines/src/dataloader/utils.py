from typing import List
import numpy as np
import os
import h5py
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



def get_means_stds_missing_values(training_years: List[int]):
    """_summary_ Returns mean and std values as tensor, computed on unaugmented and unstandardized 
    data of the indicated training years. We don't clip values, because min/max did not diverge 
    much from the 0.1 and 99.9 percentiles. Some variables are not standardized, indicated by mean=0, std=1. 
    These are specifically: All variables indicating a direction in degrees 
    (wind direction, aspect, forecast wind direction), and the categorical land cover type.

    Args:
        training_years (_type_): _description_

    Returns:
        _type_: _description_
    """

    stats_per_training_year_combo = {
        ("train1", "train2"): {
        'means': np.array([
            1.9084414e+03,  2.9467229e+03,  1.8178037e+03,  4.3490439e+03,
            2.2339529e+03,  4.8713881e-01,  3.4709952e+00,  2.2286052e+02,
            2.8317722e+02,  2.9958282e+02,  6.9358322e+01,  5.5317660e-03,
            7.3105764e+00,  1.7558824e+02,  1.4902290e+03, -2.0236363e+00,
            8.6291704e+00,  9.4792700e+00,  1.5613624e+00,  5.8093648e+00,
            1.8738892e+01,  5.4540909e-03,  2.6684472e-02], dtype=np.float32),
        'stds': np.array([
            1.1497416e+03, 1.6648529e+03, 1.9463845e+03, 2.2131379e+03, 1.0902662e+03,
            2.2098498e+00, 1.4194025e+00, 8.0483849e+01, 7.1991887e+00, 8.1728430e+00,
            1.9647594e+01, 2.1566851e-03, 6.9128695e+00, 1.0273334e+02, 7.5486322e+02,
            2.1915972e+00, 3.3333352e+00, 3.0396461e+01, 1.1140391e+00, 4.4170338e+01,
            7.0927715e+00, 1.8761724e-03, 6.6959363e-01], dtype=np.float32),
        'missing_values': np.array([
            0.02567231, 0.0256701, 0.02566863, 0.02243902, 0.02243973,
            0.01035774, 0.01035774, 0.01035774, 0.01035774, 0.01035774,
            0.01035774, 0.01035774, 0.00783755, 0.00767712, 0.00767712,
            0.01404397, 0., 0., 0., 0., 0., 0., 0.99896296], dtype=np.float32)
        }}

    years_tuple = tuple(training_years)
    means = stats_per_training_year_combo[years_tuple]["means"]
    stds = stats_per_training_year_combo[years_tuple]["stds"]
    missing_values = stats_per_training_year_combo[years_tuple]["missing_values"]

    # Zero out means and stds for degree-based features and the categorical land cover type variable
    features_to_not_standardize = get_indices_of_degree_features() + [16]

    means[features_to_not_standardize] = 0
    stds[features_to_not_standardize] = 1

    return means, stds, missing_values


def get_indices_of_degree_features():
    """
    :return: Indices of features that take values in [0,360] and thus will be transformed via sin

    """
    return [7, 13, 19]
