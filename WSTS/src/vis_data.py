
from dataloader.FireSpreadDataset import FireSpreadDataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable  

def convert_feature_names(base_feature_names):
    feature_names = {}
    for i, name in enumerate(base_feature_names):
        feature_names[name] = name
    return feature_names

# Feature lists
base_feature_names = [
    'VIIRS band M11',
    'VIIRS band I2',
    'VIIRS band I1',
    'NDVI',
    'EVI2',
    'Total precipitation',
    'Wind speed',
    'Wind direction',
    'Minimum temperature',
    'Maximum temperature',
    'Energy release component',
    'Specific humidity',
    'Slope',
    'Aspect',
    'Elevation',
    'Palmer drought severity index (PDSI)',
    'Landcover class',
    'Forecast: Total precipitation',
    'Forecast: Wind speed',
    'Forecast: Wind direction',
    'Forecast: Temperature',
    'Forecast: Specific humidity',
    'Active fire']

# Different land cover classes of feature "Landcover class"
land_cover_classes = [
    'Land cover: Evergreen Needleleaf Forests',
    'Land cover: Evergreen Broadleaf Forests',
    'Land cover: Deciduous Needleleaf Forests',
    'Land cover: Deciduous Broadleaf Forests',
    'Land cover: Mixed Forests',
    'Land cover: Closed Shrublands',
    'Land cover: Open Shrublands',
    'Land cover: Woody Savannas',
    'Land cover: Savannas',
    'Land cover: Grasslands',
    'Land cover: Permanent Wetlands',
    'Land cover: Croplands',
    'Land cover: Urban and Built-up Lands',
    'Land cover: Cropland/Natural Vegetation Mosaics',
    'Land cover: Permanent Snow and Ice',
    'Land cover: Barren',
    'Land cover: Water Bodies']

# return_features = base_feature_names[:16] + land_cover_classes + base_feature_names[17:] + ["Active fire (binary)"]

# Convert feature names properly
feature_names = convert_feature_names(base_feature_names)
print("feature_names.keys():", feature_names.keys())

# Load dataset
dataset = FireSpreadDataset(
    data_dir='../data/WildfireSpreadTS/processed',
    included_fire_years=[2021],
    n_leading_observations=5,
    crop_side_length=64,
    load_from_hdf5=True,
    # is_train=True,
    is_train=False,
    features_to_keep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                      32, 33, 34, 35, 36, 37, 38], # not including land cover classes
    remove_duplicate_features=False,
    stats_years=[2018, 2019],
    return_doy=True
)

def plot_sample(x, y, t, feature_names):
    # Plot input data (alll features in a separate subplot)
    plt.figure(figsize=(12, 6))
    # for i in range(x.shape[0]):
    for i, name in enumerate(feature_names):
        # plt.subplot(5, 5, i + 1)
        plt.subplot(4, 6, i + 1)
        plt.imshow(x[i, t, :, :], cmap='viridis')
        # plt.title(feature_names[name])
        plt.title(name)
        plt.axis('off')

    plt.subplot(4, 6, 24)
    plt.imshow(y, cmap='gray')
    plt.title("Fire Mask (Label)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Plot sample
for t in range(5):
    sample = dataset[t]
    # print("sample type:", type(sample)) # tuple
    # print("sample length:", len(sample)) # 3
    x, y, doys = sample
    # print("x.shape:", x.shape)
    # print("y.shape:", y.shape)
    plot_sample(x, y, t, feature_names)
