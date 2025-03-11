INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                        'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

OUTPUT_FEATURES = ['FireMask']

num_features = len(INPUT_FEATURES)

description = {feature_name: "float" for feature_name in INPUT_FEATURES + OUTPUT_FEATURES}

# th: wind direction
# vs: wind speed
# tmmn: min temp
# tmmx: max temp
# sph: humidity
# pr: precipitation
# pdsi: drought index
# NDVI: vegetation index (high = grass, low = snow or ice)
# erc: energy release component (available energy per unit area within the flaming front of a fire)

# topography: INPUT_FEATURES[0]
# weather: IPUT_FEATURES[1:5]
# weather & humidity: INPUT_FEATURES[5:8]
# fuel: INPUT_FEATURES[8:11]
# fire history: INPUT_FEATURES[11]