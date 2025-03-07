import json

def save_channel_stats(INPUT_FEATURES, mean_per_channel, std_per_channel):
    mean = dict()
    std = dict()
    channel_stats = {"mean": mean, "std": std}
    for i in range(len(INPUT_FEATURES)):
        channel_stats["mean"][INPUT_FEATURES[i]] = mean_per_channel[i]
        channel_stats["std"][INPUT_FEATURES[i]] = std_per_channel[i]
    print(channel_stats)

    with open("channel_stats.json", "w") as f:
        json.dump(channel_stats, f, indent=4) 
    print("Saved channel statistics to channel_stats.json")

def check_batch_shape(dataloader):
    batch = next(iter(dataloader))

    inputs, labels = batch  # (batch_size, T, num_channels, 64, 64)
    
    print("Padded Input Shape:", inputs.shape)  # (batch_size, max_T, num_channels, 64, 64)
    print("Padded Label Shape:", labels.shape)  # (batch_size, max_T, 64, 64)