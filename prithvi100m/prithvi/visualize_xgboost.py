import numpy as np
import matplotlib.pyplot as plt

# Load both predictions and ground truth
def load_predictions_and_ground_truth(pred_filename="saved_predictions.npy", gt_filename="saved_ground_truth.npy"):
    predictions = np.load(pred_filename)
    ground_truth = np.load(gt_filename)
    return predictions, ground_truth

# Plot predictions and ground truth
def plot_predictions(predictions, ground_truth):
    print(f"Predictions shape before reshaping: {predictions.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    # Reshape predictions to match ground truth
    num_maps = ground_truth.shape[0]  # Number of segmentation maps
    height, width = ground_truth.shape[1:]  # Spatial dimensions
    predictions = predictions.reshape(num_maps, height, width)

    print(f"Predictions shape after reshaping: {predictions.shape}")

    plt.figure(figsize=(10, 5))

    # Ground truth
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(ground_truth[0], cmap='gray')  # Assuming single-channel segmentation map
    plt.axis('off')

    # Predictions
    plt.subplot(1, 2, 2)
    plt.title("Predictions")
    plt.imshow(predictions[0], cmap='gray')  # Assuming single-channel segmentation map
    plt.axis('off')

    plt.show()

def main():
    # Load the saved predictions and ground truth
    predictions, ground_truth = load_predictions_and_ground_truth("saved_predictions.npy", "saved_ground_truth.npy")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    # Plot the predictions and ground truth
    plot_predictions(predictions, ground_truth)

if __name__ == '__main__':
    main()
