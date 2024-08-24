import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from PIL import Image


def visualize_data(folder_path, input_file, label_file, y_gen_file):
    """Visualizes the input, label, and generated output from separate pickle files."""

    with open(os.path.join(folder_path, input_file), "rb") as f:
        input_data = pickle.load(f)

    with open(os.path.join(folder_path, label_file), "rb") as f:
        label_data = pickle.load(f)

    with open(os.path.join(folder_path, y_gen_file), "rb") as f:
        y_gen_data = pickle.load(f)

    # Select the first photo
    input_data = input_data[0]
    label_data = label_data[0]
    y_gen_data = y_gen_data[0]

    # Reshape using the original shape
    input_data = input_data.transpose(1, 2, 0)
    label_data = label_data.reshape(label_data.shape[:-1])
    y_gen_data = y_gen_data.squeeze(0)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the input data
    axes[0].imshow(input_data, cmap="gray")
    axes[0].set_title("Input")

    # Plot the label data
    axes[1].imshow(label_data, cmap="gray")
    axes[1].set_title("Label")

    # Plot the generated output
    axes[2].imshow(y_gen_data, cmap="gray")
    axes[2].set_title("Generated Output")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    folder_path = "./evaluation"
    numbers = [0, 100, 200, 300, 400]

    for number in numbers:
        input_file = f"input_{number}.pkl"
        label_file = f"label_{number}.pkl"
        y_gen_file = f"y_gen_{number}.pkl"
        visualize_data(folder_path, input_file, label_file, y_gen_file)

    pass
