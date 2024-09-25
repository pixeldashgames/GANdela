import numpy as np
import numpy as np
import config
import numpy as np
import torch
from pix2pix.Generator import Generator
import matplotlib.pyplot as plt

import os
from DataSet.dataset import Satellite2Map_Data
from torch.utils.data import DataLoader


def get_folder_names(folder_path):
    folder_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names


def load_model(model, folder_path, lr):
    try:
        checkpoint = torch.load(
            f"{folder_path}/399_{config.CHECKPOINT_GEN}", map_location=config.DEVICE
        )
        model.load_state_dict(checkpoint["state_dict"])
    except:
        checkpoint = torch.load(
            f"{folder_path}/699_{config.CHECKPOINT_GEN}", map_location=config.DEVICE
        )
        model.load_state_dict(checkpoint["state_dict"])


def calculate_rmse(matrix1, matrix2):

    # Ensure matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape.")

    squared_differences = (matrix1 - matrix2) ** 2
    mean_squared_error = np.mean(squared_differences)
    rmse = np.sqrt(mean_squared_error)

    return rmse


def main():
    folder_path = "./versions (Gen-Disc)"

    val_dataset = Satellite2Map_Data(root=config.VAL_DIR, rgb_on=False)
    val_dl = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    tests = []

    for i, batch in enumerate(val_dl):
        if i >= 1:  # Stop after collecting 1 batch
            break
        tests = batch

    netG = Generator(in_channels=1).to(config.DEVICE)

    load_model(netG, f"{folder_path}/L1 with L1", config.LEARNING_RATE_GEN)

    errors = []

    for i in range(len(tests[0])):
        x, y = tests[0][i], tests[1][i]
        x, y = x.to(config.DEVICE).float().unsqueeze(1), y.to(
            config.DEVICE
        ).float().unsqueeze(1)
        netG.eval()
        with torch.no_grad():
            y_fake = netG(x)
        x = x.cpu().detach().numpy().squeeze()
        y_fake = y_fake.cpu().detach().numpy().squeeze()
        y = y.cpu().detach().numpy().squeeze()
        errors.append(((x, y, y_fake), calculate_rmse(y, y_fake)))

    indexed_list = list(enumerate(errors))
    sorted_list = sorted(indexed_list, key=lambda x: x[1][1])

    import cv2

    for i in range(5):
        best_x, best_y, best_y_gen = sorted_list[i][1][0]

        print(sorted_list[i][1][1])
        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Plot the input data (grayscale)
        axes[0].imshow(best_x, cmap="gray")
        axes[0].set_title("Input (Grayscale)")
        # Plot the label data
        axes[1].imshow(best_y, cmap="gray")
        axes[1].set_title("Label")
        # Plot the generated output
        axes[2].imshow(best_y_gen, cmap="gray")
        axes[2].set_title("Generated Output")
        # Adjust spacing between subplots
        plt.tight_layout()

        output_folder = "./photos"
        os.makedirs(output_folder, exist_ok=True)

        # Define the filename based on the iteration
        filename = f"plot_{i+1}.png"
        filepath = os.path.join(output_folder, filename)

        # Save the plot
        plt.savefig(filepath)
        plt.close(fig)


if __name__ == "__main__":
    main()
