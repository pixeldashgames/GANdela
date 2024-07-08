import torch
from utils.utils import output_transform, load_checkpoint, tif_to_ndarray

import config

from pix2pix.Generator import Generator
from torchvision import transforms
import os
import sys


if __name__ == '__main__':

    netG = Generator(in_channels=3).to(config.DEVICE)
    optimizerG = torch.optim.Adam(netG.parameters(), lr = config.LEARNING_RATE, betas=(config.BETA1, 0.999))

    load_checkpoint(config.CHECKPOINT_GEN, netG, optimizerG, config.LEARNING_RATE)

    file_path = sys.argv[1] if len(sys.argv) > 1 else "default_file_path"

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)


    transform = transforms.ToTensor()
    satellite_image = transform(tif_to_ndarray(file_path))
    satellite_image = satellite_image.to(config.DEVICE)

    y = netG(satellite_image)

    print(y.shape)

    y_reduced = output_transform(y)["image"]

    print(y_reduced.shape)
    print(y_reduced)

