import torch
from utils.utils import output_transform, load_checkpoint, tif_to_ndarray, desnormalice

import config
import albumentations as a
from albumentations.pytorch import ToTensorV2
from pix2pix.Generator import Generator
from torchvision import transforms
from PIL import Image


both_transform = a.Compose(
    [a.Resize(width=256, height=256), ],
)

transform_only_input = a.Compose(
    [
        a.HorizontalFlip(p=0.5),
        a.ColorJitter(p=0.2),
        a.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)

def print_image():
    pass


if __name__ == '__main__':
    image_path = "satellite_632.tif"
    satellite_original = tif_to_ndarray(image_path)

    satellite_img = Image.fromarray(satellite_original.astype('uint8'), 'RGB')

    satellite_img.show()

    satellite_image = both_transform(image=satellite_original)["image"]

    satellite_image = transform_only_input(image=satellite_image)["image"]

    transformed_image = satellite_image.unsqueeze(0)
    transformed_image = transformed_image.to(config.DEVICE)

    netG = Generator(in_channels=3).to(config.DEVICE)
    optimizerG = torch.optim.Adam(netG.parameters(), lr = config.LEARNING_RATE, betas=(config.BETA1, 0.999))

    states = [10,50,100,150,200,300,399]
    for i, state in enumerate(states):
        load_checkpoint(config.CHECKPOINT_GEN, state, netG, optimizerG, config.LEARNING_RATE)
        print(transformed_image.shape)
        fake_image = netG(transformed_image)

        y = fake_image.detach().cpu().numpy()[0][0]
        y_image = Image.fromarray(y.astype('uint8'), 'L')
        y_image.show()
    





