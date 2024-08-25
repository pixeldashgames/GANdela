import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as a
from albumentations.pytorch import ToTensorV2

import numpy as np
from sklearn.preprocessing import MinMaxScaler


import rasterio


def tif_to_ndarray(tif_path: str) -> np.ndarray:
    # Open the TIFF file
    with rasterio.open(tif_path) as src:
        data = src.read()
        data = np.transpose(data, (1, 2, 0))
        scaled_array = (
            (data - data.min()) * (1 / (data.max() - data.min()) * 255)
        ).astype("uint8")
        return scaled_array


# --------------- Augmentations ----------------
both_transform = a.Compose(
    [
        a.Resize(width=256, height=256),
    ],
)

# Create a MinMaxScaler instance
scaler = MinMaxScaler((-1, 1))


# Define the transformations
def scale_output(x, **kargs):
    return scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)


transform_only_output = a.Compose(
    [
        a.Lambda(scale_output, always_apply=True),
        ToTensorV2(),
    ]
)
transform_only_input = a.Compose(
    [
        a.HorizontalFlip(p=0.5),
        a.ColorJitter(p=0.2),
        a.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


class Satellite2Map_Data(Dataset):
    def __init__(self, root: str, rgb_on: bool):
        self.root = root
        list_files = os.listdir(self.root)
        self.rgb_on = rgb_on

        if ".ipynb_checkpoints" in list_files:
            list_files.remove(".ipynb_checkpoints")
        self.sat = list(filter(lambda x: x.startswith("satellite"), list_files))
        self.elev = list(filter(lambda x: x.startswith("elevation"), list_files))
        if len(self.sat) != len(self.elev):
            raise ValueError(
                "There are need to be the same number of Satellite and Elevation images"
            )

    def __len__(self):
        return len(self.sat)

    def __getitem__(self, idx):
        # try:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sat_name = self.sat[idx]
        elev_name = self.elev[idx]
        sat_path = os.path.join(self.root, sat_name)
        elev_path = os.path.join(self.root, elev_name)

        satellite_image = tif_to_ndarray(sat_path)

        elevation_image = tif_to_ndarray(elev_path)

        input_image = both_transform(image=satellite_image)["image"]

        output_image = both_transform(image=elevation_image)["image"]

        satellite_image = transform_only_input(image=input_image)["image"]

        if self.rgb_on:
            satellite_image = satellite_image.transpose(0, 2)
            grayscale = np.dot(satellite_image, [0.299, 0.587, 0.114])
            satellite_image = torch.from_numpy(grayscale).unsqueeze(0)

        elevation_image = transform_only_output(image=output_image)["image"]

        return (satellite_image, elevation_image)


if __name__ == "__main__":
    dataset = Satellite2Map_Data("./database/train", rgb_on=False)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print("X Shape :-", x.shape)
        print("Y Shape :-", y.shape)
        save_image(x, "satellite.png")
        save_image(y, "map.png")
        break
