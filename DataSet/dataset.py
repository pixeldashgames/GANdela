import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as a
from albumentations.pytorch import ToTensorV2
from utils.utils import tif_to_ndarray

# --------------- Augmentations ----------------
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


class Satellite2Map_Data(Dataset):
    def __init__(self, root):
        self.root = root
        list_files = os.listdir(self.root)

        if '.ipynb_checkpoints' in list_files:
            list_files.remove('.ipynb_checkpoints')
        self.sat = list(filter(lambda x: x.startswith("satellite"), list_files))
        self.elev = list(filter(lambda x: x.startswith("elevation"), list_files))

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

        elevation_image = both_transform(image=elevation_image)["image"]


        satellite_image = transform_only_input(image=input_image)["image"]

        # PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
        # satellite_image = Image.fromarray(satellite_image.astype('uint8'),'RGB')
        # map_image = Image.fromarray(map_image.astype('uint8'),'RGB')
        # if self.transform!=None:
        #     satellite_image = self.transform(satellite_image)
        #     map_image = self.transform(map_image)
        return (satellite_image, elevation_image)
        # except:
        #     if torch.is_tensor(idx):
        #         idx = idx.tolist()
        #     image_name = self.sat[idx]

        #     image_path = os.path.join(self.root, image_name)
        #     print(image_path)
        #     pass


if __name__ == "__main__":
    dataset = Satellite2Map_Data("./database/train")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print("X Shape :-", x.shape)
        print("Y Shape :-", y.shape)
        save_image(x, "satellite.png")
        save_image(y, "map.png")
        break
