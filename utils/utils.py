import os
import torch
import config
import rasterio
import numpy as np
import albumentations as a
import numpy as np
import pickle
from PIL import Image


output_transform = a.Compose(
    [a.Resize(width=256, height=256), ],
)

def save_some_examples(gen, val_loader, epoch, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        save_matrix(y_fake, folder + f"/y_gen_{epoch}.pkl")
        save_matrix(x * 0.5 + 0.5, folder + f"/input_{epoch}.pkl")
        save_matrix(y, folder + f"/label_{epoch}.pkl")
    gen.train()


def save_image(tensor, path):
    data = tensor.cpu().detach().numpy()

    image = Image.fromarray(data, 'RGB')
    image.save(path)
    

def save_matrix(tensor, path, transform_params=None, crs="EPSG:4326"):
    # Assuming tensor is a PyTorch tensor and has shape [Channels, Height, Width]
    # Convert tensor to numpy array
    data = tensor.cpu().detach().numpy()

    with open(path, 'wb') as f:
        pickle.dump(data, f)



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def tif_to_ndarray(tif_path:str) -> np.ndarray:
    # Open the TIFF file
    with rasterio.open(tif_path) as src:
        data = src.read()
        data = np.transpose(data, (1, 2, 0))
        scaled_array = ((data - data.min()) * (1/(data.max() - data.min()) * 255)).astype('uint8')
        return scaled_array