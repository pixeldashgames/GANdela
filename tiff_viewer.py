import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

file_path = 'database/train/elevation_1.tif'

# Open the TIFF file
with rasterio.open(file_path) as src:
    data = src.read()
    data = np.transpose(data, (1, 2, 0))
    scaled_array = ((data - data.min()) * (1/(data.max() - data.min()) * 255)).astype('uint8')
    image = Image.fromarray(scaled_array, 'RGB')
    image.show()