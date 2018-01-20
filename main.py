import os
import numpy as np
from PIL import Image

import keras

def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

def load_train_database:
    path_to_train_database = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'training')

    path_to_images = os.path.join(path_to_train_database, 'images')
    images = [load_image(path) for path in os.listdir(path_to_images) if path[-3:] == 'tif']

    path_to_masks = os.path.join(path_to_train_database, 'mask')
    masks = [load_image(path) for path in os.listdir(path_to_masks) if path[-3:] == 'tif']

    path_to_targets = os.path.join(path_to_train_database, '1st_manual')
    targets = [load_image(path) for path in os.listdir(path_to_targets) if path[-3:] == 'tif']

    return images, masks, targets


