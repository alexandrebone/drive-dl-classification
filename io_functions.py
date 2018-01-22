import os
import numpy as np
from PIL import Image
from sklearn import feature_extraction

def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image).astype('float64')


def load_train_database():
    path_to_train_database = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'training')

    path_to_images = os.path.join(path_to_train_database, 'images')
    images = [load_image(os.path.join(path_to_images, path))
              for path in os.listdir(path_to_images) if path[-3:] == 'tif']

    path_to_masks = os.path.join(path_to_train_database, 'mask')
    masks = [np.array(load_image(os.path.join(path_to_masks, path)))
             for path in os.listdir(path_to_masks) if path[-3:] == 'gif']

    path_to_targets = os.path.join(path_to_train_database, '1st_manual')
    targets = [load_image(os.path.join(path_to_targets, path))
               for path in os.listdir(path_to_targets) if path[-3:] == 'gif']

    return images, masks, targets


def remove_mean(images):
    mean = sum(images) / float(len(images))
    return [image - mean for image in images]


def cut_into_patches(images, targets, patch_size=100):
    image_patches = feature_extraction.image.extract_patches(
        images[0], patch_shape=(patch_size, patch_size, 3),
        extraction_step=patch_size).reshape([-1, patch_size, patch_size, 3])
    target_patches = feature_extraction.image.extract_patches(
        targets[0], patch_shape=(patch_size, patch_size),
        extraction_step=patch_size).reshape([-1, patch_size, patch_size])

    for (image, target) in zip(images[1:], targets[1:]):
        image_patches = np.append(image_patches, feature_extraction.image.extract_patches(
            image, patch_shape=(patch_size, patch_size, 3),
            extraction_step=patch_size).reshape([-1, patch_size, patch_size, 3]), axis=0)
        target_patches = np.append(target_patches, feature_extraction.image.extract_patches(
            target, patch_shape=(patch_size, patch_size),
            extraction_step=patch_size).reshape([-1, patch_size, patch_size]), axis=0)

    return image_patches, target_patches