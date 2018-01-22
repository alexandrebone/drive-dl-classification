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


def load_test_database():
    path_to_test_database = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test')

    path_to_images = os.path.join(path_to_test_database, 'images')
    images = [load_image(os.path.join(path_to_images, path))
              for path in os.listdir(path_to_images) if path[-3:] == 'tif']

    path_to_masks = os.path.join(path_to_test_database, 'mask')
    masks = [np.array(load_image(os.path.join(path_to_masks, path)))
             for path in os.listdir(path_to_masks) if path[-3:] == 'gif']

    path_to_targets_1 = os.path.join(path_to_test_database, '1st_manual')
    targets_1 = [load_image(os.path.join(path_to_targets_1, path))
                 for path in os.listdir(path_to_targets_1) if path[-3:] == 'gif']

    path_to_targets_2 = os.path.join(path_to_test_database, '2nd_manual')
    targets_2 = [load_image(os.path.join(path_to_targets_2, path))
                 for path in os.listdir(path_to_targets_2) if path[-3:] == 'gif']

    return images, masks, targets_1, targets_2


def compute_mean(images):
    number_of_pixels = float(images[0].shape[0] * images[0].shape[1])
    return sum([np.sum(np.sum(image, axis=0), axis=0) / number_of_pixels for image in images]) / float(len(images))


def remove_mean(images):
    mean = compute_mean(images)
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

def images_to_patches(images, patch_size=100, stride=100):
    """
    Zero padding.
    :param images: List of numpy 2D or 3D arrays (black & white or color).
    :param patch_size: Size of the square patches.
    :param stride:
    :return:
    """
    padding_value = images[0][0, 0]
    padded_shape = (images[0].shape[0] + patch_size, images[0].shape[1] + patch_size)
    if len(images[0].shape) == 3: padded_shape += (images[0].shape[2],)
    patches = []
    for image in images:
        padded_image = np.zeros(padded_shape) + padding_value
        padded_image[0:image.shape[0], 0:image.shape[1]] = image
        for i in range(0, image.shape[0], stride):
            for j in range(0, image.shape[1], stride):
                patch = padded_image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
    return patches

if __name__ == "__main__":
    images, masks, targets = load_train_database()
    print('Mean before: ' + str(compute_mean(images)))
    images = remove_mean(images)
    print('Mean after: ' + str(compute_mean(images)))

    patches = images_to_patches(images)
    print('Number of patches: ' + str(len(patches)))



