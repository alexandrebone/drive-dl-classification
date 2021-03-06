import os
import numpy as np
from PIL import Image, ImageFilter


def load_image(image_path, blur=False):
    image = Image.open(image_path)
    if blur: image = image.filter(ImageFilter.GaussianBlur)
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
    targets = [load_image(os.path.join(path_to_targets, path), blur=True) / 255.0
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


def normalize(images):
    mean, std = compute_mean_and_std(images)
    return [(image - mean)/std for image in images]


def compute_mean_and_std(images):
    number_of_pixels = float(images[0].shape[0] * images[0].shape[1])
    mean = sum([np.sum(np.sum(image, axis=0), axis=0) / number_of_pixels for image in images]) / float(len(images))
    std = np.sqrt(np.array(sum([np.sum(np.sum(image ** 2, axis=0), axis=0) / number_of_pixels - mean ** 2 for image in images]) / float(len(images))))
    return mean, std


def remove_mean(images):
    mean, _ = compute_mean_and_std(images)
    return [image - mean for image in images]


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
                patch = padded_image[i:i + patch_size, j:j + patch_size]
                patches.append(patch)
    return np.array(patches)


def patches_to_images(patches, image_shape, patch_size=100, stride=100, number_of_images=20):
    number_of_patches_per_image = patches.shape[0] / number_of_images
    padded_shape = (image_shape[0] + patch_size, image_shape[1] + patch_size)
    if len(image_shape) == 3: padded_shape += (image_shape[2],)
    images = []
    for k, patch in enumerate(patches):
        if not k % number_of_patches_per_image:
            if not k == 0: images.append(padded_image[0:image_shape[0], 0:image_shape[1]])
            i = 0
            j = 0
            padded_image = np.zeros(padded_shape)
        padded_image[i:i + patch_size, j:j + patch_size] = patch
        j += stride
        if j >= image_shape[1]:
            j = 0
            i += stride
            if i >= image_shape[0]:
                i = 0

    images.append(padded_image[0:image_shape[0], 0:image_shape[1]])
    return images


def write_image(image_array, path):
    image = Image.fromarray(np.uint8(image_array * 255))
    image.save(path)


def save_predictions(predictions, image_shape,
                     patch_size=100, stride=100, number_of_images=20, output_directory='outputs'):
    if not os.path.isdir(output_directory): os.mkdir(output_directory)
    patches = []
    for prediction in predictions:
        patches.append(prediction.reshape((patch_size, patch_size)))
    reconstructed_images = patches_to_images(np.array(patches), image_shape, patch_size, stride, number_of_images)
    for k, reconstructed_image in enumerate(reconstructed_images):
        write_image(reconstructed_image, os.path.join(output_directory, str(k + 1) + '_prediction.jpg'))


if __name__ == "__main__":
    images, masks, targets = load_train_database()
    print('Mean before: ' + str(compute_mean(images)))
    images = remove_mean(images)
    print('Mean after: ' + str(compute_mean(images)))

    patch_size = 100
    stride = 100

    patches = images_to_patches(images, patch_size, stride)
    print('Number of patches: ' + str(patches.shape[0]))

    reconstructed_images = patches_to_images(patches, images[0].shape, patch_size, stride)
    print('Number of images: ' + str(len(reconstructed_images)))
    print('Check for first image: ' + str((images[0] == reconstructed_images[0]).all()))

    write_image(targets[0] / 255, 'test.jpg')
