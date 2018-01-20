import os
import numpy as np
from PIL import Image
from sklearn import feature_extraction

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras

import _pickle as pickle


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


images, masks, targets = load_train_database()
images = remove_mean(images)

patch_size = 100
image_patches, target_patches = cut_into_patches(images, targets, patch_size)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(patch_size, patch_size, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', strides=1, padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=patch_size ** 2, activation='relu'))
model.add(Dense(patch_size ** 2, activation='softmax'))
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

targets_flat = []
for k in range(target_patches.shape[0]):
    targets_flat.append(target_patches[k].ravel())
targets_flat = np.array(targets_flat)

# model.fit(x=np.array(images), epochs = 25, y= np.array(targets))
model.fit(x=np.array(image_patches), epochs=25, y=np.array(targets_flat))
pickle.dump(model, open('fitted_model.p', 'wb'))

# d = pickle.load(open('fitted_model', 'rb'))