import os
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras

from io_functions import load_image, load_train_database, remove_mean, cut_into_patches

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

images_test, masks_test, targets_test = load_test_database()
image_test_patches, target__test_patches = cut_into_patches(images_test, targets_test, patch_size)
targets_test_flat = np.array(targets_test_flat)


# model.fit(x=np.array(images), epochs = 25, y= np.array(targets))
model.fit(x=np.array(image_patches), epochs=1, y=np.array(targets_flat),validation_data=(np.array(image_test_patches),targets_test_flat))
# model.save('fitted_model.h5')