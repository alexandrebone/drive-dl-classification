import os
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras

from io_functions import load_train_database, remove_mean, images_to_patches, load_test_database, save_predictions

images, masks, targets = load_train_database()
images = remove_mean(images)

patch_size = 50
stride = 50

image_patches = images_to_patches(images, patch_size, stride)
target_patches = images_to_patches(targets, patch_size, stride)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(patch_size, patch_size, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

learning_rate = 1e-4

model.add(Flatten())
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=patch_size ** 2, activation='relu'))
model.add(Dense(patch_size ** 2, activation='softmax'))
# model.compile(loss=keras.losses.mean_squared_error,
#               optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True))

images_test, masks_test, targets_test1, targets_test2 = load_test_database()
images_test_patches = images_to_patches(images_test, patch_size, stride)
targets__test_patches = images_to_patches(targets_test1, patch_size, stride)
targets_flat = []
targets_test_flat = []
for k in range(target_patches.shape[0]):
    targets_flat.append(target_patches[k].ravel())
    targets_test_flat.append(target_patches[k].ravel())
targets_flat = np.array(targets_flat)
targets_test_flat = np.array(targets_test_flat)

n_epochs = 1000

# Custom callback --------------------------------------------------------------
output_dir = 'output_' + str(patch_size) + '-' + str(stride) + '-' + str(n_epochs) + '-' + str(learning_rate) + '-categorical_crossentropy'
output_dir_1000 = 'output_' + str(patch_size) + '-' + str(stride) + '-' + str(n_epochs) + '-' + str(learning_rate) + '-categorical_crossentropy_times1000'
os.mkdir(output_dir)
os.mkdir(output_dir_1000)

class SaveReconstructions(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(image_patches)
        print('\n>> Predictions max value: ' + str(np.max(predictions)))
        save_predictions(predictions, targets[0].shape, patch_size, stride, output_directory=output_dir)
        save_predictions(predictions * 1000, targets[0].shape, patch_size, stride, output_directory=output_dir_1000)

# Fit --------------------------------------------------------------------------
# model.fit(x=np.array(images), epochs = 25, y= np.array(targets))
# model.fit(x=np.array(image_patches), epochs=25, y=np.array(targets_flat),
#           validation_data=(np.array(images_test_patches), targets_test_flat))

callback = SaveReconstructions()
model.fit(x=image_patches, epochs=n_epochs, y=np.array(targets_flat), callbacks=[callback])

# Save -------------------------------------------------------------------------
model.save(os.path.join(output_dir, 'fitted-model' + '.h5'))
predictions = model.predict(image_patches)
save_predictions(predictions, targets[0].shape, patch_size, stride, output_directory=output_dir)
