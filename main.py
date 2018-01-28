import os
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout
import keras

from io_functions import *

# Parameters -------------------------------------------------------------------
patch_size = 48
stride = 48
learning_rate = 1e-4
n_epochs = 150
loss_name = 'categorical_crossentropy'
batch_size = 32

# Load and pre-process data ----------------------------------------------------
images, masks, targets = load_train_database()
images = normalize(images)
mean, std = compute_mean_and_std(images)
print('-' * 100)
print('Mean after normalization: ', mean)
print('Std after normalization: ', std)
print('-' * 100)

image_patches = images_to_patches(images, patch_size, stride)
target_patches = images_to_patches(targets, patch_size, stride)


def channels_last_to_channels_first(patches):
    patches = np.swapaxes(patches, 1, 3)
    patches = np.swapaxes(patches, 2, 3)
    return patches


def flatten_patches(patches):
    patches_flat = []
    for k in range(patches.shape[0]):
        patches_flat.append(target_patches[k].ravel())
    return np.array(patches_flat)


def masks_Unet(masks):
    assert (len(masks.shape) == 3)
    masks = np.reshape(masks, (masks.shape[0], 1, masks.shape[1], masks.shape[2]))
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, 2))
    for i in range(masks.shape[0]):
        for j in range(im_h * im_w):
            if masks[i, j] == 0:
                new_masks[i, j, 0] = 1
                new_masks[i, j, 1] = 0
            else:
                new_masks[i, j, 0] = 0
                new_masks[i, j, 1] = 1
    return new_masks


def masks_Unet_inverse(pred, patch_height=patch_size, patch_width=patch_size, mode="original"):
    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)
    assert (pred.shape[2] == 2)  # check the classes are 2
    pred_images = np.empty((pred.shape[0], pred.shape[1]))  # (Npatches,height*width)
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i, pix] = pred[i, pix, 1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] >= 0.5:
                    pred_images[i, pix] = 1
                else:
                    pred_images[i, pix] = 0
    else:
        print("mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images, (pred_images.shape[0], patch_height, patch_width))
    return pred_images


# Define model -----------------------------------------------------------------
# X = image_patches
# Y = flatten_patches(target_patches)
#
# model = Sequential()
# model.add(Conv2D(64, (3, 3), input_shape=(patch_size, patch_size, 3), activation='relu', strides=1, padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(128, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(256, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
# model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', strides=1, padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(patch_size ** 2, activation='relu'))
# model.add(Dense(patch_size ** 2, activation='softmax'))
#
# model.compile(loss=loss_name,
#               optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True))

X = channels_last_to_channels_first(image_patches)
Y = masks_Unet(target_patches)

inputs = Input(shape=X[0].shape)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

up1 = UpSampling2D(size=(2, 2))(conv3)
up1 = concatenate([conv2, up1], axis=1)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)

up2 = UpSampling2D(size=(2, 2))(conv4)
up2 = concatenate([conv1, up2], axis=1)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)

conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
conv6 = core.Reshape((2, patch_size ** 2))(conv6)
conv6 = core.Permute((2, 1))(conv6)

conv7 = core.Activation('softmax')(conv6)

model = Model(input=inputs, output=conv7)
model.compile(optimizer='sgd', loss=loss_name, metrics=['accuracy'])

# Custom callback --------------------------------------------------------------
output_dir = 'output_' + str(patch_size) + '-' + str(stride) + '-' + str(n_epochs) + '-' \
             + str(learning_rate) + '-' + loss_name
output_dir_1000 = 'output_' + str(patch_size) + '-' + str(stride) + '-' + str(n_epochs) + '-' \
                  + str(learning_rate) + '-' + loss_name + '-times1000'
if not os.path.isdir(output_dir): os.mkdir(output_dir)
if not os.path.isdir(output_dir_1000): os.mkdir(output_dir_1000)


class SaveReconstructions(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        predictions = masks_Unet_inverse(self.model.predict(X))
        print('\n>> Predictions max value: ' + str(np.max(predictions)))
        save_predictions(predictions, targets[0].shape, patch_size, stride, output_directory=output_dir)
        save_predictions(predictions * 1000, targets[0].shape, patch_size, stride, output_directory=output_dir_1000)


# Fit --------------------------------------------------------------------------
callback = SaveReconstructions()
model.fit(X, Y, nb_epoch=n_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1,
          callbacks=[callback])

# Save -------------------------------------------------------------------------
model.save(os.path.join(output_dir, 'fitted-model' + '.h5'))
predictions = masks_Unet_inverse(model.predict(X))
save_predictions(predictions, targets[0].shape, patch_size, stride, output_directory=output_dir)
