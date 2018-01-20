import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

def load_train_database():
    path_to_train_database = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'training')

    path_to_images = os.path.join(path_to_train_database, 'images')
    images = [load_image(os.path.join(path_to_images, path)) for path in os.listdir(path_to_images) if path[-3:] == 'tif']

    path_to_masks = os.path.join(path_to_train_database, 'mask')
    masks = [load_image(os.path.join(path_to_masks, path)) for path in os.listdir(path_to_masks) if path[-3:] == 'gif']

    path_to_targets = os.path.join(path_to_train_database, '1st_manual')
    targets = [load_image(os.path.join(path_to_targets, path)) for path in os.listdir(path_to_targets) if path[-3:] == 'gif']

    return images, masks, targets

images, masks, targets = load_train_database()

print('End')


model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = (224, 224, 3), activation = 'relu',stride= 1,padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2),stride=2))
model.add(Conv2D(128, (3, 3), activation = 'relu',stride= 1,padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2),stride=2))
model.add(Conv2D(256, (3, 3), activation = 'relu',stride= 1,padding='same'))
model.add(Conv2D(256, (3, 3), activation = 'relu',stride= 1,padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2),stride=2))

model.add(Conv2D(512, (3, 3),  activation = 'relu',stride= 1,padding='same'))
model.add(Conv2D(512, (3, 3),  activation = 'relu',stride= 1,padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2),stride=2))
model.add(Conv2D(512, (3, 3),  activation = 'relu',stride= 1,padding='same'))
model.add(Conv2D(512, (3, 3),  activation = 'relu',stride= 1,padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2),stride=2))

model.add(Flatten())
model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dense(units = 329960, activation = 'relu'))
model.add(Dense(329960, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.fit_generator(training_set,
steps_per_epoch = 8000,
epochs = 25,
validation_data = test_set,
validation_steps = 2000)