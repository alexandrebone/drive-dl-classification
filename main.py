import keras
import numpy as np


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


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