import keras
import numpy as np
from io_functions import *

patch_size = 100
stride = 75

# images_test, masks_test, targets_test1, targets_test2 = load_test_database()
# images_test_patches = images_to_patches(images_test, patch_size, stride)

images, masks, targets = load_train_database()
images = remove_mean(images)
image_patches = images_to_patches(images, patch_size, stride)

model = keras.models.load_model('fitted_model_100-75-5-0.0001-categorical_crossentropy.h5')
predictions = model.predict(np.array(image_patches))
save_predictions(predictions*1000, targets[0].shape, patch_size, stride,
                 output_directory='output_100-75-5-0.0001-categorical_crossentropy_times1000')

'''
patches = []
for k in range(a.shape[0]):
    patches.append(a[k].reshape((patch_size, patch_size)))
predictions = patches_to_images(patches)

for i in predictions:
    i.save()
print('aaaa')
'''
