import keras
import numpy as np
from io_functions import *
from PIL import Image


patch_size=100
stride=100
images_test, masks_test, targets_test1, targets_test2 = load_test_database()
images_test_patches= images_to_patches(images_test,patch_size,stride)


model=keras.models.load_model('fitted_model.h5')
predictions = model.predict(np.array(images_test_patches))
save_predictions(predictions, patch_size)


'''
patches = []
for k in range(a.shape[0]):
    patches.append(a[k].reshape((patch_size, patch_size)))
predictions = patches_to_images(patches)

for i in predictions:
    i.save()
print('aaaa')
'''