import keras
import numpy as np
from io_functions import *

patch_size = 50
stride = 50
model_path = 'fitted_model_'+str(patch_size)+'-'+str(stride)+'-200-0.0001-categorical_crossentropy.h5'
output_dir = 'output_' + model_path[13:-3]

# images_test, masks_test, targets_test1, targets_test2 = load_test_database()
# images_test_patches = images_to_patches(images_test, patch_size, stride)

images, masks, targets = load_train_database()
images = remove_mean(images)
image_patches = images_to_patches(images, patch_size, stride)

model = keras.models.load_model(os.path.join(output_dir, 'fitted-model.h5'))
predictions = model.predict(image_patches)
print(predictions)
print('Max: ' + str(np.max(predictions)))
# save_predictions(predictions / np.max(predictions), targets[0].shape, patch_size, stride, output_directory=output_dir)
save_predictions(predictions*1000, targets[0].shape, patch_size, stride, output_directory=output_dir + '_times1000')
