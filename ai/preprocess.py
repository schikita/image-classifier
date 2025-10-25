from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

def prepare_data(data_dir="data", img_size=(128,128), val_split=0.2, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    train_flow = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        subset='training', class_mode='categorical'
    )
    val_flow = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        subset='validation', class_mode='categorical'
    )
    num_classes = len(train_flow.class_indices)
    return train_flow, val_flow, num_classes

def preprocess_single_image(img_path, img_size=(128,128)):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)
