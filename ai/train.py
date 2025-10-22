import tensorflow as tf
from ai.model import CNNModel

def main():
    train_dir = "data/train"
    val_dir = "data/val"
    img_size = (128,128)
    batch_size = 32

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

    train_flow = train_gen.flow_from_directory(
        train_dir, target_size = img_size, batch_size = batch_size, class_mode = 'categorical') 
    val_flow = val_gen.flow_from_directory(
        val_dir, target_size = img_size, batch_size = batch_size, class_mode = 'categorical')
    
    num_classes = len(train_flow.class_indices)

    model = CNNModel(img_size=img_size, num_classes=num_classes)
    model.train(train_flow, val_flow, epochs=10, save_path="model/model.h5")

if __name__ == "__main__":
    main()