import tensorflow as tf
from pathlib import Path

class KNNModel:
    def __init__(self, img_size = (128,128), num_classes = 2, learning_rate = 0.001):
        self.img_size = img_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()


    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(*self.img_size, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(self.num_classes, activation = 'softmax') 
        ])
        
        model.compile(
            optimizer = tf.keras.optimizers.Adam(self.learning_rate),
            loss = 'categorial_crossentropy',
            metricd = ['accuracy']
        )

        return model
    
