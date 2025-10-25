import tensorflow as tf
from pathlib import Path

class CNNModel:
    def __init__(self, img_size=(128, 128), num_classes=3, learning_rate=1e-3):
        self.img_size = img_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(*self.img_size, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',   
            metrics=['accuracy'],              
        )
        return model

    def train(self, train_flow, val_flow, epochs=10, save_path="model/model.h5"):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', mode='max'),
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ]
        history = self.model.fit(train_flow, validation_data=val_flow, epochs=epochs, callbacks=callbacks)
        return history

    def save(self, path='model/model.h5'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path='model/model.h5'):
        self.model = tf.keras.models.load_model(path)
        return self

    def predict(self, x_batch):
        return self.model.predict(x_batch)
