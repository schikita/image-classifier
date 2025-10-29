import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from ai.preprocess import preprocess_single_image
import matplotlib.pyplot as plt  # для показа картинки

class Predictor:
    def __init__(self, model_path="model/model.h5", labels_path="model/labels.json", img_size=(128,128)):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size
        self.id_to_label = None
        lp = Path(labels_path)
        if lp.exists():
            data = json.loads(lp.read_text(encoding="utf-8"))
            self.id_to_label = {int(k): v for k, v in data.items()}

    def predict(self, img_path, show=False):
        """
        Делает предсказание для одного изображения.
        :param img_path: путь к файлу изображения
        :param show: если True — покажет изображение и подпись
        :return: словарь с результатом
        """
        x = preprocess_single_image(img_path, img_size=self.img_size)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.id_to_label.get(idx) if self.id_to_label else str(idx)

        result = {
            "index": idx,
            "label": label,
            "probs": probs.tolist(),
            "top_prob": float(probs[idx])
        }

        if show:
            img = tf.keras.preprocessing.image.load_img(img_path)
            plt.imshow(img)
            plt.title(f"{label} ({probs[idx]*100:.1f}%)")
            plt.axis("off")
            plt.show()

        return result
