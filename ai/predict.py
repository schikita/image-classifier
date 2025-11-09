import json, numpy as np, tensorflow as tf
from pathlib import Path
from ai.preprocess import preprocess_single_image

class Predictor:
    def __init__(self, model_path="model/model.h5", labels_path="model/labels.json", img_size=(128,128)):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size

        lp = Path(labels_path)
        if not lp.exists():
            raise FileNotFoundError(f"labels.json not found at {lp.resolve()}")

        data = json.loads(lp.read_text(encoding="utf-8"))

        self.id_to_label = {int(k): v for k, v in data.items()}

        units = self.model.output_shape[-1]
        assert len(self.id_to_label) == units, (
            f"Mismatch: model has {units} outputs, labels.json has {len(self.id_to_label)}"
        )

    def predict(self, img_path):
        x = preprocess_single_image(img_path, img_size=self.img_size)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.id_to_label.get(idx, str(idx))
        top_prob = float(probs[idx])
       
        print("probs=", probs.tolist(), "argmax=", idx, "label=", label, "top_prob=", top_prob)
        return {
            "index": idx,
            "label": label,
            "probs": probs.tolist(),
            "top_prob": top_prob
        }
