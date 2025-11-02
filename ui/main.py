from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile, os

from ai.predict import Predictor

app = Flask(__name__)

# Загружаем модель только при первом запросе
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = Predictor(
                model_path="model/model.h5",
                labels_path="model/labels.json",
                img_size=(128, 128)
            )
        except FileNotFoundError:
            predictor = None
    return predictor


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify(error="No file uploaded (field name must be 'file')"), 400

    pred = get_predictor()
    if pred is None:
        return jsonify(error="Model files not found (model/model.h5)"), 503

    f = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix='_' + secure_filename(f.filename)) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        out = pred.predict(tmp_path)  # {'index', 'label', 'probs', 'top_prob'}
        return jsonify(out)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    
    app.run(host="127.0.0.1", port=5000, debug=True)
