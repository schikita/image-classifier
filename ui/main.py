from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.h5"
LABELS_PATH = BASE_DIR / "model" / "labels.json"


from ai.predict import Predictor

app = Flask(__name__)


predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = Predictor(
                model_path=str(MODEL_PATH),
                labels_path=str(LABELS_PATH),
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
        out = pred.predict(tmp_path)
        return jsonify(out)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    
    app.run(host="127.0.0.1", port=5000, debug=True)
