from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile, os
import os, hashlib, uuid 
from pathlib import Path
import math
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.h5"
LABELS_PATH = BASE_DIR / "model" / "labels.json"

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


from ai.predict import Predictor
from db.image_repo import ImageRepo

app = Flask(__name__)

repo = ImageRepo()

def sha256_file(path:str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def image_metadata(path: str):
    with Image.open(path) as im:
        w, h = im.size
        fmt = (im.format or "unknown").lower()
    size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
    return h, w, fmt, size_mb

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = Predictor(
            model_path=str(MODEL_PATH),
            labels_path=str(LABELS_PATH),
            img_size=(128, 128)
        )
    return predictor


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify(error="No file uploaded (field name must be 'file')"), 400

   
    f = request.files["file"]
    safe_name = secure_filename(f.filename) or "upload"
    fname = f"{Path(safe_name).stem}_{uuid.uuid4().hex}{Path(safe_name).suffix or '.bin'}"
    save_path = str(UPLOAD_DIR / fname)
    f.save(save_path)

    try:
        height, width, fmt, size_mb = image_metadata(save_path)
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return jsonify(error=f"Invalid image: {e}"), 415

    file_hash = sha256_file(save_path)

    image_id = repo.insert_pending(
        name=safe_name,
        path=save_path,
        height=height,
        width=width,
        format=fmt,
        size_mb=size_mb,
        file_hash=file_hash
    )
    if image_id is None:

        try:
            os.remove(save_path)
        except Exception:
            pass
        return jsonify(error="Duplicate file"), 409

    pred = get_predictor()
    raw = pred.predict(save_path)

    label = str(raw.get("label", "unknown"))
    conf = raw.get("confidence", raw.get("top_prob", 0.0))
    try:
        confidence = float(conf)
    except Exception:
        confidence = 0.0
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        confidence = 0.0

    probs = raw.get("probs")

    if isinstance(raw, dict):
        label = raw.get("label") or raw.get("class") or raw.get("prediction") or "unknown"
        confidence = float(raw.get("confidence") or raw.get("probability") or 0.0)
        extra = {k: v for k, v in raw.items() if k not in ("label", "class", "prediction", "confidence", "probability")}
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        label = str(raw[0])
        confidence = float(raw[1])
        extra = {}
    else:
        label = str(raw)
        confidence = 0.0
        extra = {}

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        confidence = 0.0

    repo.set_prediction(image_id, label, confidence)

    return jsonify({
    "id": image_id,
    "file": safe_name,
    "stored_path": save_path,
    "width": width,
    "height": height,
    "format": fmt,
    "size_mb": size_mb,

    # для фронта (твоя логика)
    "label": label,
    "top_prob": confidence,

    # стандартные поля, чтобы не ломать ничего в БД
    "predicted_label": label,
    "confidence": confidence,

    # если пришли вероятности
    **({"probs": raw.get("probs")} if "probs" in raw else {})
})

@app.route("/api/confirm", methods=["POST"])
def api_confirm():
    data = request.get_json(silent=True) or {}
    try:
        image_id = int(data.get("id"))
    except Exception:
        return jsonify(error="id is required (int)"), 400
    true_label = (data.get("true_label") or "").strip()
    if not true_label:
        return jsonify(error="true_label is required"), 400

    ok = repo.confirm(image_id, true_label)
    if not ok:
        return jsonify(error="not found"), 404
    return jsonify(ok=True)

if __name__ == "__main__":
    
    app.run(host="127.0.0.1", port=5000, debug=True)
