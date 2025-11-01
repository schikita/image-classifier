from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image, UnidentifiedImageError
import hashlib
import imagehash

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
    img = load_img(img_path, target_size=img_size)
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)


def filter_broken_and_tiny(data_dir = "data/raw", min_size = (64,64)):
    removed = 0
    for p in Path(data_dir).rglob("*"):
        if not (p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}):
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
            if w < min_size[0] or h < min_size[1]:
                p.unlink(missing_ok=True); removed += 1
        except (UnidentifiedImageError, OSError):
            p.unlink(missing_ok=True); removed += 1
    print(f"[filter] was deleted: {removed}")

def remove_exact_duplicates(data_dir="data/raw"):
    seen = set(); removed = 0
    for p in Path(data_dir).rglob("*"):
        if not (p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}):
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in seen:
            p.unlink(missing_ok=True); removed += 1
        else:
            seen.add(h)
    print(f"[dedupe] duplicates : {removed}")
    

def clean_near_duplicates(data_dir = "data/raw", threshold = 5):
    for class_dir in Path(data_dir).iterdir():
        if not class_dir.is_dir():
            continue
        hashes = {}
        for img_path in class_dir.iterdir():
            if not (img_path.is_file() and img_path.suffix.lower() in {".jpg", "jpeg", ".png"}):
                continue
            try:
                with Image.open(img_path) as im:
                    h = imagehash.phash(im)

                for seen_path, seen_hash in hashes.items():
                    if abs(h - seen_hash) <= threshold:
                        print(f"delete near-duplicate: {img_path} ~ {seen_path}")
                        img_path.unlink(missing_ok=True)
                        break
                else:
                    hashes[img_path] = h
            except Exception as e:
                print(f"Error in {img_path} : {e}")
