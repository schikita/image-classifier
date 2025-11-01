from pathlib import Path
import json
from ai.model import CNNModel
from ai.preprocess import (
    prepare_data,
    filter_broken_and_tiny,
    remove_exact_duplicates,
    clean_near_duplicates )

def main():
    data_dir = "data/raw"
    img_size = (128, 128)
    val_split = 0.2
    batch_size = 32

    filter_broken_and_tiny(data_dir)
    remove_exact_duplicates(data_dir)
    clean_near_duplicates(data_dir)

    train_flow, val_flow, num_classes = prepare_data(
        data_dir=data_dir, img_size=img_size, val_split=val_split, batch_size=batch_size
    )

 
    labels_path = Path("model/labels.json")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    id_to_label = {idx: cls for cls, idx in train_flow.class_indices.items()}
    labels_path.write_text(json.dumps(id_to_label, ensure_ascii=False, indent=2), encoding="utf-8")

    model = CNNModel(img_size=img_size, num_classes=num_classes)
    model.train(train_flow, val_flow, epochs=30, save_path="model/model.h5")

if __name__ == "__main__":
    main()
