from ai.predict import Predictor

img_path = r"data/tests/Image.jpg"   
predictor = Predictor(
    model_path="model/model.h5",
    labels_path="model/labels.json",
    img_size=(128, 128)
)

result = predictor.predict(img_path, show = True)

print("📸 Prediction:")
print(f"  Class: {result['label']}")
print(f"  Accuracy: {result['top_prob']*100:.2f}%")
