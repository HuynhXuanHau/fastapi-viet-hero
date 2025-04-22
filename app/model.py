import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image

# Đường dẫn cố định trong Docker
MODEL_PATH = "app/models/resnet_model.keras"


def load_resnet_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model tại {MODEL_PATH}. Hãy chắc chắn rằng Docker đã tải model đúng.")

    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
    return model


def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB").resize((224, 224))
    array = np.array(image)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def decode_prediction(preds):
    decoded = decode_predictions(preds, top=3)[0]
    return [
        {"label": label, "desc": desc, "confidence": float(score)}
        for (label, desc, score) in decoded
    ]
