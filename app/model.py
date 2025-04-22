import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image

MODEL_PATH = "resnet50_final_t4_optimized.keras"
MODEL_URL = "https://drive.google.com/uc?id=12WQjGl1bxI8NSF1btoKBaZ_a9QGWrbqB&confirm=t"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if file_size < 100:
        raise ValueError(f"Model file too small: {file_size:.2f}MB")

def load_resnet_model():
    model = load_model(MODEL_PATH, compile=False)
    model.summary()  # optional, để in kiến trúc
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
