import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests


# Fixed path for Docker
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'resnet50_final_t4_optimized.keras')
MODEL_URL = "https://link-to-your-model/resnet50_final_t4_optimized.keras"

def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(requests.get(MODEL_URL).content)
def load_resnet_model():
    try:
        # Load the model without compiling it
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB").resize((224, 224))
    array = np.array(image)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.resnet50.preprocess_input(array)
    return array

def decode_prediction(preds):
    decoded = tf.keras.applications.resnet50.decode_predictions(preds, top=3)[0]
    return [
        {"label": label, "desc": desc, "confidence": float(score)}
        for (label, desc, score) in decoded
    ]

