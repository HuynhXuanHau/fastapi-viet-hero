import os
# from unittest import result
import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import tensorflow as tf
# import logging

# logger = logging.getLogger(__name__)

# Đường dẫn cố định trong Docker
# MODEL_PATH = "app/models/resnet50_final_t4_optimized.keras"


def load_resnet_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/resnet50_final_t4_optimized.keras')
    try:
        from tensorflow.keras.models import load_model

        # Thử cách đơn giản nhất trước
        # return load_model(MODEL_PATH)
        model = load_model(MODEL_PATH)
        print("Mô hình đã được tải thành công")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e2:
            print(f"Second attempt failed: {e2}")


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


