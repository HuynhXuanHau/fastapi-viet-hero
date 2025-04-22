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

import logging
logger = logging.getLogger(__name__)

def load_model():
    logger.info("Bắt đầu tải model ResNet50...")
    try:
        # Mã tải model
        logger.info("Đang tải các trọng số model...")
        # ...
        logger.info("Model đã được tải thành công")
        return model
    except Exception as e:
        logger.error(f"Lỗi khi tải model: {str(e)}")
        # Có thể raise lại exception hoặc xử lý phù hợp
        raise

def predict_image(image, model):
    logger.info("Bắt đầu xử lý ảnh và dự đoán...")
    try:
        # Mã dự đoán
        logger.info("Hoàn tất dự đoán")
        return result
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán ảnh: {str(e)}")
        raise