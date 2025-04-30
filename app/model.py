import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

# URL để tải model
MODEL_URL = "https://huggingface.co/HXHau/fastapi-viet-hero/resolve/main/resnet50_final_t4_optimized.keras"
# Đường dẫn local để lưu và load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/resnet50_final_t4_optimized.keras')


def download_model_if_missing():
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading...")
        try:
            response = requests.get(MODEL_URL)
            response.raise_for_status()  # Gây lỗi nếu request thất bại
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise


def load_resnet_model():
    print(f"Attempting to load model from: {MODEL_PATH}")

    # Đảm bảo model đã được tải về
    try:
        download_model_if_missing()
    except Exception as e:
        print(f"Warning: Could not download model: {e}")
        # Tiếp tục và thử load model nếu file đã tồn tại

    # Thử load model với nhiều cách khác nhau
    try:
        # Cách 1: Sử dụng tf.keras.models.load_model
        print("Loading model with tf.keras.models.load_model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully with tf.keras")
        return model
    except Exception as e:
        print(f"First attempt failed: {e}")

        try:
            # Cách 2: Sử dụng keras.models.load_model
            print("Trying alternative loading method...")
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH, compile=False)
            print("Model loaded successfully with keras")
            return model
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            # Nếu cả hai cách đều thất bại, raise lỗi
            raise RuntimeError(f"Failed to load model after multiple attempts: {e}, then {e2}")


def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB").resize((224, 224))
    array = np.array(image)
    array = np.expand_dims(array, axis=0)
    # Sử dụng hàm preprocess_input từ tf.keras.applications
    array = tf.keras.applications.resnet50.preprocess_input(array)
    return array


def decode_prediction(preds):
    # Sử dụng hàm decode_predictions từ tf.keras.applications
    decoded = tf.keras.applications.resnet50.decode_predictions(preds, top=3)[0]
    return [
        {"label": label, "desc": desc, "confidence": float(score)}
        for (label, desc, score) in decoded
    ]