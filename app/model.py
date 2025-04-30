import os
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import gc
import time

# URL để tải model
MODEL_URL = "https://huggingface.co/HXHau/fastapi-viet-hero/resolve/main/resnet50_final_t4_optimized.keras"
# Đường dẫn local để lưu và load model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'resnet50_final_t4_optimized.keras')


# Cài đặt các tùy chọn TensorFlow để tiết kiệm bộ nhớ
def configure_tensorflow_memory():
    # Giới hạn phân bổ bộ nhớ GPU (nếu có)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")

    # Cấu hình giới hạn phân bổ bộ nhớ cho CPU
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def download_model_if_missing():
    """Download model chỉ khi không tìm thấy nó."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading...")
        try:
            # Trước khi tải, dọn dẹp bộ nhớ
            gc.collect()

            # Tải model theo từng phần nhỏ để tiết kiệm bộ nhớ
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):  # 8KB chunks
                        f.write(chunk)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            # Nếu tải thất bại, xóa file tạm nếu có
            if os.path.exists(MODEL_PATH):
                try:
                    os.remove(MODEL_PATH)
                except:
                    pass
            raise


_model = None


def load_resnet_model():
    """Lazy-load model khi cần, chỉ giữ một phiên bản duy nhất."""
    global _model

    if _model is not None:
        return _model

    print(f"Attempting to load model from: {MODEL_PATH}")

    # Cấu hình TensorFlow để tiết kiệm bộ nhớ
    configure_tensorflow_memory()

    # Dọn dẹp bộ nhớ trước khi tải model
    gc.collect()

    # Đảm bảo model đã được tải về
    try:
        download_model_if_missing()
    except Exception as e:
        print(f"Warning: Could not download model: {e}")

    # Thử tải model
    try:
        # Tải model với tùy chọn tối ưu hóa bộ nhớ
        print("Loading model with memory optimization...")
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,  # Không biên dịch để tiết kiệm bộ nhớ
        )
        print("Model loaded successfully.")

        # Dọn dẹp bộ nhớ sau khi tải model
        gc.collect()
        return _model
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Nếu tải không thành công, xóa biến _model
        _model = None
        # Dọn dẹp bộ nhớ
        gc.collect()
        raise RuntimeError(f"Failed to load model: {e}")


def preprocess_image(image_bytes):
    """Tiền xử lý ảnh để đưa vào model."""
    image = Image.open(image_bytes).convert("RGB").resize((224, 224))
    array = np.array(image)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.resnet50.preprocess_input(array)
    return array


def decode_prediction(preds):
    """Giải mã kết quả dự đoán."""
    decoded = tf.keras.applications.resnet50.decode_predictions(preds, top=3)[0]
    return [
        {"label": label, "desc": desc, "confidence": float(score)}
        for (label, desc, score) in decoded
    ]