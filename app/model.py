import os
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import gc
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = "https://huggingface.co/HXHau/fastapi-viet-hero/resolve/main/resnet50_final_t4_optimized.keras"
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'resnet50_final_t4_optimized.keras')
MODEL_CACHE_PATH = os.path.join(tempfile.gettempdir(), 'cached_model.keras')

# List of Vietnamese heroes (78 classes)
CLASS_NAMES = [
    "Bao_Dai", "Be_Van_Dan", "Bui_Van_Nguyen", "Bui_Xuan_Phai", "Che_Lan_Vien",
    "Cu_Chinh_Lan", "Dang_Thuy_Tram", "Do_Muoi", "Dong_Khanh", "Dong_Sy_Nguyen",
    "Duy_Tan", "Ha_Huy_Tap", "Ho_Chi_Minh", "Hoang_Dieu", "Hoang_Ngoc_Phach",
    "Hoang_Quoc_Viet", "Hoang_Quy", "Hoang_Van_Thu", "Huy_Can", "Khai_Dinh",
    "La_Van_Cau", "Le_Duan", "Le_Duc_Anh", "Le_Duc_Tho", "Le_Quang_Dao",
    "Le_Trong_Tan", "Le_Van_Dung", "Le_Van_Tam", "Luong_Dinh_Cua", "Nam_Cao",
    "Nguyen_Chi_Thanh", "Nguyen_Dinh_Thi", "Nguyen_Duy_Trinh", "Nguyen_Huu_Tho",
    "Nguyen_Luong_Bang", "Nguyen_Si_Sach", "Nguyen_Thai_Hoc", "Nguyen_Thi_Binh",
    "Nguyen_Thi_Dinh", "Nguyen_Thi_Minh_Khai", "Nguyen_Tuan", "Nguyen_Van_Bay",
    "Nguyen_Van_Linh", "Nguyen_Van_Troi", "Nguyen_Xuan_Khoat", "Pham_Duy",
    "Pham_Quynh", "Pham_Van_Lai", "Phan_Boi_Chau", "Phan_Chau_Trinh",
    "Phan_Dang_Luu", "Phan_Dinh_Giot", "Phan_Khoi", "Phan_Thanh_Gian",
    "Ta_Quang_Buu", "Thanh_Thai", "To_Huu", "Ton_Duc_Thang", "Ton_That_Thuyet",
    "Tran_Dai_Nghia", "Tran_Phu", "Tran_Trong_Kim", "Tran_Van_Tra",
    "Trinh_Dinh_Cuu", "Truong_Chinh", "Truong_Dinh", "Van_Tien_Dung",
    "Vo_Nguyen_Giap", "Vo_Thi_Sau", "Vo_Thi_Thang", "Vo_Van_Huyen",
    "Vo_Van_Kiet", "Vu_Mao", "Vu_Ngoc_Phan", "Vu_Quang_Huy", "Vu_Thu_Hien",
    "Xuan_Dieu", "Xuan_Thuy"
]


def configure_tensorflow_memory():
    """Configure TensorFlow to optimize memory usage"""
    try:
        # Set memory growth for GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Limit CPU threads
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception as e:
        logger.warning(f"Memory configuration error: {e}")


def download_model():
    """Download the model from the specified URL"""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        logger.info(f"Downloading model from {MODEL_URL}")
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()

        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Model downloaded successfully")
        return load_model_file(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise RuntimeError(f"Model download failed: {e}")


def load_model_file(model_path):
    """Load model from file with error handling"""
    try:
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


_model_instance = None


def load_resnet_model():
    """Load model with caching mechanism"""
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    configure_tensorflow_memory()
    gc.collect()

    try:
        # Try to load from cache first
        if os.path.exists(MODEL_CACHE_PATH):
            logger.info("Loading model from cache")
            _model_instance = load_model_file(MODEL_CACHE_PATH)
            return _model_instance

        # Try to load from local model path
        if os.path.exists(MODEL_PATH):
            logger.info("Loading model from local storage")
            _model_instance = load_model_file(MODEL_PATH)

            # Cache the model for future use
            _model_instance.save(MODEL_CACHE_PATH)
            logger.info(f"Model cached at {MODEL_CACHE_PATH}")
            return _model_instance

        # Download the model if not found locally
        _model_instance = download_model()

        # Cache the downloaded model
        _model_instance.save(MODEL_CACHE_PATH)
        logger.info(f"Model cached at {MODEL_CACHE_PATH}")
        return _model_instance

    except Exception as e:
        _model_instance = None
        gc.collect()
        logger.error(f"Critical error loading model: {e}")
        raise


def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    try:
        image = Image.open(image_bytes).convert("RGB").resize((224, 224))
        array = np.array(image)
        array = np.expand_dims(array, axis=0)
        array = tf.keras.applications.resnet50.preprocess_input(array)
        return array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Invalid image: {e}")


def decode_prediction(preds, class_names=CLASS_NAMES):
    """Decode model predictions to human-readable format"""
    try:
        # Get top 3 predictions
        top_indices = np.argsort(preds[0])[-3:][::-1]
        top_probs = preds[0][top_indices]

        # Normalize probabilities to sum to 1
        top_probs = top_probs / np.sum(top_probs)

        return [
            {
                "label": str(idx),
                "name": class_names[idx] if class_names else f"Class {idx}",
                "confidence": float(prob)
            }
            for idx, prob in zip(top_indices, top_probs)
        ]
    except Exception as e:
        logger.error(f"Prediction decoding failed: {e}")
        raise ValueError(f"Failed to decode predictions: {e}")