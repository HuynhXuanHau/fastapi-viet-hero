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


CLASS_NAMES = [
    "Bao_Dai",
    "Be_Van_Dan",
    "Bui_Van_Nguyen",
    "Bui_Xuan_Phai",
    "Che_Lan_Vien",
    "Cu_Chinh_Lan",
    "Dang_Thuy_Tram",
    "Do_Muoi",
    "Dong_Khanh",
    "Dong_Sy_Nguyen",
    "Duy_Tan",
    "Ha_Huy_Tap",
    "Ho_Chi_Minh",
    "Hoang_Dieu",
    "Hoang_Ngoc_Phach",
    "Hoang_Quoc_Viet",
    "Hoang_Quy",
    "Hoang_Van_Thu",
    "Huy_Can",
    "Khai_Dinh",
    "La_Van_Cau",
    "Le_Duan",
    "Le_Duc_Anh",
    "Le_Duc_Tho",
    "Le_Quang_Dao",
    "Le_Trong_Tan",
    "Le_Van_Dung",
    "Le_Van_Tam",
    "Luong_Dinh_Cua",
    "Nam_Cao",
    "Nguyen_Chi_Thanh",
    "Nguyen_Dinh_Thi",
    "Nguyen_Duy_Trinh",
    "Nguyen_Huu_Tho",
    "Nguyen_Luong_Bang",
    "Nguyen_Si_Sach",
    "Nguyen_Thai_Hoc",
    "Nguyen_Thi_Binh",
    "Nguyen_Thi_Dinh",
    "Nguyen_Thi_Minh_Khai",
    "Nguyen_Tuan",
    "Nguyen_Van_Bay",
    "Nguyen_Van_Linh",
    "Nguyen_Van_Troi",
    "Nguyen_Xuan_Khoat",
    "Pham_Duy",
    "Pham_Quynh",
    "Pham_Van_Lai",
    "Phan_Boi_Chau",
    "Phan_Chau_Trinh",
    "Phan_Dang_Luu",
    "Phan_Dinh_Giot",
    "Phan_Khoi",
    "Phan_Thanh_Gian",
    "Ta_Quang_Buu",
    "Thanh_Thai",
    "To_Huu",
    "Ton_Duc_Thang",
    "Ton_That_Thuyet",
    "Tran_Dai_Nghia",
    "Tran_Phu",
    "Tran_Trong_Kim",
    "Tran_Van_Tra",
    "Trinh_Dinh_Cuu",
    "Truong_Chinh",
    "Truong_Dinh",
    "Van_Tien_Dung",
    "Vo_Nguyen_Giap",
    "Vo_Thi_Sau",
    "Vo_Thi_Thang",
    "Vo_Van_Huyen",
    "Vo_Van_Kiet",
    "Vu_Mao",
    "Vu_Ngoc_Phan",
    "Vu_Quang_Huy",
    "Vu_Thu_Hien",
    "Xuan_Dieu",
    "Xuan_Thuy"
]


def decode_prediction(preds, class_names=CLASS_NAMES):
    """
    Giải mã kết quả dự đoán
    :param preds: Kết quả dự đoán từ model
    :param class_names: Danh sách tên class (nếu không có sẽ trả về dạng Class X)
    """
    # Lấy top 3 predictions
    top_indices = np.argsort(preds[0])[-3:][::-1]
    top_probs = preds[0][top_indices]

    if class_names is not None:
        return [
            {"label": str(idx), "name": class_names[idx], "confidence": float(prob)}
            for idx, prob in zip(top_indices, top_probs)
        ]
    else:
        return [
            {"label": str(idx), "name": f"Class {idx}", "confidence": float(prob)}
            for idx, prob in zip(top_indices, top_probs)
        ]