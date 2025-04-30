from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import io
import time
import gc

router = APIRouter()

# Lazy load model chỉ khi cần thiết
_model = None


def get_model():
    global _model
    if _model is None:
        from app.model import load_resnet_model
        _model = load_resnet_model()
    return _model


@router.get("/api")
def api_info():
    return {
        "name": "ResNet50 Image Classification API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Classify an image",
            "/health": "GET - Check API health",
            "/memory": "GET - Check memory usage"
        }
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Nhận ảnh và trả về kết quả phân loại.
    """
    # Kiểm tra định dạng file
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid file type: {file.content_type}. Only image files are accepted."}
        )

    try:
        # Đọc nội dung file
        contents = await file.read()

        # Xử lý ảnh và dự đoán
        from app.model import preprocess_image, decode_prediction

        # Tiền xử lý ảnh
        processed = preprocess_image(io.BytesIO(contents))

        # Lấy model và dự đoán
        model = get_model()
        preds = model.predict(processed)

        # Giải mã kết quả
        results = decode_prediction(preds)

        # Thêm thông tin file
        response = {
            "filename": file.filename,
            "content_type": file.content_type,
            "predictions": results
        }

        # Dọn dẹp bộ nhớ trong background
        if background_tasks:
            background_tasks.add_task(cleanup_memory)

        return response
    except Exception as e:
        # Dọn dẹp bộ nhớ nếu có lỗi
        gc.collect()
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )


def cleanup_memory():
    """Dọn dẹp bộ nhớ sau khi xử lý"""
    time.sleep(1)  # Đợi cho quá trình hoàn tất
    gc.collect()