FROM python:3.10-slim

# 1. Làm việc trong thư mục /app
WORKDIR /app

# 2. Cài các thư viện cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Cài wget
RUN apt update && apt install -y wget

# 4. Copy code FastAPI (trước!)
COPY ./app /app/app

# 5. Tạo thư mục models/ và tải model từ HuggingFace
RUN mkdir -p /app/app/models
RUN wget https://huggingface.co/HXHau/fastapi-viet-hero/resolve/main/resnet50_final_t4_optimized.keras -O /app/app/models/resnet_model.keras

# Đảm bảo không tắt logs
ENV PYTHONUNBUFFERED=1

# 6. Khởi chạy API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
