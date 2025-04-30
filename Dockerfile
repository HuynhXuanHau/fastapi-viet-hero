FROM python:3.10-slim

# Tạo thư mục làm việc
WORKDIR /app

# Cài đặt biến môi trường để giảm kích thước image và tối ưu hóa TensorFlow
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

# Cài đặt các dependencies cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt wget
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy mã nguồn
COPY ./app /app/app

# Tạo thư mục models và tải model
RUN mkdir -p /app/app/models
# Chỉ tải model khi khởi động container (không tải khi build)
# File sẽ được tải khi API nhận request đầu tiên

# Expose cổng mặc định
EXPOSE 8000

# Chạy ứng dụng với 1 worker để tiết kiệm bộ nhớ
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]