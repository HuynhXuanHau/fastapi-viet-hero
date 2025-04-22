# 1. Base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy project files
COPY ./app /app/app
COPY requirements.txt /app

# 4. Cài dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose port (Railway mặc định dùng 8000)
EXPOSE 8000

# 6. Chạy app FastAPI bằng Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
