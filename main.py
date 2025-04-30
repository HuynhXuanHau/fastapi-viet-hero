import os
from fastapi import FastAPI
import psutil

# Tạo ứng dụng FastAPI
app = FastAPI(title="ResNet50 Image Classification API")

# Import routes sau khi cấu hình để giảm thiểu việc sử dụng bộ nhớ
@app.on_event("startup")
async def startup_event():
    # Import routes sau khi khởi động để trì hoãn việc tải model
    from app.routes import router
    app.include_router(router)

@app.get("/")
def root():
    return {"message": "ResNet50 API is live! Use /predict endpoint for image classification."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/memory")
def memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return {
        "memory_usage_mb": round(memory_mb, 2),
        "memory_limit_mb": 512,  # Render Free tier limit
        "memory_percent": round((memory_mb / 512) * 100, 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))