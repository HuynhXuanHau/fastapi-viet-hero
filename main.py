import os

import numpy as np
from fastapi import FastAPI
import psutil
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router, get_model

# Tạo ứng dụng FastAPI
# Thêm CORS middleware để cho phép truy cập từ web client
app = FastAPI(title="ResNet50 Image Classification API",
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)

# Import routes sau khi cấu hình để giảm thiểu việc sử dụng bộ nhớ
@app.on_event("startup")
async def startup_event():
    """Thực hiện các tác vụ khởi tạo app."""
    # Import routes sau khi khởi động để trì hoãn việc tải model
    app.include_router(router)

@app.get("/")
def root():
    return {"message": "ResNet50 API is live! Use /predict endpoint for image classification."}

@app.get("/health")
def health_check():
    # Gọi model nhẹ để giữ nó trong RAM
    try:
        get_model().predict(np.zeros((1, 224, 224, 3)))
        return {"status": "ok"}
    except:
        return {"status": "error"}, 500

@app.get("/memory")
def memory_usage():
    """Endpoint theo dõi việc sử dụng bộ nhớ."""
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