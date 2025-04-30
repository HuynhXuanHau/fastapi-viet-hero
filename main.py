import os
from unittest import result

import psutil
from fastapi import FastAPI
from app.routes import router

app = FastAPI()
app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok"}



# Trong các endpoint
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     logger.info(f"Nhận yêu cầu dự đoán với file: {file.filename}")
#     try:
#         # Mã xử lý...
#         logger.info("Xử lý dự đoán thành công")
#         return result
#     except Exception as e:
#         logger.error(f"Lỗi khi dự đoán: {str(e)}")
#         raise


# @app.get("/memory")
# async def memory_usage():
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     memory_mb = memory_info.rss / 1024 / 1024
#     logger.info(f"Sử dụng bộ nhớ hiện tại: {memory_mb:.2f} MB")
#     return {"memory_usage_mb": memory_mb}


# import asyncio
#
# async def load_model_async():
#     # Tải model trong một task riêng biệt
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, load_model)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
