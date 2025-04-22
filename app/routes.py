from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.model import load_resnet_model, preprocess_image, decode_prediction
import io

router = APIRouter()

model = load_resnet_model()

@router.get("/")
def root():
    return {"message": "ResNet50 API is live!"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Invalid file type"})

    try:
        contents = await file.read()
        processed = preprocess_image(io.BytesIO(contents))
        preds = model.predict(processed)
        results = decode_prediction(preds)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
