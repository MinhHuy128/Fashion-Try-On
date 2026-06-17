from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import io
from PIL import Image
import sys
import os
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.inference import FitAIInferencePipeline

app = FastAPI(title="FitAI Virtual Try-On API", version="1.0.0")

pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = FitAIInferencePipeline()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/predict")
async def predict(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    try:
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        
        result = pipeline.predict(person_img, garment_img)
        
        output_io = io.BytesIO()
        result["output_image"].save(output_io, format="JPEG")
        output_bytes = output_io.getvalue()
        
        headers = {"X-Latency-Ms": str(result["latency_ms"])}
        return Response(content=output_bytes, media_type="image/jpeg", headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend at the root
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
