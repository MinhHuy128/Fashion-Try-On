from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import io
from PIL import Image
import sys
import os
from fastapi.staticfiles import StaticFiles

# Ensure parent directory and training directory are in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
TRAINING_DIR = os.path.join(ROOT_DIR, "training")
if TRAINING_DIR not in sys.path:
    sys.path.append(TRAINING_DIR)

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
    garment_image: UploadFile = File(...),
    prompt: str = Form("A highly detailed, photorealistic image of a person wearing this exact t-shirt, perfect texture and logo matching, 8k resolution")
):
    try:
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        
        result = pipeline.predict(person_img, garment_img, prompt=prompt)
        
        output_io = io.BytesIO()
        result["output_image"].save(output_io, format="JPEG")
        output_bytes = output_io.getvalue()
        
        headers = {"X-Latency-Ms": str(round(result["latency_ms"], 2))}
        return Response(content=output_bytes, media_type="image/jpeg", headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend at the root
frontend_dir = os.path.join(ROOT_DIR, "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

