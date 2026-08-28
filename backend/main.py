import time
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pipeline import RecognitionPipeline
from models.schemas import RecognitionResult, ErrorResponse

app = FastAPI(title="Cambodian Plate Recognition", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RecognitionPipeline()


@app.get("/")
async def root():
    return {"message": "Cambodian Plate Recognition API", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vehicle_model": pipeline.vehicle_model is not None,
        "plate_model": pipeline.plate_model is not None,
        "ocr": pipeline.ocr is not None
    }


@app.post("/api/recognize", response_model=RecognitionResult)
async def recognize(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        start_time = time.time()
        result = pipeline.predict(image)
        result["processing_time"] = round(time.time() - start_time, 3)

        return result

    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error=str(e)).model_dump())
    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error=str(e)).model_dump())
