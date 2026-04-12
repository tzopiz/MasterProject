import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.detector_service import TMJDetectorService
from services.dicom_processor import DICOMProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Service for DICOM Processing (3D)", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
dicom_processor = DICOMProcessor()
detector_service: Optional[TMJDetectorService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize ML model on startup"""
    global detector_service
    logger.info("Starting ML Service...")

    # Try to load best detector model
    # Search in experiments/ for best_model.pth if not specified
    model_path = os.getenv("MODEL_PATH")

    if not model_path:
        # Try to find latest detector experiment
        try:
            exp_dir = "experiments"
            if os.path.exists(exp_dir):
                experiments = sorted([d for d in os.listdir(exp_dir) if d.startswith("detector_")])
                if experiments:
                    latest_exp = experiments[-1]
                    candidate = os.path.join(exp_dir, latest_exp, "best_model.pth")
                    if os.path.exists(candidate):
                        model_path = candidate
        except Exception as e:
            logger.warning(f"Error searching for model: {e}")

    # Fallback
    if not model_path:
        model_path = "models/tmj_detector_best.pth"

    detector_service = TMJDetectorService(
        model_path if model_path and os.path.exists(model_path) else None
    )

    logger.info("ML Service started successfully")


# Response Models


class BoundingBox(BaseModel):
    center: List[float]  # [z, y, x]
    bbox: List[int]  # [z1, y1, x1, z2, y2, x2]


class TMJResult(BaseModel):
    left: BoundingBox
    right: BoundingBox


class ProcessResponse(BaseModel):
    task_id: str
    status: str
    tmj: Optional[TMJResult] = None
    volume_shape: Optional[List[int]] = None  # [depth, height, width]
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
    model_loaded: bool


class ModelStatusResponse(BaseModel):
    model_loaded: bool
    model_type: str = "tmj_detector_3d"
    model_path: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        service="ml-service",
        timestamp=datetime.now(),
        model_loaded=detector_service is not None and detector_service.is_loaded(),
    )


@app.get("/models/status", response_model=ModelStatusResponse)
async def model_status():
    """Check ML model status"""
    if detector_service and detector_service.is_loaded():
        return ModelStatusResponse(
            model_loaded=True,
            model_path=detector_service.model.model_path
            if hasattr(detector_service.model, "model_path")
            else "loaded",
        )
    return ModelStatusResponse(model_loaded=False)


@app.post("/process", response_model=ProcessResponse)
async def process_dicom(task_id: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Process DICOM Series (3D):
    1. Save uploaded files to temp directory
    2. Load 3D Volume
    3. Run TMJ Detector
    4. Return Bounding Box
    """
    logger.info(f"Processing task: {task_id}, files: {len(files)}")

    temp_dir = None
    try:
        # 1. Save files
        temp_dir = tempfile.mkdtemp(prefix=f"task_{task_id}_")
        logger.info(f"Saving {len(files)} files to {temp_dir}")

        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

        # 2. Load Series
        logger.info("Loading 3D volume...")
        dicom_data = dicom_processor.load_series(temp_dir)

        if dicom_data is None:
            raise HTTPException(
                status_code=400, detail="Failed to load DICOM series from uploaded files"
            )

        volume = dicom_data["pixel_array"]  # 3D numpy array

        # 3. Run Detector
        logger.info("Running TMJ Detector...")
        detection_result = None

        if detector_service and detector_service.is_loaded():
            detection_result = detector_service.detect(volume)
        else:
            logger.warning("Detector not loaded, cannot process")
            return ProcessResponse(
                task_id=task_id, status="failed", error_message="Model not loaded"
            )

        if detection_result is None:
            return ProcessResponse(
                task_id=task_id, status="failed", error_message="Detection failed"
            )

        # 4. Return Result
        tmj_result = TMJResult(
            left=BoundingBox(
                center=detection_result["left"]["center"], bbox=detection_result["left"]["bbox"]
            ),
            right=BoundingBox(
                center=detection_result["right"]["center"], bbox=detection_result["right"]["bbox"]
            ),
        )

        logger.info(f"Task {task_id} completed successfully")

        return ProcessResponse(
            task_id=task_id,
            status="completed",
            tmj=tmj_result,
            volume_shape=list(volume.shape),  # [D, H, W]
        )

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
        return ProcessResponse(task_id=task_id, status="failed", error_message=str(e))

    finally:
        # Cleanup temp dir
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp dir: {temp_dir}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
