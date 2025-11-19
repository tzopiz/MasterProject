from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
from datetime import datetime
import os

from services.dicom_processor import DICOMProcessor
from services.slice_finder import SliceFinder
from services.geometry_calculator import GeometryCalculator
from services.diagnosis_engine import DiagnosisEngine
from models.segmentation_model import SegmentationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Service for DICOM Processing", version="1.0.0")

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
slice_finder = SliceFinder()
geometry_calculator = GeometryCalculator()
diagnosis_engine = DiagnosisEngine()
segmentation_model: Optional[SegmentationModel] = None


@app.on_event("startup")
async def startup_event():
    """Initialize ML model on startup"""
    global segmentation_model
    logger.info("Starting ML Service...")
    
    # Try to load segmentation model if it exists
    try:
        model_path = os.getenv("MODEL_PATH", "models/segmentation_model.pth")
        if os.path.exists(model_path):
            segmentation_model = SegmentationModel(model_path)
            logger.info(f"Segmentation model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Using dummy mode.")
            segmentation_model = SegmentationModel(None)  # Dummy mode
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        segmentation_model = SegmentationModel(None)  # Dummy mode
    
    logger.info("ML Service started successfully")


# Request/Response models
class ProcessRequest(BaseModel):
    dicom_path: str
    task_id: str


class SlicesData(BaseModel):
    orthogonal: Optional[List[str]] = None
    sagittal: Optional[List[str]] = None
    frontal: Optional[List[str]] = None


class MasksData(BaseModel):
    orthogonal: Optional[List[str]] = None
    sagittal: Optional[List[str]] = None
    frontal: Optional[List[str]] = None


class GeometricParameters(BaseModel):
    fossa_height: Optional[float] = None
    head_height: Optional[float] = None
    width: Optional[float] = None
    additional_params: Optional[Dict[str, float]] = None


class DiagnosisData(BaseModel):
    status: str
    confidence: Optional[float] = None
    recommendations: Optional[List[str]] = None
    disclaimer: Optional[str] = None


class ProcessResponse(BaseModel):
    task_id: str
    status: str
    slices: Optional[SlicesData] = None
    masks: Optional[MasksData] = None
    parameters: Optional[GeometricParameters] = None
    diagnosis: Optional[DiagnosisData] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
    model_loaded: bool


class ModelStatusResponse(BaseModel):
    model_loaded: bool
    model_type: Optional[str] = None
    model_path: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        service="ml-service",
        timestamp=datetime.now(),
        model_loaded=segmentation_model is not None and segmentation_model.is_loaded()
    )


@app.get("/models/status", response_model=ModelStatusResponse)
async def model_status():
    """Check ML model status"""
    if segmentation_model and segmentation_model.is_loaded():
        return ModelStatusResponse(
            model_loaded=True,
            model_type="segmentation",
            model_path=segmentation_model.model_path
        )
    return ModelStatusResponse(
        model_loaded=False,
        model_type=None,
        model_path=None
    )


@app.post("/process", response_model=ProcessResponse)
async def process_dicom(request: ProcessRequest):
    """
    Process DICOM file:
    1. Parse DICOM file
    2. Find relevant slices
    3. Run segmentation
    4. Calculate geometric parameters
    5. Generate diagnosis
    """
    logger.info(f"Processing task: {request.task_id}, file: {request.dicom_path}")
    
    try:
        # Step 1: Parse DICOM file
        logger.info("Step 1: Parsing DICOM file...")
        dicom_data = dicom_processor.load_dicom(request.dicom_path)
        
        if dicom_data is None:
            raise HTTPException(status_code=400, reason="Failed to load DICOM file")
        
        # Step 2: Find relevant slices
        logger.info("Step 2: Finding relevant slices...")
        slices_indices = slice_finder.find_slices(dicom_data)
        
        # Extract slice images
        slices_data = SlicesData(
            orthogonal=slices_indices.get("orthogonal", []),
            sagittal=slices_indices.get("sagittal", []),
            frontal=slices_indices.get("frontal", [])
        )
        
        # Step 3: Run segmentation
        logger.info("Step 3: Running segmentation...")
        if segmentation_model and segmentation_model.is_loaded():
            masks = segmentation_model.segment(dicom_data, slices_indices)
        else:
            logger.warning("Model not loaded, using dummy masks")
            masks = {
                "orthogonal": ["dummy_mask_base64"] if slices_data.orthogonal else [],
                "sagittal": ["dummy_mask_base64"] if slices_data.sagittal else [],
                "frontal": ["dummy_mask_base64"] if slices_data.frontal else []
            }
        
        masks_data = MasksData(
            orthogonal=masks.get("orthogonal"),
            sagittal=masks.get("sagittal"),
            frontal=masks.get("frontal")
        )
        
        # Step 4: Calculate geometric parameters
        logger.info("Step 4: Calculating geometric parameters...")
        params = geometry_calculator.calculate(dicom_data, masks, slices_indices)
        
        parameters = GeometricParameters(
            fossa_height=params.get("fossa_height"),
            head_height=params.get("head_height"),
            width=params.get("width"),
            additional_params=params.get("additional_params")
        )
        
        # Step 5: Generate diagnosis
        logger.info("Step 5: Generating diagnosis...")
        diagnosis_result = diagnosis_engine.diagnose(params)
        
        diagnosis = DiagnosisData(
            status=diagnosis_result.get("status", "unknown"),
            confidence=diagnosis_result.get("confidence"),
            recommendations=diagnosis_result.get("recommendations"),
            disclaimer=diagnosis_result.get("disclaimer")
        )
        
        logger.info(f"Task {request.task_id} completed successfully")
        
        return ProcessResponse(
            task_id=request.task_id,
            status="completed",
            slices=slices_data,
            masks=masks_data,
            parameters=parameters,
            diagnosis=diagnosis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing task {request.task_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

