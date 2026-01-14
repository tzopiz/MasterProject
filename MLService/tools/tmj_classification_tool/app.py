#!/usr/bin/env python3
"""
TMJ Classification Tool - Web Application
Инструмент для разметки DICOM исследований ВНЧС с визуализацией в 3 плоскостях
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Setup paths
TOOL_DIR = Path(__file__).parent
MLSERVICE_DIR = TOOL_DIR.parent.parent

# Add MLService first, then tool directory
sys.path.insert(0, str(MLSERVICE_DIR))
sys.path.insert(1, str(TOOL_DIR))

# Import tool services with explicit path to avoid conflicts
import importlib
import importlib.util

# Import FileCleaner from tool's services
spec = importlib.util.spec_from_file_location(
    "tool_file_cleaner", 
    TOOL_DIR / "services" / "file_cleaner.py"
)
file_cleaner_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(file_cleaner_mod)
FileCleaner = file_cleaner_mod.FileCleaner

# Import DICOMLoader from tool's services  
spec = importlib.util.spec_from_file_location(
    "tool_dicom_loader",
    TOOL_DIR / "services" / "dicom_loader.py"
)
dicom_loader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dicom_loader_mod)
DICOMLoader = dicom_loader_mod.DICOMLoader

# Import AnnotationManager from tool's services
spec = importlib.util.spec_from_file_location(
    "tool_annotation_manager",
    TOOL_DIR / "services" / "annotation_manager.py"
)
annotation_manager_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotation_manager_mod)
AnnotationManager = annotation_manager_mod.AnnotationManager

# Import SliceExtractor from tool's utils
spec = importlib.util.spec_from_file_location(
    "tool_slice_extractor",
    TOOL_DIR / "utils" / "slice_extractor.py"
)
slice_extractor_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slice_extractor_mod)
SliceExtractor = slice_extractor_mod.SliceExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TMJ Classification Tool",
    description="Web tool for annotating TMJ DICOM studies",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(TOOL_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(TOOL_DIR / "templates"))

# Initialize services
file_cleaner = FileCleaner()
dicom_loader = DICOMLoader()
annotation_manager = AnnotationManager()
slice_extractor = SliceExtractor()


# Pydantic models
class ScanRequest(BaseModel):
    patients_dir: str


class CleanRequest(BaseModel):
    patients_dir: str
    dry_run: bool = False


class AnnotateRequest(BaseModel):
    patient_id: str
    study_id: str
    left_joint_tag: str
    right_joint_tag: str


class AddTagRequest(BaseModel):
    tag_name: str


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/annotate/{patient_id}/{study_id}", response_class=HTMLResponse)
async def annotate_page(request: Request, patient_id: str, study_id: str):
    """Annotation page for a specific study"""
    return templates.TemplateResponse("annotate.html", {
        "request": request,
        "patient_id": patient_id,
        "study_id": study_id
    })


@app.post("/api/scan_patients")
async def scan_patients(request: ScanRequest) -> Dict:
    """Scan patients directory and find all DICOM studies"""
    try:
        patients_dir = Path(request.patients_dir)
        if not patients_dir.exists():
            raise HTTPException(status_code=400, detail="Directory not found")
        
        logger.info(f"Scanning patients directory: {patients_dir}")
        studies = dicom_loader.scan_studies(patients_dir)
        
        # Load existing annotations
        annotations = annotation_manager.load_annotations()
        annotated_ids = {
            f"{ann['patient_id']}_{ann['study_id']}" 
            for ann in annotations.get('annotations', [])
        }
        
        # Mark annotated studies
        for study in studies:
            study_key = f"{study['patient_id']}_{study['study_id']}"
            study['is_annotated'] = study_key in annotated_ids
            if study['is_annotated']:
                # Find annotation
                for ann in annotations.get('annotations', []):
                    if ann['patient_id'] == study['patient_id'] and ann['study_id'] == study['study_id']:
                        study['left_joint_tag'] = ann.get('left_joint_tag')
                        study['right_joint_tag'] = ann.get('right_joint_tag')
                        break
        
        return {
            "success": True,
            "studies_count": len(studies),
            "annotated_count": len(annotated_ids),
            "studies": studies
        }
    
    except Exception as e:
        logger.error(f"Error scanning patients: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clean_files")
async def clean_files(request: CleanRequest) -> Dict:
    """Clean junk files from patients directories"""
    try:
        patients_dir = Path(request.patients_dir)
        if not patients_dir.exists():
            raise HTTPException(status_code=400, detail="Directory not found")
        
        logger.info(f"Cleaning files in: {patients_dir} (dry_run={request.dry_run})")
        result = file_cleaner.clean_directory(patients_dir, dry_run=request.dry_run)
        
        return {
            "success": True,
            "files_removed": result['files_removed'],
            "files_scanned": result['files_scanned'],
            "dry_run": request.dry_run
        }
    
    except Exception as e:
        logger.error(f"Error cleaning files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/studies")
async def get_studies() -> Dict:
    """Get all studies from last scan"""
    try:
        studies = dicom_loader.get_cached_studies()
        return {
            "success": True,
            "studies": studies
        }
    
    except Exception as e:
        logger.error(f"Error getting studies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/study/{patient_id}/{study_id}")
async def get_study(patient_id: str, study_id: str) -> Dict:
    """Load a specific study"""
    try:
        study_info = dicom_loader.load_study(patient_id, study_id)
        
        if not study_info:
            raise HTTPException(status_code=404, detail="Study not found")
        
        # Check if it's a decompression error
        if isinstance(study_info, dict) and 'error' in study_info:
            if study_info['error'] == 'dicom_decompression':
                raise HTTPException(
                    status_code=500, 
                    detail=f"{study_info['message']} {study_info['solution']}"
                )
        
        return {
            "success": True,
            "study": study_info
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading study: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/slice/{patient_id}/{study_id}/{plane}/{index}")
async def get_slice(patient_id: str, study_id: str, plane: str, index: int) -> Dict:
    """Get a specific slice from a study"""
    try:
        # Get volume from loader
        volume = dicom_loader.get_volume(patient_id, study_id)
        
        if volume is None:
            raise HTTPException(status_code=404, detail="Study not loaded")
        
        # Extract slice
        slice_base64 = slice_extractor.get_slice(volume, plane, index)
        
        if slice_base64 is None:
            raise HTTPException(status_code=400, detail="Invalid slice parameters")
        
        return {
            "success": True,
            "slice": slice_base64,
            "plane": plane,
            "index": index
        }
    
    except Exception as e:
        logger.error(f"Error getting slice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/annotate")
async def annotate_study(request: AnnotateRequest) -> Dict:
    """Save annotation for a study"""
    try:
        # Get study path
        study_info = dicom_loader.get_study_info(request.patient_id, request.study_id)
        
        if not study_info:
            raise HTTPException(status_code=404, detail="Study not found")
        
        # Save annotation
        annotation_manager.save_annotation(
            patient_id=request.patient_id,
            study_id=request.study_id,
            study_path=study_info['study_path'],
            left_joint_tag=request.left_joint_tag,
            right_joint_tag=request.right_joint_tag
        )
        
        return {
            "success": True,
            "message": "Annotation saved"
        }
    
    except Exception as e:
        logger.error(f"Error saving annotation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/annotations")
async def get_annotations() -> Dict:
    """Get all annotations"""
    try:
        annotations = annotation_manager.load_annotations()
        return {
            "success": True,
            "annotations": annotations
        }
    
    except Exception as e:
        logger.error(f"Error loading annotations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/add_tag")
async def add_tag(request: AddTagRequest) -> Dict:
    """Add a new tag to available tags"""
    try:
        annotation_manager.add_tag(request.tag_name)
        
        return {
            "success": True,
            "tag": request.tag_name
        }
    
    except Exception as e:
        logger.error(f"Error adding tag: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tags")
async def get_tags() -> Dict:
    """Get all available tags"""
    try:
        annotations = annotation_manager.load_annotations()
        tags = annotations.get('available_tags', [])
        
        return {
            "success": True,
            "tags": tags
        }
    
    except Exception as e:
        logger.error(f"Error loading tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting TMJ Classification Tool...")
    logger.info("Open http://localhost:8000 in your browser")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
