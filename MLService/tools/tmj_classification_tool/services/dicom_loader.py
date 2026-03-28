"""
DICOM Loader Service
Scans patient directories and loads DICOM studies
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pydicom
from datetime import datetime

# Add MLService directory to path for DICOMProcessor import
MLSERVICE_DIR = Path(__file__).parent.parent.parent
if str(MLSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(MLSERVICE_DIR))

# Import DICOMProcessor from MLService/services
from services.dicom_processor import DICOMProcessor

logger = logging.getLogger(__name__)


def convert_dicom_value(value: Any) -> Any:
    """Convert PyDICOM values to JSON-serializable Python types"""
    if value is None:
        return None
    
    # Handle dict recursively
    if isinstance(value, dict):
        return {k: convert_dicom_value(v) for k, v in value.items()}
    
    # Handle PyDICOM MultiValue
    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, dict)):
        try:
            return [convert_dicom_value(v) for v in value]
        except:
            return str(value)
    
    # Handle numpy types
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    
    # Handle standard types
    if isinstance(value, (int, float, str, bool)):
        return value
    
    # Convert to string as fallback
    return str(value)


class DICOMLoader:
    """Service for scanning and loading DICOM studies"""
    
    def __init__(self):
        self.dicom_processor = DICOMProcessor()
        self.studies_cache = []  # List of found studies
        self.volume_cache = {}   # Cache for loaded volumes {study_key: volume}
        self.study_info_cache = {}  # Cache for study metadata
        self.last_patients_dir = None  # Store last scanned directory
        self._load_last_scan_path()
    
    def _load_last_scan_path(self):
        """Load and restore last scanned directory path"""
        cache_file = Path(__file__).parent.parent / "data" / "last_scan_path.txt"
        try:
            if cache_file.exists():
                self.last_patients_dir = cache_file.read_text().strip()
                if self.last_patients_dir and Path(self.last_patients_dir).exists():
                    logger.info(f"Restoring scan from last path: {self.last_patients_dir}")
                    self.scan_studies(Path(self.last_patients_dir))
        except Exception as e:
            logger.warning(f"Could not restore last scan: {e}")
    
    def _save_last_scan_path(self, patients_dir: Path):
        """Save last scanned directory path"""
        cache_file = Path(__file__).parent.parent / "data" / "last_scan_path.txt"
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(str(patients_dir))
        except Exception as e:
            logger.warning(f"Could not save last scan path: {e}")
    
    def scan_studies(self, patients_dir: Path) -> List[Dict]:
        """
        Scan patients directory and find all DICOM studies
        
        Structure: patients_dir/patient_name/patient_id/date/series_uid/*.dcm
        
        Args:
            patients_dir: Root directory containing patient folders
            
        Returns:
            List of study dictionaries
        """
        studies = []
        patients_dir = Path(patients_dir)
        
        if not patients_dir.exists():
            logger.error(f"Patients directory not found: {patients_dir}")
            return []
        
        logger.info(f"Scanning patients directory: {patients_dir}")
        
        # Save this path for future restarts
        self._save_last_scan_path(patients_dir)
        self.last_patients_dir = str(patients_dir)
        
        # Iterate through patient folders
        for patient_folder in patients_dir.iterdir():
            if not patient_folder.is_dir():
                continue
            
            # Skip hidden folders
            if patient_folder.name.startswith('.'):
                continue
            
            patient_name = patient_folder.name
            logger.info(f"Scanning patient: {patient_name}")
            
            # Look for study structure: patient_id/date/series_uid/*.dcm
            try:
                studies_found = self._scan_patient_folder(patient_folder, patient_name)
                studies.extend(studies_found)
            except Exception as e:
                logger.error(f"Error scanning patient {patient_name}: {e}")
        
        logger.info(f"Found {len(studies)} studies total")
        self.studies_cache = studies
        return studies
    
    def _scan_patient_folder(self, patient_folder: Path, patient_name: str) -> List[Dict]:
        """Scan a single patient folder for studies"""
        studies = []
        
        # Look for patient_id folders (numeric or UUID-like)
        for patient_id_folder in patient_folder.iterdir():
            if not patient_id_folder.is_dir():
                continue
            
            if patient_id_folder.name.startswith('.'):
                continue
            
            patient_id = patient_id_folder.name
            
            # Look for date folders
            for date_folder in patient_id_folder.iterdir():
                if not date_folder.is_dir():
                    continue
                
                if date_folder.name.startswith('.'):
                    continue
                
                study_date = date_folder.name
                
                # Look for series folders (DICOM series UID)
                for series_folder in date_folder.iterdir():
                    if not series_folder.is_dir():
                        continue
                    
                    if series_folder.name.startswith('.'):
                        continue
                    
                    # Check if this folder contains DICOM files
                    dcm_files = list(series_folder.glob('*.dcm'))
                    
                    if dcm_files:
                        series_uid = series_folder.name
                        
                        # Create study info
                        study_info = {
                            'patient_name': patient_name,
                            'patient_id': patient_id,
                            'study_id': f"{patient_id}_{study_date}_{series_uid[:8]}",
                            'study_date': study_date,
                            'series_uid': series_uid,
                            'study_path': str(series_folder),
                            'num_files': len(dcm_files),
                            'is_annotated': False
                        }
                        
                        # Try to load metadata from first DICOM
                        try:
                            first_dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                            study_info['metadata'] = {
                                'modality': convert_dicom_value(getattr(first_dcm, 'Modality', 'Unknown')),
                                'series_description': convert_dicom_value(getattr(first_dcm, 'SeriesDescription', 'Unknown')),
                                'manufacturer': convert_dicom_value(getattr(first_dcm, 'Manufacturer', 'Unknown')),
                            }
                        except Exception as e:
                            logger.warning(f"Could not read DICOM metadata from {dcm_files[0]}: {e}")
                            study_info['metadata'] = {}
                        
                        studies.append(study_info)
                        logger.info(f"  Found study: {study_info['study_id']} ({len(dcm_files)} files)")
        
        return studies
    
    def get_cached_studies(self) -> List[Dict]:
        """Get list of studies from last scan"""
        return self.studies_cache
    
    def get_studies(self) -> List[Dict]:
        """Alias for get_cached_studies() for backward compatibility"""
        return self.get_cached_studies()
    
    def load_study(self, patient_id: str, study_id: str) -> Optional[Dict]:
        """
        Load a specific study and cache the volume
        
        Args:
            patient_id: Patient ID
            study_id: Study ID (full study_id from cache)
            
        Returns:
            Study info with volume loaded
        """
        # Find study in cache by study_id only (it already contains patient_id)
        study_info = None
        for study in self.studies_cache:
            if study['study_id'] == study_id:
                study_info = study
                break
        
        if not study_info:
            logger.error(f"Study not found in cache: {study_id}")
            logger.error(f"Available studies: {[s['study_id'] for s in self.studies_cache]}")
            return None
        
        study_path = Path(study_info['study_path'])
        study_key = f"{patient_id}_{study_id}"
        
        # Check if already loaded
        if study_key in self.volume_cache:
            logger.info(f"Study {study_id} already loaded (using cache)")
            return self.study_info_cache[study_key]
        
        # Load DICOM series
        logger.info(f"Loading study: {study_id} from {study_path}")
        
        try:
            dicom_data = self.dicom_processor.load_series(str(study_path))
            
            if not dicom_data:
                logger.error(f"Failed to load DICOM series from {study_path}")
                return None
            
            # Cache the volume
            volume = dicom_data['pixel_array']
            self.volume_cache[study_key] = volume
            
            # Convert PyDICOM values to JSON-serializable types
            pixel_spacing = convert_dicom_value(dicom_data.get('pixel_spacing', [1.0, 1.0]))
            slice_thickness = convert_dicom_value(dicom_data.get('slice_thickness', 1.0))
            
            # Ensure pixel_spacing is a list
            if not isinstance(pixel_spacing, list):
                pixel_spacing = [pixel_spacing, pixel_spacing]
            
            # Create study info with volume metadata
            loaded_study_info = {
                **convert_dicom_value(study_info),  # Convert all values in study_info
                'volume_shape': list(volume.shape),
                'pixel_spacing': pixel_spacing,
                'slice_thickness': slice_thickness,
            }
            
            self.study_info_cache[study_key] = loaded_study_info
            
            logger.info(f"Study loaded: {volume.shape}")
            return loaded_study_info
        
        except RuntimeError as e:
            error_msg = str(e)
            if "Unable to decompress" in error_msg and "missing dependencies" in error_msg:
                logger.error(f"DICOM decompression error: {e}")
                logger.error("Please install DICOM decompression libraries:")
                logger.error("  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg")
                return {
                    'error': 'dicom_decompression',
                    'message': 'DICOM files use JPEG compression. Please install decompression libraries.',
                    'solution': 'Run: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg'
                }
            logger.error(f"Error loading study {study_id}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error loading study {study_id}: {e}", exc_info=True)
            return None
    
    def get_volume(self, patient_id: str, study_id: str) -> Optional[np.ndarray]:
        """Get cached volume for a study"""
        study_key = f"{patient_id}_{study_id}"
        
        if study_key not in self.volume_cache:
            # Try to load it
            self.load_study(patient_id, study_id)
        
        return self.volume_cache.get(study_key)
    
    def get_study_info(self, patient_id: str, study_id: str) -> Optional[Dict]:
        """Get study info from cache"""
        study_key = f"{patient_id}_{study_id}"
        
        if study_key in self.study_info_cache:
            return self.study_info_cache[study_key]
        
        # Look in studies_cache by study_id only
        for study in self.studies_cache:
            if study['study_id'] == study_id:
                return study
        
        return None
    
    def clear_cache(self):
        """Clear all cached volumes to free memory"""
        self.volume_cache.clear()
        self.study_info_cache.clear()
        logger.info("Volume cache cleared")
