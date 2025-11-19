import pydicom
import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DICOMProcessor:
    """Handles DICOM file parsing and data extraction"""
    
    def load_dicom(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and parse DICOM file
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Dictionary containing DICOM data and metadata
        """
        try:
            logger.info(f"Loading DICOM file: {file_path}")
            
            # Read DICOM file
            dicom_dataset = pydicom.dcmread(file_path)
            
            # Extract pixel data
            pixel_array = dicom_dataset.pixel_array
            
            # Get metadata
            metadata = {
                "patient_id": getattr(dicom_dataset, 'PatientID', 'Unknown'),
                "study_date": getattr(dicom_dataset, 'StudyDate', 'Unknown'),
                "modality": getattr(dicom_dataset, 'Modality', 'Unknown'),
                "manufacturer": getattr(dicom_dataset, 'Manufacturer', 'Unknown'),
                "rows": dicom_dataset.Rows,
                "columns": dicom_dataset.Columns,
            }
            
            # Get spacing information (important for measurements)
            pixel_spacing = getattr(dicom_dataset, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(dicom_dataset, 'SliceThickness', 1.0)
            
            # Normalize pixel array to 0-255 range for processing
            pixel_array_normalized = self._normalize_pixel_array(pixel_array)
            
            # For 3D volumes (multiple slices)
            if len(pixel_array.shape) == 3:
                num_slices = pixel_array.shape[0]
                logger.info(f"3D volume detected with {num_slices} slices")
            else:
                num_slices = 1
                pixel_array_normalized = np.expand_dims(pixel_array_normalized, axis=0)
                logger.info("Single slice detected")
            
            dicom_data = {
                "pixel_array": pixel_array_normalized,
                "original_array": pixel_array,
                "metadata": metadata,
                "pixel_spacing": pixel_spacing,
                "slice_thickness": slice_thickness,
                "num_slices": num_slices,
                "shape": pixel_array_normalized.shape,
                "dataset": dicom_dataset
            }
            
            logger.info(f"DICOM loaded successfully: {metadata['modality']}, shape: {pixel_array_normalized.shape}")
            return dicom_data
            
        except FileNotFoundError:
            logger.error(f"DICOM file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading DICOM file: {str(e)}", exc_info=True)
            return None
    
    def _normalize_pixel_array(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Normalize pixel array to 0-255 range
        
        Args:
            pixel_array: Raw pixel array from DICOM
            
        Returns:
            Normalized pixel array (uint8)
        """
        # Convert to float for processing
        array_float = pixel_array.astype(np.float32)
        
        # Normalize to 0-1 range
        array_min = np.min(array_float)
        array_max = np.max(array_float)
        
        if array_max > array_min:
            normalized = (array_float - array_min) / (array_max - array_min)
        else:
            normalized = np.zeros_like(array_float)
        
        # Scale to 0-255
        normalized = (normalized * 255).astype(np.uint8)
        
        return normalized
    
    def extract_slice(self, dicom_data: Dict[str, Any], slice_index: int) -> Optional[np.ndarray]:
        """
        Extract a specific slice from DICOM data
        
        Args:
            dicom_data: DICOM data dictionary
            slice_index: Index of slice to extract
            
        Returns:
            2D numpy array of the slice
        """
        try:
            pixel_array = dicom_data["pixel_array"]
            
            if len(pixel_array.shape) == 3:
                if 0 <= slice_index < pixel_array.shape[0]:
                    return pixel_array[slice_index]
                else:
                    logger.error(f"Slice index {slice_index} out of range (0-{pixel_array.shape[0]-1})")
                    return None
            elif len(pixel_array.shape) == 2:
                if slice_index == 0:
                    return pixel_array
                else:
                    logger.error(f"Single slice available, but index {slice_index} requested")
                    return None
            else:
                logger.error(f"Unexpected pixel array shape: {pixel_array.shape}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting slice: {str(e)}")
            return None
    
    def get_physical_dimensions(self, dicom_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Get physical dimensions of the scan
        
        Args:
            dicom_data: DICOM data dictionary
            
        Returns:
            Dictionary with physical dimensions in mm
        """
        pixel_spacing = dicom_data.get("pixel_spacing", [1.0, 1.0])
        slice_thickness = dicom_data.get("slice_thickness", 1.0)
        shape = dicom_data.get("shape")
        
        if len(shape) == 3:
            return {
                "width_mm": shape[2] * pixel_spacing[1],
                "height_mm": shape[1] * pixel_spacing[0],
                "depth_mm": shape[0] * slice_thickness
            }
        else:
            return {
                "width_mm": shape[1] * pixel_spacing[1],
                "height_mm": shape[0] * pixel_spacing[0],
                "depth_mm": slice_thickness
            }

