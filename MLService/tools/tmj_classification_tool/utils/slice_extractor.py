"""
Slice Extractor Utility
Extracts slices from 3D DICOM volumes in different planes
"""

import base64
import logging
from io import BytesIO
from typing import Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SliceExtractor:
    """Utility for extracting and encoding slices from 3D volumes"""
    
    def __init__(self):
        pass
    
    def normalize_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Normalize slice for visualization
        Uses percentile-based normalization similar to roi_annotation_tool.py
        """
        # Clip extreme values (outliers)
        p2, p98 = np.percentile(slice_2d, [2, 98])
        slice_clipped = np.clip(slice_2d, p2, p98)
        
        # Normalize to 0-255
        if slice_clipped.max() > slice_clipped.min():
            normalized = (slice_clipped - slice_clipped.min()) / (slice_clipped.max() - slice_clipped.min()) * 255
        else:
            normalized = np.zeros_like(slice_clipped)
        
        return normalized.astype(np.uint8)
    
    def get_slice(self, volume: np.ndarray, plane: str, index: int) -> Optional[str]:
        """
        Extract a slice from volume and convert to base64
        
        Args:
            volume: 3D numpy array [z, y, x]
            plane: 'axial', 'sagittal', or 'coronal'
            index: Slice index
            
        Returns:
            Base64-encoded PNG image or None if invalid
        """
        try:
            # Extract slice based on plane
            if plane == 'axial':
                # Axial (horizontal): volume[z, :, :]
                if 0 <= index < volume.shape[0]:
                    slice_2d = volume[index, :, :]
                else:
                    logger.error(f"Axial index {index} out of range (0-{volume.shape[0]-1})")
                    return None
            
            elif plane == 'sagittal':
                # Sagittal (side view): volume[:, :, x]
                if 0 <= index < volume.shape[2]:
                    slice_2d = volume[:, :, index]
                else:
                    logger.error(f"Sagittal index {index} out of range (0-{volume.shape[2]-1})")
                    return None
            
            elif plane == 'coronal':
                # Coronal (frontal view): volume[:, y, :]
                if 0 <= index < volume.shape[1]:
                    slice_2d = volume[:, index, :]
                else:
                    logger.error(f"Coronal index {index} out of range (0-{volume.shape[1]-1})")
                    return None
            
            else:
                logger.error(f"Unknown plane: {plane}")
                return None
            
            # Normalize for visualization
            normalized = self.normalize_slice(slice_2d)
            
            # Convert to PIL Image
            image = Image.fromarray(normalized, mode='L')
            
            # Encode to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            base64_str = base64.b64encode(buffer.read()).decode('utf-8')
            
            return f"data:image/png;base64,{base64_str}"
        
        except Exception as e:
            logger.error(f"Error extracting slice: {e}", exc_info=True)
            return None
    
    def get_slice_dimensions(self, volume: np.ndarray, plane: str) -> dict:
        """
        Get dimensions for a specific plane
        
        Returns:
            Dictionary with max_index, width, height
        """
        if plane == 'axial':
            return {
                'max_index': volume.shape[0] - 1,
                'width': volume.shape[2],
                'height': volume.shape[1]
            }
        elif plane == 'sagittal':
            return {
                'max_index': volume.shape[2] - 1,
                'width': volume.shape[1],
                'height': volume.shape[0]
            }
        elif plane == 'coronal':
            return {
                'max_index': volume.shape[1] - 1,
                'width': volume.shape[2],
                'height': volume.shape[0]
            }
        else:
            return {'max_index': 0, 'width': 0, 'height': 0}
