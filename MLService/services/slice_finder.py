import base64
import logging
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SliceFinder:
    """Finds relevant orthogonal, sagittal, and frontal slices for TMJ analysis"""

    def find_slices(self, dicom_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Find relevant slices for TMJ analysis

        Args:
            dicom_data: Parsed DICOM data

        Returns:
            Dictionary with base64-encoded slice images
        """
        pixel_array = dicom_data["pixel_array"]
        shape = pixel_array.shape

        logger.info(f"Finding slices in volume of shape: {shape}")

        slices_result = {"orthogonal": [], "sagittal": [], "frontal": []}

        # For 3D volumes
        if len(shape) == 3:
            # Orthogonal slices (axial view) - typically best for TMJ
            # Take middle third of the volume
            start_idx = shape[0] // 3
            end_idx = 2 * shape[0] // 3
            step = max(1, (end_idx - start_idx) // 5)  # Take up to 5 slices

            for i in range(start_idx, end_idx, step):
                slice_img = pixel_array[i]
                encoded = self._encode_image(slice_img)
                if encoded:
                    slices_result["orthogonal"].append(encoded)

            # Sagittal slices (side view)
            # Extract sagittal plane through the volume
            mid_sagittal = shape[2] // 2
            # Take slices around middle
            for offset in [-10, -5, 0, 5, 10]:
                idx = mid_sagittal + offset
                if 0 <= idx < shape[2]:
                    sagittal_slice = pixel_array[:, :, idx]
                    encoded = self._encode_image(sagittal_slice)
                    if encoded:
                        slices_result["sagittal"].append(encoded)

            # Frontal slices (coronal view)
            mid_frontal = shape[1] // 2
            for offset in [-10, -5, 0, 5, 10]:
                idx = mid_frontal + offset
                if 0 <= idx < shape[1]:
                    frontal_slice = pixel_array[:, idx, :]
                    encoded = self._encode_image(frontal_slice)
                    if encoded:
                        slices_result["frontal"].append(encoded)

        # For 2D images
        elif len(shape) == 2:
            # Treat as single orthogonal slice
            encoded = self._encode_image(pixel_array)
            if encoded:
                slices_result["orthogonal"].append(encoded)

        logger.info(
            f"Found slices - Orthogonal: {len(slices_result['orthogonal'])}, "
            f"Sagittal: {len(slices_result['sagittal'])}, "
            f"Frontal: {len(slices_result['frontal'])}"
        )

        return slices_result

    def _encode_image(self, image_array: np.ndarray) -> str:
        """
        Encode numpy array as base64 PNG string

        Args:
            image_array: 2D numpy array (grayscale image)

        Returns:
            Base64-encoded PNG string
        """
        try:
            # Ensure uint8
            if image_array.dtype != np.uint8:
                # Normalize to 0-255
                img_min = image_array.min()
                img_max = image_array.max()
                if img_max > img_min:
                    image_array = ((image_array - img_min) / (img_max - img_min) * 255).astype(
                        np.uint8
                    )
                else:
                    image_array = np.zeros_like(image_array, dtype=np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(image_array, mode="L")

            # Save to bytes buffer
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode to base64
            encoded = base64.b64encode(buffer.read()).decode("utf-8")

            return encoded

        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return ""

    def find_tmj_region(self, slice_array: np.ndarray) -> Dict[str, int]:
        """
        Find TMJ region in a slice using intensity-based heuristics

        Args:
            slice_array: 2D numpy array

        Returns:
            Dictionary with bounding box coordinates
        """
        try:
            # Simple approach: find high-intensity regions (bone)
            threshold = np.percentile(slice_array, 75)
            binary = slice_array > threshold

            # Find connected components
            from scipy import ndimage

            labeled, num_features = ndimage.label(binary)

            if num_features > 0:
                # Find largest component (likely bone structure)
                sizes = ndimage.sum(binary, labeled, range(num_features + 1))
                largest_component = sizes[1:].argmax() + 1

                # Get bounding box
                mask = labeled == largest_component
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]

                return {
                    "x_min": int(cmin),
                    "x_max": int(cmax),
                    "y_min": int(rmin),
                    "y_max": int(rmax),
                }

        except Exception as e:
            logger.error(f"Error finding TMJ region: {str(e)}")

        # Return full image if detection fails
        return {
            "x_min": 0,
            "x_max": slice_array.shape[1],
            "y_min": 0,
            "y_max": slice_array.shape[0],
        }
