import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def decode_base64_image(encoded_string: str) -> np.ndarray:
    """
    Decode base64 string to numpy array
    
    Args:
        encoded_string: Base64-encoded image string
        
    Returns:
        Numpy array (grayscale image)
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(encoded_string)
        
        # Load with PIL
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        return np.zeros((256, 256), dtype=np.uint8)


def encode_image_base64(image_array: np.ndarray) -> str:
    """
    Encode numpy array as base64 PNG string
    
    Args:
        image_array: Numpy array (grayscale or RGB)
        
    Returns:
        Base64-encoded PNG string
    """
    try:
        # Ensure uint8
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if len(image_array.shape) == 2:
            pil_image = Image.fromarray(image_array, mode='L')
        else:
            pil_image = Image.fromarray(image_array)
        
        # Save to bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        
        return encoded
        
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return ""


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image array
        target_size: Target (height, width)
        
    Returns:
        Resized image
    """
    try:
        resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        return resized
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    img_min = image.min()
    img_max = image.max()
    
    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = np.zeros_like(image, dtype=np.float32)
    
    return normalized


def apply_window_level(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Apply window/level (contrast adjustment) to image
    
    Args:
        image: Input image
        window_center: Center of the window
        window_width: Width of the window
        
    Returns:
        Windowed image
    """
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    
    windowed = np.clip(image, img_min, img_max)
    windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    return windowed


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE
    
    Args:
        image: Input image (uint8)
        
    Returns:
        Contrast-enhanced image
    """
    try:
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (normalize_image(image) * 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing contrast: {str(e)}")
        return image


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay segmentation mask on image
    
    Args:
        image: Grayscale image
        mask: Binary mask
        alpha: Transparency of overlay (0-1)
        
    Returns:
        RGB image with overlay
    """
    try:
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (normalize_image(image) * 255).astype(np.uint8)
        
        # Convert grayscale to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create colored mask (red)
        mask_colored = np.zeros_like(image_rgb)
        mask_colored[:, :, 2] = mask  # Red channel
        
        # Blend
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)
        
        return overlay
        
    except Exception as e:
        logger.error(f"Error creating overlay: {str(e)}")
        return image

