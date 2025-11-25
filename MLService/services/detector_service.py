import torch
import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from skimage.transform import resize
import os
import json

from models.tmj_detector import get_detector_model

logger = logging.getLogger(__name__)

class TMJDetectorService:
    """
    Service for TMJ detection (bounding box regression).
    Uses 3D CNN to find Left and Right TMJ centers.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        self.model = None
        self.input_shape = (96, 128, 128) # D, H, W as used in training
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No model path provided for TMJDetectorService")

    def load_model(self, model_path: str):
        try:
            logger.info(f"Loading TMJ Detector from {model_path}...")
            
            # Try to detect model type from config
            model_type = 'large'  # Default to large
            
            # Check if config.json exists in experiment directory
            exp_dir = os.path.dirname(model_path)
            config_path = os.path.join(exp_dir, 'config.json')
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        model_type = config.get('model_type', 'large')
                        logger.info(f"Detected model type from config: {model_type}")
                except Exception as e:
                    logger.warning(f"Could not read config.json: {e}, using default 'large'")
            
            # Initialize model architecture
            self.model = get_detector_model(model_type=model_type)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'best_val_mae' in checkpoint:
                    logger.info(f"Model best validation MAE: {checkpoint['best_val_mae']:.2f} px")
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"TMJ Detector ({model_type}) loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load TMJ Detector: {e}", exc_info=True)
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def detect(self, volume: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
        """
        Run detection on a 3D volume.
        
        Args:
            volume: 3D numpy array (D, H, W)
            
        Returns:
            Dictionary with coordinates for 'left' and 'right' TMJ
            {
                'left': {'center': [z, y, x], 'bbox': [z1, y1, x1, z2, y2, x2]},
                'right': ...
            }
        """
        if not self.is_loaded():
            logger.warning("Detector not loaded, returning None")
            return None
            
        try:
            # 1. Preprocess
            input_tensor = self._preprocess(volume)
            
            # 2. Inference
            with torch.no_grad():
                # Output: (1, 6) -> [lz, ly, lx, rz, ry, rx] in [0, 1]
                coords_norm = self.model(input_tensor).cpu().numpy()[0]
                
            # 3. Postprocess (Denormalize)
            original_shape = volume.shape # (D, H, W)
            
            # Split coordinates
            left_norm = coords_norm[:3]
            right_norm = coords_norm[3:]
            
            # Scale to original size
            left_abs = left_norm * np.array(original_shape)
            right_abs = right_norm * np.array(original_shape)
            
            # Create Bounding Boxes (fixed size crop around center, e.g., 40mm or 60px)
            # Let's assume we want a crop of size ~50-60 voxels for display or further processing
            # Or just return the center.
            
            # Let's return Center + ROI (Region of Interest)
            roi_size = 64 # voxels
            
            result = {
                "left": {
                    "center": left_abs.tolist(), # [z, y, x]
                    "bbox": self._get_bbox(left_abs, original_shape, roi_size)
                },
                "right": {
                    "center": right_abs.tolist(),
                    "bbox": self._get_bbox(right_abs, original_shape, roi_size)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during TMJ detection: {e}", exc_info=True)
            return None

    def _preprocess(self, volume: np.ndarray) -> torch.Tensor:
        """
        Resize and normalize volume for model input.
        Target shape: (96, 128, 128)
        """
        # Normalize to [0, 1] if not already
        if volume.max() > 1.0:
            volume = volume.astype(np.float32)
            v_min, v_max = volume.min(), volume.max()
            if v_max > v_min:
                volume = (volume - v_min) / (v_max - v_min)
            else:
                volume = np.zeros_like(volume)
                
        # Resize
        # Note: skimage resize expects (D, H, W)
        volume_resized = resize(
            volume, 
            self.input_shape, 
            mode='constant', 
            preserve_range=True,
            anti_aliasing=True
        )
        
        # To Tensor: (1, 1, D, H, W)
        tensor = torch.from_numpy(volume_resized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0) # Add Batch and Channel
        
        return tensor.to(self.device)

    def _get_bbox(self, center, shape, size):
        """Calculate bbox [min_z, min_y, min_x, max_z, max_y, max_x]"""
        z, y, x = center
        half = size // 2
        
        z1 = max(0, int(z - half))
        y1 = max(0, int(y - half))
        x1 = max(0, int(x - half))
        
        z2 = min(shape[0], int(z + half))
        y2 = min(shape[1], int(y + half))
        x2 = min(shape[2], int(x + half))
        
        return [z1, y1, x1, z2, y2, x2]

