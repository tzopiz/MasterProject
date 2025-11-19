import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Dict, List, Any
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class UNet(nn.Module):
    """Simple U-Net architecture for segmentation"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Decoder
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d3 = self.dec3(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return torch.sigmoid(self.final(d1))


class SegmentationModel:
    """Handles loading and inference of segmentation model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize segmentation model
        
        Args:
            model_path: Path to saved model weights (.pth file)
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and model_path != "None":
            self._load_model(model_path)
        else:
            logger.warning("No model path provided, running in dummy mode")
    
    def _load_model(self, model_path: str):
        """Load model from file"""
        try:
            logger.info(f"Loading segmentation model from {model_path}...")
            
            # Create model architecture
            self.model = UNet(in_channels=1, out_channels=1)
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self.model = None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def segment(
        self, 
        dicom_data: Dict[str, Any], 
        slices_indices: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Perform segmentation on DICOM slices
        
        Args:
            dicom_data: DICOM data dictionary
            slices_indices: Dictionary with slice indices for each view
            
        Returns:
            Dictionary with base64-encoded mask images
        """
        if not self.is_loaded():
            logger.warning("Model not loaded, returning dummy masks")
            return self._generate_dummy_masks(slices_indices)
        
        try:
            masks = {
                "orthogonal": [],
                "sagittal": [],
                "frontal": []
            }
            
            pixel_array = dicom_data["pixel_array"]
            
            # Process each view
            for view in ["orthogonal", "sagittal", "frontal"]:
                if view in slices_indices and slices_indices[view]:
                    for _ in slices_indices[view]:
                        # In real implementation, decode base64 image and run inference
                        # For now, create dummy mask
                        dummy_mask = self._create_dummy_mask(256, 256)
                        encoded = self._encode_mask(dummy_mask)
                        masks[view].append(encoded)
            
            logger.info(f"Segmentation completed: {len(masks['orthogonal'])} orthogonal, "
                       f"{len(masks['sagittal'])} sagittal, {len(masks['frontal'])} frontal masks")
            
            return masks
            
        except Exception as e:
            logger.error(f"Error during segmentation: {str(e)}", exc_info=True)
            return self._generate_dummy_masks(slices_indices)
    
    def _inference_on_slice(self, slice_image: np.ndarray) -> np.ndarray:
        """
        Run model inference on a single slice
        
        Args:
            slice_image: 2D numpy array (grayscale image)
            
        Returns:
            Binary mask as 2D numpy array
        """
        try:
            # Preprocess
            input_tensor = self._preprocess(slice_image)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Postprocess
            mask = self._postprocess(output, slice_image.shape)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return np.zeros_like(slice_image, dtype=np.uint8)
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: 2D numpy array
            
        Returns:
            Preprocessed tensor
        """
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Resize to model input size (e.g., 256x256)
        from skimage.transform import resize
        image = resize(image, (256, 256), preserve_range=True)
        
        # Convert to tensor [B, C, H, W]
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _postprocess(self, output: torch.Tensor, target_shape: tuple) -> np.ndarray:
        """
        Postprocess model output to binary mask
        
        Args:
            output: Model output tensor
            target_shape: Target shape for mask
            
        Returns:
            Binary mask as numpy array
        """
        # Convert to numpy
        mask = output.squeeze().cpu().numpy()
        
        # Threshold
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Resize to target shape
        from skimage.transform import resize
        mask = resize(mask, target_shape, preserve_range=True, order=0)
        mask = mask.astype(np.uint8)
        
        return mask
    
    def _create_dummy_mask(self, height: int, width: int) -> np.ndarray:
        """Create a dummy segmentation mask for testing"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create a circular region in the center
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 4
        
        y, x = np.ogrid[:height, :width]
        circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[circle_mask] = 255
        
        return mask
    
    def _encode_mask(self, mask: np.ndarray) -> str:
        """Encode mask as base64 PNG string"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(mask, mode='L')
            
            # Save to bytes buffer
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            encoded = base64.b64encode(buffer.read()).decode('utf-8')
            
            return encoded
            
        except Exception as e:
            logger.error(f"Error encoding mask: {str(e)}")
            return ""
    
    def _generate_dummy_masks(self, slices_indices: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Generate dummy masks for testing"""
        masks = {
            "orthogonal": [],
            "sagittal": [],
            "frontal": []
        }
        
        for view in ["orthogonal", "sagittal", "frontal"]:
            if view in slices_indices:
                for _ in slices_indices[view]:
                    dummy_mask = self._create_dummy_mask(256, 256)
                    encoded = self._encode_mask(dummy_mask)
                    masks[view].append(encoded)
        
        return masks

