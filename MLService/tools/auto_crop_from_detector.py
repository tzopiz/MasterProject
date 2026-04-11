#!/usr/bin/env python3
"""
Automatic ROI cropping using trained TMJ detector.

This script:
1. Loads a trained detector model
2. Predicts TMJ locations on full CBCT scans
3. Extracts 3D crops around predicted coordinates
4. Saves crops for further segmentation training

Usage:
    python tools/auto_crop_from_detector.py \
        --model experiments/detector_20251124_013727/best_model.pth \
        --input data/dataset/study_0001 \
        --output data/crops \
        --crop_size 128
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pydicom
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.tmj_detector import TMJDetector, get_detector_model
# from training.datasets.tmj_detector_dataset import load_dicom_volume, preprocess_volume
from models.tmj_heatmap_detector import TMJHeatmapDetector
from training.utils.heatmap import coords_from_heatmap as _hm_coords


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dicom_volume(dicom_dir: Path) -> np.ndarray:
    """Load DICOM series from directory."""
    dicom_files = list(dicom_dir.glob('*.dcm'))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
        
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    return volume


def preprocess_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1]."""
    # Clip extreme values
    p2, p98 = np.percentile(volume, [2, 98])
    volume = np.clip(volume, p2, p98)
    
    # Normalize
    if volume.max() > volume.min():
        volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    return volume


def load_detector(model_path: str, device: str = 'mps') -> Tuple[torch.nn.Module, int]:
    """Load trained detector model (regression or heatmap)."""
    logger.info(f"Loading detector from {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Read config.json from the same directory
    model_path_obj = Path(model_path)
    config_path = model_path_obj.parent / 'config.json'

    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Could not read config: {e}, using defaults")

    if config.get('heatmap', False):
        # Heatmap 3-D U-Net model
        logger.info("Detected heatmap model (TMJHeatmapDetector)")
        model = TMJHeatmapDetector()
        model._is_heatmap = True
    else:
        # Legacy coordinate-regression model
        model_type = config.get('model_type', 'large')
        logger.info(f"Detected regression model, type={model_type}")
        model = get_detector_model(model_type=model_type)
        model._is_heatmap = False

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    logger.info(f"Model loaded from epoch {checkpoint['epoch']}")

    return model, checkpoint['epoch']


def predict_tmj_coords(
    model: TMJDetector,
    volume: np.ndarray,
    downsample_factor: int = 6,
    device: str = 'mps'
) -> Dict[str, np.ndarray]:
    """
    Predict TMJ coordinates on a full volume.
    
    Args:
        model: Trained detector model
        volume: Full 3D volume (Z, Y, X)
        downsample_factor: Factor to downsample volume (must match training)
        device: Device to run inference on
    
    Returns:
        Dictionary with 'left' and 'right' TMJ coordinates (z, y, x)
    """
    # Preprocess volume (same as training)
    volume_processed = preprocess_volume(volume)
    
    # Downsample
    D, H, W = volume_processed.shape
    new_D = D // downsample_factor
    new_H = H // downsample_factor
    new_W = W // downsample_factor
    
    volume_down = torch.nn.functional.interpolate(
        torch.from_numpy(volume_processed[None, None, ...]).float(),
        size=(new_D, new_H, new_W),
        mode='trilinear',
        align_corners=False
    )[0, 0].numpy()
    
    # To tensor
    volume_tensor = torch.from_numpy(volume_down).float().unsqueeze(0).unsqueeze(0)
    volume_tensor = volume_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(volume_tensor)

    if getattr(model, '_is_heatmap', False):
        # pred: (1, 2, D, H, W) — ch0=left, ch1=right
        hm_left  = torch.sigmoid(pred[0, 0])
        hm_right = torch.sigmoid(pred[0, 1])
        _, left_orig  = _hm_coords(hm_left,  downsample_factor)
        _, right_orig = _hm_coords(hm_right, downsample_factor)
        left_coords  = left_orig.cpu().numpy().astype(int)
        right_coords = right_orig.cpu().numpy().astype(int)
    else:
        # Legacy regression model — pred: (1, 6)
        pred = pred.cpu().numpy()[0]  # (6,)

        # Unnormalize (from [0, 1] to original downsampled size)
        pred_coords = np.array([
            pred[0] * new_D,  # left_z
            pred[1] * new_H,  # left_y
            pred[2] * new_W,  # left_x
            pred[3] * new_D,  # right_z
            pred[4] * new_H,  # right_y
            pred[5] * new_W,  # right_x
        ])

        # Upscale to original resolution
        pred_coords[[0, 3]] *= downsample_factor  # Z
        pred_coords[[1, 4]] *= downsample_factor  # Y
        pred_coords[[2, 5]] *= downsample_factor  # X

        # Round to integers
        pred_coords = pred_coords.astype(int)

        left_coords  = pred_coords[:3]
        right_coords = pred_coords[3:]

    return {
        'left': left_coords,
        'right': right_coords
    }


def extract_crop(
    volume: np.ndarray,
    center: np.ndarray,
    crop_size: int = 128
) -> np.ndarray:
    """
    Extract a 3D crop centered at given coordinates.
    
    Args:
        volume: Full 3D volume (Z, Y, X)
        center: Center coordinates (z, y, x)
        crop_size: Size of the crop (cube)
    
    Returns:
        Cropped volume of shape (crop_size, crop_size, crop_size)
    """
    D, H, W = volume.shape
    half = crop_size // 2
    
    z, y, x = center
    
    # Calculate crop boundaries
    z_start = max(0, z - half)
    z_end = min(D, z + half)
    y_start = max(0, y - half)
    y_end = min(H, y + half)
    x_start = max(0, x - half)
    x_end = min(W, x + half)
    
    # Extract crop
    crop = volume[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Pad if necessary
    if crop.shape != (crop_size, crop_size, crop_size):
        padded = np.zeros((crop_size, crop_size, crop_size), dtype=crop.dtype)
        
        # Calculate padding offsets
        pad_z = (crop_size - crop.shape[0]) // 2
        pad_y = (crop_size - crop.shape[1]) // 2
        pad_x = (crop_size - crop.shape[2]) // 2
        
        padded[
            pad_z:pad_z+crop.shape[0],
            pad_y:pad_y+crop.shape[1],
            pad_x:pad_x+crop.shape[2]
        ] = crop
        
        crop = padded
    
    return crop


def save_crop_as_nifti(crop: np.ndarray, output_path: Path):
    """Save crop as NIfTI file."""
    import nibabel as nib
    
    # Create NIfTI image
    nifti = nib.Nifti1Image(crop, affine=np.eye(4))
    nib.save(nifti, str(output_path))


def process_study(
    study_path: Path,
    model: TMJDetector,
    output_dir: Path,
    crop_size: int = 128,
    downsample_factor: int = 6,
    device: str = 'mps',
    save_format: str = 'nifti'
) -> Dict:
    """
    Process a single study: predict TMJ coords and extract crops.
    
    Returns:
        Dictionary with prediction info and crop paths
    """
    study_name = study_path.name
    logger.info(f"Processing {study_name}...")
    
    # Load volume
    logger.info("  Loading DICOM volume...")
    volume = load_dicom_volume(study_path)
    logger.info(f"  Volume shape: {volume.shape}")
    
    # Predict TMJ coordinates
    logger.info("  Predicting TMJ coordinates...")
    coords = predict_tmj_coords(
        model, volume, 
        downsample_factor=downsample_factor, 
        device=device
    )
    
    left_coord = coords['left']
    right_coord = coords['right']
    
    logger.info(f"  Predicted Left TMJ:  {left_coord}")
    logger.info(f"  Predicted Right TMJ: {right_coord}")
    
    # Extract crops
    logger.info(f"  Extracting crops (size={crop_size})...")
    left_crop = extract_crop(volume, left_coord, crop_size)
    right_crop = extract_crop(volume, right_coord, crop_size)
    
    # Save crops
    study_output = output_dir / study_name
    study_output.mkdir(parents=True, exist_ok=True)
    
    if save_format == 'nifti':
        left_path = study_output / f"{study_name}_left.nii.gz"
        right_path = study_output / f"{study_name}_right.nii.gz"
        
        save_crop_as_nifti(left_crop, left_path)
        save_crop_as_nifti(right_crop, right_path)
    else:  # numpy
        left_path = study_output / f"{study_name}_left.npy"
        right_path = study_output / f"{study_name}_right.npy"
        
        np.save(left_path, left_crop)
        np.save(right_path, right_crop)
    
    logger.info(f"  ✅ Saved crops to {study_output}")
    
    # Save metadata
    metadata = {
        'study': study_name,
        'volume_shape': list(volume.shape),
        'predicted_coords': {
            'left': left_coord.tolist(),
            'right': right_coord.tolist()
        },
        'crop_size': crop_size,
        'crop_paths': {
            'left': str(left_path.relative_to(output_dir.parent)),
            'right': str(right_path.relative_to(output_dir.parent))
        }
    }
    
    metadata_path = study_output / f"{study_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Automatic TMJ cropping using detector")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained detector model (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to study directory or dataset root')
    parser.add_argument('--output', type=str, default='data/auto_crops',
                        help='Output directory for crops')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='Size of extracted crops (default: 128)')
    parser.add_argument('--downsample_factor', type=int, default=6,
                        help='Downsample factor (must match training, default: 6)')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run inference on')
    parser.add_argument('--format', type=str, default='nifti',
                        choices=['nifti', 'numpy'],
                        help='Output format for crops (default: nifti)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all studies in input directory')
    
    args = parser.parse_args()
    
    # Setup paths
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, epoch = load_detector(str(model_path), device=args.device)
    
    # Determine studies to process
    if args.batch or input_path.is_dir() and not (input_path / 'DICOMDIR').exists():
        # Batch mode: process all subdirectories
        study_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        study_dirs = sorted(study_dirs)
    else:
        # Single study mode
        study_dirs = [input_path]
    
    logger.info(f"Found {len(study_dirs)} studies to process")
    
    # Process studies
    results = []
    for study_dir in tqdm(study_dirs, desc="Processing studies"):
        try:
            metadata = process_study(
                study_dir,
                model,
                output_dir,
                crop_size=args.crop_size,
                downsample_factor=args.downsample_factor,
                device=args.device,
                save_format=args.format
            )
            results.append(metadata)
        except Exception as e:
            logger.error(f"Failed to process {study_dir.name}: {e}")
            continue
    
    # Save summary
    summary = {
        'model': str(model_path),
        'model_epoch': epoch,
        'input': str(input_path),
        'output': str(output_dir),
        'crop_size': args.crop_size,
        'downsample_factor': args.downsample_factor,
        'format': args.format,
        'total_studies': len(study_dirs),
        'successful': len(results),
        'studies': results
    }
    
    summary_path = output_dir / 'crop_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CROPPING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Processed: {len(results)}/{len(study_dirs)} studies")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Summary: {summary_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

