"""
Comprehensive model evaluation script

Usage:
    python evaluate_model.py --model models/segmentation_model_best.pth \
                            --data data/test_crops \
                            --output validation/reports/eval_report.html
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

from models.segmentation_model import UNet
from validation.metrics import compute_all_metrics, print_metrics
from validation.visualize import plot_side_by_side, plot_metrics_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: Path, device: torch.device) -> UNet:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    model = UNet(in_channels=1, out_channels=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model


def find_test_pairs(data_dir: Path) -> List[tuple]:
    """
    Find pairs of volumes and masks in test directory.
    
    Returns:
        List of (volume_path, mask_path) tuples
    """
    data_dir = Path(data_dir)
    
    # Find all .nii.gz files that are NOT masks
    volumes = sorted([f for f in data_dir.glob('*.nii.gz') if '_mask' not in f.name])
    
    pairs = []
    for vol_path in volumes:
        # Find corresponding mask
        mask_path = vol_path.parent / f"{vol_path.stem.replace('.nii', '')}_mask.nii.gz"
        
        if mask_path.exists():
            pairs.append((vol_path, mask_path))
        else:
            logger.warning(f"No mask found for {vol_path.name}")
    
    logger.info(f"Found {len(pairs)} volume-mask pairs")
    return pairs


def preprocess_slice(slice_img: np.ndarray) -> torch.Tensor:
    """Preprocess single slice for model inference."""
    # Normalize to [0, 1]
    slice_img = slice_img.astype(np.float32)
    if slice_img.max() > 0:
        slice_img = slice_img / slice_img.max()
    
    # Resize to 256x256 if needed
    from skimage.transform import resize
    if slice_img.shape != (256, 256):
        slice_img = resize(slice_img, (256, 256), preserve_range=True)
    
    # Convert to tensor [1, 1, H, W]
    tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0)
    return tensor


def segment_volume(model: UNet, volume: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Segment entire 3D volume slice by slice.
    
    Args:
        model: Trained segmentation model
        volume: 3D numpy array (D, H, W)
        device: Computing device
        
    Returns:
        3D segmentation mask
    """
    depth, height, width = volume.shape
    pred_mask = np.zeros_like(volume, dtype=np.float32)
    
    with torch.no_grad():
        for z in range(depth):
            slice_img = volume[z]
            
            # Preprocess
            tensor = preprocess_slice(slice_img).to(device)
            
            # Inference
            output = model(tensor)
            
            # Postprocess
            mask = output.squeeze().cpu().numpy()
            
            # Resize back to original size if needed
            if mask.shape != (height, width):
                from skimage.transform import resize
                mask = resize(mask, (height, width), preserve_range=True, order=0)
            
            pred_mask[z] = mask
    
    # Threshold
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    
    return pred_mask


def evaluate_pair(model: UNet, vol_path: Path, mask_path: Path, 
                 device: torch.device, output_dir: Path) -> Dict:
    """
    Evaluate model on single volume-mask pair.
    
    Returns:
        Dictionary with metrics
    """
    logger.info(f"Evaluating {vol_path.name}")
    
    # Load volume and mask
    vol_nii = nib.load(vol_path)
    mask_nii = nib.load(mask_path)
    
    volume = vol_nii.get_fdata()
    gt_mask = mask_nii.get_fdata()
    
    # Get spacing for distance metrics
    spacing = vol_nii.header.get_zooms()
    
    # Segment
    pred_mask = segment_volume(model, volume, device)
    
    # Compute metrics (on middle slice for visualization)
    middle_slice = volume.shape[0] // 2
    metrics = compute_all_metrics(
        pred_mask[middle_slice], 
        gt_mask[middle_slice],
        spacing=(spacing[1], spacing[2])  # H, W spacing
    )
    
    # Also compute volume-level metrics
    volume_metrics = compute_all_metrics(pred_mask, gt_mask, spacing=spacing)
    metrics['volume_dice'] = volume_metrics['dice']
    metrics['volume_iou'] = volume_metrics['iou']
    
    # Save visualization
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    plot_side_by_side(
        volume[middle_slice],
        pred_mask[middle_slice],
        gt_mask[middle_slice],
        metrics,
        title=f"{vol_path.stem} (slice {middle_slice})",
        save_path=vis_dir / f"{vol_path.stem}_result.png"
    )
    
    return metrics


def evaluate_model(model_path: Path, data_dir: Path, output_dir: Path):
    """
    Main evaluation function.
    
    Args:
        model_path: Path to trained model
        data_dir: Directory with test data
        output_dir: Output directory for reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Find test pairs
    test_pairs = find_test_pairs(data_dir)
    
    if not test_pairs:
        logger.error("No test pairs found!")
        return
    
    # Evaluate each pair
    all_metrics = []
    
    for vol_path, mask_path in tqdm(test_pairs, desc="Evaluating"):
        try:
            metrics = evaluate_pair(model, vol_path, mask_path, device, output_dir)
            metrics['filename'] = vol_path.name
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Error evaluating {vol_path.name}: {e}")
            continue
    
    # Aggregate results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    # Average metrics
    avg_metrics = {}
    metric_keys = ['dice', 'iou', 'precision', 'recall', 'specificity', 
                   'volume_dice', 'volume_iou']
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if values:
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
    
    # Print summary
    print(f"\n📊 Summary Statistics (n={len(all_metrics)}):")
    print(f"  Average Dice:       {avg_metrics.get('dice', 0):.4f} ± {avg_metrics.get('dice_std', 0):.4f}")
    print(f"  Average IoU:        {avg_metrics.get('iou', 0):.4f} ± {avg_metrics.get('iou_std', 0):.4f}")
    print(f"  Average Precision:  {avg_metrics.get('precision', 0):.4f} ± {avg_metrics.get('precision_std', 0):.4f}")
    print(f"  Average Recall:     {avg_metrics.get('recall', 0):.4f} ± {avg_metrics.get('recall_std', 0):.4f}")
    print(f"  Volume-level Dice:  {avg_metrics.get('volume_dice', 0):.4f} ± {avg_metrics.get('volume_dice_std', 0):.4f}")
    
    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'model_path': str(model_path),
            'data_dir': str(data_dir),
            'n_samples': len(all_metrics),
            'average_metrics': avg_metrics,
            'per_sample_metrics': all_metrics
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {results_file}")
    logger.info(f"📊 Visualizations saved to {output_dir / 'visualizations'}")
    
    # Generate summary report
    generate_html_report(all_metrics, avg_metrics, output_dir)


def generate_html_report(all_metrics: List[Dict], avg_metrics: Dict, output_dir: Path):
    """Generate HTML report with results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; margin: 20px 0; }}
            .metric {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Segmentation Model Evaluation Report</h1>
        
        <div class="summary">
            <h2>Summary Statistics (n={len(all_metrics)})</h2>
            <p><span class="metric">Average Dice:</span> {avg_metrics.get('dice', 0):.4f} ± {avg_metrics.get('dice_std', 0):.4f}</p>
            <p><span class="metric">Average IoU:</span> {avg_metrics.get('iou', 0):.4f} ± {avg_metrics.get('iou_std', 0):.4f}</p>
            <p><span class="metric">Average Precision:</span> {avg_metrics.get('precision', 0):.4f} ± {avg_metrics.get('precision_std', 0):.4f}</p>
            <p><span class="metric">Average Recall:</span> {avg_metrics.get('recall', 0):.4f} ± {avg_metrics.get('recall_std', 0):.4f}</p>
        </div>
        
        <h2>Per-Sample Results</h2>
        <table>
            <tr>
                <th>Filename</th>
                <th>Dice</th>
                <th>IoU</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Volume Dice</th>
            </tr>
    """
    
    for m in all_metrics:
        html_content += f"""
            <tr>
                <td>{m.get('filename', 'N/A')}</td>
                <td>{m.get('dice', 0):.4f}</td>
                <td>{m.get('iou', 0):.4f}</td>
                <td>{m.get('precision', 0):.4f}</td>
                <td>{m.get('recall', 0):.4f}</td>
                <td>{m.get('volume_dice', 0):.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
        <p>See visualizations folder for detailed per-sample results.</p>
    </body>
    </html>
    """
    
    report_path = output_dir / 'evaluation_report.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📄 HTML report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output', type=str, default='validation/reports/latest', 
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    evaluate_model(
        Path(args.model),
        Path(args.data),
        Path(args.output)
    )


if __name__ == "__main__":
    main()

