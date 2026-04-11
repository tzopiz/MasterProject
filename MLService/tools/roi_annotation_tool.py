#!/usr/bin/env python3
"""
ROI Annotation Tool for TMJ Detection Dataset

Быстрый инструмент для разметки координат центров ВНЧС (левого и правого)
на полных DICOM сканах для обучения детектора.

Usage:
    python tools/roi_annotation_tool.py <dicom_dir> --output annotations/
    
Controls:
    - Left Click: Поставить точку
    - Mouse Wheel: Навигация по срезам
    - 'L': Режим "Left TMJ"
    - 'R': Режим "Right TMJ"
    - 'S': Сохранить аннотацию
    - 'N': Следующий скан (Next)
    - 'U': Отменить последнюю точку (Undo)
    - 'Q': Выход
    - 'H': Показать помощь
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import pydicom
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ROIAnnotationTool:
    """Interactive tool for annotating TMJ ROI centers"""
    
    def __init__(self, dicom_dir: str, output_dir: str):
        self.dicom_dir = Path(dicom_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.volume = None
        self.current_slice = 0
        self.scan_id = None
        self.original_shape = None
        
        # Annotations
        self.left_tmj = None  # [z, y, x]
        self.right_tmj = None  # [z, y, x]
        self.current_mode = "left"  # "left" or "right"
        
        # Display
        self.window_name = "ROI Annotation Tool"
        self.display_scale = 1.0
        self.brightness = 1.0
        self.contrast = 1.0
        
        # Stats
        self.annotated_count = 0
        self.load_existing_annotations()
        
    def load_existing_annotations(self):
        """Count how many scans already annotated"""
        if self.output_dir.exists():
            self.annotated_count = len(list(self.output_dir.glob("*_rois.json")))
            logger.info(f"Found {self.annotated_count} existing annotations")
    
    def load_dicom_series(self, dicom_dir: Path) -> np.ndarray:
        """Load DICOM series into 3D numpy array"""
        logger.info(f"Loading DICOM series from {dicom_dir}")
        
        # Find all DICOM files
        dicom_files = []
        for root, dirs, files in os.walk(dicom_dir):
            for file in files:
                filepath = Path(root) / file
                try:
                    pydicom.dcmread(filepath, stop_before_pixels=True)
                    dicom_files.append(filepath)
                except:
                    continue
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")
        
        logger.info(f"Found {len(dicom_files)} DICOM files")
        
        # Sort by Instance Number or Slice Location
        slices = []
        for filepath in tqdm(dicom_files, desc="Reading DICOM files"):
            ds = pydicom.dcmread(filepath)
            slices.append(ds)
        
        # Sort slices
        slices.sort(key=lambda x: float(getattr(x, 'InstanceNumber', 0)))
        
        # Stack into 3D volume
        logger.info("Stacking slices into 3D volume...")
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
        
        # Normalize to 0-255 for display
        volume = self._normalize_volume(volume)
        
        self.original_shape = volume.shape
        self.scan_id = self.dicom_dir.name
        
        logger.info(f"Loaded volume: {volume.shape}")
        return volume
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to 0-255 range"""
        # Clip extreme values (outliers)
        p2, p98 = np.percentile(volume, [2, 98])
        volume = np.clip(volume, p2, p98)
        
        # Normalize to 0-255
        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min()) * 255
        else:
            volume = np.zeros_like(volume)
        
        return volume.astype(np.uint8)
    
    def get_slice_display(self, slice_idx: int) -> np.ndarray:
        """Get slice for display with annotations"""
        if self.volume is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Get axial slice
        slice_2d = self.volume[slice_idx].copy()
        
        # Apply brightness/contrast
        slice_2d = np.clip(slice_2d * self.contrast + self.brightness * 50, 0, 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        slice_bgr = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2BGR)
        
        # Draw annotations
        slice_bgr = self._draw_annotations(slice_bgr, slice_idx)
        
        # Add info overlay
        slice_bgr = self._draw_info_overlay(slice_bgr, slice_idx)
        
        # Scale for display
        if self.display_scale != 1.0:
            new_size = (int(slice_bgr.shape[1] * self.display_scale),
                       int(slice_bgr.shape[0] * self.display_scale))
            slice_bgr = cv2.resize(slice_bgr, new_size, interpolation=cv2.INTER_LINEAR)
        
        return slice_bgr
    
    def _draw_annotations(self, image: np.ndarray, slice_idx: int) -> np.ndarray:
        """Draw annotation points on image"""
        # Draw left TMJ (green)
        if self.left_tmj is not None:
            z, y, x = self.left_tmj
            if abs(z - slice_idx) < 3:  # Show marker if within 3 slices
                alpha = 1.0 - abs(z - slice_idx) / 3.0
                color = (0, int(255 * alpha), 0)  # Green
                thickness = 2 if z == slice_idx else 1
                cv2.circle(image, (int(x), int(y)), 10, color, thickness)
                cv2.circle(image, (int(x), int(y)), 2, color, -1)
                if z == slice_idx:
                    cv2.putText(image, "L", (int(x) + 15, int(y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw right TMJ (blue)
        if self.right_tmj is not None:
            z, y, x = self.right_tmj
            if abs(z - slice_idx) < 3:
                alpha = 1.0 - abs(z - slice_idx) / 3.0
                color = (int(255 * alpha), 0, 0)  # Blue
                thickness = 2 if z == slice_idx else 1
                cv2.circle(image, (int(x), int(y)), 10, color, thickness)
                cv2.circle(image, (int(x), int(y)), 2, color, -1)
                if z == slice_idx:
                    cv2.putText(image, "R", (int(x) + 15, int(y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image
    
    def _draw_info_overlay(self, image: np.ndarray, slice_idx: int) -> np.ndarray:
        """Draw information overlay"""
        h, w = image.shape[:2]
        
        # Semi-transparent black bar at top
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # Scan info
        info_lines = [
            f"Scan: {self.scan_id}",
            f"Slice: {slice_idx + 1}/{self.original_shape[0]}",
            f"Mode: {'LEFT' if self.current_mode == 'left' else 'RIGHT'} TMJ",
            f"Status: L={'✓' if self.left_tmj else '✗'}  R={'✓' if self.right_tmj else '✗'}",
        ]
        
        y_offset = 20
        for line in info_lines:
            color = (0, 255, 0) if self.current_mode == "left" else (255, 0, 0)
            cv2.putText(image, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        # Help text at bottom
        help_text = "L:Left | R:Right | S:Save | N:Next | U:Undo | H:Help | Q:Quit"
        cv2.putText(image, help_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Progress
        progress_text = f"Annotated: {self.annotated_count} scans"
        cv2.putText(image, progress_text, (w - 200, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        return image
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale coordinates back to original size
            x_orig = int(x / self.display_scale)
            y_orig = int(y / self.display_scale)
            
            # Set annotation
            if self.current_mode == "left":
                self.left_tmj = [self.current_slice, y_orig, x_orig]
                logger.info(f"Set LEFT TMJ: {self.left_tmj}")
                # Auto-switch to right mode
                self.current_mode = "right"
            else:
                self.right_tmj = [self.current_slice, y_orig, x_orig]
                logger.info(f"Set RIGHT TMJ: {self.right_tmj}")
                # Auto-switch to left mode for next scan
                self.current_mode = "left"
            
            self.update_display()
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll through slices
            # Interpret flags as signed 32-bit (macOS may pass unsigned)
            signed_flags = flags if flags < 0x80000000 else flags - 0x100000000
            delta = 1 if signed_flags > 0 else -1
            self.current_slice = np.clip(
                self.current_slice + delta,
                0,
                self.original_shape[0] - 1
            )
            self.update_display()
    
    def update_display(self):
        """Update the display window"""
        display_image = self.get_slice_display(self.current_slice)
        cv2.imshow(self.window_name, display_image)
    
    def save_annotation(self) -> bool:
        """Save current annotation to JSON"""
        if self.left_tmj is None or self.right_tmj is None:
            logger.warning("Both LEFT and RIGHT TMJ must be annotated before saving!")
            return False
        
        # Convert numpy types to native Python types
        left_tmj_list = [int(x) for x in self.left_tmj]
        right_tmj_list = [int(x) for x in self.right_tmj]
        original_shape_list = [int(x) for x in self.original_shape] if isinstance(self.original_shape, (list, tuple, np.ndarray)) else list(self.original_shape)
        
        annotation = {
            "scan_id": self.scan_id,
            "dicom_dir": str(self.dicom_dir),
            "original_shape": original_shape_list,
            "annotated_at": datetime.now().isoformat(),
            "left_tmj": {
                "center": left_tmj_list,  # [z, y, x]
                "confidence": "manual"
            },
            "right_tmj": {
                "center": right_tmj_list,  # [z, y, x]
                "confidence": "manual"
            }
        }
        
        # Save to JSON
        output_file = self.output_dir / f"{self.scan_id}_rois.json"
        with open(output_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        logger.info(f"✅ Saved annotation to {output_file}")
        self.annotated_count += 1
        return True
    
    def reset_annotations(self):
        """Reset current annotations"""
        self.left_tmj = None
        self.right_tmj = None
        self.current_mode = "left"
        logger.info("Reset annotations")
    
    def show_help(self):
        """Show help window"""
        help_text = """
ROI Annotation Tool - Help
=========================

CONTROLS:
---------
Left Click       : Place annotation point
Mouse Wheel      : Navigate through slices
L                : Switch to LEFT TMJ mode (Green)
R                : Switch to RIGHT TMJ mode (Blue)
S                : Save annotation (both points required)
N                : Next scan (after saving)
U                : Undo last point
Q / ESC          : Quit
H                : Show this help

WORKFLOW:
---------
1. Navigate to axial slice where TMJ is visible
2. Click on center of LEFT TMJ condyle (Green)
3. Tool auto-switches to RIGHT mode
4. Click on center of RIGHT TMJ condyle (Blue)
5. Press 'S' to save
6. Press 'N' to load next scan (or 'Q' to quit)

TIPS:
-----
- Use middle slices where condyle is most visible
- Aim for center of condylar head
- Both points should be on same slice (or close)
- You can adjust points by re-clicking
- Press 'U' to undo last point

STATUS:
-------
✓ = Annotated
✗ = Not annotated
        """
        
        print("\n" + "="*60)
        print(help_text)
        print("="*60 + "\n")
    
    def run(self):
        """Run the annotation tool"""
        logger.info("Starting ROI Annotation Tool")
        logger.info(f"DICOM directory: {self.dicom_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Load DICOM volume
        try:
            self.volume = self.load_dicom_series(self.dicom_dir)
        except Exception as e:
            logger.error(f"Failed to load DICOM series: {e}")
            return
        
        # Set initial slice (middle of volume)
        self.current_slice = self.original_shape[0] // 2
        
        # Calculate display scale (fit to screen ~800px)
        target_size = 800
        self.display_scale = target_size / max(self.original_shape[1], self.original_shape[2])
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Show help
        self.show_help()
        
        # Initial display
        self.update_display()
        
        # Main loop
        logger.info("Ready! Click on TMJ centers to annotate.")
        logger.info("Press 'H' for help, 'Q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                logger.info("Quitting...")
                break
            
            elif key == ord('l'):  # Left mode
                self.current_mode = "left"
                logger.info("Switched to LEFT TMJ mode")
                self.update_display()
            
            elif key == ord('r'):  # Right mode
                self.current_mode = "right"
                logger.info("Switched to RIGHT TMJ mode")
                self.update_display()
            
            elif key == ord('s'):  # Save
                if self.save_annotation():
                    logger.info("Annotation saved! Press 'N' for next scan or 'Q' to quit")
            
            elif key == ord('n'):  # Next (only after saving)
                if self.left_tmj is not None and self.right_tmj is not None:
                    logger.info("Load next scan manually (re-run tool with new scan)")
                    break
                else:
                    logger.warning("Complete and save current annotation first!")
            
            elif key == ord('u'):  # Undo
                if self.current_mode == "right" and self.right_tmj is not None:
                    self.right_tmj = None
                    logger.info("Undone RIGHT TMJ")
                elif self.current_mode == "left" and self.left_tmj is not None:
                    self.left_tmj = None
                    logger.info("Undone LEFT TMJ")
                self.update_display()
            
            elif key == ord('h'):  # Help
                self.show_help()
            
            elif key == ord('+') or key == ord('='):  # Brightness up
                self.brightness = min(self.brightness + 0.1, 2.0)
                self.update_display()

            elif key == ord('-') or key == ord('_'):  # Brightness down
                self.brightness = max(self.brightness - 0.1, 0.0)
                self.update_display()

            elif key == ord('d') or key == 83:  # D or → : next slice
                self.current_slice = min(self.current_slice + 1,
                                         self.original_shape[0] - 1)
                self.update_display()

            elif key == ord('a') or key == 81:  # A or ← : previous slice
                self.current_slice = max(self.current_slice - 1, 0)
                self.update_display()

            elif key == ord('D'):  # Shift+D : next 10 slices
                self.current_slice = min(self.current_slice + 10,
                                         self.original_shape[0] - 1)
                self.update_display()

            elif key == ord('A'):  # Shift+A : previous 10 slices
                self.current_slice = max(self.current_slice - 10, 0)
                self.update_display()
        
        cv2.destroyAllWindows()
        logger.info(f"Total annotated: {self.annotated_count} scans")
        logger.info("Done!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ROI Annotation Tool for TMJ Detection Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "dicom_dir",
        type=str,
        help="Path to DICOM directory to annotate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/roi_annotations",
        help="Output directory for annotations (default: data/roi_annotations)"
    )
    
    args = parser.parse_args()
    
    # Check if DICOM directory exists
    if not Path(args.dicom_dir).exists():
        logger.error(f"DICOM directory not found: {args.dicom_dir}")
        sys.exit(1)
    
    # Run tool
    tool = ROIAnnotationTool(args.dicom_dir, args.output)
    tool.run()


if __name__ == "__main__":
    main()

