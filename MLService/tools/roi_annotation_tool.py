#!/usr/bin/env python3
"""
ROI Annotation Tool for TMJ Detection Dataset — MPR view

Трёхпанельное отображение: Axial | Coronal | Sagittal.
Клик работает в любом из трёх видов.
Перекрестия показывают позицию аннотации во всех плоскостях.

Usage:
    python tools/roi_annotation_tool.py <dicom_dir> --output annotations/

Controls:
    Left Click       : Поставить точку (в любом из трёх видов)
    Mouse Wheel      : Навигация по Axial срезам
    A / ←            : Предыдущий Axial срез
    D / →            : Следующий Axial срез
    Shift+A          : −10 срезов
    Shift+D          : +10 срезов
    L                : Режим Left TMJ
    R                : Режим Right TMJ
    S                : Сохранить аннотацию
    U                : Отменить последнюю точку
    + / -            : Яркость
    Q / ESC          : Выход
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Optional, List

import numpy as np
import cv2
import pydicom
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Each panel is rendered at this size
PANEL_H = 512
PANEL_W = 512

# Annotation colours (BGR)
COL_LEFT  = (0,   220,   0)   # green
COL_RIGHT = (220,   0,   0)   # blue
COL_CROSS = (180, 180,   0)   # yellow crosshair


class ROIAnnotationTool:
    """Interactive MPR tool for annotating TMJ ROI centers."""

    def __init__(self, dicom_dir: str, output_dir: str):
        self.dicom_dir  = Path(dicom_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.volume        = None
        self.current_slice = 0       # axial (Z) slice
        self.scan_id       = None
        self.original_shape = None   # (D, H, W)

        self.left_tmj  = None        # [z, y, x] in original voxel space
        self.right_tmj = None
        self.current_mode = "left"

        self.brightness = 1.0
        self.window_name = "ROI Annotation Tool — MPR"

        self.annotated_count = len(list(self.output_dir.glob("*_rois.json")))
        logger.info(f"Found {self.annotated_count} existing annotations")

    # ------------------------------------------------------------------ #
    #  DICOM loading                                                        #
    # ------------------------------------------------------------------ #

    def load_dicom_series(self, dicom_dir: Path) -> np.ndarray:
        dicom_files = []
        for root, _, files in os.walk(dicom_dir):
            for f in files:
                fp = Path(root) / f
                try:
                    pydicom.dcmread(fp, stop_before_pixels=True)
                    dicom_files.append(fp)
                except Exception:
                    continue

        if not dicom_files:
            raise ValueError(f"No DICOM files in {dicom_dir}")

        logger.info(f"Found {len(dicom_files)} DICOM files")
        slices = [pydicom.dcmread(str(f)) for f in
                  tqdm(dicom_files, desc="Reading DICOM")]
        slices.sort(key=lambda s: float(getattr(s, 'InstanceNumber', 0)))

        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
        p2, p98 = np.percentile(volume, [2, 98])
        volume = np.clip(volume, p2, p98)
        volume = ((volume - p2) / max(p98 - p2, 1) * 255).astype(np.uint8)

        self.original_shape = volume.shape
        self.scan_id = self.dicom_dir.name
        logger.info(f"Volume shape: {volume.shape}")
        return volume

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _active_ann(self):
        """Return [z,y,x] of the annotation being placed (or last placed)."""
        if self.current_mode == "right" and self.left_tmj is not None:
            return self.left_tmj
        if self.current_mode == "left" and self.right_tmj is not None:
            return self.right_tmj
        if self.left_tmj is not None:
            return self.left_tmj
        if self.right_tmj is not None:
            return self.right_tmj
        return None

    def _apply_brightness(self, sl: np.ndarray) -> np.ndarray:
        return np.clip(sl.astype(np.float32) * self.brightness, 0, 255).astype(np.uint8)

    def _make_panel(self, raw: np.ndarray, label: str,
                    cross_rc=None, dot_rc=None,
                    dot_color=(0, 220, 0)) -> np.ndarray:
        """
        Scale `raw` (uint8 2-D) to PANEL_H×PANEL_W, draw crosshair and dot.

        cross_rc : (row, col) in raw-space where horizontal+vertical lines cross
        dot_rc   : (row, col) in raw-space of the annotation circle
        """
        orig_h, orig_w = raw.shape
        panel = cv2.resize(raw, (PANEL_W, PANEL_H),
                           interpolation=cv2.INTER_LINEAR)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

        def to_panel(r, c):
            return (int(c * PANEL_W / orig_w),
                    int(r * PANEL_H / orig_h))

        # Crosshair at cross_rc
        if cross_rc is not None:
            px, py = to_panel(*cross_rc)
            cv2.line(panel_bgr, (0, py), (PANEL_W, py), COL_CROSS, 1)
            cv2.line(panel_bgr, (px, 0), (px, PANEL_H), COL_CROSS, 1)

        # Annotation circle at dot_rc
        if dot_rc is not None:
            px, py = to_panel(*dot_rc)
            cv2.circle(panel_bgr, (px, py), 10, dot_color, 2)
            cv2.circle(panel_bgr, (px, py),  2, dot_color, -1)

        # Label
        cv2.putText(panel_bgr, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                    cv2.LINE_AA)
        return panel_bgr

    # ------------------------------------------------------------------ #
    #  3-panel display                                                      #
    # ------------------------------------------------------------------ #

    def _build_display(self) -> np.ndarray:
        D, H, W = self.original_shape
        z = self.current_slice
        ann = self._active_ann()

        # Reference coords for cross-planes (fall back to centre)
        cy = ann[1] if ann else H // 2
        cx = ann[2] if ann else W // 2

        # --- Panel 1: Axial  volume[z, :, :]  rows=Y  cols=X
        axial_raw = self._apply_brightness(self.volume[z])
        # crosshair always at current (cy, cx)
        # show dot for left and right if on this z slice
        p1 = self._make_panel(axial_raw,
                               f"AXIAL  z={z}/{D}",
                               cross_rc=(cy, cx))
        for ann_pt, col in [(self.left_tmj, COL_LEFT),
                             (self.right_tmj, COL_RIGHT)]:
            if ann_pt is not None and abs(ann_pt[0] - z) < 3:
                r, c = ann_pt[1], ann_pt[2]
                px = int(c * PANEL_W / W)
                py = int(r * PANEL_H / H)
                alpha = 1.0 - abs(ann_pt[0] - z) / 3.0
                col_a = tuple(int(v * alpha) for v in col)
                cv2.circle(p1, (px, py), 10, col_a, 2 if ann_pt[0] == z else 1)
                cv2.circle(p1, (px, py),  2, col_a, -1)
                tag = "L" if col == COL_LEFT else "R"
                cv2.putText(p1, tag, (px + 14, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_a, 2)

        # --- Panel 2: Coronal  volume[:, cy, :]  rows=Z  cols=X
        cor_raw   = self._apply_brightness(self.volume[:, cy, :])
        cor_dot   = (ann[0], ann[2]) if ann else None
        p2 = self._make_panel(cor_raw,
                               f"CORONAL  y={cy}/{H}",
                               cross_rc=(z, cx),
                               dot_rc=cor_dot,
                               dot_color=COL_LEFT if self.current_mode == "left"
                                         else COL_RIGHT)

        # --- Panel 3: Sagittal  volume[:, :, cx]  rows=Z  cols=Y
        sag_raw   = self._apply_brightness(self.volume[:, :, cx])
        sag_dot   = (ann[0], ann[1]) if ann else None
        p3 = self._make_panel(sag_raw,
                               f"SAGITTAL  x={cx}/{W}",
                               cross_rc=(z, cy),
                               dot_rc=sag_dot,
                               dot_color=COL_LEFT if self.current_mode == "left"
                                         else COL_RIGHT)

        combined = np.hstack([p1, p2, p3])

        # --- Info bar at bottom
        bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
        mode_col = COL_LEFT if self.current_mode == "left" else COL_RIGHT
        status = (f"Scan: {self.scan_id}  |  "
                  f"Mode: {'LEFT' if self.current_mode == 'left' else 'RIGHT'}  |  "
                  f"L={'✓' if self.left_tmj else '✗'}  "
                  f"R={'✓' if self.right_tmj else '✗'}  |  "
                  f"Done: {self.annotated_count}")
        cv2.putText(bar, status, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_col, 1, cv2.LINE_AA)
        controls = "A/D:slice  L/R:mode  S:save  U:undo  +/-:brightness  Q:quit"
        cv2.putText(bar, controls, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        return np.vstack([combined, bar])

    def update_display(self):
        img = self._build_display()
        cv2.imshow(self.window_name, img)

    # ------------------------------------------------------------------ #
    #  Mouse callback                                                       #
    # ------------------------------------------------------------------ #

    def mouse_callback(self, event, x, y, flags, param):
        D, H, W = self.original_shape

        if event == cv2.EVENT_MOUSEWHEEL:
            signed = flags if flags < 0x80000000 else flags - 0x100000000
            delta = 1 if signed > 0 else -1
            self.current_slice = int(np.clip(self.current_slice + delta, 0, D - 1))
            self.update_display()
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Determine which panel (x is in the combined image)
        # Panels: [0..PW) axial, [PW..2PW) coronal, [2PW..3PW) sagittal
        # Info bar is below: y >= PANEL_H → ignore
        if y >= PANEL_H:
            return

        panel_idx = x // PANEL_W
        local_x   = x % PANEL_W
        local_y   = y                # row in panel

        ann = self._active_ann()

        if panel_idx == 0:           # Axial: rows=Y  cols=X
            orig_z = self.current_slice
            orig_y = int(local_y * H / PANEL_H)
            orig_x = int(local_x * W / PANEL_W)

        elif panel_idx == 1:         # Coronal: rows=Z  cols=X
            orig_z = int(local_y * D / PANEL_H)
            orig_x = int(local_x * W / PANEL_W)
            orig_y = ann[1] if ann else H // 2
            self.current_slice = orig_z   # sync axial view

        elif panel_idx == 2:         # Sagittal: rows=Z  cols=Y
            orig_z = int(local_y * D / PANEL_H)
            orig_y = int(local_x * H / PANEL_W)
            orig_x = ann[2] if ann else W // 2
            self.current_slice = orig_z

        else:
            return

        # Clamp
        orig_z = int(np.clip(orig_z, 0, D - 1))
        orig_y = int(np.clip(orig_y, 0, H - 1))
        orig_x = int(np.clip(orig_x, 0, W - 1))

        if self.current_mode == "left":
            self.left_tmj = [orig_z, orig_y, orig_x]
            logger.info(f"LEFT TMJ: z={orig_z} y={orig_y} x={orig_x}")
            self.current_mode = "right"
        else:
            self.right_tmj = [orig_z, orig_y, orig_x]
            logger.info(f"RIGHT TMJ: z={orig_z} y={orig_y} x={orig_x}")
            self.current_mode = "left"

        self.update_display()

    # ------------------------------------------------------------------ #
    #  Save / reset                                                         #
    # ------------------------------------------------------------------ #

    def save_annotation(self) -> bool:
        if self.left_tmj is None or self.right_tmj is None:
            logger.warning("Both LEFT and RIGHT TMJ must be annotated!")
            return False

        annotation = {
            "scan_id": self.scan_id,
            "dicom_dir": str(self.dicom_dir),
            "original_shape": [int(v) for v in self.original_shape],
            "annotated_at": datetime.now().isoformat(),
            "left_tmj":  {"center": [int(v) for v in self.left_tmj],
                          "confidence": "manual"},
            "right_tmj": {"center": [int(v) for v in self.right_tmj],
                          "confidence": "manual"},
        }
        out = self.output_dir / f"{self.scan_id}_rois.json"
        with open(out, "w") as f:
            json.dump(annotation, f, indent=2)
        logger.info(f"✓ Saved → {out}")
        self.annotated_count += 1
        return True

    # ------------------------------------------------------------------ #
    #  Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        logger.info(f"Loading: {self.dicom_dir}")
        try:
            self.volume = self.load_dicom_series(self.dicom_dir)
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            return

        self.current_slice = self.original_shape[0] // 2

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Resize window to show all 3 panels
        cv2.resizeWindow(self.window_name, PANEL_W * 3, PANEL_H + 50)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()

        logger.info("Ready — click to annotate. A/D = slices, S = save, Q = quit")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('s'):
                if self.save_annotation():
                    logger.info("Saved. Press Q to close.")

            elif key == ord('l'):
                self.current_mode = "left"
                self.update_display()

            elif key == ord('r'):
                self.current_mode = "right"
                self.update_display()

            elif key == ord('u'):
                if self.current_mode == "right" and self.right_tmj:
                    self.right_tmj = None; self.current_mode = "right"
                elif self.left_tmj:
                    self.left_tmj = None; self.current_mode = "left"
                self.update_display()

            elif key == ord('n'):
                if self.left_tmj and self.right_tmj:
                    break
                else:
                    logger.warning("Annotate and save both points first.")

            elif key in (ord('+'), ord('=')):
                self.brightness = min(self.brightness + 0.1, 3.0)
                self.update_display()

            elif key in (ord('-'), ord('_')):
                self.brightness = max(self.brightness - 0.1, 0.1)
                self.update_display()

            # Slice navigation
            elif key in (ord('d'), 83):   # D or →
                self.current_slice = min(self.current_slice + 1,
                                         self.original_shape[0] - 1)
                self.update_display()

            elif key in (ord('a'), 81):   # A or ←
                self.current_slice = max(self.current_slice - 1, 0)
                self.update_display()

            elif key == ord('D'):         # Shift+D
                self.current_slice = min(self.current_slice + 10,
                                         self.original_shape[0] - 1)
                self.update_display()

            elif key == ord('A'):         # Shift+A
                self.current_slice = max(self.current_slice - 10, 0)
                self.update_display()

        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ROI Annotation Tool for TMJ Detection Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("dicom_dir", type=str,
                        help="Path to DICOM directory")
    parser.add_argument("--output", type=str,
                        default="data/roi_annotations",
                        help="Output directory (default: data/roi_annotations)")
    args = parser.parse_args()

    if not Path(args.dicom_dir).exists():
        logger.error(f"Not found: {args.dicom_dir}")
        sys.exit(1)

    ROIAnnotationTool(args.dicom_dir, args.output).run()


if __name__ == "__main__":
    main()
