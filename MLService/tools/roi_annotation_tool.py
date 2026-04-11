#!/usr/bin/env python3
"""
ROI Annotation Tool for TMJ Detection Dataset — MPR view, step-by-step workflow

Workflow:
  1. Поставь LEFT точку (можно двигать пока не нажал S)
  2. S → фиксируем левый, переходим к RIGHT
  3. Поставь RIGHT точку
  4. S → сохраняем, показываем превью обоих суставов
  5. S ещё раз → выход (batch_annotate переходит к следующему)

Controls:
    Left Click       : Поставить точку текущей стадии
    A / ←            : Предыдущий Axial срез
    D / →            : Следующий Axial срез
    Shift+A / Shift+D: ±10 срезов
    Mouse Wheel      : Навигация по Axial срезам
    + / -            : Яркость
    S                : Следующий шаг (фиксация / сохранение / выход)
    U                : Отменить текущую точку
    Q / ESC          : Выход без сохранения
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime

import numpy as np
import cv2
import pydicom
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PANEL_H = 512
PANEL_W = 512

COL_LEFT    = (0,   220,   0)    # green
COL_RIGHT   = (220,   0,   0)    # blue
COL_CROSS   = (180, 180,   0)    # yellow
COL_LOCKED  = (120, 120, 120)    # grey (locked annotation)

# Stages
STAGE_LEFT    = "LEFT"       # placing left joint
STAGE_RIGHT   = "RIGHT"      # left locked, placing right
STAGE_PREVIEW = "PREVIEW"    # both saved, showing result


class ROIAnnotationTool:
    def __init__(self, dicom_dir: str, output_dir: str):
        self.dicom_dir   = Path(dicom_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.volume         = None
        self.current_slice  = 0
        self.scan_id        = None
        self.original_shape = None

        self.left_tmj  = None   # [z, y, x]
        self.right_tmj = None
        self.stage     = STAGE_LEFT

        self.brightness = 1.0
        self.window_name = "ROI Annotation Tool"

        self.annotated_count = len(list(self.output_dir.glob("*_rois.json")))

    # ------------------------------------------------------------------ #
    #  DICOM loading                                                        #
    # ------------------------------------------------------------------ #

    def load_dicom_series(self, dicom_dir: Path) -> np.ndarray:
        files = []
        for root, _, fnames in os.walk(dicom_dir):
            for f in fnames:
                fp = Path(root) / f
                try:
                    pydicom.dcmread(fp, stop_before_pixels=True)
                    files.append(fp)
                except Exception:
                    continue
        if not files:
            raise ValueError(f"No DICOM files in {dicom_dir}")

        logger.info(f"Found {len(files)} DICOM files")
        slices = [pydicom.dcmread(str(f)) for f in
                  tqdm(files, desc="Reading DICOM")]
        slices.sort(key=lambda s: float(getattr(s, 'InstanceNumber', 0)))

        vol = np.stack([s.pixel_array.astype(np.float32) for s in slices])
        p2, p98 = np.percentile(vol, [2, 98])
        vol = np.clip(vol, p2, p98)
        vol = ((vol - p2) / max(p98 - p2, 1) * 255).astype(np.uint8)

        self.original_shape = vol.shape
        self.scan_id = self.dicom_dir.name
        logger.info(f"Volume: {vol.shape}")
        return vol

    def _load_existing_annotation(self):
        ann_file = self.output_dir / f"{self.scan_id}_rois.json"
        if not ann_file.exists():
            return
        try:
            with open(ann_file) as f:
                data = json.load(f)
            self.left_tmj  = data["left_tmj"]["center"]
            self.right_tmj = data["right_tmj"]["center"]
            self.current_slice = self.left_tmj[0]
            self.stage = STAGE_PREVIEW   # show existing annotation immediately
            logger.info(f"Loaded: L={self.left_tmj}  R={self.right_tmj}")
        except Exception as e:
            logger.warning(f"Could not load annotation: {e}")

    # ------------------------------------------------------------------ #
    #  Display helpers                                                      #
    # ------------------------------------------------------------------ #

    def _adj(self, sl: np.ndarray) -> np.ndarray:
        return np.clip(sl.astype(np.float32) * self.brightness, 0, 255).astype(np.uint8)

    def _make_panel(self, raw, label, cross_rc=None,
                    dots=None) -> np.ndarray:
        """
        raw    : (H, W) uint8
        dots   : list of (row, col, color, label_str)
        cross_rc: (row, col) in raw coords
        """
        oh, ow = raw.shape
        panel = cv2.resize(raw, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

        def to_p(r, c):
            return (int(c * PANEL_W / ow), int(r * PANEL_H / oh))

        if cross_rc is not None:
            px, py = to_p(*cross_rc)
            cv2.line(bgr, (0, py), (PANEL_W, py), COL_CROSS, 1)
            cv2.line(bgr, (px, 0), (px, PANEL_H), COL_CROSS, 1)

        for dot in (dots or []):
            r, c, col, tag = dot
            px, py = to_p(r, c)
            cv2.circle(bgr, (px, py), 10, col, 2)
            cv2.circle(bgr, (px, py),  2, col, -1)
            if tag:
                cv2.putText(bgr, tag, (px + 14, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        cv2.putText(bgr, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return bgr

    def _build_display(self) -> np.ndarray:
        D, H, W = self.original_shape
        z = self.current_slice

        # Reference point for coronal/sagittal: active annotation
        active = self.left_tmj if self.stage in (STAGE_LEFT, STAGE_RIGHT) and self.left_tmj else \
                 self.right_tmj if self.right_tmj else None
        if self.stage == STAGE_RIGHT and self.right_tmj:
            active = self.right_tmj
        ref_y = active[1] if active else H // 2
        ref_x = active[2] if active else W // 2

        # ── AXIAL panel (volume[z, :, :]) ──
        axial_raw = self._adj(self.volume[z])
        axial_dots = []
        for ann, col, tag in [(self.left_tmj, COL_LEFT, "L"),
                               (self.right_tmj, COL_RIGHT, "R")]:
            if ann is None:
                continue
            # in STAGE_LEFT, left is being placed → bright; right N/A
            # in STAGE_RIGHT, left is locked → grey; right is active → colour
            effective_col = col
            if self.stage == STAGE_RIGHT and tag == "L":
                effective_col = COL_LOCKED
            if abs(ann[0] - z) < 4:
                alpha = 1.0 - abs(ann[0] - z) / 4.0
                c = tuple(int(v * alpha) for v in effective_col)
                axial_dots.append((ann[1], ann[2], c, tag if ann[0] == z else ""))
        p_axial = self._make_panel(axial_raw, f"AXIAL  z={z}/{D}",
                                   cross_rc=(ref_y, ref_x) if active else None,
                                   dots=axial_dots)

        # ── CORONAL panel (volume[:, ref_y, :]) rows=Z, cols=X ──
        cor_raw  = self._adj(self.volume[:, ref_y, :])
        cor_dots = []
        if self.left_tmj:
            col = COL_LOCKED if self.stage == STAGE_RIGHT else COL_LEFT
            cor_dots.append((self.left_tmj[0], self.left_tmj[2], col, "L"))
        if self.right_tmj:
            cor_dots.append((self.right_tmj[0], self.right_tmj[2], COL_RIGHT, "R"))
        p_cor = self._make_panel(cor_raw, f"CORONAL  y={ref_y}/{H}",
                                 cross_rc=(z, ref_x) if active else (z, W // 2),
                                 dots=cor_dots)

        # ── SAGITTAL panel (volume[:, :, ref_x]) rows=Z, cols=Y ──
        sag_raw  = self._adj(self.volume[:, :, ref_x])
        sag_dots = []
        if self.left_tmj:
            col = COL_LOCKED if self.stage == STAGE_RIGHT else COL_LEFT
            sag_dots.append((self.left_tmj[0], self.left_tmj[1], col, "L"))
        if self.right_tmj:
            sag_dots.append((self.right_tmj[0], self.right_tmj[1], COL_RIGHT, "R"))
        p_sag = self._make_panel(sag_raw, f"SAGITTAL  x={ref_x}/{W}",
                                 cross_rc=(z, ref_y) if active else (z, H // 2),
                                 dots=sag_dots)

        combined = np.hstack([p_axial, p_cor, p_sag])

        # ── Info bar ──
        bar = np.zeros((56, combined.shape[1], 3), dtype=np.uint8)

        if self.stage == STAGE_LEFT:
            stage_txt = "ШАग 1/2: поставь LEFT сустав (зелёный)  →  S для перехода к правому"
            col = COL_LEFT
        elif self.stage == STAGE_RIGHT:
            stage_txt = "ШАГИ 2/2: поставь RIGHT сустав (синий)  →  S для сохранения и превью"
            col = COL_RIGHT
        else:
            stage_txt = "ПРЕВЬЮ: проверь оба сустава  →  S для подтверждения и выхода"
            col = (220, 220, 100)

        cv2.putText(bar, stage_txt, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

        status = (f"Scan: {self.scan_id}  |  "
                  f"L={'✓' if self.left_tmj else '–'}  "
                  f"R={'✓' if self.right_tmj else '–'}  |  "
                  f"Done: {self.annotated_count}")
        cv2.putText(bar, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        controls = "A/D:срез  S:следующий шаг  U:отмена точки  +/-:яркость  Q:выход"
        cv2.putText(bar, controls, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        return np.vstack([combined, bar])

    def update_display(self):
        cv2.imshow(self.window_name, self._build_display())

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
        if y >= PANEL_H:     # info bar — ignore
            return
        if self.stage == STAGE_PREVIEW:
            return           # no edits in preview

        panel_idx = x // PANEL_W
        lx = x % PANEL_W
        ly = y

        # Reference for coronal/sagittal (same logic as display)
        active = self.left_tmj if self.stage == STAGE_LEFT else self.right_tmj
        ref_y = active[1] if active else H // 2
        ref_x = active[2] if active else W // 2

        if panel_idx == 0:       # Axial: rows=Y, cols=X
            oz = self.current_slice
            oy = int(ly * H / PANEL_H)
            ox = int(lx * W / PANEL_W)

        elif panel_idx == 1:     # Coronal ([:, ref_y, :]): rows=Z, cols=X
            oz = int(ly * D / PANEL_H)
            ox = int(lx * W / PANEL_W)
            oy = ref_y
            self.current_slice = oz

        elif panel_idx == 2:     # Sagittal ([:, :, ref_x]): rows=Z, cols=Y
            oz = int(ly * D / PANEL_H)
            oy = int(lx * H / PANEL_W)
            ox = ref_x
            self.current_slice = oz

        else:
            return

        oz = int(np.clip(oz, 0, D - 1))
        oy = int(np.clip(oy, 0, H - 1))
        ox = int(np.clip(ox, 0, W - 1))

        if self.stage == STAGE_LEFT:
            self.left_tmj = [oz, oy, ox]
            logger.info(f"LEFT: z={oz} y={oy} x={ox}")
        else:  # STAGE_RIGHT
            self.right_tmj = [oz, oy, ox]
            logger.info(f"RIGHT: z={oz} y={oy} x={ox}")

        self.update_display()

    # ------------------------------------------------------------------ #
    #  Save                                                                 #
    # ------------------------------------------------------------------ #

    def _save_json(self):
        ann = {
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
            json.dump(ann, f, indent=2)
        logger.info(f"✓ Saved → {out}")
        self.annotated_count += 1

    def _handle_s(self) -> bool:
        """Returns True when tool should exit."""
        if self.stage == STAGE_LEFT:
            if self.left_tmj is None:
                logger.warning("Поставь LEFT точку сначала!")
                return False
            self.stage = STAGE_RIGHT
            # Jump to left z-slice for reference
            self.current_slice = self.left_tmj[0]
            logger.info("LEFT зафиксирован → теперь размечай RIGHT")
            self.update_display()
            return False

        elif self.stage == STAGE_RIGHT:
            if self.right_tmj is None:
                logger.warning("Поставь RIGHT точку сначала!")
                return False
            self._save_json()
            self.stage = STAGE_PREVIEW
            # Show both annotations together
            # Jump to midpoint between L and R
            mid_z = (self.left_tmj[0] + self.right_tmj[0]) // 2
            self.current_slice = mid_z
            logger.info("Сохранено! Проверь превью → S для выхода")
            self.update_display()
            return False

        else:  # STAGE_PREVIEW
            return True   # exit

    # ------------------------------------------------------------------ #
    #  Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        logger.info(f"Loading: {self.dicom_dir}")
        try:
            self.volume = self.load_dicom_series(self.dicom_dir)
        except Exception as e:
            logger.error(f"Failed: {e}")
            return

        self.current_slice = self.original_shape[0] // 2
        self._load_existing_annotation()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, PANEL_W * 3, PANEL_H + 56)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()

        logger.info("Шаг 1: поставь LEFT сустав, потом S")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('s'):
                if self._handle_s():
                    break

            elif key == ord('u'):
                if self.stage == STAGE_LEFT:
                    self.left_tmj = None
                elif self.stage == STAGE_RIGHT:
                    self.right_tmj = None
                # PREVIEW: no undo
                self.update_display()

            elif key in (ord('+'), ord('=')):
                self.brightness = min(self.brightness + 0.1, 3.0)
                self.update_display()

            elif key in (ord('-'), ord('_')):
                self.brightness = max(self.brightness - 0.1, 0.1)
                self.update_display()

            elif key in (ord('d'), 83):
                self.current_slice = min(self.current_slice + 1,
                                         self.original_shape[0] - 1)
                self.update_display()

            elif key in (ord('a'), 81):
                self.current_slice = max(self.current_slice - 1, 0)
                self.update_display()

            elif key == ord('D'):
                self.current_slice = min(self.current_slice + 10,
                                         self.original_shape[0] - 1)
                self.update_display()

            elif key == ord('A'):
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
    parser.add_argument("dicom_dir", type=str)
    parser.add_argument("--output", type=str, default="data/roi_annotations")
    args = parser.parse_args()

    if not Path(args.dicom_dir).exists():
        logger.error(f"Not found: {args.dicom_dir}")
        sys.exit(1)

    ROIAnnotationTool(args.dicom_dir, args.output).run()


if __name__ == "__main__":
    main()
