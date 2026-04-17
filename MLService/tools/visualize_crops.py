#!/usr/bin/env python3
"""
Visualize detector-generated NIfTI crops.

Для каждого исследования показывает 3 среза (axial, coronal, sagittal)
для левого и правого кропа. Позволяет убедиться что мыщелок
виден и кроп центрирован правильно.

Usage:
    # Показать 5 случайных исследований
    ./venv/bin/python tools/visualize_crops.py

    # Конкретное исследование
    ./venv/bin/python tools/visualize_crops.py --study study_0001

    # Сохранить PNG вместо показа
    ./venv/bin/python tools/visualize_crops.py --save-dir /tmp/crop_check
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

CROPS_DIR = Path(__file__).parent.parent / "data" / "detector_crops"


def load_crop(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)


def normalize(vol: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(vol, [2, 98])
    vol = np.clip(vol, p2, p98)
    denom = p98 - p2
    return (vol - p2) / denom if denom > 0 else np.zeros_like(vol)


def show_study(study_id: str, crops_dir: Path, save_dir: Path | None = None):
    left_path = crops_dir / study_id / f"{study_id}_left.nii.gz"
    right_path = crops_dir / study_id / f"{study_id}_right.nii.gz"

    missing = [p for p in (left_path, right_path) if not p.exists()]
    if missing:
        print(f"[{study_id}] Не найдены: {[str(m) for m in missing]}")
        return

    left = normalize(load_crop(left_path))
    right = normalize(load_crop(right_path))

    D, H, W = left.shape
    mid = (D // 2, H // 2, W // 2)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(study_id, fontsize=14, fontweight="bold")

    slices = [
        ("Axial", left[mid[0], :, :], right[mid[0], :, :]),
        ("Coronal", left[:, mid[1], :], right[:, mid[1], :]),
        ("Sagittal", left[:, :, mid[2]], right[:, :, mid[2]]),
    ]

    for col, (plane, l_sl, r_sl) in enumerate(slices):
        for row, (sl, side) in enumerate([(l_sl, "Left"), (r_sl, "Right")]):
            ax = axes[row][col]
            ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.set_title(f"{side} — {plane}", fontsize=10)
            # Crosshair на центре
            ch, cw = sl.shape[0] // 2, sl.shape[1] // 2
            ax.axhline(ch, color="red", alpha=0.4, linewidth=0.8)
            ax.axvline(cw, color="red", alpha=0.4, linewidth=0.8)
            ax.axis("off")

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{study_id}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Сохранено: {out}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize TMJ NIfTI crops")
    parser.add_argument(
        "--crops-dir",
        default=str(CROPS_DIR),
        help="Папка с кропами (по умолчанию data/detector_crops)",
    )
    parser.add_argument(
        "--study", default=None, help="study_id для показа (иначе случайная выборка)"
    )
    parser.add_argument(
        "-n", type=int, default=5, help="Количество случайных исследований (default: 5)"
    )
    parser.add_argument("--save-dir", default=None, help="Сохранить PNG в папку вместо показа")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    save_dir = Path(args.save_dir) if args.save_dir else None

    if not crops_dir.exists():
        print(f"Папка не найдена: {crops_dir}")
        return

    # Собрать доступные исследования (есть оба кропа)
    studies = sorted(
        d.name
        for d in crops_dir.iterdir()
        if d.is_dir()
        and (d / f"{d.name}_left.nii.gz").exists()
        and (d / f"{d.name}_right.nii.gz").exists()
    )

    if not studies:
        print(f"Кропов не найдено в {crops_dir}")
        return

    print(f"Найдено исследований с кропами: {len(studies)}")

    if args.study:
        selected = [args.study]
    else:
        n = min(args.n, len(studies))
        selected = random.sample(studies, n)
        print(f"Показываю {n} случайных: {selected}")

    for sid in selected:
        print(f"\n--- {sid} ---")
        show_study(sid, crops_dir, save_dir)


if __name__ == "__main__":
    main()
