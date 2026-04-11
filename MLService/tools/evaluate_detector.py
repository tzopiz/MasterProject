#!/usr/bin/env python3
"""
Evaluate a trained TMJHeatmapDetector on the test split.

Reports MAE in downsampled pixels, original pixels, mm (if voxel spacing known),
and percentage of predictions within 5mm / 10mm thresholds.

Usage:
    ./venv/bin/python tools/evaluate_detector.py \\
        --model  experiments/heatmap_detector_XXXXXX/best_model.pth \\
        --split  data/detector_split.json \\
        --annotations data/roi_annotations \\
        --dataset data/dataset_cbct_public
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tmj_heatmap_detector import TMJHeatmapDetector
from training.datasets.tmj_heatmap_dataset import TMJHeatmapDataset
from training.utils.heatmap import coords_from_heatmap
from torch.utils.data import DataLoader


def evaluate(model, loader, device, ds_factor, voxel_mm=None):
    model.eval()
    errs_left, errs_right = [], []

    with torch.no_grad():
        for vols, targets in loader:
            vols = vols.to(device)
            pred = torch.sigmoid(model(vols)).cpu()

            B = pred.shape[0]
            for b in range(B):
                for ch, err_list in enumerate([errs_left, errs_right]):
                    pc, _ = coords_from_heatmap(pred[b, ch],    ds_factor)
                    tc, _ = coords_from_heatmap(targets[b, ch], ds_factor)
                    err_ds = torch.sqrt(((pc - tc) ** 2).sum()).item()
                    err_list.append(err_ds)

    all_errs = errs_left + errs_right
    results = {
        "n_samples":          len(errs_left),
        "mae_left_ds":        float(np.mean(errs_left)),
        "mae_right_ds":       float(np.mean(errs_right)),
        "mae_overall_ds":     float(np.mean(all_errs)),
        "mae_left_orig":      float(np.mean(errs_left)  * ds_factor),
        "mae_right_orig":     float(np.mean(errs_right) * ds_factor),
        "mae_overall_orig":   float(np.mean(all_errs) * ds_factor),
    }
    if voxel_mm:
        results["mae_overall_mm"]  = results["mae_overall_orig"] * voxel_mm
        results["pct_within_5mm"]  = 100 * float(np.mean([e * ds_factor * voxel_mm < 5  for e in all_errs]))
        results["pct_within_10mm"] = 100 * float(np.mean([e * ds_factor * voxel_mm < 10 for e in all_errs]))

    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    model = TMJHeatmapDetector().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from epoch {ckpt['epoch']} (val MAE={ckpt['best_val_mae']:.2f} ds px)")

    with open(args.split) as f:
        split = json.load(f)

    test_ds = TMJHeatmapDataset(
        split["test"], args.annotations,
        dataset_dir=args.dataset, is_train=False,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test set: {len(test_ds)} studies")

    results = evaluate(model, loader, device, ds_factor=6, voxel_mm=args.voxel_mm)

    print("\n=== TEST RESULTS ===")
    print(f"  MAE left  (ds px):     {results['mae_left_ds']:.2f}")
    print(f"  MAE right (ds px):     {results['mae_right_ds']:.2f}")
    print(f"  MAE overall (ds px):   {results['mae_overall_ds']:.2f}")
    print(f"  MAE left  (orig px):   {results['mae_left_orig']:.1f}")
    print(f"  MAE right (orig px):   {results['mae_right_orig']:.1f}")
    print(f"  MAE overall (orig px): {results['mae_overall_orig']:.1f}")
    if args.voxel_mm:
        print(f"  MAE overall (mm):      {results['mae_overall_mm']:.2f}")
        print(f"  % within 5mm:          {results['pct_within_5mm']:.1f}%")
        print(f"  % within 10mm:         {results['pct_within_10mm']:.1f}%")

    out = Path(args.model).parent / "test_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate TMJ Heatmap Detector on test split")
    p.add_argument("--model",       required=True, help="Path to best_model.pth")
    p.add_argument("--split",       default="data/detector_split.json")
    p.add_argument("--annotations", default="data/roi_annotations")
    p.add_argument("--dataset",     default="data/dataset_cbct_public")
    p.add_argument("--voxel-mm",    type=float, default=None,
                   help="Voxel spacing in mm for MAE_mm and %%within_Nmm metrics")
    main(p.parse_args())
