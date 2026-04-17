#!/usr/bin/env python3
"""
Quick script to show training results
"""

import sys

import torch

checkpoint_path = (
    sys.argv[1] if len(sys.argv) > 1 else "experiments/detector_20251124_013727/best_model.pth"
)

# Load model with weights_only=False
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("╔═══════════════════════════════════════════════════════════════╗")
print("║          🎯 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ TMJ DETECTOR                 ║")
print("╚═══════════════════════════════════════════════════════════════╝")
print()
print(f"📌 Эпоха лучшей модели: {checkpoint.get('epoch', 'N/A')}")

val_mae = checkpoint.get("val_mae", None)
if val_mae is not None and val_mae != "N/A":
    print(f"✅ Best Val MAE: {val_mae:.2f} px")
else:
    print(f"✅ Best Val MAE: {val_mae}")

val_loss = checkpoint.get("val_loss", None)
if val_loss is not None and val_loss != "N/A":
    print(f"📊 Best Val Loss: {val_loss:.6f}")
else:
    print(f"📊 Best Val Loss: {val_loss}")
print()

# Per-side breakdown
if "val_mae_left" in checkpoint and "val_mae_right" in checkpoint:
    print(f"   Left TMJ MAE:  {checkpoint['val_mae_left']:.2f} px")
    print(f"   Right TMJ MAE: {checkpoint['val_mae_right']:.2f} px")
    print()

# Per-axis breakdown
if "val_mae_z" in checkpoint:
    print(f"   Z-axis MAE: {checkpoint['val_mae_z']:.2f} px")
    print(f"   Y-axis MAE: {checkpoint['val_mae_y']:.2f} px")
    print(f"   X-axis MAE: {checkpoint['val_mae_x']:.2f} px")
    print()

print("📈 История обучения:")
print("─" * 60)
history = checkpoint.get("history", {})
if history:
    train_mae = history.get("train_mae", [])
    val_mae_hist = history.get("val_mae", [])

    # Show first 5
    print("\nПервые 5 эпох:")
    for i in range(min(5, len(val_mae_hist))):
        print(f"  Epoch {i + 1:3d}: Train={train_mae[i]:6.2f} px | Val={val_mae_hist[i]:6.2f} px")

    if len(val_mae_hist) > 10:
        print("\n  ...")

    # Show last 10
    print(f"\nПоследние {min(10, len(val_mae_hist))} эпох:")
    start_idx = max(0, len(val_mae_hist) - 10)
    for i in range(start_idx, len(val_mae_hist)):
        indicator = "✅" if val_mae_hist[i] == min(val_mae_hist) else "  "
        print(
            f"  {indicator} Epoch {i + 1:3d}: Train={train_mae[i]:6.2f} px | Val={val_mae_hist[i]:6.2f} px"
        )
else:
    print("  История не найдена в checkpoint")

print()
print(f"💾 Модель: {checkpoint_path}")
print()

# Show all keys for debugging
print("📋 Доступные ключи в checkpoint:")
for key in sorted(checkpoint.keys()):
    if key not in ["model_state_dict", "optimizer_state_dict"]:
        value = checkpoint[key]
        if isinstance(value, (int, float, str)):
            print(f"   - {key}: {value}")
        elif isinstance(value, dict):
            print(f"   - {key}: dict with {len(value)} keys")
        elif isinstance(value, list):
            print(f"   - {key}: list with {len(value)} items")
        else:
            print(f"   - {key}: {type(value).__name__}")
