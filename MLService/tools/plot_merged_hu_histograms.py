#!/usr/bin/env python3
"""
Merge two HU (Hounsfield units) histograms on one axis — same bins, transparent bars,
fitted normal curves and mean markers.

Usage:
  python plot_merged_hu_histograms.py --demo -o hu_merged.png

  python plot_merged_hu_histograms.py --hu-a a.npy --hu-b b.npy -o hu_merged.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from scipy import stats

# Акцент — насыщенный синий; второй ряд — в той же гамме, но спокойнее
COLOR_ACCENT_FILL = "#2563eb"
COLOR_ACCENT_EDGE = "#1d4ed8"
COLOR_SECOND_FILL = "#bfdbfe"
COLOR_SECOND_EDGE = "#60a5fa"
COLOR_CURVE_ACCENT = "#1e3a8a"
COLOR_CURVE_SECOND = "#3b82f6"
COLOR_MEAN_ACCENT = "#1d4ed8"
COLOR_MEAN_SECOND = "#64748b"
COLOR_GRID = "#e2e8f0"
COLOR_SPINE = "#cbd5e1"


def _load_hu(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def plot_merged(
    hu_a: np.ndarray,
    hu_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: Path,
    bins: int = 72,
    dpi: int = 200,
) -> None:
    combined = np.concatenate([hu_a, hu_b])
    lo, hi = float(np.percentile(combined, 0.5)), float(np.percentile(combined, 99.5))
    edges = np.linspace(lo, hi, bins + 1)
    width = float(edges[1] - edges[0])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "SF Pro Display",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#fafbfc",
            "axes.edgecolor": COLOR_SPINE,
            "axes.linewidth": 1.0,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    # Сначала второй ряд (фон), затем акцентный синий поверх
    ax.hist(
        hu_b,
        bins=edges,
        alpha=0.92,
        color=COLOR_SECOND_FILL,
        label=label_b,
        edgecolor=COLOR_SECOND_EDGE,
        linewidth=0.55,
        zorder=2,
    )
    ax.hist(
        hu_a,
        bins=edges,
        alpha=0.88,
        color=COLOR_ACCENT_FILL,
        label=label_a,
        edgecolor=COLOR_ACCENT_EDGE,
        linewidth=0.65,
        zorder=3,
    )

    x = np.linspace(lo, hi, 600)
    curve_outline = [pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()]

    for hu, color, z in ((hu_b, COLOR_CURVE_SECOND, 4), (hu_a, COLOR_CURVE_ACCENT, 5)):
        mu, sigma = stats.norm.fit(hu)
        y = stats.norm.pdf(x, mu, sigma) * hu.size * width
        ax.plot(
            x,
            y,
            color=color,
            linestyle="-",
            linewidth=2.35 if color == COLOR_CURVE_ACCENT else 1.85,
            alpha=0.95 if color == COLOR_CURVE_ACCENT else 0.82,
            zorder=z,
            path_effects=curve_outline,
        )

    mu_a, _ = stats.norm.fit(hu_a)
    mu_b, _ = stats.norm.fit(hu_b)
    ax.axvline(
        mu_b,
        color=COLOR_MEAN_SECOND,
        linestyle=(0, (4, 3)),
        linewidth=1.6,
        alpha=0.85,
        zorder=6,
        label=f"Среднее ({label_b}): {mu_b:.1f}",
    )
    ax.axvline(
        mu_a,
        color=COLOR_MEAN_ACCENT,
        linestyle=(0, (6, 2)),
        linewidth=2.0,
        alpha=0.95,
        zorder=7,
        label=f"Среднее ({label_a}): {mu_a:.1f}",
    )

    ax.set_xlabel("HU", fontweight="600", color="#334155", labelpad=8)
    ax.set_ylabel("Количество пикселей", fontweight="600", color="#334155", labelpad=8)
    ax.tick_params(colors="#475569", length=5, width=0.8)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.7, color=COLOR_GRID, zorder=0)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_SPINE)

    leg = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.96,
        edgecolor="#e2e8f0",
        facecolor="#ffffff",
        borderpad=0.85,
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout(pad=1.1)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


def demo_arrays(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Roughly match the two screenshots (means ~1213 and ~453)."""
    a = rng.normal(1213.2, 220.0, size=120_000)
    b = rng.normal(452.6, 140.0, size=55_000)
    a = np.clip(a, 200, 2200)
    b = np.clip(b, -450, 1250)
    return a, b


def main() -> None:
    p = argparse.ArgumentParser(description="Merge two HU histograms on one figure.")
    p.add_argument("--hu-a", type=Path, help="Path to .npy with HU values (dataset A)")
    p.add_argument("--hu-b", type=Path, help="Path to .npy with HU values (dataset B)")
    p.add_argument(
        "--label-a",
        type=str,
        default="Губчатая кость (1400)",
    )
    p.add_argument(
        "--label-b",
        type=str,
        default="Кортикальная кость (1500)",
    )
    p.add_argument("-o", "--output", type=Path, default=Path("hu_merged.png"))
    p.add_argument("--bins", type=int, default=72)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--demo", action="store_true", help="Use synthetic HU matching prior means")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.demo:
        hu_a, hu_b = demo_arrays(rng)
    else:
        if not args.hu_a or not args.hu_b:
            p.error("Provide both --hu-a and --hu-b, or use --demo")
        hu_a = _load_hu(args.hu_a)
        hu_b = _load_hu(args.hu_b)

    plot_merged(
        hu_a,
        hu_b,
        args.label_a,
        args.label_b,
        args.output,
        bins=args.bins,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
