#!/usr/bin/env python3
"""
Visualize 3D CBCT with TMJ Bounding Boxes.

This script renders a 3D isosurface of the skull from the CBCT volume
and overlays the detected TMJ bounding boxes.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import pydicom

# Add parent directory to path for imports if needed
sys.path.append(str(Path(__file__).parent.parent))

def load_dicom_volume(dicom_dir: Path) -> np.ndarray:
    """Load DICOM series from directory."""
    dicom_files = list(dicom_dir.glob('*.dcm'))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
        
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    return volume

def load_crop_summary(summary_path: Path) -> dict:
    with open(summary_path, 'r') as f:
        return json.load(f)

def get_study_coords(summary: dict, study_id: str):
    for study in summary['studies']:
        if study['study'] == study_id:
            return study['predicted_coords'], study['crop_size'], study['volume_shape']
    raise ValueError(f"Study {study_id} not found in summary")

def plot_cube(ax, center, size, color='r', linewidth=2):
    """Draw a wireframe cube."""
    x, y, z = center
    h = size / 2
    
    # Corners
    x_range = [x - h, x + h]
    y_range = [y - h, y + h]
    z_range = [z - h, z + h]
    
    # Draw edges - X aligned
    for y_k in y_range:
        for z_k in z_range:
            ax.plot3D(x_range, [y_k, y_k], [z_k, z_k], color=color, linewidth=linewidth)
            
    # Y aligned
    for x_k in x_range:
        for z_k in z_range:
            ax.plot3D([x_k, x_k], y_range, [z_k, z_k], color=color, linewidth=linewidth)
            
    # Z aligned
    for x_k in x_range:
        for y_k in y_range:
            ax.plot3D([x_k, x_k], [y_k, y_k], z_range, color=color, linewidth=linewidth)

def visualize_3d(volume, coords_left, coords_right, crop_size, output_path=None, threshold=None, downsample=4, azim=90):
    """
    Render 3D isosurface and bounding boxes.
    Two separate meshes: background CBCT and highlighted TMJ region with lower threshold.
    
    azim: Azimuth angle for view (0=front, 90=right side, 180=back, 270=left side)
    """
    print("Preprocessing volume for 3D rendering...")
    
    # Convert TMJ coordinates
    zr, yr, xr = coords_right
    zl, yl, xl = coords_left
    h = crop_size // 2
    
    # Downsample for performance
    vol_small = volume[::downsample, ::downsample, ::downsample]
    
    # Determine thresholds
    print(f"Volume range: {volume.min()} to {volume.max()}")
    
    # Background threshold (higher - less detail, just skull outline)
    threshold_bg = threshold if threshold else 500
    # TMJ region threshold (much lower - more detail inside joint)
    threshold_tmj = 100  # Very low to capture soft tissue and fine bone detail
    
    print(f"Background threshold: {threshold_bg}")
    print(f"TMJ region threshold: {threshold_tmj}")
    
    max_z = volume.shape[0]
    
    # Setup plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    from matplotlib.colors import LinearSegmentedColormap
    
    # ========== MESH 1: Background CBCT (high threshold, transparent) ==========
    print("Generating background mesh...")
    try:
        verts_bg, faces_bg, _, _ = measure.marching_cubes(vol_small, level=threshold_bg)
    except ValueError:
        threshold_bg = np.percentile(vol_small, 85)
        verts_bg, faces_bg, _, _ = measure.marching_cubes(vol_small, level=threshold_bg)
    
    verts_bg = verts_bg * downsample
    
    # Convert to plot coords
    verts_bg_plot = np.zeros_like(verts_bg)
    verts_bg_plot[:, 0] = verts_bg[:, 2]
    verts_bg_plot[:, 1] = verts_bg[:, 1]
    verts_bg_plot[:, 2] = max_z - verts_bg[:, 0]
    
    # Color background mesh - muted teal/coral
    face_centers_bg = verts_bg_plot[faces_bg].mean(axis=1)
    z_norm_bg = (face_centers_bg[:, 2] - face_centers_bg[:, 2].min()) / (face_centers_bg[:, 2].max() - face_centers_bg[:, 2].min() + 1e-8)
    
    cmap_bg = LinearSegmentedColormap.from_list('bg', [
        (0.1, 0.5, 0.6),
        (0.2, 0.65, 0.7),
        (0.7, 0.4, 0.5),
        (0.85, 0.35, 0.45),
    ])
    face_colors_bg = cmap_bg(z_norm_bg)
    face_colors_bg[:, 3] = 0.25  # Very transparent background
    
    mesh_bg = Poly3DCollection(verts_bg_plot[faces_bg], facecolors=face_colors_bg)
    mesh_bg.set_edgecolor('none')
    ax.add_collection3d(mesh_bg)
    
    # ========== MESH 2: TMJ Region (low threshold, bright, solid) ==========
    print("Generating TMJ region mesh...")
    
    # Extract just the TMJ region from volume
    z_min, z_max = max(0, zr - h), min(volume.shape[0], zr + h)
    y_min, y_max = max(0, yr - h), min(volume.shape[1], yr + h)
    x_min, x_max = max(0, xr - h), min(volume.shape[2], xr + h)
    
    tmj_region = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Lower downsample for TMJ region (more detail)
    ds_tmj = max(1, downsample // 2)
    tmj_small = tmj_region[::ds_tmj, ::ds_tmj, ::ds_tmj]
    
    try:
        verts_tmj, faces_tmj, _, _ = measure.marching_cubes(tmj_small, level=threshold_tmj)
    except ValueError:
        threshold_tmj = np.percentile(tmj_small, 50)
        verts_tmj, faces_tmj, _, _ = measure.marching_cubes(tmj_small, level=threshold_tmj)
    
    # Scale and offset to original position
    verts_tmj = verts_tmj * ds_tmj
    verts_tmj[:, 0] += z_min  # Z offset
    verts_tmj[:, 1] += y_min  # Y offset
    verts_tmj[:, 2] += x_min  # X offset
    
    # Convert to plot coords
    verts_tmj_plot = np.zeros_like(verts_tmj)
    verts_tmj_plot[:, 0] = verts_tmj[:, 2]
    verts_tmj_plot[:, 1] = verts_tmj[:, 1]
    verts_tmj_plot[:, 2] = max_z - verts_tmj[:, 0]
    
    # Color TMJ mesh - bright golden/orange gradient
    face_centers_tmj = verts_tmj_plot[faces_tmj].mean(axis=1)
    z_norm_tmj = (face_centers_tmj[:, 2] - face_centers_tmj[:, 2].min()) / (face_centers_tmj[:, 2].max() - face_centers_tmj[:, 2].min() + 1e-8)
    
    cmap_tmj = LinearSegmentedColormap.from_list('tmj', [
        (1.0, 0.6, 0.1),      # Deep orange
        (1.0, 0.8, 0.2),      # Golden
        (1.0, 0.95, 0.4),     # Bright yellow
        (1.0, 0.85, 0.3),     # Gold
    ])
    face_colors_tmj = cmap_tmj(z_norm_tmj)
    face_colors_tmj[:, 3] = 0.9  # Very solid/opaque
    
    # Add lighting to TMJ mesh
    v0 = verts_tmj_plot[faces_tmj[:, 0]]
    v1 = verts_tmj_plot[faces_tmj[:, 1]]
    v2 = verts_tmj_plot[faces_tmj[:, 2]]
    normals_tmj = np.cross(v1 - v0, v2 - v0)
    normals_tmj = normals_tmj / (np.linalg.norm(normals_tmj, axis=1, keepdims=True) + 1e-8)
    
    light_dir = np.array([0.3, 0.5, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shading_tmj = np.abs(np.dot(normals_tmj, light_dir))
    shading_tmj = 0.6 + 0.4 * shading_tmj
    
    face_colors_tmj[:, 0] *= shading_tmj
    face_colors_tmj[:, 1] *= shading_tmj
    face_colors_tmj[:, 2] *= shading_tmj
    face_colors_tmj = np.clip(face_colors_tmj, 0, 1)
    
    mesh_tmj = Poly3DCollection(verts_tmj_plot[faces_tmj], facecolors=face_colors_tmj)
    mesh_tmj.set_edgecolor('none')
    ax.add_collection3d(mesh_tmj)
    
    print(f"TMJ mesh: {len(faces_tmj)} faces")
    
    # Zoom very close to the right TMJ - joint fills the frame
    padding = crop_size * 0.9  # Very tight around the box
    ax.set_xlim(xr - padding, xr + padding)
    ax.set_ylim(yr - padding, yr + padding)
    ax.set_zlim(max_z - zr - padding, max_z - zr + padding)
    
    # Plot only RIGHT TMJ box with bright neon color that pops on dark bg
    plot_cube(ax, (xr, yr, max_z - zr), crop_size, color='#00FF88', linewidth=5)  # Bright neon green
    
    # Add center marker for right TMJ - bright with glow effect
    ax.scatter([xr], [yr], [max_z - zr], color='#00FF88', s=250, marker='o', label='Right TMJ', edgecolors='white', linewidths=2)
    
    # Labels
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    
    # View angle - slightly elevated, azimuth from parameter
    ax.view_init(elev=15, azim=azim)
    
    # Clean look for presentation
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Legend
    ax.legend(loc='upper left', fontsize=11)
    
    # Title
    ax.set_title("3D CBCT with TMJ Detection Boxes", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        print(f"Saving to {output_path}...")
        # Dark gradient background - stylish dark theme
        fig.patch.set_facecolor('#0d1117')  # GitHub dark bg
        ax.set_facecolor('#0d1117')
        
        # Make panes dark/transparent
        ax.xaxis.pane.set_facecolor((0.05, 0.07, 0.1, 0.8))
        ax.yaxis.pane.set_facecolor((0.05, 0.07, 0.1, 0.8))
        ax.zaxis.pane.set_facecolor((0.05, 0.07, 0.1, 0.8))
        
        # Light colored axes/labels for dark bg
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.zaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#f0f6fc')
        
        # Legend with dark style
        legend = ax.legend(loc='upper left', fontsize=11, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
        print(f"✅ Saved: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="3D Visualization of CBCT with TMJ Boxes")
    parser.add_argument('--study', type=str, default='study_0001', help='Study ID')
    parser.add_argument('--dataset', type=str, default='data/dataset', help='Path to dataset')
    parser.add_argument('--summary', type=str, default='data/auto_crops/crop_summary.json', help='Path to crop summary')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--downsample', type=int, default=4, help='Downsample factor (increase for speed, decrease for quality)')
    parser.add_argument('--threshold', type=float, default=None, help='Bone threshold (HU/intensity)')
    
    args = parser.parse_args()
    
    study_id = args.study
    dataset_dir = Path(args.dataset)
    summary_path = Path(args.summary)
    
    # Load summary
    if not summary_path.exists():
        print(f"Error: Summary file {summary_path} not found.")
        return
    
    summary = load_crop_summary(summary_path)
    
    # Get coords
    try:
        coords, crop_size, vol_shape = get_study_coords(summary, study_id)
    except ValueError as e:
        print(e)
        return
        
    coords_left = coords['left']
    coords_right = coords['right']
    
    # Load volume
    study_path = dataset_dir / study_id
    print(f"Loading volume for {study_id}...")
    volume = load_dicom_volume(study_path)
    
    # Output path
    if args.output is None:
        output_path = Path(f"{study_id}_3d_viz.png")
    else:
        output_path = Path(args.output)
        
    # Generate 9 views: full 360° rotation every 40 degrees
    views = [(i * 40, f"{i * 40:03d}deg") for i in range(9)]  # 0, 40, 80, 120, 160, 200, 240, 280, 320
    
    for azim, view_name in views:
        if args.output:
            out_path = Path(str(args.output).replace('.png', f'_{view_name}.png'))
        else:
            out_path = Path(f"{study_id}_3d_{view_name}.png")
        
        print(f"\n--- Generating {view_name} view (azim={azim}) ---")
        visualize_3d(volume, coords_left, coords_right, crop_size, out_path, args.threshold, args.downsample, azim=azim)
    
    print("\n✅ All 5 views generated!")

if __name__ == "__main__":
    main()

