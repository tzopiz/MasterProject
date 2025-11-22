import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy import ndimage
import pydicom
from tqdm import tqdm

def load_volume(path):
    # Reuse logic from analyze_dicom_series
    slices = [pydicom.dcmread(str(p)) for p in Path(path).glob("*.dcm")]
    if not slices: return None
    
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: x.filename)
        
    # Basic HU conversion
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    
    intercept = slices[0].RescaleIntercept if hasattr(slices[0], 'RescaleIntercept') else 0
    slope = slices[0].RescaleSlope if hasattr(slices[0], 'RescaleSlope') else 1
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    
    image += np.int16(intercept)
    return image, slices[0].PixelSpacing, slices[0].SliceThickness

def find_condyles(volume, spacing, debug_plot=True):
    """
    Heuristic to find Left and Right condyles.
    Assumes standard orientation: Z (depth), Y (height/front-back), X (width/left-right)
    """
    z, y, x = volume.shape
    
    # 1. Region of Interest (ROI) Definition
    # Condyles are typically in the upper half of the scan (if neck included) or middle
    # and on the sides.
    
    # Heuristic: 
    # X: Outer 25% on each side
    # Y: Middle-to-Back third (roughly)
    # Z: Middle third (roughly)
    
    mid_x = x // 2
    
    # Parameters for search
    bone_threshold = 400  # HU for dense bone
    
    def find_peak_in_roi(roi_volume, offset_z, offset_y, offset_x):
        # Thresholding
        binary = roi_volume > bone_threshold
        
        # Erosion to separate connected thin bones
        binary = ndimage.binary_erosion(binary, iterations=2)
        
        # Label components
        labeled, num_features = ndimage.label(binary)
        
        if num_features == 0:
            return None
            
        # Find component with highest max intensity in original volume (densest bone)
        # Or simply largest volume
        
        # Let's try finding the "highest" point (condyle head is usually superior)
        # But highest point in ROI might be skull base.
        # Condyle is distinct because it's surrounded by space (joint space).
        
        # Simple approach: Center of Mass of the largest dense object
        sizes = ndimage.sum(binary, labeled, range(num_features + 1))
        largest_label = sizes[1:].argmax() + 1
        
        center = ndimage.center_of_mass(binary, labeled, largest_label)
        
        # Add offsets back
        global_center = (
            int(center[0] + offset_z),
            int(center[1] + offset_y),
            int(center[2] + offset_x)
        )
        return global_center

    # Right ROI (Patient's Right is usually Left side of image in DICOM, but let's check limits)
    # Let's assume X=0 is Right side of PATIENT (standard medical imaging view from feet up)
    # But in numpy array: 0 is left, X_max is right.
    
    # Let's search LEFT side of the array (0 to mid_x)
    # ROI limits:
    x_start_L = 0
    x_end_L = x // 4  # Outer quarter
    
    # Y limits (Front-Back): Condyles are somewhat posterior
    # Assuming Y=0 is front (or back? need to check). Usually Y increases towards back in many recons,
    # but in DICOM usually Y=0 is Top (Coronal) or Front.
    # Let's look at the whole Y range but mask out the very front (teeth) and very back (spine).
    y_start = y // 3
    y_end = 2 * y // 3
    
    # Z limits: Skip very top (skull cap) and bottom (jaw/neck)
    z_start = z // 3
    z_end = 2 * z // 3
    
    print(f"Searching Left ROI: X[{x_start_L}:{x_end_L}], Y[{y_start}:{y_end}], Z[{z_start}:{z_end}]")
    roi_L = volume[z_start:z_end, y_start:y_end, x_start_L:x_end_L]
    center_L = find_peak_in_roi(roi_L, z_start, y_start, x_start_L)
    
    # Search RIGHT side of the array (mid to end)
    x_start_R = x - (x // 4)
    x_end_R = x
    
    print(f"Searching Right ROI: X[{x_start_R}:{x_end_R}], Y[{y_start}:{y_end}], Z[{z_start}:{z_end}]")
    roi_R = volume[z_start:z_end, y_start:y_end, x_start_R:x_end_R]
    center_R = find_peak_in_roi(roi_R, z_start, y_start, x_start_R)
    
    return center_L, center_R

def save_visualization(volume, center, name, output_dir, crop_size=128):
    if center is None:
        print(f"⚠️ No center found for {name}")
        return

    cz, cy, cx = center
    r = crop_size // 2
    
    # Visualization: Draw box on full slice
    import matplotlib.patches as patches
    
    def draw_box(ax, center_x, center_y, color='red'):
        # Create a Rectangle patch
        # xy is bottom left corner
        rect = patches.Rectangle((center_x - r, center_y - r), crop_size, crop_size, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.plot(center_x, center_y, 'x', color=color)

    # 1. Axial (Top-down) - Show full slice at Z=cz
    plt.figure(figsize=(10, 10))
    plt.imshow(volume[cz, :, :], cmap='gray')
    draw_box(plt.gca(), cx, cy) # x is x-axis, y is y-axis (which is volume y dim)
    plt.title(f"{name} Condyle - Axial View (Slice {cz})")
    plt.savefig(f"{output_dir}/viz_{name}_axial.png")
    plt.close()
    
    # 2. Sagittal (Side) - Show full slice at X=cx
    plt.figure(figsize=(10, 10))
    # In sagittal view (slice along X), the axes are: Y (horizontal in array, but usually vertical in plot) vs Z (vertical in array)
    # Wait, volume[z, y, x].
    # Sagittal slice: volume[:, :, cx] -> Shape (Z, Y)
    # Usually plotted as imshow(slice), where dim 0 is Y-axis (vertical), dim 1 is X-axis (horizontal)
    # So plotting volume[:, :, cx] puts Z on Y-axis, Y on X-axis. 
    # DICOM coords: Z is Head-Feet. Y is Front-Back.
    plt.imshow(volume[:, :, cx], cmap='gray') 
    # Center is (cz, cy).
    # On plot: x-axis is Y-dim (cy), y-axis is Z-dim (cz).
    draw_box(plt.gca(), cy, cz)
    plt.title(f"{name} Condyle - Sagittal View (Slice {cx})")
    plt.savefig(f"{output_dir}/viz_{name}_sagittal.png")
    plt.close()
    
    # 3. Coronal (Front) - Show full slice at Y=cy
    plt.figure(figsize=(10, 10))
    # Coronal slice: volume[:, cy, :] -> Shape (Z, X)
    plt.imshow(volume[:, cy, :], cmap='gray')
    # Center is (cz, cx).
    # On plot: x-axis is X-dim (cx), y-axis is Z-dim (cz).
    draw_box(plt.gca(), cx, cz)
    plt.title(f"{name} Condyle - Coronal View (Slice {cy})")
    plt.savefig(f"{output_dir}/viz_{name}_coronal.png")
    plt.close()

    print(f"✅ Saved full visualizations for {name} at {center}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to DICOM series")
    args = parser.parse_args()
    
    print(f"Loading volume from {args.path}...")
    volume, spacing, thickness = load_volume(args.path)
    if volume is None: sys.exit(1)
    
    print(f"Volume loaded: {volume.shape}")
    
    left_center, right_center = find_condyles(volume, spacing)
    
    print(f"Found Left Center: {left_center}")
    print(f"Found Right Center: {right_center}")
    
    save_visualization(volume, left_center, "Left", ".")
    save_visualization(volume, right_center, "Right", ".")

