import os
import sys
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

def load_scan(path):
    # Load all DICOM files
    slices = [pydicom.dcmread(str(p)) for p in Path(path).glob("*.dcm")]
    
    if not slices:
        print(f"❌ No DICOM files found in {path}")
        return None
        
    # Sort by ImagePositionPatient Z-coordinate
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        print("⚠️ Warning: No ImagePositionPatient found, sorting by filename")
        slices.sort(key=lambda x: x.filename)
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to Hounsfield Units (HU)
    # Usually RescaleIntercept is -1024 for CT, but dental CBCT varies
    intercept = slices[0].RescaleIntercept if hasattr(slices[0], 'RescaleIntercept') else 0
    slope = slices[0].RescaleSlope if hasattr(slices[0], 'RescaleSlope') else 1
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def analyze_series(series_path, output_preview="preview.png"):
    print(f"📂 Analyzing series: {series_path}")
    
    slices = load_scan(series_path)
    if not slices:
        return

    print(f"   Found {len(slices)} slices.")
    
    # Metadata info
    first = slices[0]
    print(f"   Modality: {first.get('Modality', 'Unknown')}")
    print(f"   Manufacturer: {first.get('Manufacturer', 'Unknown')}")
    
    pixel_spacing = first.get("PixelSpacing", [0, 0])
    slice_thickness = first.get("SliceThickness", 0)
    print(f"   Pixel Spacing: {pixel_spacing} mm")
    print(f"   Slice Thickness: {slice_thickness} mm")
    
    # Volume info
    volume = get_pixels_hu(slices)
    print(f"   Volume Shape: {volume.shape} (Depth, Height, Width)")
    print(f"   Intensity Range: {volume.min()} to {volume.max()} HU")

    # Generate Previews
    print("🖼️  Generating previews...")
    
    # 1. Axial (Top-down) - Middle slice
    axial_idx = volume.shape[0] // 2
    axial = volume[axial_idx, :, :]
    
    # 2. Coronal (Front) - Middle slice
    coronal_idx = volume.shape[1] // 2
    coronal = volume[:, coronal_idx, :]
    # Correct aspect ratio for display
    aspect_coronal = slice_thickness / pixel_spacing[0]

    # 3. Sagittal (Side) - Middle slice
    sagittal_idx = volume.shape[2] // 2
    sagittal = volume[:, :, sagittal_idx]
    aspect_sagittal = slice_thickness / pixel_spacing[1]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(axial, cmap='gray')
    axs[0].set_title(f'Axial (Slice {axial_idx})')
    
    axs[1].imshow(coronal, cmap='gray', aspect=aspect_coronal)
    axs[1].set_title(f'Coronal (Slice {coronal_idx})')
    axs[1].invert_yaxis() # DICOM coords usually need flip for coronal/sagittal
    
    axs[2].imshow(sagittal, cmap='gray', aspect=aspect_sagittal)
    axs[2].set_title(f'Sagittal (Slice {sagittal_idx})')
    axs[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_preview)
    print(f"✅ Preview saved to {output_preview}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to directory containing .dcm files")
    args = parser.parse_args()
    
    analyze_series(args.path)

