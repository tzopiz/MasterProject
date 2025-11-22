import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import nibabel as nib
from pathlib import Path

# Import helpers from analyze_dicom_series
try:
    from analyze_dicom_series import load_scan, get_pixels_hu
except ImportError:
    # Fallback if import fails (e.g. path issues)
    import pydicom
    def load_scan(path):
        slices = [pydicom.dcmread(str(p)) for p in Path(path).glob("*.dcm")]
        if not slices: return None
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: x.filename)
        return slices

    def get_pixels_hu(slices):
        image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
        intercept = slices[0].RescaleIntercept if hasattr(slices[0], 'RescaleIntercept') else 0
        slope = slices[0].RescaleSlope if hasattr(slices[0], 'RescaleSlope') else 1
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

class TMJCropTool:
    def __init__(self, volume, spacing, output_dir, scan_name, crop_size=128):
        self.volume = volume
        self.spacing = spacing
        self.output_dir = Path(output_dir)
        self.scan_name = scan_name
        self.crop_size = crop_size
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.z_idx = volume.shape[0] // 2
        self.selected_points = {'Left': None, 'Right': None} # (z, y, x)
        self.active_side = None # 'Left' or 'Right'
        
        self.setup_gui()

    def setup_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.2)
        
        self.im_plot = self.ax.imshow(self.volume[self.z_idx], cmap='gray', vmin=-400, vmax=1000)
        self.ax.set_title(f"Scan: {self.scan_name} | Slice: {self.z_idx}")
        self.marker_plot, = self.ax.plot([], [], 'rx', markersize=12)
        self.rect_patches = []

        # Slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.volume.shape[0]-1, valinit=self.z_idx, valstep=1)
        self.slider.on_changed(self.update_slice)
        
        # Buttons
        ax_btn_l = plt.axes([0.1, 0.025, 0.15, 0.04])
        self.btn_l = Button(ax_btn_l, 'Set Left TMJ')
        self.btn_l.on_clicked(lambda x: self.set_active('Left'))
        
        ax_btn_r = plt.axes([0.3, 0.025, 0.15, 0.04])
        self.btn_r = Button(ax_btn_r, 'Set Right TMJ')
        self.btn_r.on_clicked(lambda x: self.set_active('Right'))
        
        ax_btn_save = plt.axes([0.7, 0.025, 0.15, 0.04])
        self.btn_save = Button(ax_btn_save, 'Save Crops')
        self.btn_save.on_clicked(self.save_crops)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print(f"Instructions:")
        print(f"1. Scroll to find TMJ (Up/Down keys or Slider)")
        print(f"2. Click 'Set Left TMJ' then click on the Left Condyle center")
        print(f"3. Click 'Set Right TMJ' then click on the Right Condyle center")
        print(f"4. Click 'Save Crops'")
        
        plt.show()

    def update_slice(self, val):
        self.z_idx = int(val)
        self.im_plot.set_data(self.volume[self.z_idx])
        self.ax.set_title(f"Scan: {self.scan_name} | Slice: {self.z_idx}")
        self.redraw_markers()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'up' or event.key == 'right':
            if self.z_idx < self.volume.shape[0] - 1:
                self.slider.set_val(self.z_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            if self.z_idx > 0:
                self.slider.set_val(self.z_idx - 1)

    def set_active(self, side):
        self.active_side = side
        print(f"👉 Click on the {side} TMJ center...")
        self.ax.set_title(f"Click on center of {side} TMJ")
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        if self.active_side is None: return
        
        x, y = int(event.xdata), int(event.ydata)
        z = self.z_idx
        
        print(f"📍 Selected {self.active_side}: ({z}, {y}, {x})")
        self.selected_points[self.active_side] = (z, y, x)
        self.active_side = None # Reset after selection
        self.redraw_markers()

    def redraw_markers(self):
        # Clear old patches
        [p.remove() for p in self.rect_patches]
        self.rect_patches = []
        
        xs, ys = [], []
        
        import matplotlib.patches as patches
        
        for side, point in self.selected_points.items():
            if point is None: continue
            pz, py, px = point
            
            # Show marker if within range of current slice? 
            # Or always show projected marker?
            # Let's show if close (within +/- crop_size/2)
            if abs(pz - self.z_idx) < self.crop_size // 2:
                xs.append(px)
                ys.append(py)
                
                # Draw box
                rect = patches.Rectangle((px - self.crop_size//2, py - self.crop_size//2), 
                                       self.crop_size, self.crop_size, 
                                       linewidth=1, edgecolor='r' if side=='Right' else 'b', facecolor='none')
                self.ax.add_patch(rect)
                self.rect_patches.append(rect)
        
        self.marker_plot.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def save_crops(self, event):
        print("\n💾 Saving crops...")
        
        for side, point in self.selected_points.items():
            if point is None:
                print(f"⚠️ Skipping {side} - not selected")
                continue
                
            z, y, x = point
            r = self.crop_size // 2
            
            # Calculate bounds with padding checks
            z_start = max(0, z - r)
            z_end = min(self.volume.shape[0], z + r)
            y_start = max(0, y - r)
            y_end = min(self.volume.shape[1], y + r)
            x_start = max(0, x - r)
            x_end = min(self.volume.shape[2], x + r)
            
            crop_vol = self.volume[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Pad if near borders to ensure fixed size
            pad_z = (r*2) - crop_vol.shape[0]
            pad_y = (r*2) - crop_vol.shape[1]
            pad_x = (r*2) - crop_vol.shape[2]
            
            if pad_z > 0 or pad_y > 0 or pad_x > 0:
                # Simple zero padding at the end
                crop_vol = np.pad(crop_vol, ((0, pad_z), (0, pad_y), (0, pad_x)), 'constant', constant_values=-1000)
            
            # Create NIfTI image
            # We need an affine. We can construct a basic one or use identity if we don't care about world coords yet.
            # Better: preserve spacing.
            affine = np.eye(4)
            affine[0,0] = self.spacing[0] # PixelSpacing X
            affine[1,1] = self.spacing[1] # PixelSpacing Y
            affine[2,2] = self.spacing[2] if len(self.spacing) > 2 else 1.0 # SliceThickness
            
            nifti_img = nib.Nifti1Image(crop_vol, affine)
            
            filename = f"{self.scan_name}_{side.lower()}.nii.gz"
            save_path = self.output_dir / filename
            nib.save(nifti_img, save_path)
            print(f"✅ Saved {save_path}")
            
        print("Done! You can close the window.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-automatic TMJ Cropping Tool")
    parser.add_argument("path", help="Path to DICOM series folder")
    parser.add_argument("--output", default="data/crops", help="Output directory")
    parser.add_argument("--size", type=int, default=128, help="Crop size (cube side)")
    
    args = parser.parse_args()
    
    print(f"📂 Loading DICOM from {args.path}...")
    slices = load_scan(args.path)
    
    if not slices:
        print("❌ No DICOM files found.")
        sys.exit(1)
        
    volume = get_pixels_hu(slices)
    
    # Get spacing
    try:
        spacing = [float(x) for x in slices[0].PixelSpacing] + [float(slices[0].SliceThickness)]
    except:
        spacing = [1.0, 1.0, 1.0]
        print("⚠️ Warning: Could not determine spacing, using 1.0mm")
        
    scan_name = Path(args.path).parent.name + "_" + Path(args.path).name
    # Simplify name if it's a long UID
    if len(scan_name) > 30:
        scan_name = Path(args.path).name[:10]
    
    print(f"Volume loaded: {volume.shape}")
    
    tool = TMJCropTool(volume, spacing, args.output, scan_name, crop_size=args.size)
    tool.start()

