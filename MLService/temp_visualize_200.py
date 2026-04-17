import sys
from pathlib import Path

sys.path.append('tools')
from visualize_detector import *  # noqa: F403

# Load volume
volume = load_dicom_volume(Path('data/dataset/study_0001'))  # noqa: F405

# Load metadata with crop_size=128
meta_path = Path('data/auto_crops/study_0001/study_0001_metadata.json')
with open(meta_path) as f:
    meta = json.load(f)  # noqa: F405

coords_left = meta['predicted_coords']['left']
coords_right = meta['predicted_coords']['right']

# Load manual if exists
manual = load_manual_annotation(Path('data/dataset'), 'study_0001')  # noqa: F405
manual_left = manual['left_tmj']['center'] if manual else None
manual_right = manual['right_tmj']['center'] if manual else None

# Plot with crop_size=200
plot_prediction(volume, coords_left, coords_right, crop_size=200,   # noqa: F405
                output_path='data/auto_crops/study_0001_viz_200.png',
                manual_left=manual_left, manual_right=manual_right)

print("✅ Visualization with 200x200 box created!")
