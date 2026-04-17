#!/usr/bin/env python3
"""
Extract Crops from Annotated ROIs

Uses manual ROI annotations to extract 3D crops around TMJ for 3D segmentation training.

Usage:
    python tools/extract_crops_from_annotations.py \\
        --annotations data/roi_annotations \\
        --dataset data/dataset \\
        --output data/processed_crops \\
        --crop_size 128
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pydicom

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CropExtractor:
    """Extract 3D crops from DICOM based on ROI annotations"""

    def __init__(
        self, annotations_dir: str, dataset_dir: str, output_dir: str, crop_size: int = 128
    ):
        self.annotations_dir = Path(annotations_dir)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {"annotations_found": 0, "crops_extracted": 0, "errors": 0}

    def load_dicom_volume(self, dicom_dir: Path) -> np.ndarray:
        """Load DICOM series into 3D numpy array"""
        logger.info(f"Loading DICOM from {dicom_dir.name}...")

        # Find all DICOM files
        dicom_files = list(dicom_dir.glob("*.dcm"))

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")

        # Read and sort slices
        slices = []
        for dcm_file in dicom_files:
            ds = pydicom.dcmread(dcm_file)
            slices.append(ds)

        # Sort by Instance Number
        slices.sort(key=lambda x: int(x.InstanceNumber))

        # Stack into 3D volume
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

        logger.info(f"  Loaded volume: {volume.shape}")
        return volume

    def extract_crop(
        self, volume: np.ndarray, center: List[int], crop_size: int
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract a 3D crop around the center point

        Args:
            volume: 3D numpy array (Z, Y, X)
            center: [z, y, x] coordinates
            crop_size: size of the crop (cubic)

        Returns:
            crop: 3D numpy array of shape (crop_size, crop_size, crop_size)
            actual_center: adjusted center coordinates
        """
        z, y, x = center
        half_size = crop_size // 2

        # Calculate crop boundaries
        z_start = max(0, z - half_size)
        z_end = min(volume.shape[0], z + half_size)
        y_start = max(0, y - half_size)
        y_end = min(volume.shape[1], y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(volume.shape[2], x + half_size)

        # Extract crop
        crop = volume[z_start:z_end, y_start:y_end, x_start:x_end]

        # Pad if necessary (crop at boundaries)
        if crop.shape != (crop_size, crop_size, crop_size):
            padded_crop = np.zeros((crop_size, crop_size, crop_size), dtype=crop.dtype)

            # Calculate padding
            z_pad = (crop_size - crop.shape[0]) // 2
            y_pad = (crop_size - crop.shape[1]) // 2
            x_pad = (crop_size - crop.shape[2]) // 2

            # Place crop in center
            padded_crop[
                z_pad : z_pad + crop.shape[0],
                y_pad : y_pad + crop.shape[1],
                x_pad : x_pad + crop.shape[2],
            ] = crop

            crop = padded_crop
            logger.warning(
                f"  Padded crop from {(z_end - z_start, y_end - y_start, x_end - x_start)} to {crop.shape}"
            )

        actual_center = [z, y, x]
        return crop, actual_center

    def save_crop_as_nifti(self, crop: np.ndarray, output_path: Path, metadata: Dict = None):
        """Save crop as NIfTI file"""
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(crop, np.eye(4))

        # Save
        nib.save(nifti_img, output_path)
        logger.info(f"  Saved: {output_path.name}")

        # Save metadata as JSON
        if metadata:
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    def process_annotation(self, annotation_file: Path):
        """Process a single annotation and extract crops"""
        try:
            # Load annotation
            with open(annotation_file, "r") as f:
                annotation = json.load(f)

            study_id = annotation["scan_id"]
            dicom_dir = self.dataset_dir / study_id

            if not dicom_dir.exists():
                logger.error(f"DICOM directory not found: {dicom_dir}")
                self.stats["errors"] += 1
                return

            logger.info(f"\n{'=' * 70}")
            logger.info(f"Processing: {study_id}")
            logger.info(f"{'=' * 70}")

            # Load DICOM volume
            volume = self.load_dicom_volume(dicom_dir)

            # Extract LEFT crop
            left_center = annotation["left_tmj"]["center"]
            logger.info(f"Extracting LEFT crop (center: {left_center})...")
            left_crop, left_actual_center = self.extract_crop(volume, left_center, self.crop_size)

            left_output = self.output_dir / f"{study_id}_left.nii.gz"
            left_metadata = {
                "study_id": study_id,
                "side": "left",
                "center": left_actual_center,
                "crop_size": self.crop_size,
                "original_shape": list(volume.shape),
                "annotation_file": str(annotation_file.name),
            }
            self.save_crop_as_nifti(left_crop, left_output, left_metadata)
            self.stats["crops_extracted"] += 1

            # Extract RIGHT crop
            right_center = annotation["right_tmj"]["center"]
            logger.info(f"Extracting RIGHT crop (center: {right_center})...")
            right_crop, right_actual_center = self.extract_crop(
                volume, right_center, self.crop_size
            )

            right_output = self.output_dir / f"{study_id}_right.nii.gz"
            right_metadata = {
                "study_id": study_id,
                "side": "right",
                "center": right_actual_center,
                "crop_size": self.crop_size,
                "original_shape": list(volume.shape),
                "annotation_file": str(annotation_file.name),
            }
            self.save_crop_as_nifti(right_crop, right_output, right_metadata)
            self.stats["crops_extracted"] += 1

            logger.info(f"✅ Successfully extracted 2 crops from {study_id}")

        except Exception as e:
            logger.error(f"Error processing {annotation_file.name}: {e}")
            self.stats["errors"] += 1

    def run(self):
        """Process all annotations"""
        logger.info("=" * 70)
        logger.info("TMJ CROP EXTRACTOR")
        logger.info("=" * 70)
        logger.info(f"Annotations: {self.annotations_dir}")
        logger.info(f"Dataset: {self.dataset_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Crop size: {self.crop_size}³")
        logger.info("=" * 70 + "\n")

        # Find all annotation files
        annotation_files = sorted(list(self.annotations_dir.glob("*_rois.json")))

        if not annotation_files:
            logger.error(f"No annotation files found in {self.annotations_dir}")
            return

        self.stats["annotations_found"] = len(annotation_files)
        logger.info(f"Found {len(annotation_files)} annotations\n")

        # Process each annotation
        for annotation_file in annotation_files:
            self.process_annotation(annotation_file)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Annotations processed: {self.stats['annotations_found']}")
        logger.info(f"Crops extracted:       {self.stats['crops_extracted']}")
        logger.info(f"Errors:                {self.stats['errors']}")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 70 + "\n")

        if self.stats["crops_extracted"] > 0:
            logger.info("NEXT STEPS:")
            logger.info("-" * 70)
            logger.info("1. Open crops in ITK-SNAP to annotate 3D masks:")
            logger.info(f"   open -a ITK-SNAP {self.output_dir}/study_0001_left.nii.gz")
            logger.info("")
            logger.info("2. For each crop:")
            logger.info("   - Segmentation → Add Label → 'TMJ'")
            logger.info("   - Paint the condyle and fossa (3D brush)")
            logger.info("   - Save as: *_mask.nii.gz")
            logger.info("   - Example: study_0001_left_mask.nii.gz")
            logger.info("")
            logger.info("3. After annotating 10-20 crops:")
            logger.info("   python train_3d.py --data_dir data/processed_crops")
            logger.info("-" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D crops from ROI annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/roi_annotations",
        help="Directory with ROI annotation JSON files (default: data/roi_annotations)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset",
        help="Directory with organized DICOM dataset (default: data/dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_crops",
        help="Output directory for crops (default: data/processed_crops)",
    )
    parser.add_argument(
        "--crop_size", type=int, default=128, help="Size of cubic crop (default: 128)"
    )

    args = parser.parse_args()

    # Check inputs exist
    if not Path(args.annotations).exists():
        logger.error(f"Annotations directory not found: {args.annotations}")
        sys.exit(1)

    if not Path(args.dataset).exists():
        logger.error(f"Dataset directory not found: {args.dataset}")
        sys.exit(1)

    # Run extractor
    extractor = CropExtractor(args.annotations, args.dataset, args.output, args.crop_size)
    extractor.run()


if __name__ == "__main__":
    main()
