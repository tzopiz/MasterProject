#!/usr/bin/env python3
"""
Dataset Organizer for TMJ CBCT Data

This script:
1. Scans all patient directories in data/
2. Finds all DICOM series
3. Removes unnecessary files (exe, dll, jpg, db, bin, etc.) if --clean
4. Organizes all series into a clean dataset structure with sequential naming
5. Creates a manifest file with mapping
6. With --anonymize: strips PHI in DICOM (see dicom_phi_strip.py), writes a public
   manifest without patient names/paths and manifest_private.json for local linkage.

Usage:
    python tools/organize_dataset.py --input data/ --output data/dataset --clean
    python tools/organize_dataset.py --input data/cbct_public_extracted --output data/dataset_cbct_public --anonymize
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Files/extensions to remove
JUNK_FILES = [
    'Runthis.exe',
    'Vision.exe',
    'VisionRUS.dll',
    'Autorun.inf',
    'license.txt',
    '.DS_Store',
]

JUNK_EXTENSIONS = [
    '.exe',
    '.dll',
    '.inf',
    '.db',
    '.jpg',
    '.jpeg',
    '.png',
    '.bin',  # Remove if not DICOM
]


class DatasetOrganizer:
    """Organize CBCT dataset into clean structure"""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        clean_input: bool = False,
        anonymize: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.clean_input = clean_input
        self.anonymize = anonymize
        
        # Statistics
        self.stats = {
            'patients_found': 0,
            'series_found': 0,
            'files_removed': 0,
            'files_copied': 0,
            'errors': 0
        }
        
        # Manifest: mapping from study_id to original path
        self.manifest = []
    
    def find_dicom_series(self) -> List[Tuple[str, Path]]:
        """
        Find all DICOM series in input directory
        
        Returns:
            List of tuples: (patient_name, series_path)
        """
        series_list = []
        
        logger.info(f"Scanning {self.input_dir} for DICOM series...")
        
        # Iterate through patient directories
        for patient_dir in self.input_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            
            # Skip special directories
            if patient_dir.name in ['processed_crops', 'roi_annotations', 'dataset', 'dummy']:
                continue
            
            patient_name = patient_dir.name
            self.stats['patients_found'] += 1
            
            logger.info(f"  Patient: {patient_name}")
            
            # Find patient ID directories (e.g., 00000172, 11844)
            for patient_id_dir in patient_dir.iterdir():
                if not patient_id_dir.is_dir():
                    continue
                
                # Find date directories (e.g., 20190312)
                for date_dir in patient_id_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    
                    # Find DICOM series directories (e.g., 2.16.840.../*.dcm)
                    # Либо *.dcm прямо в series_dir, либо ещё один уровень UID (типично для CBCT с Диска)
                    for series_dir in date_dir.iterdir():
                        if not series_dir.is_dir():
                            continue

                        dcm_files = list(series_dir.glob("*.dcm"))
                        if dcm_files:
                            series_list.append((patient_name, series_dir))
                            self.stats["series_found"] += 1
                            logger.info(
                                f"    Found series: {series_dir.name[:30]}... ({len(dcm_files)} DICOM files)"
                            )
                            continue

                        for uid_dir in series_dir.iterdir():
                            if not uid_dir.is_dir():
                                continue
                            dcm_files = list(uid_dir.glob("*.dcm"))
                            if dcm_files:
                                series_list.append((patient_name, uid_dir))
                                self.stats["series_found"] += 1
                                logger.info(
                                    f"    Found series: {uid_dir.name[:30]}... ({len(dcm_files)} DICOM files)"
                                )
        
        logger.info(f"\nTotal: {self.stats['patients_found']} patients, {self.stats['series_found']} series")
        return series_list
    
    def clean_directory(self, directory: Path) -> int:
        """
        Remove junk files from directory
        
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        for item in directory.rglob('*'):
            if not item.is_file():
                continue
            
            # Check if junk file
            should_remove = False
            
            if item.name in JUNK_FILES:
                should_remove = True
            
            if item.suffix.lower() in JUNK_EXTENSIONS:
                # Don't remove if it's actually a DICOM file misnamed as .bin
                if item.suffix == '.bin':
                    # Skip for now, handle carefully
                    pass
                else:
                    should_remove = True
            
            if should_remove:
                try:
                    item.unlink()
                    removed_count += 1
                    logger.debug(f"    Removed: {item.name}")
                except Exception as e:
                    logger.warning(f"    Failed to remove {item.name}: {e}")
        
        return removed_count
    
    def organize_dataset(self, series_list: List[Tuple[str, Path]]) -> Dict:
        """
        Organize series into clean dataset structure
        
        Structure:
            dataset/
                study_0001/
                    *.dcm
                study_0002/
                    *.dcm
                ...
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nOrganizing dataset into {self.output_dir}")

        if self.anonymize:
            from pydicom.uid import generate_uid

            from dicom_phi_strip import write_anonymized_dicom

        for idx, (patient_name, series_path) in enumerate(series_list, start=1):
            study_id = f"study_{idx:04d}"
            subject_id = f"sub_{idx:04d}"
            output_path = self.output_dir / study_id
            
            try:
                # Create output directory
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Copy DICOM files
                dcm_files = list(series_path.glob('*.dcm'))
                
                mode = "anonymized" if self.anonymize else "copy"
                logger.info(
                    f"  [{idx}/{len(series_list)}] {study_id} ({subject_id}, {mode}) <- {patient_name} ({len(dcm_files)} files)"
                )

                study_uid = series_uid = frame_uid = ""
                if self.anonymize:
                    study_uid = generate_uid()
                    series_uid = generate_uid()
                    frame_uid = generate_uid()

                for dcm_file in dcm_files:
                    dest_file = output_path / dcm_file.name
                    if self.anonymize:
                        write_anonymized_dicom(
                            dcm_file,
                            dest_file,
                            study_id=study_id,
                            study_instance_uid=study_uid,
                            series_instance_uid=series_uid,
                            frame_of_reference_uid=frame_uid,
                        )
                    else:
                        shutil.copy2(dcm_file, dest_file)
                    self.stats['files_copied'] += 1
                
                # Add to manifest (full row; public export may omit PHI fields)
                self.manifest.append({
                    'study_id': study_id,
                    'subject_id': subject_id,
                    'patient_name': patient_name,
                    'original_path': str(series_path.relative_to(self.input_dir)),
                    'num_files': len(dcm_files),
                    'organized_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"  Error processing {series_path}: {e}")
                self.stats['errors'] += 1
        
        return self.manifest
    
    def save_manifest(self):
        """Save manifest to JSON file (and manifest_private.json if --anonymize)."""
        manifest_file = self.output_dir / 'manifest.json'

        if self.anonymize:
            public_studies = [
                {
                    'study_id': row['study_id'],
                    'subject_id': row['subject_id'],
                    'num_files': row['num_files'],
                    'organized_at': row['organized_at'],
                }
                for row in self.manifest
            ]
            manifest_data = {
                'anonymized': True,
                'created_at': datetime.now().isoformat(),
                'output_dir': str(self.output_dir),
                'note': 'DICOM PHI stripped; no patient names or paths in this file.',
                'statistics': self.stats,
                'studies': public_studies,
            }
            private_file = self.output_dir / 'manifest_private.json'
            private_data = {
                'warning': 'Содержит ФИО и пути — не публиковать и не коммитить.',
                'created_at': datetime.now().isoformat(),
                'input_dir': str(self.input_dir),
                'studies': self.manifest,
            }
            with open(private_file, 'w', encoding='utf-8') as f:
                json.dump(private_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Private linkage manifest (PHI): {private_file}")
        else:
            manifest_data = {
                'created_at': datetime.now().isoformat(),
                'input_dir': str(self.input_dir),
                'output_dir': str(self.output_dir),
                'statistics': self.stats,
                'studies': self.manifest,
            }

        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ Manifest saved to {manifest_file}")
    
    def create_annotation_script(self):
        """Create a batch annotation script"""
        script_path = self.output_dir / 'annotate_all.sh'
        
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('# Batch ROI Annotation Script\n')
            f.write('# Auto-generated by organize_dataset.py\n\n')
            f.write('set -e\n\n')
            f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
            f.write('MLSERVICE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"\n')
            f.write('OUTPUT_DIR="$MLSERVICE_DIR/data/roi_annotations"\n\n')
            f.write('# Activate virtual environment\n')
            f.write('if [ -f "$MLSERVICE_DIR/venv/bin/activate" ]; then\n')
            f.write('    source "$MLSERVICE_DIR/venv/bin/activate"\n')
            f.write('fi\n\n')
            f.write('echo "Starting batch ROI annotation..."\n')
            f.write('echo "Output directory: $OUTPUT_DIR"\n')
            f.write('echo ""\n\n')
            
            for idx, item in enumerate(self.manifest, start=1):
                study_id = item['study_id']
                who = item["subject_id"] if self.anonymize else item["patient_name"]
                f.write(f'# {study_id} ({who})\n')
                f.write(f'echo "=== [{idx}/{len(self.manifest)}] Processing {study_id} ==="\n')
                f.write(f'python3 "$MLSERVICE_DIR/tools/roi_annotation_tool.py" \\\n')
                f.write(f'    "$SCRIPT_DIR/{study_id}" \\\n')
                f.write(f'    --output "$OUTPUT_DIR"\n\n')
                f.write(f'if [ $? -ne 0 ]; then\n')
                f.write(f'    echo "❌ Failed to annotate {study_id}"\n')
                f.write(f'    exit 1\n')
                f.write(f'fi\n\n')
            
            f.write('echo ""\n')
            f.write('echo "✅ All studies annotated!"\n')
            f.write('echo "Annotations saved to: $OUTPUT_DIR"\n')
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"✅ Annotation script created: {script_path}")
    
    def print_summary(self):
        """Print summary statistics"""
        logger.info("\n" + "="*70)
        logger.info("DATASET ORGANIZATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Patients found:    {self.stats['patients_found']}")
        logger.info(f"Series found:      {self.stats['series_found']}")
        logger.info(f"Files copied:      {self.stats['files_copied']}")
        logger.info(f"Files removed:     {self.stats['files_removed']}")
        logger.info(f"Errors:            {self.stats['errors']}")
        logger.info("="*70)
        logger.info(f"Output directory:  {self.output_dir}")
        logger.info(f"Manifest:          {self.output_dir / 'manifest.json'}")
        if self.anonymize:
            logger.info(f"Private manifest:  {self.output_dir / 'manifest_private.json'}")
        logger.info(f"Annotation script: {self.output_dir / 'annotate_all.sh'}")
        logger.info("="*70 + "\n")
    
    def run(self) -> bool:
        """Run the full organization process. Returns False if nothing to organize."""
        logger.info("="*70)
        logger.info("TMJ CBCT DATASET ORGANIZER")
        logger.info("="*70 + "\n")
        
        # Step 1: Find all DICOM series
        series_list = self.find_dicom_series()
        
        if not series_list:
            logger.error("No DICOM series found!")
            return False
        
        # Step 2: Clean input directory (if requested)
        if self.clean_input:
            logger.info("\nCleaning input directory...")
            removed = self.clean_directory(self.input_dir)
            self.stats['files_removed'] = removed
            logger.info(f"✅ Removed {removed} junk files")
        
        # Step 3: Organize into clean structure
        self.organize_dataset(series_list)
        
        # Step 4: Save manifest
        self.save_manifest()
        
        # Step 5: Create annotation script
        self.create_annotation_script()
        
        # Step 6: Print summary
        self.print_summary()
        
        # Step 7: Usage instructions
        logger.info("NEXT STEPS:")
        logger.info("-" * 70)
        logger.info("1. Review the manifest:")
        logger.info(f"   cat {self.output_dir / 'manifest.json'}")
        logger.info("")
        logger.info("2. Start batch annotation:")
        logger.info(f"   cd {self.output_dir}")
        logger.info("   ./annotate_all.sh")
        logger.info("")
        logger.info("3. Or annotate manually one by one:")
        logger.info("   python tools/roi_annotation_tool.py \\")
        logger.info(f"       {self.output_dir / 'study_0001'} \\")
        logger.info("       --output data/roi_annotations")
        logger.info("-" * 70 + "\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Organize TMJ CBCT dataset into clean structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data',
        help='Input directory containing patient data (default: data/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/dataset',
        help='Output directory for organized dataset (default: data/dataset/)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Remove junk files from input directory (exe, dll, jpg, etc.)'
    )
    parser.add_argument(
        '--anonymize',
        action='store_true',
        help='Strip PHI in DICOM; public manifest without names/paths; writes manifest_private.json',
    )

    args = parser.parse_args()
    
    # Check input directory exists
    if not Path(args.input).exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)
    
    # Run organizer
    organizer = DatasetOrganizer(args.input, args.output, args.clean, anonymize=args.anonymize)
    if not organizer.run():
        sys.exit(1)


if __name__ == '__main__':
    main()

