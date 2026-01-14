"""
File Cleaner Service
Removes junk files from patient directories
"""

import os
import logging
from pathlib import Path
from typing import Dict, List
import pydicom

logger = logging.getLogger(__name__)

# Files/extensions to remove (from organize_dataset.py)
JUNK_FILES = [
    'Runthis.exe',
    'Vision.exe',
    'VisionRUS.dll',
    'Autorun.inf',
    'license.txt',
    '.DS_Store',
    'Thumbs.db',
]

JUNK_EXTENSIONS = [
    '.exe',
    '.dll',
    '.inf',
    '.db',
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.bmp',
    '.txt',
    '.pdf',
    '.bin',
]


class FileCleaner:
    """Service for cleaning junk files from patient directories"""
    
    def __init__(self):
        self.junk_files = [f.lower() for f in JUNK_FILES]
        self.junk_extensions = [ext.lower() for ext in JUNK_EXTENSIONS]
    
    def is_junk_file(self, filepath: Path) -> bool:
        """Check if file should be removed"""
        filename = filepath.name.lower()
        
        # Check exact filename match
        if filename in self.junk_files:
            return True
        
        # Check extension match
        if filepath.suffix.lower() in self.junk_extensions:
            # Special case: .bin files might be DICOM, check first
            if filepath.suffix.lower() == '.bin':
                try:
                    pydicom.dcmread(str(filepath), stop_before_pixels=True)
                    return False  # It's a DICOM file, keep it
                except:
                    return True  # Not a DICOM, remove it
            return True
        
        return False
    
    def clean_directory(self, root_dir: Path, dry_run: bool = False) -> Dict:
        """
        Clean junk files from directory tree
        
        Args:
            root_dir: Root directory to clean
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'files_scanned': 0,
            'files_removed': 0,
            'files_to_remove': [],
            'errors': 0
        }
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleaning directory: {root_dir}")
        
        # Walk through all files
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                filepath = Path(root) / filename
                stats['files_scanned'] += 1
                
                try:
                    if self.is_junk_file(filepath):
                        stats['files_to_remove'].append(str(filepath))
                        
                        if not dry_run:
                            # Actually remove the file
                            filepath.unlink()
                            logger.info(f"Removed: {filepath}")
                        else:
                            logger.info(f"Would remove: {filepath}")
                        
                        stats['files_removed'] += 1
                
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {e}")
                    stats['errors'] += 1
        
        logger.info(f"Cleaning complete: {stats['files_removed']} files {'would be ' if dry_run else ''}removed")
        return stats
    
    def clean_specific_files(self, file_paths: List[str]) -> Dict:
        """
        Remove specific files
        
        Args:
            file_paths: List of file paths to remove
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'files_removed': 0,
            'errors': 0
        }
        
        for filepath_str in file_paths:
            try:
                filepath = Path(filepath_str)
                if filepath.exists() and filepath.is_file():
                    filepath.unlink()
                    stats['files_removed'] += 1
                    logger.info(f"Removed: {filepath}")
            except Exception as e:
                logger.error(f"Error removing {filepath_str}: {e}")
                stats['errors'] += 1
        
        return stats
