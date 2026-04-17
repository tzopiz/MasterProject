"""
Annotation Manager Service
Manages saving and loading of TMJ joint classification annotations
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default location for annotations file
DEFAULT_ANNOTATIONS_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "tmj_classifications.json"
)


class AnnotationManager:
    """Service for managing TMJ classification annotations"""

    def __init__(self, annotations_path: Optional[Path] = None):
        self.annotations_path = annotations_path or DEFAULT_ANNOTATIONS_PATH
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file if it doesn't exist
        if not self.annotations_path.exists():
            self._initialize_annotations_file()

    def _initialize_annotations_file(self):
        """Create initial annotations file with default structure"""
        initial_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "available_tags": ["normal", "osteoarthritis", "dislocation", "ankylosis"],
            "annotations": [],
        }

        with open(self.annotations_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Initialized annotations file: {self.annotations_path}")

    def load_annotations(self) -> Dict:
        """
        Load all annotations from file

        Returns:
            Dictionary with annotations and metadata
        """
        try:
            if not self.annotations_path.exists():
                self._initialize_annotations_file()

            with open(self.annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data

        except Exception as e:
            logger.error(f"Error loading annotations: {e}", exc_info=True)
            return {"version": "1.0", "available_tags": [], "annotations": []}

    def save_annotation(
        self,
        patient_id: str,
        study_id: str,
        study_path: str,
        left_joint_tag: str,
        right_joint_tag: str,
    ) -> bool:
        """
        Save or update annotation for a study

        Args:
            patient_id: Patient ID
            study_id: Study ID
            study_path: Path to study
            left_joint_tag: Tag for left joint
            right_joint_tag: Tag for right joint

        Returns:
            True if successful
        """
        try:
            # Load existing annotations
            data = self.load_annotations()

            # Check if annotation already exists
            existing_idx = None
            for idx, ann in enumerate(data["annotations"]):
                if ann["patient_id"] == patient_id and ann["study_id"] == study_id:
                    existing_idx = idx
                    break

            # Create annotation
            annotation = {
                "patient_id": patient_id,
                "study_id": study_id,
                "study_path": study_path,
                "left_joint_tag": left_joint_tag,
                "right_joint_tag": right_joint_tag,
                "annotated_at": datetime.now().isoformat(),
                "annotated_by": "user",
            }

            # Update or append
            if existing_idx is not None:
                data["annotations"][existing_idx] = annotation
                logger.info(f"Updated annotation for {study_id}")
            else:
                data["annotations"].append(annotation)
                logger.info(f"Added new annotation for {study_id}")

            # Save to file
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Error saving annotation: {e}", exc_info=True)
            return False

    def get_annotation(self, patient_id: str, study_id: str) -> Optional[Dict]:
        """Get annotation for a specific study"""
        try:
            data = self.load_annotations()

            for ann in data["annotations"]:
                if ann["patient_id"] == patient_id and ann["study_id"] == study_id:
                    return ann

            return None

        except Exception as e:
            logger.error(f"Error getting annotation: {e}", exc_info=True)
            return None

    def add_tag(self, tag_name: str) -> bool:
        """
        Add a new tag to available tags

        Args:
            tag_name: Name of the tag to add

        Returns:
            True if successful
        """
        try:
            data = self.load_annotations()

            # Check if tag already exists
            if tag_name in data.get("available_tags", []):
                logger.warning(f"Tag '{tag_name}' already exists")
                return True

            # Add tag
            if "available_tags" not in data:
                data["available_tags"] = []

            data["available_tags"].append(tag_name)

            # Save to file
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Added new tag: {tag_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding tag: {e}", exc_info=True)
            return False

    def get_tags(self) -> List[str]:
        """Get list of available tags"""
        try:
            data = self.load_annotations()
            return data.get("available_tags", [])

        except Exception as e:
            logger.error(f"Error getting tags: {e}", exc_info=True)
            return []

    def get_statistics(self) -> Dict:
        """Get annotation statistics"""
        try:
            data = self.load_annotations()
            annotations = data.get("annotations", [])

            # Count by tags
            left_tags_count = {}
            right_tags_count = {}

            for ann in annotations:
                left_tag = ann.get("left_joint_tag")
                right_tag = ann.get("right_joint_tag")

                if left_tag:
                    left_tags_count[left_tag] = left_tags_count.get(left_tag, 0) + 1

                if right_tag:
                    right_tags_count[right_tag] = right_tags_count.get(right_tag, 0) + 1

            return {
                "total_annotations": len(annotations),
                "left_tags_distribution": left_tags_count,
                "right_tags_distribution": right_tags_count,
                "available_tags": data.get("available_tags", []),
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}", exc_info=True)
            return {
                "total_annotations": 0,
                "left_tags_distribution": {},
                "right_tags_distribution": {},
                "available_tags": [],
            }
