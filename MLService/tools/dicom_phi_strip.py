#!/usr/bin/env python3
"""Remove/replace PHI in DICOM files for a de-identified export dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pydicom
from pydicom.uid import generate_uid

logger = logging.getLogger(__name__)

# DICOM keywords to drop (PatientID set separately).
_PHI_KEYWORDS = (
    "PatientName",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "EthnicGroup",
    "Occupation",
    "AdditionalPatientHistory",
    "PatientComments",
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
    "StationName",
    "StudyDescription",
    "SeriesDescription",
    "InstitutionalDepartmentName",
    "DeviceSerialNumber",
    "StudyID",
    "IssuerOfPatientID",
    "IssuerOfPatientIDQualifiersSequence",
)


def _del_keyword(ds: pydicom.Dataset, keyword: str) -> None:
    try:
        if keyword in ds:
            del ds[keyword]
    except Exception as exc:
        logger.debug("drop %s: %s", keyword, exc)


def write_anonymized_dicom(
    src: Path,
    dst: Path,
    *,
    study_id: str,
    study_instance_uid: str,
    series_instance_uid: str,
    frame_of_reference_uid: str,
) -> None:
    """
    Read DICOM from src, strip PHI and stable UIDs that could link to source PACS,
    write to dst. New SOPInstanceUID per file; shared study/series/frame UIDs per series folder.
    """
    ds = pydicom.dcmread(str(src), force=True)
    try:
        ds.remove_private_tags()
    except Exception as exc:
        logger.debug("remove_private_tags skipped: %s", exc)

    for kw in _PHI_KEYWORDS:
        _del_keyword(ds, kw)

    ds.PatientID = study_id
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    sop = generate_uid()
    ds.SOPInstanceUID = sop
    if "FrameOfReferenceUID" in ds:
        ds.FrameOfReferenceUID = frame_of_reference_uid

    if ds.file_meta and len(ds.file_meta):
        try:
            ds.file_meta.MediaStorageSOPInstanceUID = sop
        except Exception as exc:
            logger.debug("file_meta MediaStorageSOPInstanceUID: %s", exc)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(dst), write_like_original=False)
