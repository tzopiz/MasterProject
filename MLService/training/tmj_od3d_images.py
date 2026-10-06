"""Native DICOM -> author-ROI slice bags; no assertion of regular 3D geometry."""
from __future__ import annotations
import contextlib
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pydicom
from scipy.ndimage import zoom
from training.tmj_od3d_inputs import parse_annotation, OD3DInputError

SOURCE_DOI = '10.57760/sciencedb.37727'
CODEBOOK_COMMIT = 'a4bce89c89b7175e330193f04a282887b5e3f8b0'
TASK = 'tmj-osseous-author-roi-v1'

class PreparationError(ValueError):
    pass

def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()

def _fail(code):
    raise PreparationError(code)

@contextlib.contextmanager
def _private_decoder():
    log = logging.getLogger('pydicom'); old = log.level
    log.setLevel(logging.CRITICAL+1)
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            yield
        if notices: _fail('unsupported_dicom_encoding')
    finally: log.setLevel(old)

def _inventory(folder):
    files = sorted(folder.glob('*.dcm'))
    if not files or len(files)>4096: _fail('invalid_dicom_count')
    rows, series, sop, instances = [], set(), set(), set()
    source = hashlib.sha256(); total = 0
    with _private_decoder():
        for path in files:
            if path.is_symlink() or not path.is_file(): _fail('unsafe_source_path')
            size = path.stat().st_size; total += size
            if size>8*1024**2 or total>1024**3: _fail('source_byte_limit')
            raw = path.read_bytes()
            if raw[128:132]!=b'DICM': _fail('unsupported_dicom_encoding')
            header = pydicom.dcmread(path, stop_before_pixels=True)
            syntax = str(header.file_meta.TransferSyntaxUID)
            if syntax not in ('1.2.840.10008.1.2','1.2.840.10008.1.2.1','1.2.840.10008.1.2.2'):
                _fail('unsupported_dicom_encoding')
            number = float(header.InstanceNumber)
            if not math.isfinite(number) or not number.is_integer(): _fail('invalid_instance')
            number = int(number)
            if number in instances or str(header.SOPInstanceUID) in sop: _fail('duplicate_dicom')
            instances.add(number); sop.add(str(header.SOPInstanceUID)); series.add(str(header.SeriesInstanceUID))
            if (int(getattr(header,'NumberOfFrames',1))!=1 or int(header.SamplesPerPixel)!=1
                or header.PhotometricInterpretation not in ('MONOCHROME1','MONOCHROME2')):
                _fail('unsupported_dicom_profile')
            if not 1<=int(header.Rows)<=1024 or not 1<=int(header.Columns)<=1024:
                _fail('source_shape_limit')
            signature = (int(header.Rows),int(header.Columns),str(header.PhotometricInterpretation),
                         tuple(float(v) for v in header.ImageOrientationPatient),
                         tuple(float(v) for v in header.PixelSpacing))
            if not np.isfinite(signature[3]+signature[4]).all() or len(signature[3])!=6 or len(signature[4])!=2:
                _fail('invalid_dicom_geometry')
            if min(signature[4])<=0: _fail('invalid_dicom_geometry')
            if rows:
                oldsig = rows[0][3]
                if signature[:3]!=oldsig[:3] or not np.allclose(signature[3],oldsig[3],atol=1e-5,rtol=0) or not np.allclose(signature[4],oldsig[4],atol=1e-5,rtol=0):
                    _fail('mixed_dicom_geometry')
            source.update(hashlib.sha256(raw).digest())
            rows.append((number,path,header,signature))
    if len(series)!=1: _fail('mixed_series')
    ordered = sorted(rows,key=lambda row:row[0])
    pixel_hash = hashlib.sha256()
    for row in ordered:
        pixels = _decoded_image(row)
        pixel_hash.update(pixels.tobytes(order='C'))
    return ordered,source.hexdigest(),pixel_hash.hexdigest()

def _decoded_image(row):
    _,path,header,_ = row
    with _private_decoder():
        ds = pydicom.dcmread(path)
        image = ds.pixel_array.astype(np.float32)
    if image.shape!=(int(header.Rows),int(header.Columns)): _fail('invalid_pixel_shape')
    slope = float(getattr(header,'RescaleSlope',1)); intercept = float(getattr(header,'RescaleIntercept',0))
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope==0: _fail('invalid_rescale')
    image = image*slope+intercept
    if not np.isfinite(image).all(): _fail('invalid_pixels')
    return image

def _image(row, bbox, image_size):
    image = _decoded_image(row)
    header = row[2]
    low,high = np.percentile(image,(1,99))
    if high<=low: _fail('constant_source_image')
    image = np.clip((image-low)/(high-low),0,1)
    if header.PhotometricInterpretation=='MONOCHROME1': image=1-image
    x0,y0,x1,y1 = bbox
    crop = image[math.floor(y0):math.ceil(y1), math.floor(x0):math.ceil(x1)]
    if not crop.size: _fail('empty_crop')
    out = zoom(crop,(image_size/crop.shape[0],image_size/crop.shape[1]),order=1,prefilter=False)
    if out.shape!=(image_size,image_size): _fail('resize_shape_mismatch')
    return np.clip(out,0,1).astype(np.float16)[None]

def prepare_patient(folder, metadata, output_root, slice_count=16, image_size=96):
    """Return side records, exclusion reasons; caller owns private patient namespace."""
    try:
        folder,root = Path(folder),Path(output_root)
        if folder.is_symlink() or not folder.is_dir(): _fail('unsafe_source_path')
        if type(slice_count)!=int or not 1<=slice_count<=64 or type(image_size)!=int or not 16<=image_size<=256:
            _fail('invalid_preprocessing')
        inventory,source_hash,source_pixel_hash = _inventory(folder)
        patient = hashlib.sha256((SOURCE_DOI+'\0'+folder.name).encode()).hexdigest()
        lookup = {row[0]:row for row in inventory}; basenames = {row[1].name:row for row in inventory}
        records,excluded = [],{}
        for side,key in [('L','left'),('R','right')]:
            expected = metadata[key]
            if expected is None:
                excluded['missing_label']=excluded.get('missing_label',0)+1; continue
            annotation = folder/f'{side}-label.json'
            if not annotation.is_file():
                excluded['missing_annotation']=excluded.get('missing_annotation',0)+1; continue
            if annotation.is_symlink(): _fail('unsafe_source_path')
            try:
                boxes = parse_annotation(annotation,side)
            except OD3DInputError as error:
                if error.code != 'empty_annotation': raise
                excluded['empty_annotation'] = excluded.get('empty_annotation',0)+1
                continue
            observed = sorted({c for box in boxes for c in box['codes']})
            if list(expected)!=observed: _fail('csv_annotation_label_mismatch')
            candidates = []; reference_missing = 0
            for box in boxes:
                ref = basenames.get(box['reference_basename'])
                if ref is None: reference_missing += 1
                elif not box['instance_start']<=ref[0]<=box['instance_end']: _fail('reference_outside_range')
                if (box['height'],box['width'])!=inventory[0][3][:2]: _fail('annotation_frame_mismatch')
                x0,y0,x1,y1=box['bbox_xyxy']
                if not 0<=x0<x1<=box['width'] or not 0<=y0<y1<=box['height']: _fail('annotation_bounds')
                found = [(row,box['bbox_xyxy']) for row in inventory if box['instance_start']<=row[0]<=box['instance_end']]
                if not found: _fail('empty_annotation_range')
                candidates.extend(found)
            picks = np.rint(np.linspace(0,len(candidates)-1,slice_count)).astype(int)
            selected = [candidates[i] for i in picks]
            images = np.stack([_image(row,bbox,image_size) for row,bbox in selected])
            destination = root/'data'/f'{patient}-{side}.npz'
            destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            if destination.exists(): _fail('prepared_crop_exists')
            with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.npz',delete=False) as f:
                temporary = Path(f.name)
            try:
                np.savez_compressed(temporary,images=images)
                temporary.chmod(0o600); temporary.replace(destination)
            finally: temporary.unlink(missing_ok=True)
            records.append({'patient_id':patient,'side':side,'codes':list(expected),
                            'binary_target':int(expected!=(0,)), 'crop_path':str(destination.relative_to(root)),
                            'crop_sha256':digest_file(destination), 'source_sha256':source_hash,
                            'annotation_sha256':digest_file(annotation),
                            'source_pixel_sha256':source_pixel_hash,
                            'pixel_sha256':hashlib.sha256(images.tobytes(order='C')).hexdigest(),
                            'instance_numbers':[row[0] for row,_ in selected],
                            'reference_images_not_released':reference_missing,
                            'roi_source':'author_annotation','assessment_input':'oracle_roi'})
        return records,excluded
    except OD3DInputError as error: raise PreparationError(error.code) from None
    except PreparationError: raise
    except Exception: raise PreparationError('unreadable_patient_source') from None
