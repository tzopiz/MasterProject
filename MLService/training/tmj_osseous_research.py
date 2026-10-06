"""Patient holdout research for author-provided osseous ROI slice bags.

This oracle-ROI baseline has no localization or position-classification claims.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

TASK = 'tmj-osseous-author-roi-v1'
DOI = '10.57760/sciencedb.37727'
CODEBOOK = 'a4bce89c89b7175e330193f04a282887b5e3f8b0'
PARTITIONS = ('train', 'validation', 'test')
HEX = re.compile(r'^[0-9a-f]{64}$')


class ResearchError(ValueError):
    """Only aggregate error codes may cross the public boundary."""


def _fail(code):
    raise ResearchError(code)


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        _fail('invalid_json')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(path):
    try:
        with Path(path).open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        _fail('unreadable_artifact')


def load_config(config_path):
    """Resolve config-owned paths; never interpret crop paths outside the index."""
    path = Path(config_path).resolve()
    config = _json(path)
    if not isinstance(config, dict):
        _fail('invalid_config')
    for key in ('index_path', 'output_dir', 'split_path'):
        if key in config:
            if not isinstance(config[key], str) or not config[key]:
                _fail('invalid_config_path')
            config[key] = str((path.parent / config[key]).resolve())
    return _config(config)


def _config(value):
    if not isinstance(value, dict):
        return load_config(value)
    config = dict(mode='binary', epochs=40, patience=8, batch_size=4, seed=42,
                  learning_rate=.001, max_runtime_seconds=14400, bootstrap_draws=200, device='cpu')
    config.update(copy.deepcopy(value))
    for key in ('index_path', 'output_dir'):
        if not isinstance(config.get(key), str) or not config[key]:
            _fail('missing_config_path')
    if config['mode'] not in ('binary', 'multilabel') or config['device'] not in ('cpu', 'cuda'):
        _fail('unsupported_mode_or_device')
    for key, upper in (('epochs', 40), ('patience', 8), ('batch_size', 4), ('max_runtime_seconds', 14400)):
        if type(config[key]) is not int or not 1 <= config[key] <= upper:
            _fail('invalid_resource_bound')
    if config['seed'] != 42 or type(config['seed']) is not int or config['learning_rate'] != .001:
        _fail('unsupported_baseline_parameters')
    if type(config['bootstrap_draws']) is not int or config['bootstrap_draws'] not in (0, 200):
        _fail('invalid_bootstrap_draws')
    reserved = config.get('development_only_patients', [])
    if not isinstance(reserved, list) or not all(isinstance(p, str) for p in reserved):
        _fail('invalid_development_patients')
    return config


def load_crop(path, checksum, shape):
    """Read exactly one float16 finite normalized slice bag, bound to its bytes."""
    if _sha(path) != checksum:
        _fail('crop_checksum_mismatch')
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'images'}:
                _fail('invalid_crop_keys')
            images = archive['images']
            if images.dtype != np.float16 or images.shape != tuple(shape):
                _fail('invalid_crop_shape_or_dtype')
            if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                _fail('invalid_crop_values')
            return images.copy()
    except ResearchError:
        raise
    except Exception:
        _fail('invalid_crop_archive')


def build_split(records, seed=42, development_only_patients=()):
    """Freeze a deterministic 70/15/15 patient split; sides never separate."""
    if seed != 42:
        _fail('unsupported_split_seed')
    patients = sorted({r['patient_id'] for r in records})
    labels = {p: max(r['binary_target'] for r in records if r['patient_id'] == p) for p in patients}
    try:
        train_ids, holdout = train_test_split(patients, test_size=.3, random_state=seed,
                                             stratify=[labels[p] for p in patients])
        val_ids, test_ids = train_test_split(holdout, test_size=.5, random_state=seed,
                                            stratify=[labels[p] for p in holdout])
    except ValueError:
        _fail('infeasible_patient_split')
    reserved = set(development_only_patients)
    if not reserved <= set(patients):
        _fail('unknown_development_patient')
    for patient in sorted(set(test_ids) & reserved):
        candidates = [(part, p) for part in (train_ids, val_ids) for p in sorted(part)
                      if p not in reserved and labels[p] == labels[patient]]
        if not candidates:
            _fail('infeasible_development_patient_split')
        part, replacement = candidates[0]
        part.remove(replacement)
        part.append(patient)
        test_ids.remove(patient)
        test_ids.append(replacement)
    return {key: sorted(ids) for key, ids in zip(PARTITIONS, (train_ids, val_ids, test_ids))}


def _targets(records, ids, mode):
    return np.array([[records[i]['binary_target']] if mode == 'binary' else
                    [int(c in records[i]['codes']) for c in range(1, 7)] for i in ids], dtype=np.float32)


def _prepare(value):
    config = _config(value)
    index_path = Path(config['index_path']).resolve()
    index = _json(index_path)
    if not isinstance(index, dict) or index.get('schema_version') != 1 or index.get('task') != TASK:
        _fail('incompatible_index_task')
    if index.get('source_doi') != DOI or index.get('codebook_commit') != CODEBOOK:
        _fail('incompatible_codebook')
    preprocessing = index.get('preprocessing', {})
    if not isinstance(preprocessing, dict):
        _fail('invalid_preprocessing')
    k, size = preprocessing.get('slice_count'), preprocessing.get('image_size')
    if type(k) is not int or not 1 <= k <= 16 or type(size) is not int or not 1 <= size <= 96:
        _fail('invalid_preprocessing_bounds')
    if preprocessing.get('intensity') != 'per-slice-p1-p99':
        _fail('incompatible_intensity')
    records = index.get('records')
    if not isinstance(records, list) or not records:
        _fail('empty_records')
    keys, paths, pixel_owners = set(), [], {}
    for record in records:
        if not isinstance(record, dict):
            _fail('invalid_record')
        patient, side = record.get('patient_id'), record.get('side')
        if not isinstance(patient, str) or not re.fullmatch(r'(?:[a-zA-Z0-9_.-]+:)?[0-9a-f]{64}', patient):
            _fail('invalid_patient_namespace')
        if side not in ('L', 'R') or (patient, side) in keys:
            _fail('duplicate_or_invalid_patient_side')
        keys.add((patient, side))
        codes = record.get('codes')
        if (not isinstance(codes, list) or not codes or any(type(c) is not int for c in codes)
                or len(set(codes)) != len(codes) or not (codes == [0] or all(1 <= c <= 6 for c in codes))):
            _fail('invalid_osseous_codes')
        if type(record.get('binary_target')) is not int or record['binary_target'] != int(codes != [0]):
            _fail('inconsistent_binary_target')
        if record.get('roi_source') != 'author_annotation' or record.get('assessment_input') != 'oracle_roi':
            _fail('incompatible_roi_provenance')
        if any(not isinstance(record.get(key), str) or not HEX.fullmatch(record[key])
               for key in ('crop_sha256', 'source_sha256', 'annotation_sha256')):
            _fail('invalid_artifact_digest')
        instances = record.get('instance_numbers')
        if not isinstance(instances, list) or len(instances) != k or any(type(i) is not int or i <= 0 for i in instances):
            _fail('invalid_slice_instances')
        relative = record.get('crop_path')
        if not isinstance(relative, str) or Path(relative).is_absolute():
            _fail('invalid_crop_path')
        path = (index_path.parent / relative).resolve()
        if not path.is_relative_to(index_path.parent) or path.suffix != '.npz':
            _fail('unsafe_crop_path')
        images = load_crop(path, record['crop_sha256'], (k, 1, size, size))
        pixel_digest = hashlib.sha256(images.tobytes()).hexdigest()
        if 'pixel_sha256' in record and record['pixel_sha256'] != pixel_digest:
            _fail('prepared_pixel_digest_mismatch')
        for field in ('source_pixel_sha256', 'pixel_sha256'):
            if field in record and (not isinstance(record[field], str) or not HEX.fullmatch(record[field])):
                _fail('invalid_pixel_digest')
        for kind, digest in (('prepared', pixel_digest), ('source', record.get('source_pixel_sha256'))):
            if digest is not None and pixel_owners.setdefault((kind, digest), patient) != patient:
                _fail('duplicate_pixel_patient_identity')
        paths.append(path)
    if config.get('split_path'):
        document = _json(config['split_path'])
        split = document.get('membership') if isinstance(document, dict) else None
        if not isinstance(document, dict) or document.get('task') != TASK or document.get('seed') != 42:
            _fail('incompatible_frozen_split')
        if document.get('split_digest') != _digest(split):
            _fail('frozen_split_digest_mismatch')
    else:
        split = build_split(records, development_only_patients=config.get('development_only_patients', []))
    patients = {r['patient_id'] for r in records}
    if not isinstance(split, dict) or set(split) != set(PARTITIONS):
        _fail('invalid_frozen_split')
    flat = []
    for key in PARTITIONS:
        if not isinstance(split[key], list) or not all(isinstance(p, str) for p in split[key]):
            _fail('invalid_frozen_split')
        flat += split[key]
    if len(flat) != len(set(flat)) or set(flat) != patients:
        _fail('patient_partition_leakage')
    reserved = set(config.get('development_only_patients', []))
    if not reserved <= patients:
        _fail('unknown_development_patient')
    if reserved & set(split['test']):
        _fail('development_patient_in_test')
    ownership = {p: part for part, ids in split.items() for p in ids}
    groups = {part: [i for i, r in enumerate(records) if ownership[r['patient_id']] == part] for part in PARTITIONS}
    for ids in groups.values():
        if {records[i]['binary_target'] for i in ids} != {0, 1}:
            _fail('missing_partition_binary_class')
    if config['mode'] == 'multilabel':
        train_targets, validation_targets = (_targets(records, groups[p], 'multilabel')
                                            for p in ('train', 'validation'))
        if not ((np.ptp(train_targets, axis=0) > 0) & (np.ptp(validation_targets, axis=0) > 0)).any():
            _fail('no_validation_supported_labels')
    for field in ('source_sha256', 'crop_sha256'):
        seen = {}
        for record in records:
            owner = ownership[record['patient_id']]
            previous = seen.setdefault(record[field], (owner, record['patient_id']))
            if previous[0] != owner:
                _fail('cross_partition_duplicate')
            if field == 'source_sha256' and previous[1] != record['patient_id']:
                _fail('duplicate_source_patient_identity')
    binding = dict(task=TASK, mode=config['mode'], codebook_commit=CODEBOOK,
                   input_digest=_sha(index_path), split_digest=_digest(split),
                   target_mapping_digest=_digest({'normal': 0, 'pathology_codes': [1, 2, 3, 4, 5, 6]}))
    return config, index, paths, split, groups, binding


def preflight(config):
    config, index, _, _, groups, binding = _prepare(config)
    records = index['records']
    return dict(status='ready', task=TASK, mode=config['mode'], assessment_input='oracle_roi',
                preprocessing=index['preprocessing'], bindings=binding,
                partitions={p: dict(patients=len({records[i]['patient_id'] for i in ids}),
                    joints=len(ids), positive_joints=sum(records[i]['binary_target'] for i in ids))
                    for p, ids in groups.items()})
