"""Patient holdout research for author-provided osseous ROI slice bags.

This oracle-ROI baseline has no localization or position-classification claims.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


def _architecture(name):
    if not isinstance(name, str) or name not in ('global_mean', 'spatial_head'):
        _fail('unsupported_architecture')
    return name


def _config(value):
    if not isinstance(value, dict):
        return load_config(value)
    config = dict(mode='binary', architecture='global_mean', epochs=40, patience=8, batch_size=4, seed=42,
                  learning_rate=.001, max_runtime_seconds=14400, bootstrap_draws=200, device='cpu')
    config.update(copy.deepcopy(value))
    _architecture(config['architecture'])
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


def _frozen_split(config):
    document = _json(config['split_path'])
    split = document.get('membership') if isinstance(document, dict) else None
    if not isinstance(document, dict) or document.get('task') != TASK or document.get('seed') != 42:
        _fail('incompatible_frozen_split')
    if document.get('split_digest') != _digest(split):
        _fail('frozen_split_digest_mismatch')
    if not isinstance(split, dict) or set(split) != set(PARTITIONS):
        _fail('invalid_frozen_split')
    if any(not isinstance(ids, list) or not all(isinstance(p, str) for p in ids) for ids in split.values()):
        _fail('invalid_frozen_split')
    return split


def _prepare(value, *, development=False):
    config = _config(value)
    if development and not config.get('split_path'):
        _fail('development_requires_frozen_split')
    split = _frozen_split(config) if config.get('split_path') else None
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
        pixel_digest = record.get('pixel_sha256')
        for field in ('source_pixel_sha256', 'pixel_sha256'):
            if field in record and (not isinstance(record[field], str) or not HEX.fullmatch(record[field])):
                _fail('invalid_pixel_digest')
        for kind, digest in (('prepared', pixel_digest), ('source', record.get('source_pixel_sha256'))):
            if digest is not None and pixel_owners.setdefault((kind, digest), patient) != patient:
                _fail('duplicate_pixel_patient_identity')
        paths.append(path)
    if split is None:
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
    for part, ids in groups.items():
        if development and part == 'test':
            continue
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
    # Ownership must be fully validated before reading any payload.
    for i, record in enumerate(records):
        if development and ownership[record['patient_id']] not in ('train', 'validation'):
            continue
        images = load_crop(paths[i], record['crop_sha256'], (k, 1, size, size))
        pixel_digest = hashlib.sha256(images.tobytes()).hexdigest()
        if 'pixel_sha256' in record and record['pixel_sha256'] != pixel_digest:
            _fail('prepared_pixel_digest_mismatch')
        if pixel_owners.setdefault(('prepared', pixel_digest), record['patient_id']) != record['patient_id']:
            _fail('duplicate_pixel_patient_identity')
    binding = dict(task=TASK, mode=config['mode'], codebook_commit=CODEBOOK,
                   input_digest=_sha(index_path), split_digest=_digest(split),
                   target_mapping_digest=_digest({'normal': 0, 'pathology_codes': [1, 2, 3, 4, 5, 6]}))
    if config['architecture'] != 'global_mean':
        binding['architecture'] = config['architecture']
    if development:
        groups.pop('test')
        binding['evaluation_scope'] = 'development'
    return config, index, paths, split, groups, binding


def preflight(config, *, development=False):
    config, index, _, _, groups, binding = _prepare(config, development=development)
    records = index['records']
    return dict(status='ready', task=TASK, mode=config['mode'], architecture=config['architecture'], assessment_input='oracle_roi',
                preprocessing=index['preprocessing'], bindings=binding,
                partitions={p: dict(patients=len({records[i]['patient_id'] for i in ids}),
                    joints=len(ids), positive_joints=sum(records[i]['binary_target'] for i in ids))
                    for p, ids in groups.items()})


class SliceBagClassifier(nn.Module):
    """Shared 2D slice CNN plus mean bag pooling; no spatial slice adjacency."""
    def __init__(self, outputs=1):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(16, outputs)

    def forward(self, images):
        b, k, c, h, w = images.shape
        features = self.features(images.reshape(b * k, c, h, w)).reshape(b, k, -1).mean(1)
        return self.head(features)


class SpatialSliceBagClassifier(SliceBagClassifier):
    """Experimental spatial head; no independent generalization evidence."""
    def __init__(self, outputs=1):
        super().__init__(outputs)
        self.features[-1] = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(nn.Linear(16 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, outputs))


def build_model(outputs=1, architecture='global_mean'):
    name = _architecture(architecture)
    return (SliceBagClassifier if name == 'global_mean' else SpatialSliceBagClassifier)(outputs)


class _Bags(Dataset):
    def __init__(self, records, paths, ids, shape, mode):
        self.records, self.paths, self.ids, self.shape, self.mode = records, paths, ids, shape, mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        i = self.ids[item]
        record = self.records[i]
        images = load_crop(self.paths[i], record['crop_sha256'], self.shape).astype(np.float32)
        target = _targets(self.records, [i], self.mode)[0]
        return torch.from_numpy(images), torch.from_numpy(target)


def _deadline(start, config):
    if time.monotonic() - start >= config['max_runtime_seconds']:
        _fail('runtime_limit_reached')


def _predict(model, loader, device, start, config, *, collect_logits=False):
    model.eval()
    values, logits_values = [], []
    with torch.no_grad():
        for images, _ in loader:
            _deadline(start, config)
            logits = model(images.to(device))
            values.append(torch.sigmoid(logits).cpu().numpy())
            if collect_logits:
                logits_values.append(logits.cpu().numpy())
    probabilities = np.concatenate(values)
    return (probabilities, np.concatenate(logits_values)) if collect_logits else probabilities


def _auc(target, probability):
    if len(set(target.tolist())) < 2:
        return None, None
    return float(roc_auc_score(target, probability)), float(average_precision_score(target, probability))


def _thresholds(targets, probabilities, supported=None):
    thresholds = []
    for label in range(targets.shape[1]):
        y, p = targets[:, label], probabilities[:, label]
        if (supported is not None and not supported[label]) or len(np.unique(y)) != 2:
            thresholds.append(.5)
            continue
        fpr, tpr, candidates = roc_curve(y, p)
        valid = np.isfinite(candidates) & (candidates <= 1)
        choice = np.flatnonzero(valid)[np.argmax((tpr - fpr)[valid])]
        thresholds.append(float(candidates[choice]))
    return thresholds


def _metrics(targets, probabilities, thresholds, supported=None):
    per_label = []
    for label in range(targets.shape[1]):
        y, p = targets[:, label], probabilities[:, label]
        auroc, auprc = _auc(y, p) if supported is None or supported[label] else (None, None)
        tn, fp, fn, tp = confusion_matrix(y, p >= thresholds[label], labels=[0, 1]).ravel().tolist()
        per_label.append(dict(supported_for_evaluation=auroc is not None, auroc=auroc, auprc=auprc, confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp),
            sensitivity=tp / (tp + fn) if tp + fn else None,
            specificity=tn / (tn + fp) if tn + fp else None))
    valid = [v for v in per_label if v['auroc'] is not None]
    return dict(auroc=float(np.mean([v['auroc'] for v in valid])) if valid else None,
                auprc=float(np.mean([v['auprc'] for v in valid])) if valid else None, labels=per_label)


def _bootstrap(targets, probabilities, thresholds, patients, draws, supported=None, start=None, config=None):
    """Resample patients with replacement, preserving all joints of each draw."""
    if not draws:
        return dict(requested_draws=0, valid_draws=0, auroc=None, auprc=None)
    unique = sorted(set(patients))
    groups = {p: np.flatnonzero(np.array(patients) == p) for p in unique}
    rng, values = np.random.default_rng(42), []
    for _ in range(draws):
        if start is not None:
            _deadline(start, config)
        ids = np.concatenate([groups[p] for p in rng.choice(unique, len(unique), replace=True)])
        metric = _metrics(targets[ids], probabilities[ids], thresholds, supported)
        if metric['auroc'] is not None:
            values.append([metric['auroc'], metric['auprc']])
    return dict(requested_draws=draws, valid_draws=len(values),
                auroc=np.quantile(np.array(values)[:, 0], [.025, .975]).tolist() if values else None,
                auprc=np.quantile(np.array(values)[:, 1], [.025, .975]).tolist() if values else None)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    path.chmod(0o600)


def load_checkpoint(path, device='cpu'):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        metadata = checkpoint['metadata']
        if metadata['bindings']['task'] != TASK or metadata['bindings']['codebook_commit'] != CODEBOOK:
            _fail('incompatible_checkpoint')
        architecture = _architecture(metadata.get('architecture', 'global_mean'))
        configured = _architecture(metadata.get('config', {}).get('architecture', 'global_mean'))
        bound = _architecture(metadata['bindings'].get('architecture', 'global_mean'))
        if architecture != configured or architecture != bound:
            _fail('checkpoint_architecture_mismatch')
        if metadata['mode'] not in ('binary', 'multilabel'):
            _fail('incompatible_checkpoint')
        model = build_model(1 if metadata['mode'] == 'binary' else 6, architecture).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model, metadata
    except ResearchError:
        raise
    except Exception:
        _fail('invalid_checkpoint')


def train(config, *, development=False):
    """Select on validation; development never opens the frozen test payload."""
    try:
        return _train(config, development=development)
    except ResearchError:
        raise
    except Exception:
        raise ResearchError('research_execution_failed') from None


def _train(config, *, development=False):
    start = time.monotonic()
    config, index, paths, split, groups, binding = _prepare(config, development=development)
    output = Path(config['output_dir'])
    if output.exists() and any(output.iterdir()):
        _fail('output_directory_not_empty')
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    _write(output / 'split.private.json', dict(task=TASK, seed=42, membership=split,
                                              split_digest=binding['split_digest']))
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        _fail('cuda_unavailable')
    device = torch.device(config['device'])
    records, preprocessing = index['records'], index['preprocessing']
    shape = (preprocessing['slice_count'], 1, preprocessing['image_size'], preprocessing['image_size'])
    datasets = {p: _Bags(records, paths, ids, shape, config['mode']) for p, ids in groups.items()}
    loaders = {p: DataLoader(dataset, batch_size=config['batch_size'], shuffle=p == 'train',
        generator=torch.Generator().manual_seed(42), num_workers=0) for p, dataset in datasets.items()}
    targets = {p: _targets(records, ids, config['mode']) for p, ids in groups.items()}
    prevalence = targets['train'].mean(0)
    mask = (prevalence > 0) & (prevalence < 1)
    validation_prevalence = targets['validation'].mean(0)
    validation_mask = (validation_prevalence > 0) & (validation_prevalence < 1)
    evaluation_mask = mask & validation_mask
    label_limitations, threshold_sources = [], []
    for train_ok, validation_ok in zip(mask.tolist(), validation_mask.tolist()):
        label_limitations.append([reason for supported, reason in (
            (train_ok, 'no_train_class_support'), (validation_ok, 'no_validation_class_support')) if not supported])
        if not train_ok:
            threshold_sources.append('fallback_0.5_no_train_support')
        elif not validation_ok:
            threshold_sources.append('fallback_0.5_no_validation_support')
        else:
            threshold_sources.append('validation_youden_j')
    calibration_support = dict(train_supported_label_mask=mask.tolist(),
        validation_supported_label_mask=validation_mask.tolist(), evaluation_supported_label_mask=evaluation_mask.tolist(),
        threshold_sources=threshold_sources, label_limitations=label_limitations)
    if not mask.any():
        _fail('no_train_supported_labels')
    weights = np.divide(1 - prevalence, prevalence, out=np.ones_like(prevalence), where=prevalence > 0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=device), reduction='none')
    label_mask = torch.tensor(mask, device=device, dtype=torch.float32)
    model = build_model(targets['train'].shape[1], config['architecture']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    best_score, best_epoch, stale, history, best_state = -math.inf, 0, 0, [], None
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        loss_sum, sample_count = 0., 0
        for images, target in loaders['train']:
            _deadline(start, config)
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = (loss_fn(logits, target.to(device)) * label_mask).sum() / (len(images) * label_mask.sum())
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(images)
            sample_count += len(images)
        probabilities, validation_logits = _predict(model, loaders['validation'], device, start, config, collect_logits=True)
        metric = _metrics(targets['validation'], probabilities, [.5] * len(prevalence), evaluation_mask)
        score = metric['auroc'] if config['mode'] == 'binary' else metric['auprc']
        if score is None:
            _fail('no_validation_supported_labels')
        y = targets['validation']
        with torch.no_grad():
            validation_loss = loss_fn(torch.from_numpy(validation_logits).to(device), torch.from_numpy(y).to(device))
            validation_loss = float(((validation_loss * label_mask).sum() / (len(y) * label_mask.sum())).cpu())
        history.append(dict(epoch=epoch, validation_selection_metric=score,
            train_weighted_loss=loss_sum / sample_count,
            validation_weighted_loss=validation_loss,
            validation_probability_std=float(probabilities[:, evaluation_mask].std())))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    validation = _predict(model, loaders['validation'], device, start, config)
    thresholds = _thresholds(targets['validation'], validation, evaluation_mask)
    metadata = dict(mode=config['mode'], architecture=config['architecture'], evaluation_scope='development' if development else 'holdout', selected_epoch=best_epoch, thresholds=thresholds,
                    preprocessing=preprocessing, bindings=binding, train_prevalence=prevalence.tolist(),
                    **calibration_support, config=config)
    checkpoint = output / 'checkpoint.private.pt'
    torch.save(dict(state_dict=best_state, metadata=metadata), checkpoint)
    checkpoint.chmod(0o600)
    reloaded, _ = load_checkpoint(checkpoint, str(device))
    replay = _predict(reloaded, loaders['validation'], device, start, config)
    if not np.array_equal(validation, replay):
        _fail('checkpoint_reload_mismatch')
    evaluated_part = 'validation' if development else 'test'
    evaluated = validation if development else _predict(reloaded, loaders['test'], device, start, config)
    baseline_validation = np.broadcast_to(prevalence, validation.shape)
    baseline_thresholds = _thresholds(targets['validation'], baseline_validation, evaluation_mask)
    baseline = np.broadcast_to(prevalence, evaluated.shape)
    evaluated_metrics = {}
    for name, probabilities, cutoffs in (('model', evaluated, thresholds),
            ('constant_train_prevalence', baseline, baseline_thresholds)):
        result = _metrics(targets[evaluated_part], probabilities, cutoffs, evaluation_mask)
        for label, limitations, source in zip(result['labels'], label_limitations, threshold_sources):
            label['limitations'] = list(limitations)
            label['threshold_source'] = source
        result['bootstrap'] = _bootstrap(targets[evaluated_part], probabilities, cutoffs,
            [records[i]['patient_id'] for i in groups[evaluated_part]], config['bootstrap_draws'], evaluation_mask, start, config)
        evaluated_metrics[name] = result
    predictions = []
    # Reuse evaluated outputs; train inference is reporting only.
    train_loader = DataLoader(datasets['train'], batch_size=config['batch_size'], shuffle=False)
    partition_predictions = dict(train=_predict(reloaded, train_loader, device, start, config),
                                 validation=validation)
    if not development:
        partition_predictions['test'] = evaluated
    for part, ids in groups.items():
        for row, i in enumerate(ids):
            predictions.append(dict(patient_id=records[i]['patient_id'], side=records[i]['side'],
                partition=part, target=targets[part][row].tolist(), probability=partition_predictions[part][row].tolist()))
    _deadline(start, config)
    _write(output / 'predictions.private.json', dict(bindings=binding, records=predictions))
    active_patients = {records[i]['patient_id'] for ids in groups.values() for i in ids}
    _write(output / 'manifest.private.json', dict(bindings=binding, metadata=metadata,
        crop_bindings=[dict(patient_id=r['patient_id'], side=r['side'], crop_sha256=r['crop_sha256'],
                           source_sha256=r['source_sha256'], annotation_sha256=r['annotation_sha256']) for r in records if r['patient_id'] in active_patients]))
    report = dict(status='complete', task=TASK, mode=config['mode'], architecture=config['architecture'], assessment_input='oracle_roi',
        preprocessing=preprocessing, bindings=binding, selected_epoch=best_epoch,
        selection_history=history, thresholds=thresholds, baseline_thresholds=baseline_thresholds,
        train_prevalence=prevalence.tolist(), **calibration_support,
        checkpoint_reload_verified=True, evaluation_scope='development' if development else 'holdout',
        interpretation=('development: validation used for model/threshold selection; no unbiased quality estimate'
                        if development else 'oracle ROI patient holdout; no clinical or localization validation'))
    report[evaluated_part] = evaluated_metrics
    if development:
        report['train'] = dict(model=_metrics(targets['train'], partition_predictions['train'], thresholds, evaluation_mask))
    _write(output / 'report.json', report)
    _write(output / 'completion.json', dict(status='complete', bindings=binding,
        artifact_digests={p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return report
