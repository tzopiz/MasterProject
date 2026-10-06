#!/usr/bin/env python3
"""Apply reviewed exclusions to a complete release or an explicit private archive prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_tmj_od3d import ARCHIVE_BYTES, MAX_PREPARED_BYTES, METADATA_SHA256
from training.tmj_od3d_images import CODEBOOK_COMMIT, SOURCE_DOI, TASK, digest_file
from training.tmj_od3d_inputs import load_metadata

EXPECTED_PATIENT_COUNT = 1043
RECORD_FIELDS = {'patient_id','side','codes','binary_target','crop_path','crop_sha256','source_sha256',
    'annotation_sha256','instance_numbers','roi_source','assessment_input','source_pixel_sha256',
    'pixel_sha256','reference_images_not_released'}

class CurationError(ValueError):
    """Fixed public codes only; source identities and policy reasons stay private."""

def require(condition, code):
    if not condition: raise CurationError(code)

def _read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024**2, 'invalid_private_file')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key'); result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(CurationError('invalid_json_constant')))
    require(isinstance(value, dict), 'invalid_private_file')
    return value

def _write(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')

def curate(source_root, policy_path, output_root):
    """No network, source mutation, clinical inference, or consumer readiness claim."""
    try: return _curate(Path(source_root).resolve(), Path(policy_path).resolve(), Path(output_root).resolve())
    except CurationError: raise
    except Exception: raise CurationError('curation_failed') from None

def _curate(source, policy_path, output):
    require(not output.is_relative_to(source) and not source.is_relative_to(output), 'input_output_overlap')
    require(not output.exists(), 'output_exists')
    require(not (source/'preparation.lock').exists(), 'source_locked')
    files = {'source_preparation_sha256': source/'preparation.private.json',
             'source_index_sha256': source/'index.private.json', 'metadata_sha256': source/'metadata.private.csv',
             'policy_sha256': policy_path}
    require(all(p.is_file() and not p.is_symlink() for p in files.values()), 'invalid_private_file')
    before = {key: digest_file(path) for key, path in files.items()}
    require(before['metadata_sha256'] == METADATA_SHA256, 'source_metadata_changed')
    state, index, policy = [_read(files[key]) for key in ('source_preparation_sha256','source_index_sha256','policy_sha256')]
    require(type(state.get('schema_version')) is int and state['schema_version']==1 and state.get('task')==TASK
        and state.get('source_doi')==SOURCE_DOI and state.get('codebook_commit')==CODEBOOK_COMMIT
        and state.get('metadata_sha256')==METADATA_SHA256, 'incompatible_preparation')
    selected_cohort = policy.get('cohort')
    expected_count, expected_end, source_complete = EXPECTED_PATIENT_COUNT, ARCHIVE_BYTES, True
    if 'cohort' in policy:
        require(isinstance(selected_cohort,dict) and set(selected_cohort)=={
            'kind','source_patient_count','source_end_offset'} and selected_cohort.get('kind')=='archive-prefix',
            'invalid_cohort_policy')
        expected_count, expected_end = selected_cohort['source_patient_count'], selected_cohort['source_end_offset']
        require(type(expected_count) is int and 0<expected_count<EXPECTED_PATIENT_COUNT
            and type(expected_end) is int and 0<expected_end<ARCHIVE_BYTES, 'invalid_cohort_policy')
        source_complete = False
    require(state.get('complete') is source_complete and type(state.get('next_offset')) is int
        and state['next_offset']==expected_end, 'source_incomplete')
    names, receipts, records = state.get('processed_patients'), state.get('patient_receipts'), state.get('records')
    metadata = load_metadata(files['metadata_sha256'])
    require(isinstance(names,list) and all(isinstance(name,str) for name in names)
        and len(names)==expected_count and len(set(names))==len(names) and len(metadata)==EXPECTED_PATIENT_COUNT
        and set(names)<=set(metadata), 'source_coverage_mismatch')
    require(isinstance(receipts,list) and len(receipts)==len(names) and isinstance(records,list), 'receipt_coverage_mismatch')
    expected_index = {k:v for k,v in state.items() if k not in ('processed_patients','next_offset','patient_receipts')}
    expected_index['patient_count'] = len(names)
    require(index==expected_index, 'source_index_mismatch')
    ownership, failed, failures, previous = {}, {}, Counter(), None
    for name, receipt in zip(names, receipts):
        patient = hashlib.sha256((SOURCE_DOI+'\0'+name).encode()).hexdigest()
        require(receipt.get('patient_id')==patient and patient not in ownership, 'receipt_identity_mismatch')
        lo, hi, count = receipt.get('tar_start'), receipt.get('tar_end'), receipt.get('side_count')
        require(type(lo) is int and type(hi) is int and 0<=lo<hi<=ARCHIVE_BYTES
            and (previous is None or lo==previous) and type(count) is int and 0<=count<=2, 'invalid_receipt')
        previous = hi
        if receipt.get('status')=='failed':
            require(count==0 and receipt.get('code')=='annotation_side_conflict', 'unsupported_failure_code')
            failed[patient]=receipt['code']; failures[receipt['code']]+=1
        else: require(receipt.get('status')=='prepared' and receipt.get('code')=='ok', 'invalid_receipt')
        ownership[patient]=count
    require(previous==expected_end and state.get('failures')==dict(failures), 'source_failure_mismatch')
    require(policy.get('schema_version')==1 and policy.get('task')==TASK
        and policy.get('source_preparation_sha256')==before['source_preparation_sha256'], 'policy_source_mismatch')
    reviews = policy.get('reviews'); require(isinstance(reviews,list), 'invalid_policy')
    reviewed = {}
    for row in reviews:
        require(isinstance(row,dict) and row.get('patient_id') not in reviewed
            and row.get('disposition')=='exclude' and row.get('code')=='annotation_side_conflict'
            and isinstance(row.get('reason'),str) and bool(row['reason'].strip()), 'invalid_policy_review')
        reviewed[row['patient_id']]=row['code']
    require(reviewed==failed, 'policy_receipt_mismatch')
    selected, counts, keys, total = [], Counter(), set(), 0
    require(not (source/'data').is_symlink(), 'unsafe_crop_path')
    for record in records:
        patient, side = record.get('patient_id'), record.get('side'); key = (patient,side)
        require(patient in ownership and patient not in failed and side in ('L','R') and key not in keys, 'record_receipt_mismatch')
        relative = f'data/{patient}-{side}.npz'; path = source/relative
        require(record.get('crop_path')==relative and path.is_file() and not path.is_symlink(), 'unsafe_crop_path')
        checksum = record.get('crop_sha256')
        require(isinstance(checksum,str) and re.fullmatch('[0-9a-f]{64}',checksum)
            and digest_file(path)==checksum, 'crop_changed')
        total += path.stat().st_size; keys.add(key); counts[patient]+=1
        selected.append({k:v for k,v in record.items() if k in RECORD_FIELDS})
    require(all(counts[p]==n for p,n in ownership.items()), 'record_count_mismatch')
    require(total<=MAX_PREPARED_BYTES, 'prepared_storage_limit')
    output.mkdir(mode=0o700, parents=True); (output/'data').mkdir(mode=0o700)
    try:
        for record in selected:
            src, dest = source/record['crop_path'], output/record['crop_path']
            shutil.copyfile(src,dest); dest.chmod(0o600)
            require(digest_file(src)==record['crop_sha256']==digest_file(dest), 'crop_changed')
        for record in selected:
            require(digest_file(source/record['crop_path'])==record['crop_sha256']
                ==digest_file(output/record['crop_path']), 'crop_changed')
        require(not (source/'preparation.lock').exists()
            and before=={key:digest_file(path) for key,path in files.items()}, 'source_changed_during_curation')
        curated = {k:state[k] for k in ('schema_version','task','source_doi','codebook_commit','metadata_sha256','preprocessing','exclusions')}
        curated.update(complete=True, records=selected, failures={}, source_failures=dict(failures),
            patient_count=len(counts), accepted_source_patient_count=len(names)-len(failed),
            curation=dict(before,source_patient_count=len(names),excluded_patient_count=len(failed)))
        report = dict(status='curated',source_patients=len(names),accepted_patients=len(counts),
            accepted_source_patients=len(names)-len(failed),excluded_patients=len(failed),accepted_sides=len(selected),
            copied_bytes=total,source_failures=dict(failures),consumer_preflight_required=True)
        if selected_cohort is not None:
            cohort_metadata = dict(selected_cohort, source_complete=False, release_patient_count=EXPECTED_PATIENT_COUNT)
            curated['cohort'] = cohort_metadata; report['cohort'] = cohort_metadata
        _write(output/'curation-report.json',report); _write(output/'index.private.json',curated)
        return report
    except Exception:
        shutil.rmtree(output); raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True); parser.add_argument('--policy',required=True)
    parser.add_argument('--output-root',required=True); args=parser.parse_args(argv)
    try:
        print(json.dumps(curate(args.source_root,args.policy,args.output_root),sort_keys=True)); return 0
    except CurationError as error:
        print(json.dumps({'status':'rejected','code':str(error)})); return 1

if __name__=='__main__': raise SystemExit(main())
