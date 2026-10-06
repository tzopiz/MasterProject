#!/usr/bin/env python3
"""Prepare/confirm a private frozen osseous research job; prepare makes no cloud IO."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.datasphere_research import (
    CloudLaunchError,
    _check_bundle,
    _cli_prefix,
    _digest,
    _hash_file,
    _identifier,
    _invoke,
    _job,
    _manifest,
    _read,
    _write,
    cancel_job,
    confirm_bundle,
    job_status,
    reconcile_submission,
)
from training.tmj_osseous_research import CODEBOOK, TASK, build_split, load_config, preflight

SOURCE_FILES = ('training/tmj_osseous_research.py',
                'tools/run_osseous_research.py',
                'tools/datasphere_osseous.py','tools/datasphere_research.py',
                'tools/osseous_cloud_bootstrap.sh')
MAX_INDEX_BYTES=16*1024**2
DOCKER_IMAGE='system-python-3-10'


def _read_index(path):
    """Larger source index only; lifecycle/config readers retain their 1 MiB cap."""
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError
            result[key]=value
        return result
    def invalid_constant(value):raise ValueError
    def finite_float(value):
        number=float(value)
        if not math.isfinite(number):raise ValueError
        return number
    try:
        with Path(path).open('rb') as stream:raw=stream.read(MAX_INDEX_BYTES+1)
        if len(raw)>MAX_INDEX_BYTES:raise ValueError
        return json.loads(raw.decode('utf-8'),object_pairs_hook=unique,parse_constant=invalid_constant,parse_float=finite_float)
    except (OSError,ValueError,TypeError,RecursionError):
        raise CloudLaunchError('invalid_private_index') from None


def prepare_bundle(config_path,bundle_dir,*,project_id,profile='tmj-master',resource='gt4.1',
                   hourly_price=168.48,currency='RUB',price_as_of='2026-10-05T00:00:00+03:00'):
    """Local only. No staging can launch implicitly; confirmation binds exact bytes."""
    bundle=Path(bundle_dir).resolve()
    if bundle.exists():raise CloudLaunchError('bundle_exists')
    config=load_config(config_path);index=_read_index(config['index_path']);summary=preflight(config)
    if index.get('complete') is not True or index.get('failures'):
        raise CloudLaunchError('dataset_preparation_incomplete')
    if resource!='gt4.1' or currency!='RUB' or not math.isfinite(float(hourly_price)) or float(hourly_price)<=0:
        raise CloudLaunchError('invalid_resource_quote')
    if datetime.fromisoformat(price_as_of).utcoffset() is None:raise CloudLaunchError('invalid_price_date')
    _identifier(project_id)
    if profile!='tmj-master':raise CloudLaunchError('unexpected_account_profile')
    source_index=Path(config['index_path']);source_root=source_index.parent
    if bundle.is_relative_to(source_root) or source_root.is_relative_to(bundle):
        raise CloudLaunchError('bundle_input_overlap')
    split=_read(config['split_path']) if config.get('split_path') else {
        'task':TASK,'seed':42,'membership':build_split(index['records'],development_only_patients=config.get('development_only_patients',[]))}
    split['split_digest']=_digest(split['membership'])
    bundle.mkdir(mode=0o700,parents=True);payload=bundle/'payload';payload.mkdir(mode=0o700)
    try:
        for record in index['records']:
            target=payload/record['crop_path'];target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            shutil.copyfile(source_root/record['crop_path'],target)
        _write(payload/'index.private.json',index);_write(payload/'split.private.json',split)
        remote=dict(config,index_path='index.private.json',split_path='split.private.json',
                    output_dir='../results/research',device='cuda')
        _write(payload/'research.private.json',remote)
        root=Path(__file__).resolve().parents[1]
        for relative in SOURCE_FILES:
            target=payload/relative;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            shutil.copyfile(root/relative,target)
        shutil.copyfile(root/'tools/osseous_cloud_requirements.txt',payload/'requirements.txt')
        staged_summary=preflight(load_config(payload/'research.private.json'))
        if staged_summary['bindings']['split_digest']!=summary['bindings']['split_digest']:
            raise CloudLaunchError('staged_split_changed')
        manifest=_manifest(payload);size=sum(row['bytes'] for row in manifest.values())
        if size>5*1024**3:raise CloudLaunchError('upload_size_limit')
        runtime=config['max_runtime_seconds']
        job={'name':'tmj-osseous-research','cmd':f'bash payload/tools/osseous_cloud_bootstrap.sh payload/research.private.json {runtime}',
             'inputs':['payload'],'outputs':['results'],'cloud-instance-types':[resource],
             'env':{'docker':DOCKER_IMAGE}}
        _write(bundle/'job.yaml',job)
        plan={'schema_version':1,'task':TASK,'project_id':project_id,'profile':profile,
              'resource':resource,'python':'3.10','docker_image':DOCKER_IMAGE,'manifest':manifest,'job_sha256':_hash_file(bundle/'job.yaml'),
              'bindings':staged_summary['bindings'],'source_input_digest':summary['bindings']['input_digest'],'split_digest':split['split_digest'],'upload_bytes':size,
              'training_runtime_seconds':runtime,'hourly_price':float(hourly_price),'currency':currency,
              'price_as_of':price_as_of,'training_window_compute_estimate':float(hourly_price)*runtime/3600,
              'money_cap_guaranteed':False,'excluded_costs':['environment_setup','storage','egress'],
              'assessment_input':'oracle_roi','partitions':summary['partitions']}
        plan['plan_sha256']=_digest(plan);_write(bundle/'plan.private.json',plan)
        for path in payload.rglob('*'):path.chmod(0o500 if path.is_dir() else 0o400)
        payload.chmod(0o500)
        return plan
    except CloudLaunchError:raise
    except Exception:raise CloudLaunchError('bundle_prepare_failed') from None


def artifacts_complete(directory, expected_bindings=None):
    try:
        root=Path(directory);completion=_read(root/'completion.json');report=_read(root/'report.json')
        if completion['status']!='complete' or report['status']!='complete' or report['task']!=TASK:
            return False
        if completion['bindings']!=report['bindings'] or report.get('checkpoint_reload_verified') is not True:
            return False
        bindings=report['bindings']
        if (bindings.get('task')!=TASK or bindings.get('codebook_commit')!=CODEBOOK
            or report.get('mode') not in ('binary','multilabel') or bindings.get('mode')!=report['mode']
            or bindings.get('target_mapping_digest')!=_digest({'normal':0,'pathology_codes':[1,2,3,4,5,6]})
            or any(not isinstance(bindings.get(k),str) or not re.fullmatch('[0-9a-f]{64}',bindings[k]) for k in ('input_digest','split_digest'))):
            return False
        if expected_bindings is not None and bindings!=expected_bindings:return False
        digests=completion['artifact_digests']
        required={'checkpoint.private.pt','predictions.private.json','split.private.json','manifest.private.json','report.json'}
        if not required<=set(digests):return False
        for name,digest in digests.items():
            path=root/name
            if Path(name).name!=name or path.is_symlink() or not path.is_file() or _hash_file(path)!=digest:return False
        return True
    except Exception:return False


def worker(config_path,max_runtime_seconds):
    config_path=Path(config_path).resolve();root=Path(__file__).resolve().parents[1]
    results=config_path.parent.parent/'results';results.mkdir(mode=0o700,exist_ok=True)
    if type(max_runtime_seconds)!=int or not 1<=max_runtime_seconds<=14400:raise CloudLaunchError('invalid_runtime')
    expected=preflight(load_config(config_path))['bindings']
    status,code='failed',2
    with (results/'training.private.log').open('x') as log:
        try:
            process=subprocess.Popen([sys.executable,'-m','tools.run_osseous_research','--config',str(config_path),'--train'],
                                     cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                     env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
            try:
                code=process.wait(timeout=max_runtime_seconds)
                status='complete' if code==0 and artifacts_complete(results/'research',expected) else 'failed'
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(process.pid,signal.SIGKILL)
                code=process.wait();status='timeout'
        except (OSError,ValueError):pass
    _write(results/'training-status.json',{'schema_version':1,'status':status,'exit_code':code})
    return {'training_status':status,'training_exit_code':code}


def results(bundle_dir,destination,cli_path='datasphere'):
    bundle=Path(bundle_dir).resolve();destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(bundle) or bundle.is_relative_to(destination):
        raise CloudLaunchError('invalid_results_destination')
    plan=_read(bundle/'plan.private.json');_check_bundle(bundle,plan['plan_sha256'])
    job=_job(bundle);destination.mkdir(mode=0o700,parents=True)
    _invoke(_cli_prefix(cli_path,bundle)+['project','job','download-files','--id',job,'--output-dir',str(destination),'--with-logs'],cwd=bundle)
    status=_read(destination/'results/training-status.json')
    ready=status.get('status')=='complete' and status.get('exit_code')==0 and artifacts_complete(destination/'results/research',plan['bindings'])
    return {'results_ready':ready,'training_status':status.get('status'),'artifact_status':'complete' if ready else 'partial'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--config',required=True);prep.add_argument('--bundle',required=True);prep.add_argument('--project-id',required=True)
    confirm=sub.add_parser('confirm');confirm.add_argument('--bundle',required=True);confirm.add_argument('--digest',required=True);confirm.add_argument('--cli-path',default='datasphere')
    work=sub.add_parser('worker');work.add_argument('--config',required=True);work.add_argument('--max-runtime-seconds',type=int,required=True)
    for action in ('status','cancel','results','reconcile'):
        lifecycle=sub.add_parser(action);lifecycle.add_argument('--bundle',required=True);lifecycle.add_argument('--cli-path',default='datasphere')
        if action=='results':lifecycle.add_argument('--destination',required=True)
        if action=='reconcile':
            lifecycle.add_argument('--job-id',required=True);lifecycle.add_argument('--operation-id',required=True)
            lifecycle.add_argument('--acknowledge-match',action='store_true',required=True)
    args=parser.parse_args()
    try:
        if args.action=='prepare':
            plan=prepare_bundle(args.config,args.bundle,project_id=args.project_id)
            result={k:plan[k] for k in ('plan_sha256','task','upload_bytes','training_runtime_seconds','training_window_compute_estimate','currency')}
        elif args.action=='confirm':result=confirm_bundle(args.bundle,args.digest,cli_path=args.cli_path)
        elif args.action=='status':result=job_status(args.bundle,cli_path=args.cli_path)
        elif args.action=='cancel':result=cancel_job(args.bundle,cli_path=args.cli_path)
        elif args.action=='results':result=results(args.bundle,args.destination,cli_path=args.cli_path)
        elif args.action=='reconcile':result=reconcile_submission(args.bundle,args.job_id,args.operation_id,acknowledge_match=args.acknowledge_match,cli_path=args.cli_path)
        else:result=worker(args.config,args.max_runtime_seconds)
        print(json.dumps(result,sort_keys=True));return 0
    except Exception as error:
        print(json.dumps({'ready':False,'code':str(error) if isinstance(error,CloudLaunchError) else 'cloud_preparation_failed'}));return 1

if __name__=='__main__':
    raise SystemExit(main())
