#!/usr/bin/env python3
"""Private, resumable streaming preparation; never retain the full public TAR."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.tmj_od3d_images import (
    CODEBOOK_COMMIT,
    SOURCE_DOI,
    TASK,
    PreparationError,
    digest_file,
    prepare_patient,
)
from training.tmj_od3d_inputs import load_metadata

ARCHIVE_URL='https://china.scidb.cn/download?fileId=b83163bb83cdf7dc4a278ded176debb2'
ARCHIVE_BYTES=74370498560
METADATA_URL='https://china.scidb.cn/download?fileId=1837fcc6b95cec32f86e965dbf45e8db'
METADATA_SHA256='0fc2a3393b4931e72b77484a1f64f02aa7b6228d331605065a0732ee6664e8c6'
RANGE_CHUNK_BYTES=8*1024**2
MAX_PREPARED_BYTES=5*1024**3

class RangeReader:
    """Four ordered 8 MiB ranges, no seek/full archive and bounded read output."""
    def __init__(self,url,start,end,total,chunk_size=None,opener=None):
        chunk_size=RANGE_CHUNK_BYTES if chunk_size is None else chunk_size
        if any(type(v)!=int for v in (start,end,total,chunk_size)) or not 0<=start<=end<total or not 1<=chunk_size<=8*1024**2:
            raise PreparationError('invalid_download_budget')
        self.url,self.next,self.end,self.total,self.chunk_size=url,start,end,total,chunk_size
        self.opener=opener or urllib.request.urlopen
        self.stopped=threading.Event();self.pool=ThreadPoolExecutor(max_workers=4)
        self.pending=deque();self.buffer=b'';self.position=0;self._fill()

    def _fill(self):
        while len(self.pending)<4 and self.next<=self.end and not self.stopped.is_set():
            lo=self.next;hi=min(self.end,lo+self.chunk_size-1);self.next=hi+1
            self.pending.append(self.pool.submit(self._fetch,lo,hi))

    def _wait_retry(self,seconds):
        return self.stopped.wait(seconds)

    def _fetch(self,lo,hi):
        code='source_range_failed'
        for attempt in range(3):
            if self.stopped.is_set():break
            request=urllib.request.Request(self.url,headers={'Range':f'bytes={lo}-{hi}','Accept-Encoding':'identity'})
            try:
                with self.opener(request,timeout=30) as response:
                    if (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{self.total}' or
                        response.headers.get('Content-Length')!=str(hi-lo+1) or response.headers.get('Content-Encoding','identity')!='identity'):
                        raise PreparationError('source_range_not_confirmed')
                    data=response.read(hi-lo+2)
                    if len(data)!=hi-lo+1:raise PreparationError('truncated_range_response')
                    return data
            except urllib.error.HTTPError as error:
                code='source_rate_limited' if error.code==429 else 'source_range_failed'
                retry_after=(error.headers or {}).get('Retry-After','')
                error.close()
                if code=='source_rate_limited' and attempt<2:
                    delay=30*(attempt+1)
                    try:
                        if re.fullmatch(r'[0-9]{1,9}',retry_after): delay=int(retry_after)
                        else:
                            stamp=parsedate_to_datetime(retry_after)
                            if stamp.tzinfo is not None: delay=(stamp-datetime.now(timezone.utc)).total_seconds()
                    except (ValueError,TypeError,OverflowError): pass
                    if self._wait_retry(max(1,min(60,delay))): break
            except Exception as error:
                code=str(error) if isinstance(error,PreparationError) else 'source_range_failed' 
        raise PreparationError(code) from None

    def read(self,n):
        if self.stopped.is_set():raise PreparationError('range_reader_closed')
        if type(n)!=int or n<0:raise PreparationError('invalid_read_size')
        n=min(n,self.chunk_size);parts=[];count=0
        while count<n:
            if self.position==len(self.buffer):
                self.buffer=b'';self.position=0
                if not self.pending:break
                self.buffer=self.pending.popleft().result();self._fill()
            size=min(n-count,len(self.buffer)-self.position)
            parts.append(memoryview(self.buffer)[self.position:self.position+size]);self.position+=size;count+=size
        return b''.join(parts)

    def close(self):
        self.stopped.set()
        for future in self.pending:future.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True);self.pending.clear();self.buffer=b''

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def _orphans(root,state):
    if (root/'data').is_symlink():raise PreparationError('unsafe_prepared_orphan')
    known={record['crop_path'] for record in state['records']};orphans=[]
    for path in (root/'data').glob('*'):
        if str(path.relative_to(root)) in known:continue
        if path.is_symlink() or not path.is_file() or not re.fullmatch(r'[0-9a-f]{64}-[LR]\.npz',path.name):
            raise PreparationError('unsafe_prepared_orphan')
        orphans.append(path)
    return orphans

def reconcile_stale_lock(output_dir):
    """Explicit action; live/unknown owner refused, absent lock acquired exclusively."""
    root=Path(output_dir).resolve();lock=root/'preparation.lock';owned=False
    try:
        try:fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError:
            try:
                info=json.loads(lock.read_text());pid=info['pid']
                if type(pid)!=int or pid<=0:raise ValueError
            except (OSError,ValueError,TypeError,KeyError):raise PreparationError('unverifiable_preparation_lock') from None
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            except OSError:raise PreparationError('preparation_already_locked') from None
            else:raise PreparationError('preparation_already_locked')
        else:
            owned=True
            with os.fdopen(fd,'w') as stream:json.dump({'pid':os.getpid()},stream)
        state_path=root/'preparation.private.json';state={'records':[]}
        if state_path.exists():
            try:
                state=json.loads(state_path.read_text())
                if not isinstance(state,dict) or state.get('schema_version')!=1 or state.get('task')!=TASK or state.get('metadata_sha256')!=METADATA_SHA256 or not isinstance(state.get('records'),list):raise ValueError
                for record in state['records']:
                    path=root/record['crop_path']
                    if not path.resolve().is_relative_to(root) or path.is_symlink() or digest_file(path)!=record['crop_sha256']:raise ValueError
            except (OSError,ValueError,TypeError,KeyError):raise PreparationError('invalid_preparation_state') from None
        orphans=_orphans(root,state);previous=list(root.glob('orphan-quarantine-*/*.npz'))
        if any(path.is_symlink() or not path.resolve().is_relative_to(root) for path in previous):raise PreparationError('unsafe_prepared_orphan')
        if sum(path.stat().st_size for path in [*orphans,*previous])>MAX_PREPARED_BYTES:raise PreparationError('prepared_storage_limit')
        if orphans:
            quarantine=Path(tempfile.mkdtemp(prefix='orphan-quarantine-',dir=root))
            for path in orphans:path.replace(quarantine/path.name)
        print(json.dumps({'quarantined_orphans':len(orphans)}),flush=True)
        lock.unlink();owned=False
    finally:
        if owned:lock.unlink(missing_ok=True)

def _write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
    temporary.chmod(0o600);temporary.replace(path)

def _publish_index(root,state):
    index={k:v for k,v in state.items() if k not in ('processed_patients','next_offset','patient_receipts')}
    index['patient_count']=len(state['processed_patients'])
    _write(root/'index.private.json',index)

def _existing_dataset(root,state):
    total=0
    for record in state['records']:
        relative=Path(record['crop_path']);path=root/relative
        if relative.is_absolute() or not path.resolve().is_relative_to(root) or not path.is_file() or path.is_symlink() or digest_file(path)!=record['crop_sha256']:
            raise PreparationError('prepared_artifact_changed')
        total+=path.stat().st_size
    if _orphans(root,state):raise PreparationError('prepared_orphan_artifact')
    if total>MAX_PREPARED_BYTES:raise PreparationError('prepared_storage_limit')
    return total

def prepare_archive(output_dir,metadata_path=None,max_patients=None,max_download_bytes=ARCHIVE_BYTES,
                    slice_count=16,image_size=96):
    if type(max_download_bytes)!=int or max_download_bytes<=0 or (max_patients is not None and (type(max_patients)!=int or max_patients<1)):
        raise PreparationError('invalid_download_budget')
    os.umask(0o077)
    root=Path(output_dir).resolve();root.mkdir(mode=0o700,parents=True,exist_ok=True)
    lock=root/'preparation.lock'
    try:fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:raise PreparationError('preparation_already_locked') from None
    with os.fdopen(fd,'w') as stream:json.dump({'pid':os.getpid()},stream)
    cache=root/'raw-cache'
    try:
        state_path=root/'preparation.private.json'
        metadata_file=root/'metadata.private.csv'
        if metadata_path:
            if not metadata_file.exists():shutil.copyfile(metadata_path,metadata_file)
        elif not metadata_file.exists():
            with urllib.request.urlopen(METADATA_URL,timeout=30) as response:
                raw=response.read(100001)
            if len(raw)>100000:raise PreparationError('metadata_byte_limit')
            metadata_file.write_bytes(raw)
        if digest_file(metadata_file)!=METADATA_SHA256:raise PreparationError('source_metadata_changed')
        metadata=load_metadata(metadata_file)
        preprocessing={'slice_count':slice_count,'image_size':image_size,'intensity':'per-slice-p1-p99'}
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state['preprocessing']!=preprocessing:raise PreparationError('preprocessing_changed')
            prepared_bytes=_existing_dataset(root,state)
        else:
            state={'schema_version':1,'task':TASK,'source_doi':SOURCE_DOI,'codebook_commit':CODEBOOK_COMMIT,
                   'metadata_sha256':METADATA_SHA256,'preprocessing':preprocessing,'next_offset':0,
                   'records':[],'processed_patients':[],'exclusions':{},'failures':{},'complete':False}
        state.setdefault('patient_receipts',[])
        if not state_path.exists():prepared_bytes=_existing_dataset(root,state)
        if state['complete'] or state['next_offset']==ARCHIVE_BYTES:
            if state['next_offset']!=ARCHIVE_BYTES or set(state['processed_patients'])!=set(metadata):
                raise PreparationError('source_patient_coverage_mismatch')
            state['complete']=True
            _write(state_path,state)
            _publish_index(root,state)
            return state
        if max_patients is not None and len(state['processed_patients'])>=max_patients:
            _publish_index(root,state)
            return state
        if cache.exists():shutil.rmtree(cache)
        cache.mkdir(mode=0o700)
        base=state['next_offset'];end=min(ARCHIVE_BYTES-1,base+max_download_bytes-1)
        if base>end:raise PreparationError('invalid_download_budget')
        current=None;current_start=None;folder=None;patient_bytes=0;seen=set();started=time.monotonic()
        def finish(next_offset):
            nonlocal current,current_start,folder,patient_bytes,prepared_bytes
            if current is None:return
            identity=current;receipt_code='ok';side_count=0
            if identity not in metadata:raise PreparationError('unknown_source_patient')
            if identity in state['processed_patients']:raise PreparationError('repeated_source_patient')
            try:
                if len(list(folder.glob('*.dcm')))!=metadata[identity]['slice_count']:
                    raise PreparationError('csv_dicom_count_mismatch')
                records,reasons=prepare_patient(folder,metadata[identity],root,slice_count,image_size)
                added_bytes=sum((root/record['crop_path']).stat().st_size for record in records)
                if prepared_bytes+added_bytes>MAX_PREPARED_BYTES:
                    for record in records:(root/record['crop_path']).unlink(missing_ok=True)
                    raise PreparationError('prepared_storage_limit')
                side_count=len(records)
                prepared_bytes+=added_bytes
                state['records'].extend(records)
                for reason,count in reasons.items():state['exclusions'][reason]=state['exclusions'].get(reason,0)+count
            except PreparationError as error:
                if str(error) in ('prepared_crop_exists','prepared_storage_limit'):raise
                code=str(error);receipt_code=code
                # Remove incomplete crops so failed patient has no publishable partial side.
                token=hashlib.sha256((SOURCE_DOI+'\0'+identity).encode()).hexdigest()
                for side in ('L','R'):(root/'data'/f'{token}-{side}.npz').unlink(missing_ok=True)
                state['failures'][code]=state['failures'].get(code,0)+1
            state['patient_receipts'].append({'patient_id':hashlib.sha256((SOURCE_DOI+'\0'+identity).encode()).hexdigest(),
                'tar_start':current_start,'tar_end':next_offset,'status':'prepared' if receipt_code=='ok' else 'failed',
                'code':receipt_code,'side_count':side_count})
            state['processed_patients'].append(identity)
            state['next_offset']=next_offset
            _write(state_path,state)
            _publish_index(root,state)
            shutil.rmtree(folder);current=None;current_start=None;folder=None;patient_bytes=0
            print(json.dumps({'processed':len(state['processed_patients']),'sides':len(state['records']),
                              'failures':state['failures'],'seconds':round(time.monotonic()-started)}),flush=True)
        with RangeReader(ARCHIVE_URL,base,end,ARCHIVE_BYTES) as response:
            with tarfile.open(fileobj=response,mode='r|') as archive:
                for member in archive:
                    path=PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts or '\\' in member.name or member.issym() or member.islnk():
                        raise PreparationError('unsafe_archive_member')
                    if not path.parts or path.parts[0]!='tmj' or not (member.isfile() or member.isdir()):raise PreparationError('unsafe_archive_member')
                    if member.name.rstrip('/')!=str(path) or len(path.parts)>3 or (member.isdir() and len(path.parts)>2) or (member.isfile() and len(path.parts)!=3):raise PreparationError('unsupported_archive_layout')
                    if str(path) in seen:raise PreparationError('duplicate_archive_file')
                    seen.add(str(path))
                    if len(path.parts)<2:continue
                    patient=path.parts[1]
                    if current is not None and patient!=current:
                        finish(base+member.offset)
                        if max_patients and len(state['processed_patients'])>=max_patients:return state
                    if current is None:
                        current=patient;current_start=base+member.offset;folder=cache/patient;folder.mkdir(mode=0o700)
                    if not member.isfile():
                        if not member.isdir():raise PreparationError('unsafe_archive_member')
                        continue
                    name=path.name
                    if member.size>8*1024**2:raise PreparationError('source_file_byte_limit')
                    patient_bytes+=member.size
                    if patient_bytes>1024**3:raise PreparationError('source_patient_byte_limit')
                    if Path(name).suffix.lower() not in ('.dcm','.json'):continue
                    destination=folder/name
                    with archive.extractfile(member) as src,destination.open('xb') as dest:
                        shutil.copyfileobj(src,dest,1024**2)
                    if destination.stat().st_size!=member.size:raise PreparationError('truncated_source_file')
        # EOF is authoritative only when the requested range reaches archive end.
        if end!=ARCHIVE_BYTES-1:raise PreparationError('download_budget_exhausted')
        finish(ARCHIVE_BYTES)
        if set(state['processed_patients'])!=set(metadata):raise PreparationError('source_patient_coverage_mismatch')
        state['complete']=True
        _write(state_path,state)
        _publish_index(root,state)
        return state
    except (tarfile.TarError,EOFError):
        raise PreparationError('download_budget_exhausted' if end<ARCHIVE_BYTES-1 else 'truncated_source_archive') from None
    finally:
        if cache.exists():shutil.rmtree(cache)
        lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--metadata')
    parser.add_argument('--max-patients',type=int);parser.add_argument('--max-download-gib',type=float,default=70)
    parser.add_argument('--reconcile-stale-lock',action='store_true')
    parser.add_argument('--slice-count',type=int,default=16);parser.add_argument('--image-size',type=int,default=96)
    args=parser.parse_args()
    try:
        if args.reconcile_stale_lock:reconcile_stale_lock(args.output_dir)
        state=prepare_archive(args.output_dir,args.metadata,args.max_patients,
                              int(args.max_download_gib*1024**3),args.slice_count,args.image_size)
        print(json.dumps({'complete':state['complete'],'patients':len(state['processed_patients']),
                          'sides':len(state['records']),'failures':state['failures'],'exclusions':state['exclusions']}))
    except Exception as error:
        code=str(error) if isinstance(error,PreparationError) else 'preparation_interrupted'
        print(json.dumps({'ready':False,'code':code}));return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
