"""Persistent lossless Stage S samples and bounded deterministic CPU prefetch.

Cache values are exactly Synthetic4DDataset samples (24-phase observations and
GT); no geometry/SDF/loss is reimplemented. All filesystem writes stay in ROOT.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid
import cv2
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from data.synthetic_4d import _sha256

ROOT = Path(__file__).resolve().parents[1]


def workspace_path(path):
    path = Path(path)
    path = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError('Cache/output/runtime files must stay inside workspace')
    return path


def configure_runtime_temp(path):
    path = workspace_path(path)
    # AF_UNIX addresses are limited to108 bytes; multiprocessing appends names.
    if len(os.fsencode(path))>65:path=ROOT/'outputs/.runtime'
    path.mkdir(parents=True,exist_ok=True)
    # multiprocessing resource_sharer and third-party libraries use tempfile.
    for key in ('TMPDIR','TMP','TEMP'):os.environ[key] = str(path)
    tempfile.tempdir = str(path)
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def source_stamp(row):
    result = {}
    for key in ('path','qc_path'):
        stat = Path(row[key]).stat()
        result[key] = [stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
    return result


class CachedStageSamples(Dataset):
    """One lossless float32 SDF build per identity; atomic, process-safe writes.

    Hash raw source/QC once per accessing process/stamp, then stat-check hits.
    Cache bytes have a SHA256 sidecar; verify once per process/file stamp.
    Corrupt entries fail closed; source/config changes cannot silently reuse one.
    """
    def __init__(self, dataset, indices, cache_dir):
        if not dataset.verify:raise ValueError('Persistent cache requires source verification')
        self.dataset,self.indices = dataset,list(indices)
        self.root = workspace_path(cache_dir);self.root.mkdir(parents=True,exist_ok=True)
        transform=dataset.mask_transform
        self.identity=dict(version='stage-s-sample-cache-v1',
            manifest=dataset.manifest_sha256,asset=dataset.asset_sha256,
            template=dataset.template_sha256,norm_to_mm=dataset.norm_to_mm,
            frames=dataset.expected_phases,reference_frame=dataset.reference_frame,
            remove_duplicate_endpoint=dataset.remove_duplicate_endpoint,
            transform=dict(options=transform.options,use_sdf=transform.use_sdf,
                           clip_mm=transform.sdf_clip_mm,normalize=transform.sdf_normalize),
            implementation={name:_sha256(ROOT/name) for name in
                            ('data/synthetic_4d.py','data/mask_transform.py','utils/staged_data.py')},
            versions=dict(torch=str(torch.__version__),numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__))
        self.namespace=self.root/digest(self.identity);self.namespace.mkdir(exist_ok=True)
        self._verified_sources={};self._verified_cache={}

    def __len__(self):return len(self.indices)

    def entry(self,index):
        row=self.dataset.sources[self.indices[index]]
        key=digest(dict(identity=self.identity,source=row))
        return row,self.namespace/(key+'.pth')

    def __getitem__(self,index):
        row,path=self.entry(index);stamp=source_stamp(row);source_key=str(row['path'])
        if self._verified_sources.get(source_key)!=stamp:
            for filekey,hashkey in (('path','sha256'),('qc_path','qc_sha256')):
                if _sha256(row[filekey])!=row[hashkey]:raise ValueError('Modified source file: '+str(row[filekey]))
            self._verified_sources[source_key]=stamp
        # Exclusive lock also prevents reading a mismatched data/sidecar pair.
        with path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            side=path.with_suffix('.json')
            if not path.exists() or not side.exists():
                sample=self.dataset[self.indices[index]]
                if source_stamp(row)!=stamp:raise ValueError('Source changed during preprocessing')
                temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
                meta_temp=side.with_name(side.name+'.'+uuid.uuid4().hex+'.partial')
                try:
                    torch.save(sample,temp)
                    meta_temp.write_text(json.dumps(dict(key=path.stem,sha256=_sha256(temp),source_stamp=stamp)))
                    os.replace(temp,path);os.replace(meta_temp,side)
                finally:
                    temp.unlink(missing_ok=True);meta_temp.unlink(missing_ok=True)
                return sample
            stat=path.stat();meta_stat=side.stat()
            signature=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,meta_stat.st_mtime_ns)
            if self._verified_cache.get(str(path))!=signature:
                meta=json.loads(side.read_text())
                if meta['key']!=path.stem or _sha256(path)!=meta['sha256']:
                    raise ValueError('Corrupt Stage S cache: '+str(path))
                self._verified_cache[str(path)]=signature
            sample=torch.load(path,map_location='cpu',weights_only=True)
            if sample['sample_sha256']!=row['sha256'] or sample['template_sha256']!=self.dataset.template_sha256:
                raise ValueError('Cached sample identity differs')
            return sample


class EpochOrder(Sampler):
    """Exactly the original randperm(seed+epoch), including mid-epoch resume."""
    def __init__(self,count,seed):self.count,self.seed,self.epoch,self.offset=count,seed,0,0
    def set_position(self,epoch,offset):self.epoch,self.offset=epoch,offset
    def __iter__(self):
        return iter(torch.randperm(self.count,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()[self.offset:])
    def __len__(self):return self.count-self.offset


def worker_init(_):
    from multiprocessing import util, resource_sharer
    # Close descriptor-sharing sockets before NFS temp-directory cleanup.
    util.Finalize(None,resource_sharer.stop,exitpriority=10)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    configure_runtime_temp(os.environ['TMPDIR'])


def make_loader(samples,*,workers=4,prefetch=2,seed=2026,sampler=None,collate_fn=None,pin_memory=True):
    if workers<0 or prefetch<1:raise ValueError('Invalid loader workers/prefetch')
    kwargs=dict(dataset=samples,batch_size=1,sampler=sampler,num_workers=workers,
                pin_memory=pin_memory,collate_fn=collate_fn,
                generator=torch.Generator().manual_seed(seed))
    if workers:
        kwargs.update(persistent_workers=True,prefetch_factor=prefetch,
                      multiprocessing_context='spawn',worker_init_fn=worker_init)
    return DataLoader(**kwargs)
