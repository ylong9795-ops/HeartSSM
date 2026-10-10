"""Bounded CPU input/reference caches and parallel original-label evaluation."""
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate


def reference_signature(arrays, identity=None):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode())
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def recursive_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: recursive_numpy(v) for k, v in value.items()}
    return value


def recursive_tensors(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {k: recursive_tensors(v) for k, v in value.items()}
    return value


class CPUCaseEvaluator:
    def __init__(self, geometry, limit):
        self.model = SimpleNamespace(**recursive_tensors(geometry))
        self.limit = limit
        self.cases = OrderedDict()

    def evaluate(self, key, prediction, source):
        from train_neural_ffd import exact_metrics, surface_metrics
        from data.mask_transform import CHAMBERS
        if key not in self.cases:
            if source is None:
                raise ValueError('Missing original evaluation reference')
            self.cases[key] = (recursive_tensors(source), {}, {})
        batch, reference, surfaces = self.cases[key]
        self.cases.move_to_end(key)
        while len(self.cases) > self.limit:
            self.cases.popitem(last=False)
        reused = len(reference.get('cells', {}))
        exact = exact_metrics(self.model, torch.from_numpy(prediction), batch, reference)
        values = {}
        for chamber, (left, right) in zip(CHAMBERS, self.model.chamber_ranges.tolist()):
            values[chamber] = surface_metrics(prediction[0, 0, left:right],
                batch['vertices_gt'][0, 0, left:right].numpy(),
                self.model.faces[chamber].numpy(), reference_cache=surfaces)
        return dict(exact=exact, surfaces=values, gt_cells_reused=reused,
                    gt_cells_verified=len(reference.get('cells', {}))-reused)


_worker = None
_thread_limit = None


def _initialize_worker(geometry, limit):
    global _worker, _thread_limit
    import cv2
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    _thread_limit = threadpool_limits(limits=1)
    _worker = CPUCaseEvaluator(geometry, limit)


def _evaluate_worker(key, prediction, source):
    return _worker.evaluate(key, prediction, source)


class ValidationRuntime:
    def __init__(self, workers=4, cache_cases=32):
        if workers < 0 or cache_cases < 1:
            raise ValueError('Nonnegative workers and positive cache capacity required')
        self.workers, self.cache_cases = workers, cache_cases
        self.max_pending = max(1, 2*workers)
        self.inputs = OrderedDict()
        self.lock = Lock()
        self.loader = ThreadPoolExecutor(max_workers=1)
        self.pools, self.known = [], []
        self.context = None

    def bind(self, model, dataset):
        from utils.slice_evaluation import source_renderer_identity
        geometry = dict(faces=recursive_numpy(model.faces),
                        chamber_ranges=recursive_numpy(model.chamber_ranges), norm_to_mm=model.norm_to_mm)
        context = reference_signature([geometry['chamber_ranges'], np.asarray(model.norm_to_mm),
            *geometry['faces'].values()], dict(renderer=source_renderer_identity(), dataset=id(dataset),
            manifest=getattr(dataset, 'manifest_sha256', None),
            transform=repr(vars(dataset.mask_transform)) if hasattr(dataset, 'mask_transform') else None,
            phases=getattr(dataset, 'expected_phases', None),
            reference=getattr(dataset, 'reference_frame', None),
            version=getattr(dataset, 'cache_version', None)))
        if context != self.context:
            for pool in self.pools:
                pool.shutdown(wait=True, cancel_futures=True)
            self.inputs.clear()
            self.limit = math.ceil(self.cache_cases/max(1, self.workers))
            self.known = [OrderedDict() for _ in range(max(1, self.workers))]
            self.pools = [ProcessPoolExecutor(max_workers=1,
                mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker,
                initargs=(geometry, self.limit)) for _ in range(self.workers)]
            self.inline = CPUCaseEvaluator(geometry, self.limit) if not self.workers else None
            self.context = context
        self.dataset = dataset
        self.input_hits = self.input_misses = 0

    def _stamp(self, index):
        if not hasattr(self.dataset, 'sources'):
            return None
        row = self.dataset.sources[index]
        files = []
        for field in ('path', 'qc_path'):
            stat = Path(row[field]).stat()
            files.append((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return repr(row), files

    def _load(self, index):
        stamp = repr(self._stamp(index))
        lookup = (index, stamp)
        with self.lock:
            if lookup in self.inputs:
                self.input_hits += 1
                self.inputs.move_to_end(lookup)
                return self.inputs[lookup]
        batch = default_collate([self.dataset[index]])
        source = recursive_numpy({key:batch[key] for key in
            ('vertices_gt', 'native_masks', 'native_pixel_spacing', 'pose_gt', 'view_present', 'annotated')})
        key = reference_signature([source['vertices_gt'], source['native_masks'],
            source['native_pixel_spacing'], source['view_present'], source['annotated'],
            *source['pose_gt'].values()], dict(context=self.context, index=index, stamp=stamp))
        item = batch, source, key
        with self.lock:
            self.input_misses += 1
            self.inputs[lookup] = item
            while len(self.inputs) > self.cache_cases:
                self.inputs.popitem(last=False)
        return item

    def batches(self, count):
        pending = OrderedDict()
        next_index = 0
        try:
            for index in range(count):
                while next_index < min(count, index+2):
                    pending[next_index] = self.loader.submit(self._load, next_index)
                    next_index += 1
                yield index, pending.pop(index).result()
        finally:
            for future in pending.values():
                if not future.cancel():
                    future.result()

    def submit(self, index, prediction, source, key):
        shard = index % max(1, self.workers)
        known = self.known[shard]
        reference = None if key in known else source
        known[key] = None
        known.move_to_end(key)
        while len(known) > self.limit:
            known.popitem(last=False)
        if self.workers:
            return self.pools[shard].submit(_evaluate_worker, key, prediction, reference)
        result = Future()
        try:
            result.set_result(self.inline.evaluate(key, prediction, reference))
        except Exception as error:
            result.set_exception(error)
        return result

    def close(self):
        self.loader.shutdown(wait=True, cancel_futures=True)
        for pool in self.pools:
            pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
