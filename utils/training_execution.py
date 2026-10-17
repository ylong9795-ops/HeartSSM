"""Runtime-only throughput controls; network/optimizer/checkpoint math unchanged."""
import json
from pathlib import Path

import torch


def configure_execution(model, slicer, *, objective_mode, memory_profile='default',
                         cnn_chunk_size=None, checkpoint_cnn=None, slice_cell_chunk=None,
                         slice_query_chunk=None, boundary_only=None):
    """Spend more activation/temporary memory on larger parallel GPU work.

    These attributes are execution controls, not trainable architecture. Do
    not mutate encoder_config/config dictionaries: old checkpoints still load
    strictly. Batch size is deliberately handled separately by the caller.
    Boundary-only is safe solely for grouped_gt_boundary_v1, which consumes
    cell_terms.boundary_mm and does not use probability Dice/BCE objectives.
    """
    if memory_profile not in ('default', 'throughput'):
        raise ValueError('Unknown GPU memory profile')
    fast = memory_profile == 'throughput'
    cnn = cnn_chunk_size if cnn_chunk_size is not None else (72 if fast else model.encoder.cnn_chunk_size)
    cells = slice_cell_chunk if slice_cell_chunk is not None else (64 if fast else slicer.cell_chunk_size)
    queries = slice_query_chunk if slice_query_chunk is not None else (1024 if fast else slicer.chunk_size)
    checkpoint = checkpoint_cnn if checkpoint_cnn is not None else (False if fast else model.encoder.checkpoint_cnn)
    boundary = boundary_only if boundary_only is not None else (True if fast else slicer.boundary_only)
    if any(type(n) is not int or n < 1 for n in (cnn, cells, queries)) or type(checkpoint) is not bool or type(boundary) is not bool:
        raise ValueError('Positive integer execution chunks and boolean checkpoint/boundary modes required')
    if boundary and (objective_mode != 'grouped_gt_boundary_v1' or slicer.boundary_objective != 'paired_boundary_distance_v1'):
        raise ValueError('Boundary-only speed path requires the grouped GT geometric boundary objective')
    model.encoder.cnn_chunk_size, model.encoder.checkpoint_cnn = cnn, checkpoint
    slicer.cell_chunk_size, slicer.chunk_size, slicer.boundary_only = cells, queries, boundary
    return dict(memory_profile=memory_profile, cnn_chunk_size=cnn, checkpoint_cnn=checkpoint,
                slice_cell_chunk=cells, slice_query_chunk=queries, boundary_only=boundary,
                objective_unchanged=True)


class StepProfiler:
    """Optional bounded CPU/CUDA trace for real forward/backward/optimizer work."""
    def __init__(self, steps, output_dir, device):
        if type(steps) is not int or steps < 0:
            raise ValueError('Nonnegative profiler step budget required')
        self.steps, self.count, self.profiler = steps, 0, None
        self.output_dir = Path(output_dir)
        if steps:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.device(device).type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)
            self.profiler.start()

    def step(self):
        if self.profiler is not None:
            self.profiler.step()
            self.count += 1
            if self.count >= self.steps:
                self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.profiler is not None:
            self.profiler.stop()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.profiler.export_chrome_trace(str(self.output_dir/'training_trace.json'))
            (self.output_dir/'training_profile.txt').write_text(self.profiler.key_averages().table(
                sort_by='self_cpu_time_total', row_limit=50))
            (self.output_dir/'profile_scope.json').write_text(json.dumps(dict(steps=self.count,
                scope='profiling adds overhead; use an unprofiled run for throughput comparisons'))+'\n')
            self.profiler = None
