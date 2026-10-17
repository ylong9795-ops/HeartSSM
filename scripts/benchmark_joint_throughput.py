"""Reproducible geometric slice microbenchmark; no checkpoint or dataset writes."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import trimesh

from data.mask_transform import CHAMBERS
from utils.slice_supervision import MeshSliceSupervisor, exact_section_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--subdivisions', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.frames, args.repeats) < 1 or args.subdivisions < 1:
        parser.error('Positive frame/repeat/subdivision counts required')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    mesh = trimesh.creation.icosphere(subdivisions=args.subdivisions, radius=6.)
    mesh.apply_translation([.137, -.253, .31])
    v = torch.tensor(mesh.vertices.copy(), dtype=torch.float64, device=device)
    f = {c: torch.tensor(mesh.faces.copy(), device=device) for c in CHAMBERS}
    ranges = torch.tensor([[i*len(v), (i+1)*len(v)] for i in range(4)], device=device)
    gt = v.repeat(4, 1)[None, None].repeat(1, args.frames, 1, 1)
    mask = exact_section_mask(mesh.vertices, mesh.faces, np.eye(3), np.zeros(3), [.4, .6], (224, 224))
    masks = torch.tensor(mask, device=device).expand(1, args.frames, 3, 4, 224, 224)
    annotated = torch.tensor([[1,1,0,0], [1,1,0,0], [1,1,1,1]], dtype=torch.bool, device=device)[None]
    obs = dict(masks=masks, native_masks=masks, vertices_gt=gt,
        pixel_spacing=v.new_tensor([.4, .6]).expand(1,args.frames,3,2),
        native_pixel_spacing=v.new_tensor([.4, .6]).expand(1,args.frames,3,2),
        view_present=torch.ones(1,3,dtype=torch.bool,device=device), annotated=annotated,
        pose_gt=dict(probe_R=torch.eye(3,device=device,dtype=v.dtype).expand(1,args.frames,3,3,3),
                     probe_t=v.new_zeros(1,args.frames,3,3)),
        phases=torch.arange(args.frames,device=device,dtype=v.dtype)[None]/args.frames)
    reports, values, gradients = {}, [], []
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for name, fast, cells, queries in (('original',False,16,256), ('throughput',True,64,1024)):
        slicer = MeshSliceSupervisor(ranges,f,1.,target_mode='paired_soft_geometry_v1',supervision_grid='native_grid',
            boundary_objective='paired_boundary_distance_v1',boundary_sample_step_mm=.5,
            execution_backend='batched',cell_chunk_size=cells,chunk_size=queries,boundary_only=fast).to(device)
        elapsed = []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for repetition in range(args.repeats+1):
            pred = (gt+v.new_tensor([.7,-.4,.1])).requires_grad_()
            sync()
            start = time.perf_counter()
            result = slicer(pred, obs)
            # This is the boundary vector actually consumed by grouped GT loss.
            result['cell_terms']['boundary_mm'].sum().backward()
            sync()
            if repetition:
                elapsed.append(time.perf_counter()-start)
        values.append(result['cell_terms']['boundary_mm'].detach())
        gradients.append(pred.grad.detach())
        reports[name] = dict(median_seconds=statistics.median(elapsed), samples_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        del result, pred, slicer
    torch.testing.assert_close(*values, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(*gradients, rtol=1e-8, atol=1e-9)
    report = dict(scope='synthetic slice forward/backward microbenchmark; not epoch or end-to-end speed',
        device=str(device),frames=args.frames,valid_cells=args.frames*8,triangle_count=len(mesh.faces),
        results=reports, speedup=reports['original']['median_seconds']/reports['throughput']['median_seconds'],
        max_boundary_abs_error=float((values[0]-values[1]).abs().max()),
        max_gradient_abs_error=float((gradients[0]-gradients[1]).abs().max()))
    rendered = json.dumps(report,indent=2)+'\n'
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered)


if __name__ == '__main__':
    main()
