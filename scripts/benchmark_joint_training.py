"""Compare real-case Joint execution profiles without taking optimizer steps."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from train_neural_ffd import load_config, dataset_from_config, load_checkpoint, build_model, training_collate, to_device
from utils.neural_ffd_losses import NeuralFFDLoss
from utils.slice_supervision import MeshSliceSupervisor
from utils.training_execution import configure_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cnn-chunk-size', type=int, default=72)
    parser.add_argument('--with-refiner', action='store_true', help='Include physical observations, splat and the shared Refiner')
    parser.add_argument('--monitor-gpu-index', type=int, help='Optional PHYSICAL nvidia-smi index for utilisation sampling')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.repeats, args.cnn_chunk_size) < 1 or args.case_index < 0:
        parser.error('Positive batch/repeat/chunk and nonnegative case index required')
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    config, data_config = load_config(args.config)
    dataset = dataset_from_config(data_config)
    if args.case_index+args.batch_size > len(dataset):
        parser.error('Case batch outside TRAIN membership')
    batch = to_device(training_collate([dataset[i] for i in range(args.case_index, args.case_index+args.batch_size)], 'joint'), device)
    model = load_checkpoint(args.checkpoint, device)[0] if args.checkpoint else build_model(dataset, config).to(device)
    base_model = model
    if args.with_refiner:
        from net.joint_ffd_refiner import JointFFDRefiner
        model = JointFFDRefiner(base_model).to(device)
    model.train()
    criterion = NeuralFFDLoss(base_model.chamber_ranges, base_model.faces, **config['loss']).to(device)
    reports, losses, gradients = {}, [], []
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for profile in ('default', 'throughput'):
        slicer = MeshSliceSupervisor(base_model.chamber_ranges, base_model.faces, base_model.norm_to_mm,
            **config['slice'], execution_backend='batched', cell_chunk_size=16).to(device)
        execution = configure_execution(base_model, slicer, objective_mode=criterion.objective_mode,
            memory_profile=profile, cnn_chunk_size=6 if profile=='default' else args.cnn_chunk_size,
            checkpoint_cnn=profile=='default', slice_cell_chunk=16 if profile=='default' else 64,
            slice_query_chunk=256 if profile=='default' else 1024, boundary_only=profile=='throughput')
        timings = []
        monitor = None
        if args.monitor_gpu_index is not None:
            monitor = subprocess.Popen(['nvidia-smi', f'--id={args.monitor_gpu_index}',
                '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '--loop-ms=200'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for repeat in range(args.repeats+1):
            model.zero_grad(set_to_none=True)
            sync()
            start = time.perf_counter()
            decoded = model(batch, pose=batch['pose_gt']) if args.with_refiner else model(batch)
            sync()
            forward_end = time.perf_counter()
            slices = slicer(decoded['vertices_4d'], batch)
            sync()
            slice_end = time.perf_counter()
            result = criterion(decoded, batch['vertices_gt'], slice_result=slices,
                control_spacings_mm=base_model.control_spacings_mm,
                regional_spacing_mm=base_model.regional_spacing_mm if base_model.regional_shape else None,
                regional_motion_spacings_mm=base_model.regional_motion_spacings_mm if base_model.regional_motion else None, stage='joint')
            sync()
            loss_end = time.perf_counter()
            result['total'].backward()
            sync()
            end = time.perf_counter()
            if repeat:
                timings.append(dict(forward=forward_end-start, slice=slice_end-forward_end,
                    gt_loss=loss_end-slice_end, backward=end-loss_end, total=end-start))
        losses.append(float(result['total'].detach()))
        gradients.append({name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None})
        reports[profile] = dict(execution=execution, median_seconds={k: statistics.median(t[k] for t in timings) for k in timings[0]},
            samples=timings, peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        if monitor is not None:
            monitor.terminate()
            samples, _ = monitor.communicate(timeout=5)
            utilisation = [float(line.strip()) for line in samples.splitlines() if line.strip().isdigit()]
            reports[profile]['gpu_utilization_percent'] = dict(samples=utilisation,
                mean=statistics.mean(utilisation) if utilisation else None,
                scope='device-level sampled utilisation including warmup; includes other processes on this GPU')
        del decoded, slices, result, slicer
    gradient_error = max(float((gradients[0][k]-gradients[1][k]).abs().max()) for k in gradients[0])
    gradient_scale = max(float(g.abs().max()) for g in gradients[0].values())
    report = dict(scope='TRAIN fixed-case full forward + active GT/geometric objective + backward; no optimizer update, no epoch claim',
        device=str(device), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        case_index=args.case_index,batch_size=args.batch_size,with_refiner=args.with_refiner,case_ids=batch.get('case_id'),
        results=reports, losses=losses, loss_absolute_difference=abs(losses[0]-losses[1]),
        max_gradient_absolute_difference=gradient_error, max_gradient_relative_to_global_peak=gradient_error/max(gradient_scale,1e-12),
        speedup=reports['default']['median_seconds']['total']/reports['throughput']['median_seconds']['total'])
    report['parity_pass'] = (abs(losses[0]-losses[1]) <= 2e-5*max(1.,abs(losses[0]))
                            and gradient_error <= 1e-3*max(1e-6,gradient_scale))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rendered = json.dumps(report,indent=2)+'\n'
    args.output.write_text(rendered)
    print(rendered)
    if not report['parity_pass']:
        raise SystemExit('Numerical parity gate failed; inspect the saved report before using this profile')


if __name__ == '__main__':
    main()
