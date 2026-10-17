"""Initialize a NEW shared-observation Joint from a trained, frozen Base.

Short learning experiment by default: fixed TRAIN16, VAL8, 200 Adam updates.
No TEST, no patient optimization, no selection/tuning on validation outcomes.
Caches observations/Base/features only because Base remains frozen in eval.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import default_collate
import yaml

from data.synthetic_4d import _sha256
from net.joint_ffd_refiner import JointFFDRefiner
from train_neural_ffd import (data_identity, dataset_from_config, exact_metrics, load_checkpoint,
    reconstruction_metrics, repository_path, to_device)
from utils.joint_refiner_training import (DEFAULT_REGULARIZATION, _cpu_tree, configure_refiner_training,
    load_refiner_checkpoint, refiner_objective, sampled_composed_jacobians,
    save_refiner_checkpoint, tail_summary)
from utils.neural_ffd_losses import NeuralFFDLoss
from utils.slice_supervision import MeshSliceSupervisor
from utils.training_execution import configure_execution


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def cat_cached(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values)
    if isinstance(first, dict):
        return {k: cat_cached([v[k] for v in values]) for k in first}
    if first is None:
        return None
    raise ValueError('Only batch tensors/dicts/None may enter the frozen cache')


def dataset_case_ids(dataset):
    """Same stable fallback identity as Synthetic4DDataset._read_case."""
    return [s.get('case_id', f"{dataset.asset_sha256}:{s['candidate_id']}:{s['coordinate_sha256']}")
            for s in dataset.sources]


@torch.no_grad()
def prepare_cases(model, dataset, indices, device):
    model.eval()
    result = []
    for index in indices:
        sample = dataset[index]
        batch = to_device(default_collate([sample]), device)
        base = model.base_model(batch)
        splat = model.observation_features(base, batch, pose=batch['pose_gt'])
        keep = {k: base[k] for k in ('vertices_4d', 'vertices_ref', 'z_shape', 'z_motion', 'control_grids')}
        result.append(dict(sample=sample, base=_cpu_tree(keep), features=splat['features'].cpu(),
            recovery_cells=int(splat['cell_needs_recovery'].sum()),
            recovery_available_cells=int(splat['cell_recovery_available'].sum())))
        print(f"Prepared {sample['case_id']} ({len(result)}/{len(indices)})", flush=True)
    return result


def cached_batch(cases, indices, device):
    chosen = [cases[i] for i in indices]
    batch = to_device(default_collate([v['sample'] for v in chosen]), device)
    base = to_device(cat_cached([v['base'] for v in chosen]), device)
    splat = dict(features=torch.cat([v['features'] for v in chosen]).to(device))
    return batch, base, splat


@torch.no_grad()
def evaluate_cases(model, cases, device, output_dir, split, *, criterion, slicer, weights, jacobians=True):
    model.eval()
    all_rows = {'base': [], 'refiner': []}
    records, trajectories = [], []
    for index, case in enumerate(cases):
        batch, base, splat = cached_batch(cases, [index], device)
        refined = model.refine_base(base, splat, phases=batch['phases'])
        case_id = case['sample']['case_id']
        record = dict(case_id=case_id, recovery_cells=case['recovery_cells'],
                      recovery_available_cells=case['recovery_available_cells'])
        reference = {}
        base_loss_output = dict(base, control_regularization={k: base['vertices_4d'].sum()*0 for k in weights})
        for name, prediction_output in (('base', base_loss_output), ('refiner', refined)):
            prediction = prediction_output['vertices_4d']
            record[name] = reconstruction_metrics(prediction, batch['vertices_gt'],
                model.base_model.chamber_ranges, model.base_model.faces)
            objective = refiner_objective(model, prediction_output, batch, criterion, slicer, weights)
            record[name+'_objective'] = dict(total=float(objective['total']),
                supervised=float(objective['supervised']), residual_penalty=float(objective['residual_penalty']))
            exact = exact_metrics(model.base_model, prediction, batch, reference_cache=reference)
            for row in exact['cells']:
                all_rows[name].append(dict(row, case_id=case_id))
        if jacobians:
            record['sampled_composed_jacobian'] = sampled_composed_jacobians(model.base_model, base,
                refined['refinement']['controls_res_mm'])
        controls = refined['refinement']
        record['maximum_control_norm_mm'] = float(controls['controls_res_mm'].norm(dim=-1).max())
        record['dynamic_reference_max_abs_mm'] = float(controls['controls_dynamic_mm'][:, 0].abs().max())
        records.append(record)
        trajectories.append(dict(case_id=case_id, base=base['vertices_4d'][0].cpu().numpy(),
                                 refiner=refined['vertices_4d'][0].cpu().numpy()))
        print(f"Evaluated {split} {case_id} ({index+1}/{len(cases)})", flush=True)
    summary = dict(cases=records, tails={k: tail_summary(v) for k, v in all_rows.items()},
        scope='fixed full24 cases; GT Pose; original native labels; no TEST')
    keys = ('reference_vertex_rmse_mm', 'trajectory_vertex_rmse_mm', 'relative_motion_rmse_mm', 'volume_mae_ml')
    summary['metrics'] = {name: {key: float((np.sqrt(np.mean(
        [r[name][c][key]**2 for r in records for c in ('LV','LA','RV','RA')])) if key.endswith('rmse_mm')
        else np.mean([r[name][c][key] for r in records for c in ('LV','LA','RV','RA')])))
        for key in keys} for name in ('base', 'refiner')}
    summary['objectives'] = {name: {key: float(np.mean([r[name+'_objective'][key] for r in records]))
        for key in ('total', 'supervised', 'residual_penalty')} for name in ('base', 'refiner')}
    write_json(output_dir/(split+'_comparison.json'), summary)
    write_json(output_dir/(split+'_native_cells.json'), all_rows)
    return summary, trajectories


_GEOMETRY_GUARD = None


def initialize_geometry(checkpoint):
    global _GEOMETRY_GUARD
    torch.set_num_threads(1)
    from tto.geometry.guard import GeometryGuard
    model, _ = load_refiner_checkpoint(checkpoint)
    _GEOMETRY_GUARD = GeometryGuard(model.base_model)


def audit_geometry(item):
    return dict(case_id=item['case_id'], base=_GEOMETRY_GUARD.audit(item['base']),
                refiner=_GEOMETRY_GUARD.audit(item['refiner']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--initialize', type=Path)
    source.add_argument('--resume', type=Path)
    source.add_argument('--evaluate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--train-cases', type=int, default=16)
    parser.add_argument('--val-cases', type=int, default=8)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--geometry-workers', type=int, default=2)
    args = parser.parse_args()
    if min(args.train_cases, args.val_cases, args.steps, args.batch_size, args.geometry_workers) < 1:
        parser.error('Positive case counts, updates, batch size and geometry workers required')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Positive finite learning rate required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.initialize:
        if any(output.iterdir()):
            raise ValueError('New initialization requires a fresh output directory')
        digest = _sha256(args.initialize)
        base, source_saved = load_checkpoint(args.initialize, args.device)
        if _sha256(args.initialize) != digest:
            raise ValueError('Source checkpoint changed during reading; retry with a stable checkpoint')
        model = JointFFDRefiner(base).to(args.device)
        experiment = dict(source_path=str(args.initialize.resolve()), source_sha256=digest,
            source_epoch=source_saved['epoch'], source_step=source_saved['step'], seed=args.seed,
            train_indices=list(range(args.train_cases)), val_indices=list(range(args.val_cases)),
            steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
            regularization=dict(DEFAULT_REGULARIZATION), freeze='base_frozen_refiner_only',
            pose_source='synthetic_GT', checkpoint_selection='fixed endpoint; no validation tuning',
            geometry_scope='all selected TRAIN and VAL cases, full24 and closing chord',
            test_used=False)
        root = Path(__file__).resolve().parent
        tracked_sources = ('train_joint_refiner.py', 'train_neural_ffd.py', 'net/joint_ffd_refiner.py',
            'utils/joint_refiner_training.py', 'utils/observation_residual.py',
            'utils/ffd_observation_splat.py', 'utils/ffd_physical.py', 'utils/ffd.py',
            'utils/slice_supervision.py', 'utils/neural_ffd_losses.py', 'utils/training_execution.py',
            'utils/geometry_pair_policy.py', 'tto/geometry/guard.py')
        experiment['source_files_sha256'] = {f: _sha256(root/f) for f in tracked_sources}
        for f in tracked_sources:
            destination = output/'source'/f
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/f, destination)
        step = 0
        saved = None
    else:
        model, saved = load_refiner_checkpoint(args.resume or args.evaluate, args.device)
        source_saved, experiment, step = saved['base_checkpoint'], saved['experiment'], saved['step']
    base = model.base_model
    config = source_saved['config']
    data_config = yaml.safe_load(repository_path(config['data_config']).read_text())
    if data_identity(data_config) != source_saved['data_identity']:
        raise ValueError('Source dataset/geometry identity changed')
    train, validation = dataset_from_config(data_config), dataset_from_config(data_config, 'val')
    all_train_ids, all_val_ids = dataset_case_ids(train), dataset_case_ids(validation)
    train_ids = [all_train_ids[i] for i in experiment['train_indices']]
    val_ids = [all_val_ids[i] for i in experiment['val_indices']]
    if set(all_train_ids) & set(all_val_ids):
        raise ValueError('TRAIN/VAL overlap')
    for key, value in (('train_case_ids', train_ids), ('val_case_ids', val_ids)):
        if key in experiment and experiment[key] != value:
            raise ValueError('Fixed subset identity changed')
        experiment[key] = value
    write_json(output/'protocol.json', experiment)
    criterion = NeuralFFDLoss(base.chamber_ranges, base.faces, **config['loss']).to(args.device)
    slicer = MeshSliceSupervisor(base.chamber_ranges, base.faces, base.norm_to_mm, **config['slice'],
                                execution_backend='batched', cell_chunk_size=64).to(args.device)
    execution = configure_execution(base, slicer, objective_mode=criterion.objective_mode,
                                    memory_profile='throughput', cnn_chunk_size=144)
    # Keep one physical quadrature policy throughout this bounded experiment.
    slicer.set_joint_progress(0.)
    write_json(output/'execution.json', dict(execution, device=args.device,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), frozen_base_cache=True))
    train_cases = prepare_cases(model, train, experiment['train_indices'], args.device)
    val_cases = prepare_cases(model, validation, experiment['val_indices'], args.device)
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=experiment['learning_rate'])
    generator = torch.Generator().manual_seed(experiment['seed'])
    configure_refiner_training(model)
    if saved and not args.evaluate:
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['sampler_state'])
        torch.set_rng_state(saved['rng'])
        if saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'], args.device)
    if not args.evaluate:
        history_mode = 'a' if args.resume else 'w'
        with (output/'history.jsonl').open(history_mode) as history:
            while step < experiment['steps']:
                start = time.perf_counter()
                # Sampling is shared-population SGD; IDs only select source data.
                ids = torch.randperm(len(train_cases), generator=generator)[:experiment['batch_size']].tolist()
                batch, cached_base, splat = cached_batch(train_cases, ids, args.device)
                optimizer.zero_grad(set_to_none=True)
                result = model.refine_base(cached_base, splat, phases=batch['phases'])
                losses = refiner_objective(model, result, batch, criterion, slicer, experiment['regularization'])
                losses['total'].backward()
                norms = torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                record = dict(step=step, case_ids=batch['case_id'], total=float(losses['total'].detach()),
                    supervised=float(losses['supervised'].detach()), residual_penalty=float(losses['residual_penalty'].detach()),
                    residual_terms={k: float(v.detach()) for k, v in losses['residual_terms'].items()},
                    supervised_terms={k: float(v.detach()) for k, v in losses['supervised_terms'].items()},
                    gradient_norm=float(norms), seconds=time.perf_counter()-start,
                    maximum_control_norm_mm=float(result['refinement']['controls_res_mm'].detach().norm(dim=-1).max()))
                if torch.device(args.device).type == 'cuda':
                    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
                    record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(args.device)
                history.write(json.dumps(record, allow_nan=False)+'\n'); history.flush()
                if step == 1 or step % 10 == 0:
                    print(f"Update {step}/{experiment['steps']} loss={record['total']:.6f} "
                          f"control_max={record['maximum_control_norm_mm']:.3f} mm", flush=True)
                del result, losses, batch, cached_base, splat
                if step % 50 == 0 or step == experiment['steps']:
                    save_refiner_checkpoint(output/'last.pth', model, optimizer,
                        source_checkpoint=source_saved, experiment=experiment, step=step,
                        sampler_state=generator.get_state())
    checkpoint = args.evaluate or output/'last.pth'
    # Verify a trained, nonzero model survives a fully self-contained reload,
    # and cached extraction matches the real online inference path.
    model.eval()
    restored, _ = load_refiner_checkpoint(checkpoint, args.device)
    restored.eval()
    with torch.no_grad():
        batch, cached_base, splat = cached_batch(val_cases, [0], args.device)
        expected = model.refine_base(cached_base, splat, phases=batch['phases'])['vertices_4d']
        online = restored(batch, pose=batch['pose_gt'])['vertices_4d']
        error = float((expected-online).abs().max())
        if not torch.allclose(expected, online, atol=2e-4, rtol=1e-6):
            raise ValueError('Checkpoint reload / online-cached inference parity failed')
    write_json(output/'reload_verification.json', dict(maximum_vertex_difference_mm=error, passed=True))
    del restored, batch, cached_base, splat, expected, online
    comparisons, geometry_jobs = {}, []
    for split, cases in (('train', train_cases), ('val', val_cases)):
        comparison, trajectories = evaluate_cases(model, cases, args.device, output, split,
            criterion=criterion, slicer=slicer, weights=experiment['regularization'])
        comparisons[split] = comparison
        for item in trajectories:
            item['split'] = split
            geometry_jobs.append(item)
    print('Starting complete selected-case mesh/chord geometry audits', flush=True)
    geometry = []
    with ProcessPoolExecutor(max_workers=args.geometry_workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_geometry, initargs=(str(checkpoint),)) as pool:
        jobs = {pool.submit(audit_geometry, item): item for item in geometry_jobs}
        for future in as_completed(jobs):
            item = jobs[future]
            record = dict(future.result(), split=item['split'])
            geometry.append(record)
            write_json(output/'geometry.json', geometry)
            print(f"Geometry {item['split']} {item['case_id']} Base={record['base']['passed']} "
                  f"Refiner={record['refiner']['passed']} ({len(geometry)}/{len(geometry_jobs)})", flush=True)
    decision = dict(scope='bounded shared learning diagnostic, not full-cohort acceptance',
        train_supervised_objective_improved=comparisons['train']['objectives']['refiner']['supervised'] <
                                           comparisons['train']['objectives']['base']['supervised'],
        train_improved=comparisons['train']['metrics']['refiner']['trajectory_vertex_rmse_mm'] <
                       comparisons['train']['metrics']['base']['trajectory_vertex_rmse_mm'],
        val_trajectory_improved=comparisons['val']['metrics']['refiner']['trajectory_vertex_rmse_mm'] <
                                comparisons['val']['metrics']['base']['trajectory_vertex_rmse_mm'],
        val_right_heart_p05_improved=comparisons['val']['tails']['refiner']['right_heart']['p05_dice'] >
                                     comparisons['val']['tails']['base']['right_heart']['p05_dice'],
        val_below_098_reduced=comparisons['val']['tails']['refiner']['all']['below_098'] <
                              comparisons['val']['tails']['base']['all']['below_098'],
        all_selected_refiner_geometry_pass=all(r['refiner']['passed'] for r in geometry),
        geometry_regression_cases=[r['case_id'] for r in geometry if r['base']['passed'] and not r['refiner']['passed']],
        all_selected_sampled_jacobians_positive=all(r['sampled_composed_jacobian']['refiner']['nonpositive'] == 0
            and r['sampled_composed_jacobian']['refiner']['finite'] for v in comparisons.values() for r in v['cases']),
        test_used=False, source_epoch=experiment['source_epoch'], updates=step)
    write_json(output/'decision.json', decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    main()
