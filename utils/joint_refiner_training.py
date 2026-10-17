"""Shared Refiner training contracts, independent of historical Base resumes.

Uses the existing supervised/paired-boundary objective and actual geometry
backends. No per-patient variables or new deformation operator are introduced.
"""
import copy
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from data.mask_transform import CHAMBERS
from net.joint_ffd_refiner import JointFFDRefiner, ObservationFFDRefiner
from utils.ffd_physical import bspline_ffd_physical

VERSION = 'observation-joint-ffd-refiner-v1'
DEFAULT_REGULARIZATION = dict(spatial_gradient_l2=.01, cyclic_velocity_l2=.0001,
                              cyclic_acceleration_l2=.0000001)


def configure_refiner_training(model):
    """First experiment: a frozen inherited Base and one population Refiner."""
    model.base_model.requires_grad_(False)
    model.base_model.eval()
    model.refiner.requires_grad_(True)
    model.refiner.train()


def refiner_objective(model, output, batch, criterion, slicer, weights=None):
    base = model.base_model
    slices = slicer(output['vertices_4d'], batch)
    losses = criterion(output, batch['vertices_gt'], slice_result=slices,
        control_spacings_mm=base.control_spacings_mm,
        regional_spacing_mm=base.regional_spacing_mm if base.regional_shape else None,
        regional_motion_spacings_mm=base.regional_motion_spacings_mm if base.regional_motion else None,
        stage='joint')
    weights = dict(DEFAULT_REGULARIZATION if weights is None else weights)
    regularizers = output['control_regularization']
    if (set(weights) != set(DEFAULT_REGULARIZATION) or
            any(not math.isfinite(v) or v < 0 for v in weights.values())):
        raise ValueError('Explicit nonnegative spatial/velocity/acceleration residual weights required')
    penalty = sum((regularizers[k]*weights[k] for k in weights), losses['total']*0)
    return dict(total=losses['total']+penalty, supervised=losses['total'], residual_penalty=penalty,
        residual_terms=regularizers, residual_weights=weights, supervised_terms=losses['terms'])


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def save_refiner_checkpoint(path, model, optimizer, *, source_checkpoint, experiment, step,
                            sampler_state=None):
    """Self-contained frozen Base, Refiner, Adam and sampling/RNG state."""
    embedded = _cpu_tree(source_checkpoint)
    embedded['model'] = _cpu_tree(model.base_model.state_dict())
    embedded['optimizer'] = None  # inherited Adam is never a new Joint resume
    payload = dict(version=VERSION, base_checkpoint=embedded,
        refiner_config=dict(model.refiner.configuration), pipeline_config=dict(model.configuration),
        refiner=_cpu_tree(model.refiner.state_dict()), optimizer=None if optimizer is None else _cpu_tree(optimizer.state_dict()),
        experiment=copy.deepcopy(experiment), step=int(step), sampler_state=sampler_state,
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(next(model.parameters()).device)
        if next(model.parameters()).is_cuda else None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_refiner_checkpoint(path, device='cpu'):
    from train_neural_ffd import model_from_checkpoint
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['version'] != VERSION:
        raise ValueError('Unsupported shared Refiner checkpoint')
    base = model_from_checkpoint(saved['base_checkpoint'])
    model = JointFFDRefiner(base, ObservationFFDRefiner(**saved['refiner_config']), **saved['pipeline_config'])
    model.refiner.load_state_dict(saved['refiner'], strict=True)
    return model.to(device), saved


def tail_summary(rows):
    """Cell tails and worst-case identities; never average away right-heart failures."""
    if not rows:
        raise ValueError('Nonempty exact cell records required')
    def aggregate(selected):
        if not selected:
            return None
        values = np.asarray([r['exact_dice'] for r in selected])
        by_case = {}
        for row in selected:
            by_case.setdefault(row['case_id'], []).append(row['exact_dice'])
        boundary = [r for r in selected if r.get('boundary_assd_mm') is not None]
        worst = sorted(selected, key=lambda r: r['exact_dice'])[:10]
        return dict(cells=len(values), mean_dice=float(values.mean()), minimum_dice=float(values.min()),
            p01_dice=float(np.quantile(values, .01)), p05_dice=float(np.quantile(values, .05)),
            below_095=int((values < .95).sum()), below_097=int((values < .97).sum()),
            below_098=int((values < .98).sum()), cases=len(by_case),
            cases_all_cells_ge_098=sum(min(v) >= .98 for v in by_case.values()),
            boundary_evaluable_cells=len(boundary),
            boundary_assd_mm=float(np.mean([r['boundary_assd_mm'] for r in boundary])) if boundary else None,
            boundary_hd95_mm=float(np.mean([r['boundary_hd95_mm'] for r in boundary])) if boundary else None,
            worst_cells=[{k: r[k] for k in ('case_id', 'frame', 'view', 'chamber', 'exact_dice')} for r in worst])
    return dict(all=aggregate(rows), right_heart=aggregate([r for r in rows if r['chamber'] in ('RV', 'RA')]),
        chambers={c: aggregate([r for r in rows if r['chamber'] == c]) for c in CHAMBERS})


@torch.no_grad()
def sampled_composed_jacobians(base, base_output, controls, *, per_chamber=64, epsilon_mm=.02):
    """Central differences of the ACTUAL frozen-control composition at template points.

    Replays coarse/medium/Fine0/region and recurrent Fine/regional motion,
    then the absolute residual at each CURRENT phase. This is a sampled local
    diagnostic, not a whole-volume injectivity certificate. Discrete triangle
    and inter-frame chord audits are separate.
    """
    if base_output['vertices_4d'].shape[0] != 1:
        raise ValueError('Jacobian audit accepts one patient at a time')
    grids = {k: None if v is None else v.double() for k, v in base_output['control_grids'].items()}
    controls = controls.double()
    origin, spacing = base.control_origins_mm, base.control_spacings_mm
    baseline, refined, maximum_replay_error = [], [], 0.
    shifts = torch.cat((torch.zeros(1, 3), torch.eye(3)*epsilon_mm, -torch.eye(3)*epsilon_mm)).to(controls)
    for chamber, (left, right) in enumerate(base.chamber_ranges.tolist()):
        ids = torch.linspace(left, right-1, min(per_chamber, right-left), device=controls.device).round().long()
        points = (base.template_vertices_mm[ids].double()[:, None]+shifts[None]).reshape(1, -1, 3)
        def deform(p, c, o, h):
            return p+bspline_ffd_physical(p, c, o, h)
        points = deform(points, grids['coarse'], origin[0], spacing[0])
        points = deform(points, grids['medium'], origin[1], spacing[1])
        points = deform(points, grids['fine_motion'][:, 0], origin[2], spacing[2])
        if grids['regions'] is not None:
            points = deform(points, grids['regions'][:, chamber], base.regional_origin_mm, base.regional_spacing_mm)
        for phase in range(base.frames):
            if phase:
                points = deform(points, grids['fine_motion'][:, phase], origin[2], spacing[2])
                if grids['regional_motion'] is not None:
                    points = deform(points, grids['regional_motion'][:, phase, chamber],
                                    base.regional_motion_origins_mm[chamber], base.regional_motion_spacings_mm[chamber])
            final = deform(points, controls[:, phase], origin[2], spacing[2])
            centers = points.reshape(-1, 7, 3)[:, 0]
            expected = base_output['vertices_4d'][0, phase, ids].double()
            maximum_replay_error = max(maximum_replay_error, float((centers-expected).abs().max()))
            for target, collection in ((points, baseline), (final, refined)):
                p = target.reshape(-1, 7, 3)
                derivative = ((p[:, 1:4]-p[:, 4:7])/(2*epsilon_mm)).transpose(1, 2)
                collection.append(torch.linalg.det(derivative))
    if maximum_replay_error > .002:
        raise ValueError('Composed Jacobian mapping does not reproduce the base trajectory')
    def summary(values):
        x = torch.cat(values)
        return dict(samples=x.numel(), minimum=float(x.min()), p01=float(torch.quantile(x, .01)),
                    nonpositive=int((x <= 0).sum()), finite=bool(torch.isfinite(x).all()))
    return dict(base=summary(baseline), refiner=summary(refined),
                epsilon_mm=epsilon_mm, maximum_base_replay_error_mm=maximum_replay_error,
                scope='sampled full composition at fixed template vertices, every phase; not a whole-field certificate')
