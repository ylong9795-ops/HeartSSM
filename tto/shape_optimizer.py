"""Latent-only Shape updates with atomic rollback and legal best-state selection."""
from copy import deepcopy
import numpy as np
import torch

from .slice_loss import NativeSliceLoss
from .motion_lock import FrozenMotionPropagator
from .evaluator import score
from .io import cpu


def unobserved_vertices(model, reference, pose, patient, band_mm=5.):
    unknown = torch.ones(len(reference), device=reference.device, dtype=torch.bool)
    native = patient.native(0)
    for ci, (a, b) in enumerate(model.chamber_ranges.tolist()):
        for vi in range(3):
            if not native['annotated'][vi, ci]:continue
            h, w = native['masks'][vi].shape[-2:];dy, dx = native['spacing'][vi]
            local = (reference[a:b]-pose['center_mm'][vi]) @ pose['R'][vi]
            visible = (local[:, 2].abs() <= band_mm) & (local[:, 0].abs() <= w*dx/2) & (local[:, 1].abs() <= h*dy/2)
            unknown[a:b] &= ~visible
    return unknown


class LatentShapeOptimizer:
    def __init__(self, model, config, motion_limits, guard, evaluator, shape_std, loss_config=None):
        self.model, self.config, self.motion_limits = model, config, motion_limits
        self.guard, self.evaluator = guard, evaluator
        self.loss_config = loss_config
        self.std = torch.as_tensor(shape_std, device=model.template_vertices_mm.device, dtype=torch.float32)
        if self.std.shape != (64,) or (self.std <= 0).any() or not torch.isfinite(self.std).all():
            raise ValueError('Positive finite synthetic Shape latent standard deviations required')

    @torch.no_grad()
    def project(self, parameter, initial):
        delta = ((parameter-initial)/self.std).clamp(-self.config['coordinate_limit'], self.config['coordinate_limit'])
        delta *= min(1., self.config['rms_limit']/float(delta.square().mean().sqrt().clamp_min(1e-8)))
        parameter.copy_(initial+self.std*delta)

    def optimize(self, patients, states, *, resumes=None, checkpoint=None):
        cfg = self.config;device = self.model.template_vertices_mm.device
        resumes = resumes or [None]*len(patients)
        initial = torch.stack([s['z_shape_initial'] for s in states])
        base = torch.stack([s['prior4d'][0] for s in states])
        prior = torch.stack([s['prior4d'] for s in states])
        fields = torch.stack([s['fields'] for s in states])
        poses = [s['pose'] for s in states]
        pose_batch = {k: torch.stack([p[k] for p in poses]) for k in ('R', 'center_mm')}
        unknown = [unobserved_vertices(self.model, base[i], poses[i], patient, cfg['observed_band_mm']) for i, patient in enumerate(patients)]
        supervisor = NativeSliceLoss(self.model, [p.native(0) for p in patients], cfg['stride'], self.loss_config)
        motion = FrozenMotionPropagator(self.model, fields, prior, self.motion_limits)
        parameters, optimizers, histories, best, best_scores, begins = [], [], [], [], [], []
        admissions = []
        for i, (patient, state, resume) in enumerate(zip(patients, states, resumes)):
            admitted = state['geometry']['passed']
            admissions.append(admitted)
            parameter = torch.nn.Parameter(initial[i].detach().clone())
            optimizer = torch.optim.Adam([parameter], lr=cfg['lr'])
            rows = self.evaluator.cells(prior[i].cpu().numpy(), {k:v.cpu().numpy() for k,v in poses[i].items()}, patient, 'before_shape', phases=[0])
            if resume:
                with torch.no_grad():parameter.copy_(resume['parameter'].to(device))
                optimizer.load_state_dict(resume['optimizer'])
            parameters.append(parameter);optimizers.append(optimizer)
            histories.append(resume['history'] if resume else [])
            best.append(resume['best'].to(device) if resume else initial[i].detach().clone())
            best_scores.append(resume['best_score'] if resume else list(score(rows)))
            begins.append(resume['step'] if resume else 0)
        for step in range(cfg['updates']):
            active = [i for i, admitted in enumerate(admissions) if admitted and step >= begins[i]]
            if not active:continue
            old = [p.detach().clone() for p in parameters]
            old_opt = [deepcopy(o.state_dict()) for o in optimizers]
            for opt in optimizers:opt.zero_grad()
            reference = self.model.decode_shape(torch.stack(parameters))
            losses, _, missing = supervisor(reference, pose_batch)
            terms = []
            for i, parameter in enumerate(parameters):
                drift = reference[i]-base[i]
                unseen = drift[unknown[i]].square().mean() if unknown[i].any() else drift.new_zeros(())
                terms.append(losses[i]+cfg['latent_weight']*((parameter-initial[i])/self.std).square().mean()
                    +cfg['unknown_weight']*unseen+cfg['all_vertex_weight']*drift.square().mean())
            sum(terms[i] for i in active).backward()
            for i in active:
                grad = parameters[i].grad
                if grad is None or not torch.isfinite(grad).all():
                    raise ValueError(f'Invalid Shape gradient for {patients[i].patient_id}')
                optimizers[i].step();self.project(parameters[i], initial[i])
            with torch.no_grad():
                candidates = self.model.decode_shape(torch.stack(parameters))
                trajectories = motion.propagate(candidates)
                motion_reports = motion.check(trajectories)
            for i in active:
                drift = (candidates[i]-base[i]).norm(dim=-1)
                unseen_max = float(drift[unknown[i]].max()) if unknown[i].any() else 0.
                maximum = float(drift.max());reasons = []
                if unseen_max > cfg['unknown_max_mm']:reasons.append('unobserved_3D_drift')
                if maximum > cfg['all_vertex_max_mm']:reasons.append('maximum_3D_drift')
                if not motion_reports[i]['passed']:reasons.append('motion_lock')
                geometry = None
                if not reasons:
                    geometry = self.guard.audit(trajectories[i].cpu().numpy())
                    if not geometry['passed']:reasons.append('geometry')
                legal = not reasons
                proposed = parameters[i].detach().clone()
                candidate_score = None
                if legal:
                    rows = self.evaluator.cells(trajectories[i].cpu().numpy(), {k:v.cpu().numpy() for k,v in poses[i].items()}, patients[i], 'shape', phases=[0])
                    candidate_score = list(score(rows))
                    if candidate_score > best_scores[i]:
                        best[i] = parameters[i].detach().clone();best_scores[i] = candidate_score
                else:
                    with torch.no_grad():parameters[i].copy_(old[i])
                    optimizers[i].load_state_dict(old_opt[i])
                histories[i].append(dict(update=step+1, objective=float(terms[i].detach()),
                    gradient_norm=float(parameters[i].grad.norm()), accepted=legal, rejected_by=reasons,
                    unknown_max_mm=unseen_max, maximum_drift_mm=maximum, motion=motion_reports[i], geometry=geometry,
                    z_shape_proposed=cpu(proposed).tolist(), z_shape_retained=cpu(parameters[i]).tolist(),
                    native_reference_score=candidate_score,
                    standardized_delta_rms=float(((proposed-initial[i])/self.std).square().mean().sqrt())))
                if checkpoint:
                    checkpoint(patients[i], dict(step=step+1, parameter=parameters[i], optimizer=optimizers[i].state_dict(),
                        best=best[i], best_score=best_scores[i], history=histories[i]))
        results = []
        with torch.no_grad():
            reference = self.model.decode_shape(torch.stack(best));trajectory = motion.propagate(reference)
        for i, patient in enumerate(patients):
            geometry = self.guard.audit(trajectory[i].cpu().numpy())
            results.append(dict(z_shape=best[i], vertices_ref=reference[i], trajectory=trajectory[i], geometry=geometry,
                history=histories[i], best_score=best_scores[i], admitted=admissions[i],
                status='COMPLETE' if admissions[i] else 'NOT_ADMITTED',
                reason=None if admissions[i] else 'Initial4D geometry failed or unverified; no legal rollback state',
                pose_frozen=True, motion_frozen=True, network_frozen=True))
        return results
