"""Small-step Shape/Pose refinement after separate stages, with full4D admission."""
from copy import deepcopy
import torch

from .evaluator import score
from .io import cpu
from .motion_lock import FrozenMotionPropagator
from .pose_optimizer import map_pose, project_pose
from .shape_optimizer import unobserved_vertices
from .slice_loss import NativeSliceLoss


class GuardedJointOptimizer:
    def __init__(self, model, config, shape_optimizer, guard, evaluator, motion_limits):
        self.model, self.config, self.shape = model, config, shape_optimizer
        self.guard, self.evaluator, self.motion_limits = guard, evaluator, motion_limits

    def optimize(self, patients, states, *, resumes=None, checkpoint=None):
        resumes = resumes or [None]*len(patients)
        cfg = self.config;shape_cfg = self.shape.config
        z0 = torch.stack([s['z_shape_initial'] for s in states])
        prior = torch.stack([s['prior4d'] for s in states])
        fields = torch.stack([s['fields'] for s in states])
        bases = [s['pose'] for s in states]
        pose_base = {k: torch.stack([p[k] for p in bases]) for k in ('R', 'center_mm')}
        unknown = [unobserved_vertices(self.model, prior[i, 0], bases[i], p, shape_cfg['observed_band_mm']) for i, p in enumerate(patients)]
        supervisor = NativeSliceLoss(self.model, [p.native(0) for p in patients], cfg['stride'], self.shape.loss_config)
        motion = FrozenMotionPropagator(self.model, fields, prior, self.motion_limits)
        parameters, deltas, optimizers, best, histories, begins = [], [], [], [], [], []
        admitted = [s['geometry']['passed'] and s['shape_status'] == 'COMPLETE' and s['motion_report']['passed'] for s in states]
        for i, (state, resume, patient) in enumerate(zip(states, resumes, patients)):
            z = torch.nn.Parameter(state['z_shape'].detach().clone())
            delta = torch.nn.Parameter(torch.zeros(3, 6, device=z.device))
            opt = torch.optim.Adam([dict(params=[z], lr=cfg['shape_lr']), dict(params=[delta], lr=cfg['pose_lr'])])
            pose = {k: v.cpu().numpy() for k, v in bases[i].items()}
            rows = self.evaluator.cells(state['trajectory'].cpu().numpy(), pose, patient, 'before_joint', phases=[0])
            baseline = dict(z_shape=z.detach().clone(), delta=delta.detach().clone(), score=list(score(rows)))
            if resume:
                with torch.no_grad():z.copy_(resume['z_shape'].to(z.device));delta.copy_(resume['delta'].to(z.device))
                opt.load_state_dict(resume['optimizer'])
            parameters.append(z);deltas.append(delta);optimizers.append(opt)
            best.append(resume['best'] if resume else baseline)
            histories.append(resume['history'] if resume else [])
            begins.append(resume['step'] if resume else 0)
        for step in range(cfg['updates']):
            active = [i for i in range(len(patients)) if admitted[i] and step >= begins[i]]
            if not active:continue
            old = [(z.detach().clone(), d.detach().clone(), deepcopy(o.state_dict())) for z, d, o in zip(parameters, deltas, optimizers)]
            for opt in optimizers:opt.zero_grad()
            pose = map_pose(torch.stack(deltas), pose_base)
            reference = self.model.decode_shape(torch.stack(parameters))
            losses, _, _ = supervisor(reference, pose)
            terms = []
            for i, z in enumerate(parameters):
                drift = reference[i]-prior[i, 0]
                unseen = drift[unknown[i]].square().mean() if unknown[i].any() else drift.new_zeros(())
                terms.append(losses[i]+shape_cfg['latent_weight']*((z-z0[i])/self.shape.std).square().mean()
                    +shape_cfg['unknown_weight']*unseen+shape_cfg['all_vertex_weight']*drift.square().mean()
                    +cfg['pose_prior']*deltas[i].square().mean())
            sum(terms[i] for i in active).backward()
            for i in active:
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in (parameters[i], deltas[i])):
                    raise ValueError(f'Invalid Joint gradient: {patients[i].patient_id}')
                optimizers[i].step();self.shape.project(parameters[i], z0[i])
                project_pose(deltas[i], cfg['translation_limit_mm'], cfg['rotation_limit_degrees'])
            with torch.no_grad():
                reference = self.model.decode_shape(torch.stack(parameters))
                trajectories = motion.propagate(reference)
                poses = map_pose(torch.stack(deltas), pose_base)
                motions = motion.check(trajectories)
            for i in active:
                drift = (reference[i]-prior[i, 0]).norm(dim=-1);reasons = []
                if float(drift.max()) > shape_cfg['all_vertex_max_mm']:reasons.append('maximum_3D_drift')
                if unknown[i].any() and float(drift[unknown[i]].max()) > shape_cfg['unknown_max_mm']:reasons.append('unobserved_3D_drift')
                if not motions[i]['passed']:reasons.append('motion_lock')
                geometry = None
                if not reasons:
                    geometry = self.guard.audit(trajectories[i].cpu().numpy())
                    if not geometry['passed']:reasons.append('geometry')
                legal = not reasons
                proposed_shape = cpu(parameters[i]).tolist()
                proposed_pose_delta = cpu(deltas[i]).tolist()
                current = None
                if legal:
                    estimate = {k: v[i].cpu().numpy() for k, v in poses.items()}
                    rows = self.evaluator.cells(trajectories[i].cpu().numpy(), estimate, patients[i], 'joint', phases=[0])
                    current = list(score(rows))
                    if current > best[i]['score']:
                        best[i] = dict(z_shape=parameters[i].detach().clone(), delta=deltas[i].detach().clone(), score=current)
                else:
                    with torch.no_grad():parameters[i].copy_(old[i][0]);deltas[i].copy_(old[i][1])
                    optimizers[i].load_state_dict(old[i][2])
                histories[i].append(dict(update=step+1, objective=float(terms[i].detach()), accepted=legal,
                    rejected_by=reasons, shape_gradient_norm=float(parameters[i].grad.norm()),
                    pose_gradient_norm=float(deltas[i].grad.norm()), geometry=geometry, motion=motions[i],
                    z_shape_proposed=proposed_shape, pose_delta_proposed=proposed_pose_delta,
                    z_shape_retained=cpu(parameters[i]).tolist(), pose_delta_retained=cpu(deltas[i]).tolist(),
                    native_reference_score=current))
                if checkpoint:
                    checkpoint(patients[i], dict(step=step+1, z_shape=parameters[i], delta=deltas[i],
                        optimizer=optimizers[i].state_dict(), best=best[i], history=histories[i]))
        results = []
        for i, state in enumerate(states):
            with torch.no_grad():
                z = best[i]['z_shape'].to(z0.device)
                pose = map_pose(best[i]['delta'].to(z0.device), bases[i])
                # Single-patient replay retains its own immutable Motion fields.
                reference = self.model.decode_shape(z[None])
                trajectory = FrozenMotionPropagator(self.model, fields[i:i+1]).propagate(reference)[0]
            results.append(dict(z_shape=z, pose=cpu(pose), trajectory=trajectory,
                geometry=self.guard.audit(trajectory.cpu().numpy()), history=histories[i], admitted=admitted[i],
                status='COMPLETE' if admitted[i] else 'NOT_ADMITTED', motion_frozen=True))
        return results
