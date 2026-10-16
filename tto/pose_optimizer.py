"""Independent per-patient multistart SO(3)/millimetre Pose optimization."""
import hashlib
import torch

from .evaluator import score
from .slice_loss import NativeSliceLoss
from .io import cpu


def so3(degrees):
    x, y, z = torch.deg2rad(degrees).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*degrees.shape[:-1], 3, 3)
    return torch.matrix_exp(skew)


def map_pose(delta, base):
    return dict(R=so3(delta[..., 3:]) @ base['R'], center_mm=base['center_mm']+delta[..., :3])


@torch.no_grad()
def project_pose(delta, translation_limit, rotation_limit):
    for part, limit in ((delta[..., :3], translation_limit), (delta[..., 3:], rotation_limit)):
        part.mul_((float(limit)/part.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1))


class MultiViewPoseOptimizer:
    def __init__(self, model, config, evaluator, seed=2026, loss_config=None):
        self.model, self.config, self.evaluator, self.seed = model, config, evaluator, seed
        self.loss_config = loss_config

    def optimize(self, patients, references, poses, *, resumes=None, checkpoint=None):
        cfg = self.config;device = references.device;starts = cfg['starts']
        references = references.detach()
        resumes = resumes or [None]*len(patients)
        parameters, optimizers, histories, begins = [], [], [], []
        for patient, resume in zip(patients, resumes):
            seed = self.seed+int.from_bytes(hashlib.sha256(patient.patient_id.encode()).digest()[:4], 'little')
            generator = torch.Generator().manual_seed(seed)
            initial = torch.zeros(starts, 3, 6)
            initial[1:, :, :3] = torch.randn(starts-1, 3, 3, generator=generator)*cfg['start_translation_mm']
            initial[1:, :, 3:] = torch.randn(starts-1, 3, 3, generator=generator)*cfg['start_rotation_degrees']
            parameter = torch.nn.Parameter(initial.to(device))
            optimizer = torch.optim.Adam([parameter], lr=cfg['levels'][0]['lr'])
            if resume:
                with torch.no_grad():parameter.copy_(resume['parameter'].to(device))
                optimizer.load_state_dict(resume['optimizer'])
            parameters.append(parameter);optimizers.append(optimizer)
            histories.append(resume['history'] if resume else [])
            begins.append(resume['step'] if resume else 0)
        schedule = [level for level in cfg['levels'] for _ in range(level['updates'])]
        repeated_natives = [patient.native(0) for patient in patients for _ in range(starts)]
        supervisors = {level['stride']: NativeSliceLoss(self.model, repeated_natives, level['stride'], self.loss_config) for level in cfg['levels']}
        base = {k: torch.stack([p[k] for p in poses])[:, None].expand(-1, starts, *poses[0][k].shape).reshape(-1, *poses[0][k].shape) for k in ('R', 'center_mm')}
        fixed = references[:, None].expand(-1, starts, -1, -1).reshape(-1, *references.shape[1:])
        for step, level in enumerate(schedule):
            active = [i for i in range(len(patients)) if step >= begins[i]]
            if not active:continue
            for optimizer in optimizers:
                optimizer.zero_grad();optimizer.param_groups[0]['lr'] = level['lr']
            delta = torch.stack(parameters).reshape(-1, 3, 6)
            pose = map_pose(delta, base)
            losses, _, missing = supervisors[level['stride']](fixed, pose)
            prior = cfg['translation_prior']*delta[..., :3].square().mean((1, 2))+cfg['rotation_prior']*delta[..., 3:].square().mean((1, 2))
            # Preserve the declared relative three-plane configuration while allowing unknown per-view Pose.
            relative = delta-delta.mean(1, keepdim=True)
            prior = prior+cfg['relative_prior']*relative.square().mean((1, 2))
            objectives = (losses+prior).reshape(len(patients), starts)
            objectives[active].sum().backward()
            for i in active:
                gradient = parameters[i].grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise ValueError(f'Invalid Pose gradient for {patients[i].patient_id}')
                optimizers[i].step()
                project_pose(parameters[i], cfg['translation_limit_mm'], cfg['rotation_limit_degrees'])
                histories[i].append(dict(update=step+1, objective=cpu(objectives[i]).tolist(),
                    translation_gradient_norm=float(gradient[..., :3].norm()), rotation_gradient_norm=float(gradient[..., 3:].norm()),
                    missing_sections=sum(owner//starts == i for owner, _, _ in missing), stride=level['stride'],
                    bounded_parameters=cpu(parameters[i]).tolist()))
                if checkpoint:
                    checkpoint(patients[i], dict(step=step+1, parameter=parameters[i], optimizer=optimizers[i].state_dict(), history=histories[i]))
        results = []
        for i, patient in enumerate(patients):
            candidates = []
            for j in range(starts):
                pose = map_pose(parameters[i][j].detach(), poses[i])
                estimate = {k: v.cpu().numpy() for k, v in pose.items()}
                # All candidates share exactly the same frozen reference Shape.
                rows = self.evaluator.cells(references[i:i+1].expand(24, -1, -1).cpu().numpy(), estimate, patient, 'pose', phases=[0])
                candidates.append(dict(candidate=j, pose=cpu(pose), score=list(score(rows)), rows=rows))
            winner = max(candidates, key=lambda c: c['score'])
            results.append(dict(pose=winner['pose'], selected_candidate=winner['candidate'],
                candidates=candidates, history=histories[i], shape_frozen=True, motion_frozen=True,
                status='COMPLETE', updates=len(schedule)))
        return results
