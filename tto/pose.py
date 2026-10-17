"""SO(3) estimated poses and full-cycle multistart initialization."""
import hashlib
import torch

from .evaluator import score
from .io import cpu
from .losses import RealTTOLoss
from .pose_optimizer import so3, project_pose


def estimated_pose(delta, base):
    # Local right increments implement R_initial Exp(omega), in degrees/mm.
    return dict(R=base['R'] @ so3(delta[...,3:]), center_mm=base['center_mm']+delta[...,:3])


class MultiViewPoseInitializer:
    def __init__(self, model, config, loss_config, policy, evaluator, seed=2026):
        self.model,self.config,self.loss_config,self.policy = model,config,loss_config,policy
        self.evaluator,self.seed = evaluator,seed

    def optimize(self, patients, trajectories, poses, *, resumes=None, checkpoint=None):
        cfg = self.config;device = trajectories.device;starts = cfg['starts']
        trajectories = trajectories.detach()
        resumed = resumes or [None]*len(patients)
        params, optimizers, histories, begins = [],[],[],[]
        for patient, state in zip(patients,resumed):
            generator = torch.Generator().manual_seed(self.seed+int.from_bytes(hashlib.sha256(patient.patient_id.encode()).digest()[:4],'little'))
            delta = torch.zeros(starts,3,6)
            delta[1:,:,:3] = torch.randn(starts-1,3,3,generator=generator)*cfg['start_translation_mm']
            delta[1:,:,3:] = torch.randn(starts-1,3,3,generator=generator)*cfg['start_rotation_degrees']
            delta = torch.nn.Parameter(delta.to(device));optimizer = torch.optim.Adam([delta],lr=cfg['levels'][0]['lr'])
            if state:
                with torch.no_grad():delta.copy_(state['delta'].to(device))
                optimizer.load_state_dict(state['optimizer'])
            params.append(delta);optimizers.append(optimizer)
            histories.append(state['history'] if state else []);begins.append(state['step'] if state else 0)
        repeat_patients = [p for p in patients for _ in range(starts)]
        fixed = trajectories[:,None].expand(-1,starts,-1,-1,-1).reshape(-1,*trajectories.shape[1:])
        base = {key:torch.stack([p[key] for p in poses])[:,None].expand(-1,starts,*poses[0][key].shape).reshape(-1,*poses[0][key].shape)
                for key in ('R','center_mm')}
        schedule = [level for level in cfg['levels'] for _ in range(level['updates'])]
        supervisors = {level['stride']:RealTTOLoss(self.model,repeat_patients,self.loss_config,self.policy,stride=level['stride']) for level in cfg['levels']}
        for step,level in enumerate(schedule):
            active = [i for i in range(len(patients)) if step >= begins[i]]
            if not active:continue
            for opt in optimizers:opt.zero_grad();opt.param_groups[0]['lr'] = level['lr']
            delta = torch.stack(params).reshape(-1,3,6)
            result = supervisors[level['stride']](fixed,estimated_pose(delta,base))
            relative = delta-delta.mean(1,keepdim=True)
            objectives = (result['slice']+self.loss_config['capture_prior_weight']*result['capture_prior']+
                cfg['translation_prior']*delta[...,:3].square().mean((1,2))+
                cfg['rotation_prior']*delta[...,3:].square().mean((1,2))+
                cfg['relative_prior']*relative.square().mean((1,2))).reshape(len(patients),starts)
            objectives[active].sum().backward()
            for index in active:
                gradient = params[index].grad
                if gradient is None or not torch.isfinite(gradient).all():raise ValueError('Invalid pose-initialization gradient')
                optimizers[index].step();project_pose(params[index],cfg['translation_limit_mm'],cfg['rotation_limit_degrees'])
                histories[index].append(dict(update=step+1,objective=cpu(objectives[index]),
                    translation_gradient_norm=float(gradient[...,:3].norm()),rotation_gradient_norm=float(gradient[...,3:].norm()),
                    supervision_cells=len(supervisors[level['stride']].active)//len(patients),phases=list(range(24))))
                if checkpoint:checkpoint(patients[index],dict(step=step+1,delta=params[index],optimizer=optimizers[index].state_dict(),history=histories[index]))
        outputs = []
        with torch.no_grad():all_poses = estimated_pose(torch.stack(params).reshape(-1,3,6),base)
        for owner,patient in enumerate(patients):
            candidates = []
            for start in range(starts):
                index = owner*starts+start
                pose = {key:value[index] for key,value in all_poses.items()}
                rows = self.evaluator.selection_cells(trajectories[owner].cpu().numpy(),
                    {key:value.cpu().numpy() for key,value in pose.items()},patient,'pose_init')
                candidates.append(dict(pose=cpu(pose),score=list(score(rows)),start=start))
            chosen = max(candidates,key=lambda x:x['score'])
            outputs.append(dict(**chosen,candidates=candidates,history=histories[owner],shared_across_phases=True))
        return outputs
