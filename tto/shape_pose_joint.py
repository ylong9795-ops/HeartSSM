"""Main full-cycle Shape/Pose solver with patient-local Adam and atomic rollback."""
from copy import deepcopy
import random

import numpy as np
import torch
from data.mask_transform import VIEWS,CHAMBERS

from utils.neural_ffd_losses import NeuralFFDLoss
from .evaluator import score, dice_stats
from .io import cpu
from .losses import RealTTOLoss,native_cell,interpolate_cell
from .motion import MotionPropagation
from .pose import estimated_pose
from .pose_optimizer import project_pose

ARMS = ('frozen_prior','pose_only','shape_only','alternate','joint','original_motion','ed_only')


def rng_state(device):
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(),python=random.getstate(),
                numpy=(state[0],state[1].tolist(),*state[2:]),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else None)


def restore_rng(state, device):
    torch.set_rng_state(state['torch'].cpu());random.setstate(state['python'])
    numpy_state=state['numpy'];np.random.set_state((numpy_state[0],np.array(numpy_state[1],dtype=np.uint32),*numpy_state[2:]))
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'].cpu(),device)


def rollback(z_shape, delta, optimizer, previous):
    with torch.no_grad():z_shape.copy_(previous[0]);delta.copy_(previous[1])
    optimizer.load_state_dict(previous[2])


class JointShapePoseOptimizer:
    def __init__(self, prior, config, loss_config, policy, limits, guard, evaluator, calibration):
        self.prior,self.model,self.config = prior,prior.model,config
        self.loss_config,self.policy,self.limits = loss_config,policy,limits
        self.guard,self.evaluator = guard,evaluator
        self.std = torch.as_tensor(calibration['shape_std'],device=prior.device,dtype=torch.float32)
        self.mean = torch.as_tensor(calibration['shape_mean'],device=prior.device,dtype=torch.float32)
        if self.std.shape!=(64,) or not torch.isfinite(self.std).all() or (self.std<=0).any():
            raise ValueError('Measured finite positive TRAIN latent standard deviations required')
        self.control_regularizer = NeuralFFDLoss(self.model.chamber_ranges,self.model.faces)

    @torch.no_grad()
    def project_shape(self, z, initial):
        delta = ((z-initial)/self.std).clamp(-self.config['coordinate_limit'],self.config['coordinate_limit'])
        delta *= min(1.,self.config['rms_limit']/float(delta.square().mean().sqrt().clamp_min(1e-8)))
        z.copy_(initial+self.std*delta)

    def unobserved(self, patients, initial, bases, rows):
        unknown = torch.ones(initial.shape[0],initial.shape[2],dtype=torch.bool,device=initial.device)
        with torch.no_grad():
            for row in rows:
                if not row['used']:continue
                owner,phase,view,chamber = (row[k] for k in ('owner','phase','view','chamber'))
                a,b = self.model.chamber_ranges[chamber].tolist()
                mask,_,spacing=native_cell(patients[owner],row);h,w=mask.shape;dy,dx=spacing
                local = (interpolate_cell(initial[owner],row)[a:b]-bases['center_mm'][owner,view]) @ bases['R'][owner,view]
                visible = ((local[:,2].abs()<=self.config['observed_band_mm']) &
                           (local[:,0].abs()<=w*dx/2) & (local[:,1].abs()<=h*dy/2))
                unknown[owner,a:b] &= ~visible
        return unknown

    def optimize(self, patients, initial, poses, *, arm='joint', resumes=None, checkpoint=None):
        if arm not in ARMS:raise ValueError(arm)
        cfg=self.config;device=self.prior.device;batch=len(patients)
        initial_z=initial['z_shape_initial'];baseline=initial['vertices_4d_initial']
        motion=MotionPropagation(self.prior,initial,self.limits)
        mode='original_decoder' if arm=='original_motion' else 'frozen_field'
        phases=[0] if arm=='ed_only' else range(24)
        supervisor=RealTTOLoss(self.model,patients,self.loss_config,self.policy,phases=phases)
        ed_keys=[{(VIEWS[r['view']],CHAMBERS[r['chamber']],r['native_frame']) for r in supervisor.active if r['owner']==i}
                 for i in range(batch)]
        base={key:torch.stack([pose[key].to(device) for pose in poses]) for key in ('R','center_mm')}
        unknown=self.unobserved(patients,baseline,base,supervisor.rows)
        z_values,deltas,optimizers,histories,best,begins,admitted,baseline_stats = [],[],[],[],[],[],[],[]
        geometry_initial=[];lr_factors=[]
        for owner,patient in enumerate(patients):
            z=torch.nn.Parameter(initial_z[owner].clone());delta=torch.nn.Parameter(torch.zeros(3,6,device=device))
            optimizer=torch.optim.Adam([dict(params=[z],lr=cfg['shape_lr']),dict(params=[delta],lr=cfg['pose_lr'])])
            rows=self.evaluator.selection_cells(baseline[owner].cpu().numpy(),
                {key:value[owner].cpu().numpy() for key,value in base.items()},patient,'before_joint')
            if arm=='ed_only':rows=[r for r in rows if (r['view'],r['chamber'],r['native_frame']) in ed_keys[owner]]
            geometry=self.guard.audit(baseline[owner].cpu().numpy());geometry_initial.append(geometry)
            state=(resumes or [None]*batch)[owner]
            best_state=dict(z_shape=z.detach().clone(),delta=delta.detach().clone(),score=list(score(rows)),geometry=geometry)
            if state:
                if not torch.equal(state['frozen_fields'].to(device),motion.fields[owner]):
                    raise ValueError('Resume motion fields differ from immutable initialization')
                if not torch.equal(state['z_motion_initial'].to(device),motion.z_motion[owner]):
                    raise ValueError('Resume Motion latent differs from immutable initialization')
                with torch.no_grad():z.copy_(state['z_shape'].to(device));delta.copy_(state['delta'].to(device))
                optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'],device)
                best_state=state['best']
            z_values.append(z);deltas.append(delta);optimizers.append(optimizer)
            best.append(best_state);histories.append(state['history'] if state else []);begins.append(state['step'] if state else 0)
            baseline_stats.append(dice_stats(rows));admitted.append(geometry['passed'])
            lr_factors.append(state.get('lr_factor',1.) if state else 1.)
        for step in range(0 if arm=='frozen_prior' else cfg['updates']):
            active=[i for i in range(batch) if admitted[i] and step>=begins[i]]
            if not active:continue
            previous=[(z.detach().clone(),d.detach().clone(),deepcopy(opt.state_dict())) for z,d,opt in zip(z_values,deltas,optimizers)]
            for owner,opt in enumerate(optimizers):
                opt.zero_grad()
                opt.param_groups[0]['lr']=cfg['shape_lr']*lr_factors[owner]
                opt.param_groups[1]['lr']=cfg['pose_lr']*lr_factors[owner]
            z=torch.stack(z_values);delta=torch.stack(deltas)
            trajectory=motion.decode(z,mode);pose=estimated_pose(delta,base)
            observations=supervisor(trajectory,pose)
            preservation=motion.preservation_loss(trajectory)
            objectives=[];prior_terms=[]
            for owner in range(batch):
                drift=trajectory[owner,0]-baseline[owner,0]
                unseen=drift[unknown[owner]].square().mean() if unknown[owner].any() else drift.sum()*0
                latent=((z[owner]-initial_z[owner])/self.std).square().mean()
                distribution=((z[owner]-self.mean)/self.std).square().mean()
                relative=delta[owner]-delta[owner].mean(0,keepdim=True)
                # Reuse only the GT-independent physical control component.
                condition={key: {k:v[owner:owner+1] for k,v in value.items()} for key,value in initial['conditions'].items()}
                _,_,coarse,medium,fine,regions=self.model._reference(z[owner:owner+1],condition.get('reference_spatial'))
                grids=dict(coarse=coarse,medium=medium,fine_motion=fine[:,None],regions=regions)
                control,_=self.control_regularizer._control_objectives(grids,trajectory[owner:owner+1],
                    self.model.control_spacings_mm,getattr(self.model,'regional_spacing_mm',None),None,'shape')
                terms=dict(latent_initial_prior=latent,latent_train_distribution_prior=distribution,
                    unobserved_structure_prior=unseen,reference_drift_prior=drift.square().mean(),ffd_control_prior=control,
                    translation_prior=delta[owner,:,:3].square().mean(),rotation_prior=delta[owner,:,3:].square().mean(),
                    relative_view_prior=relative.square().mean(),motion_preservation_prior=preservation[owner])
                objective=observations['slice'][owner]+self.loss_config['capture_prior_weight']*observations['capture_prior'][owner]
                objective=objective+sum(cfg['regularization'][key]*value for key,value in terms.items())
                objectives.append(objective);prior_terms.append(terms)
            sum(objectives[i] for i in active).backward()
            gradients=[]
            for owner in active:
                zg,dg=z_values[owner].grad,deltas[owner].grad
                if zg is None or dg is None or not torch.isfinite(zg).all() or not torch.isfinite(dg).all():
                    raise ValueError('Joint Shape/Pose gradients must be finite and present')
                gradients.append((owner,dict(shape=float(zg.norm()),translation=float(dg[:,:3].norm()),rotation=float(dg[:,3:].norm()))))
                if arm=='pose_only' or (arm=='alternate' and step%2==0):z_values[owner].grad=None
                if arm=='shape_only' or (arm=='alternate' and step%2==1):deltas[owner].grad=None
                optimizers[owner].step();self.project_shape(z_values[owner],initial_z[owner])
                project_pose(deltas[owner],cfg['translation_limit_mm'],cfg['rotation_limit_degrees'])
            with torch.no_grad():
                candidates=motion.decode(torch.stack(z_values),mode)
                proposed_poses=estimated_pose(torch.stack(deltas),base);motion_reports=motion.check(candidates)
            for owner,gradient in gradients:
                drift=(candidates[owner,0]-baseline[owner,0]).norm(dim=-1);reasons=[]
                if float(drift.max())>cfg['all_vertex_max_mm']:reasons.append('reference_drift')
                if unknown[owner].any() and float(drift[unknown[owner]].max())>cfg['unknown_max_mm']:reasons.append('unobserved_structure_drift')
                if not motion_reports[owner]['passed']:reasons.append('motion_preservation')
                geometry=None;native_score=None;stats=None
                if not reasons:
                    geometry=self.guard.audit(candidates[owner].cpu().numpy())
                    if not geometry['passed']:reasons.append('geometry')
                if not reasons:
                    rows=self.evaluator.selection_cells(candidates[owner].cpu().numpy(),
                        {key:value[owner].cpu().numpy() for key,value in proposed_poses.items()},patients[owner],'joint_candidate')
                    if arm=='ed_only':rows=[r for r in rows if (r['view'],r['chamber'],r['native_frame']) in ed_keys[owner]]
                    stats=dice_stats(rows);native_score=list(score(rows));base_stats=baseline_stats[owner]
                    if stats['minimum']<base_stats['minimum']-cfg['tail_max_drop']:reasons.append('native_minimum_degradation')
                    if stats['P05']<base_stats['P05']-cfg['tail_max_drop']:reasons.append('native_P05_degradation')
                    if stats['ge098']<base_stats['ge098']:reasons.append('native_098_coverage_degradation')
                proposed=dict(z_shape=cpu(z_values[owner]),delta=cpu(deltas[owner]))
                proposed_lr_factor=lr_factors[owner]
                if reasons:
                    rollback(z_values[owner],deltas[owner],optimizers[owner],previous[owner])
                    # Adam/parameters are restored in full. Only the declared
                    # separate step-size policy advances after a rejection.
                    lr_factors[owner]=max(cfg.get('minimum_lr_factor',.001),
                        lr_factors[owner]*cfg.get('rejection_lr_backoff',.5))
                elif native_score>best[owner]['score']:
                    best[owner]=dict(z_shape=z_values[owner].detach().clone(),delta=deltas[owner].detach().clone(),score=native_score,geometry=geometry)
                histories[owner].append(dict(update=step+1,objective=float(objectives[owner].detach()),
                    components={key:float(value[owner].detach()) for key,value in observations['components'].items()},
                    priors=cpu(prior_terms[owner]),gradients=gradient,accepted=not reasons,rejected_by=reasons,
                    proposed=proposed,retained=dict(z_shape=cpu(z_values[owner]),delta=cpu(deltas[owner])),
                    native_stats=stats,geometry=geometry,motion=motion_reports[owner],
                    proposed_lr_factor=proposed_lr_factor,next_lr_factor=lr_factors[owner],
                    parameter_groups_updated=(['pose'] if arm=='pose_only' or (arm=='alternate' and step%2==0) else
                        ['shape'] if arm=='shape_only' or (arm=='alternate' and step%2==1) else ['shape','pose']),
                    supervision_cells=sum(r['owner']==owner for r in supervisor.active),
                    supervised_phases=sorted({phase for r in supervisor.active if r['owner']==owner
                        for phase in ([r['phase'],r['mesh_upper']] if r['mesh_alpha']>0 else [r['phase']])})))
                if checkpoint:
                    checkpoint(patients[owner],dict(step=step+1,z_shape=z_values[owner],delta=deltas[owner],
                        optimizer=optimizers[owner].state_dict(),best=best[owner],history=histories[owner],
                        frozen_fields=motion.fields[owner],z_motion_initial=motion.z_motion[owner],
                        lr_factor=lr_factors[owner],rng=rng_state(device)))
        with torch.no_grad():
            final_z=torch.stack([state['z_shape'].to(device) for state in best])
            final_delta=torch.stack([state['delta'].to(device) for state in best])
            final_pose=estimated_pose(final_delta,base)
            frozen_trajectory=motion.decode(final_z,'frozen_field')
            original_trajectory=motion.decode(final_z,'original_decoder')
            original_controls=self.prior.decode_original(final_z,motion.z_motion,motion.conditions)['control_grids']
            final_trajectory=original_trajectory if mode=='original_decoder' else frozen_trajectory
            final_motion=motion.check(final_trajectory)
            original_motion=motion.check(original_trajectory)
        motion.assert_frozen();self.prior.assert_frozen()
        return [dict(z_shape=final_z[i],pose={k:v[i] for k,v in final_pose.items()},
            trajectory=final_trajectory[i],frozen_field_trajectory=frozen_trajectory[i],original_decoder_trajectory=original_trajectory[i],
            geometry=best[i]['geometry'],motion_report=final_motion[i],history=histories[i],
            original_decoder_motion_report=original_motion[i],
            original_decoder_control_fields=original_controls['fine_motion'][i],
            supervision_manifest=[row for row in supervisor.rows if row['owner']==i],
            status='COMPLETE' if admitted[i] else 'INITIAL_PRIOR_NOT_ADMITTED',arm=arm,motion_mode=mode,
            unobserved_vertices=unknown[i],initial_geometry=geometry_initial[i]) for i in range(batch)]
