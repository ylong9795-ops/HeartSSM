"""Versioned full-cycle Shape/Pose pipeline, shared by real and hidden-GT cases."""
import csv
import importlib.metadata
import platform
import random
import shutil
import time
import traceback
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

from .config import resolve
from .dataset import RealTTODataset, collate_patients
from .evaluation import RealTTOEvaluator,proportions
from .evaluator import ExactNativeEvaluator
from .geometry_guard import TTOGeometryGuard
from .io import (ROOT, cpu, device_tree, digest_json, load_state, read_json,
                 save_state, sha256, write_json)
from .losses import RealTTOLoss, observation_cells
from .motion import MotionPropagation
from .pose import MultiViewPoseInitializer, estimated_pose
from .prior import FrozenNeuralFFDPrior
from .shape_pose_joint import JointShapePoseOptimizer

STAGES=('initial','pose_init','joint','evaluate','export')


def split_batch(value, index):
    if isinstance(value,torch.Tensor):return value[index:index+1].detach().clone()
    if isinstance(value,dict):return {k:split_batch(v,index) for k,v in value.items()}
    return value


def concatenate(values):
    if isinstance(values[0],torch.Tensor):return torch.cat(values,0)
    if isinstance(values[0],dict):return {key:concatenate([value[key] for value in values]) for key in values[0]}
    if any(v!=values[0] for v in values[1:]):raise ValueError('Batch metadata/condition identity differs')
    return values[0]


class RealTTOPipeline:
    def __init__(self,config):
        self.config=config;self.output=resolve(config['output']['root']);self.output.mkdir(parents=True,exist_ok=True)
        torch.set_num_threads(config['runtime']['threads'])
        torch.manual_seed(config['runtime']['seed']);np.random.seed(config['runtime']['seed']);random.seed(config['runtime']['seed'])
        self.device=torch.device(config['runtime']['device'])
        if self.device.type=='cuda':
            torch.cuda.set_device(self.device);torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        self.sources={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'tto').glob('*.py'))}
        reused=('train_neural_ffd.py','net/neural_ffdnet.py','net/multiview_encoder.py','data/mask_transform.py',
                'data/synthetic_4d.py','utils/ffd_physical.py','utils/slice_supervision.py','utils/slice_evaluation.py',
                'utils/neural_ffd_losses.py','utils/geometry_pair_policy.py','utils/geometry_v2_checks.py')
        self.sources.update({p:sha256(ROOT/p) for p in reused})
        scientific={k:v for k,v in config.items() if k not in ('runtime','output')}
        validation=deepcopy(scientific)
        validation['experiment'].pop('arm',None)
        validation['experiment'].pop('initialization_source',None)
        validation['synthetic'].pop('validation_gate',None)
        self.protocol=dict(version=config['version'],scientific_config=scientific,config_sha256=digest_json(scientific),
            validation_sha256=digest_json(validation),sources=self.sources,source_sha256=digest_json(self.sources),seed=config['runtime']['seed'])
        protocol_path=self.output/'protocol.json'
        if protocol_path.exists() and read_json(protocol_path)!=self.protocol:
            raise ValueError('Output protocol/source identity changed; use a new versioned directory')
        write_json(protocol_path,self.protocol)
        for name,expected in self.sources.items():
            target=self.output/'source_snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha256(target)!=expected:raise ValueError('Archived source identity differs')
            else:shutil.copyfile(ROOT/name,target)
        self.prior=FrozenNeuralFFDPrior(resolve(config['prior']['checkpoint']),device=self.device,
                                       expected_sha256=config['prior']['sha256'])
        self.model=self.prior.model;self.calibration=read_json(resolve(config['prior']['calibration']))
        saved_slice=self.prior.saved['config']['slice']
        for key in ('sigma_mm','pixel_stride','chunk_size','dice_weight','bce_weight','boundary_weight'):
            if config['loss'][key]!=saved_slice[key]:
                raise ValueError(f'Primary baseline must retain checkpoint-embedded slice setting {key}')
        if self.calibration['checkpoint_sha256']!=self.prior.checkpoint_sha256:raise ValueError('Foreign TRAIN calibration')
        self.guard=TTOGeometryGuard(self.model);self.evaluator=RealTTOEvaluator(self.model,config['supervision'])
        self.pose_initializer=MultiViewPoseInitializer(self.model,config['pose'],config['loss'],config['supervision'],
                                                     self.evaluator,config['runtime']['seed'])
        self.optimizer=JointShapePoseOptimizer(self.prior,config['joint'],config['loss'],config['supervision'],
            config['motion_limits'],self.guard,self.evaluator,self.calibration)
        packages={}
        for package in ('torch','torchvision','numpy','scipy','trimesh','pydicom','cgal','ipctk'):
            try:packages[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:packages[package]='UNAVAILABLE'
        write_json(self.output/'model_identity.json',dict(checkpoint_sha256=self.prior.checkpoint_sha256,
            parameter_buffer_sha256=self.prior.fingerprint,specification=self.prior.saved['specification'],
            saved_config=self.prior.saved.get('config'),python=platform.python_version(),packages=packages,
            torch_cuda=torch.version.cuda,source_sha256=self.sources,
            trained_conditions=dict(reference_spatial=self.model.spatial_fusion is not None,
                                    dynamic_spatial=self.model.dynamic_conditioner is not None)))

    def identity(self,patient):
        return dict(patient=patient.patient_id,input_sha256=patient.identity,
            protocol_sha256=digest_json(self.protocol),checkpoint_sha256=self.prior.checkpoint_sha256,
            calibration_sha256=sha256(resolve(self.config['prior']['calibration'])))

    def patient_dir(self,patient):return self.output/'patients'/patient.patient_id

    def checkpoint_callback(self,stage):
        def callback(patient,progress):
            save_state(self.output/'checkpoints'/patient.patient_id/(stage+'.pth'),
                       dict(identity=self.identity(patient),stage=stage,progress=progress))
        return callback

    def resumes(self,patients,stage,resume):
        values=[]
        for patient in patients:
            path=self.output/'checkpoints'/patient.patient_id/(stage+'.pth')
            values.append(load_state(path,self.identity(patient),self.device)['progress'] if resume and path.exists() else None)
        return values

    def import_initialization(self,patient,stage):
        """Reuse only matching NEW-protocol initial/Pose states across ablations."""
        source=self.config['experiment'].get('initialization_source')
        if not source or stage not in ('initial','pose_init'):return None
        if self.config['experiment']['arm']=='frozen_prior' and stage=='pose_init':return None
        source=resolve(source);protocol=read_json(source/'protocol.json')
        if (protocol.get('validation_sha256')!=self.protocol['validation_sha256'] or
            protocol['source_sha256']!=self.protocol['source_sha256'] or protocol['seed']!=self.protocol['seed']):
            raise ValueError('Ablation initialization data/algorithm/budget identity differs')
        path=source/'patients'/patient.patient_id/'states'/(stage+'.pth')
        if not path.exists():return None
        saved=load_state(path,device=self.device);identity=saved['identity']
        if identity['input_sha256']!=patient.identity or identity['checkpoint_sha256']!=self.prior.checkpoint_sha256:
            raise ValueError('Foreign patient/model initialization cannot be reused')
        state=saved['state']
        if 'result' in state:raise ValueError('Optimized endpoints are forbidden as Joint initializations')
        write_json(self.patient_dir(patient)/f'{stage}_reuse.json',dict(source=str(path),sha256=sha256(path),
            source_identity=identity,target_identity=self.identity(patient),scope='identical initialization, never Shape/Joint endpoint'))
        save_state(self.patient_dir(patient)/'states'/(stage+'.pth'),dict(identity=self.identity(patient),stage=stage,state=state))
        if stage=='initial':
            write_json(self.patient_dir(patient)/'input_manifest.json',patient.metadata)
            write_json(self.patient_dir(patient)/'supervision_manifest.json',observation_cells([patient],self.config['supervision']))
        return state

    def initialize(self,patients):
        initial=self.prior.initialize(collate_patients(patients)['observations'])
        motion=MotionPropagation(self.prior,initial,self.config['motion_limits'])
        with torch.no_grad():replay=motion.decode(initial['z_shape_initial'])
        if not torch.equal(replay,initial['vertices_4d_initial']):raise AssertionError('Frozen-field initial replay differs')
        states=[]
        for index,patient in enumerate(patients):
            estimate=patient.metadata['contract'].get('initial_pose',self.calibration['initial_pose'])
            pose={key:torch.as_tensor(estimate[key],device=self.device,dtype=torch.float32) for key in ('R','center_mm')}
            if pose['R'].shape!=(3,3,3) or pose['center_mm'].shape!=(3,3):raise ValueError('Estimated pose dimensions differ')
            r=pose['R']
            if not torch.isfinite(r).all() or not torch.isfinite(pose['center_mm']).all() or not torch.allclose(r.transpose(-1,-2)@r,torch.eye(3,device=self.device).expand(3,3,3),atol=1e-4,rtol=0) or (torch.linalg.det(r)<=0).any():
                raise ValueError('Estimated pose must be finite SO(3) and physical millimetres')
            geometry=self.guard.audit(replay[index].cpu().numpy())
            states.append(dict(initial=split_batch(initial,index),pose_initial=pose,pose=pose,
                geometry_initial=geometry,forward_consistency=initial['forward_consistency'],
                initial_motion_replay_bitwise=True))
            write_json(self.patient_dir(patient)/'input_manifest.json',patient.metadata)
            write_json(self.patient_dir(patient)/'supervision_manifest.json',observation_cells([patient],self.config['supervision']))
        return states

    def process(self,patients,*,stage='all',resume=False):
        """The sole optimization route; synthetic adapters call this exact method."""
        requested=STAGES if stage=='all' else STAGES[:STAGES.index(stage)+1]
        states=[None]*len(patients)
        for current in requested:
            pending=[]
            for index,patient in enumerate(patients):
                path=self.patient_dir(patient)/'states'/(current+'.pth')
                if path.exists():
                    if not resume:raise FileExistsError(f'Use --resume or a fresh directory: {path}')
                    states[index]=load_state(path,self.identity(patient),self.device)['state']
                else:
                    imported=self.import_initialization(patient,current)
                    if imported is None:pending.append(index)
                    else:states[index]=imported
            if not pending:continue
            subset=[patients[i] for i in pending];inputs=[states[i] for i in pending]
            print('JOINT_TTO_STAGE',current,[p.patient_id for p in subset],flush=True)
            if current=='initial':outputs=self.initialize(subset)
            else:
                outputs=inputs
                initial=concatenate([state['initial'] for state in inputs])
                if current=='pose_init':
                    if self.config['experiment']['arm']=='frozen_prior':
                        results=[dict(pose=cpu(s['pose_initial']),history=[],shared_across_phases=True,
                                      optimization='none: Frozen Prior control') for s in inputs]
                    else:
                        results=self.pose_initializer.optimize(subset,initial['vertices_4d_initial'],[s['pose_initial'] for s in inputs],
                            resumes=self.resumes(subset,current,resume),checkpoint=self.checkpoint_callback(current))
                    for state,result in zip(outputs,results):
                        state.update(pose=device_tree(result['pose'],self.device),pose_init_result=result)
                elif current=='joint':
                    results=self.optimizer.optimize(subset,initial,[s['pose'] for s in inputs],arm=self.config['experiment']['arm'],
                        resumes=self.resumes(subset,current,resume),checkpoint=self.checkpoint_callback(current))
                    for state,result in zip(outputs,results):state.update(result=result)
                elif current=='evaluate':
                    for patient,state in zip(subset,outputs):self.evaluate(patient,state)
                elif current=='export':
                    for patient,state in zip(subset,outputs):self.export(patient,state)
            for index,patient,state in zip(pending,subset,outputs):
                self.prior.assert_frozen()
                save_state(self.patient_dir(patient)/'states'/(current+'.pth'),dict(identity=self.identity(patient),stage=current,state=state))
                states[index]=state
        return states

    def evaluate(self,patient,state):
        initial=state['initial'];result=state['result'];folder=self.patient_dir(patient)/'evaluation'
        comparisons={}
        for name,trajectory,pose in (
            ('initial',initial['vertices_4d_initial'][0],state['pose_initial']),
            ('pose_only_initialization',initial['vertices_4d_initial'][0],state['pose']),
            ('final',result['trajectory'],result['pose']),
            ('original_decoder_at_final_shape',result['original_decoder_trajectory'],result['pose'])):
            rows=self.evaluator.annotated_cells(trajectory.detach().cpu().numpy(),
                {key:value.detach().cpu().numpy() for key,value in pose.items()},patient,name)
            comparisons[name]=self.evaluator.summarize(rows)
            path=folder/(name+'_cells.csv');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['patient']);writer.writeheader();writer.writerows(rows)
        report=dict(patient=patient.patient_id,status=result['status'],arm=result['arm'],comparisons=comparisons,
            geometry=result['geometry'],motion_relative_to_frozen_prior=result['motion_report'],
            initial_geometry=state['geometry_initial'],forward_consistency=state['forward_consistency'],
            pose_initial=state['pose_initial'],pose_final=result['pose'],
            shape_latent_standardized_drift=float(((result['z_shape']-initial['z_shape_initial'][0])/self.optimizer.std).square().mean().sqrt()),
            acquisition_status=patient.metadata['contract'].get('acquisition_status','UNVERIFIED'),
            phase_semantics='Real ED anchor is an estimate; synthetic frame0 is not guaranteed ED',
            real_3d_gt_metrics='UNAVAILABLE; not fabricated')
        report['original_decoder_comparison']=dict(
            geometry=self.guard.audit(result['original_decoder_trajectory'].detach().cpu().numpy()),
            motion=result['original_decoder_motion_report'],
            scope='Original Decoder at the selected Shape; independent from Frozen-Field geometry verdict')
        report['sampling_contract']='Primary: each original frame inside selected RR once; piecewise-linear mesh at its actual timestamp. Legacy 24-slot nearest-label evaluation is separate.'
        compatibility=ExactNativeEvaluator(self.model)
        report['legacy_model24_nearest_compatibility']={}
        for name,trajectory,pose in [('initial',initial['vertices_4d_initial'][0],state['pose_initial']),('final',result['trajectory'],result['pose'])]:
            rows=compatibility.cells(trajectory.detach().cpu().numpy(),{key:value.detach().cpu().numpy() for key,value in pose.items()},patient,name)
            report['legacy_model24_nearest_compatibility'][name]=proportions(rows)
        write_json(folder/'metrics.json',report);write_json(folder/'optimization_trace.json',result['history'])
        state['evaluation']=report

    def export(self,patient,state):
        import trimesh
        from data.mask_transform import CHAMBERS
        result=state['result'];initial=state['initial'];folder=self.patient_dir(patient)/'export';folder.mkdir(parents=True,exist_ok=True)
        arrays=dict(z_shape_initial=initial['z_shape_initial'][0],z_motion_initial=initial['z_motion_initial'][0],
            z_shape_final=result['z_shape'],vertices_reference_initial=initial['vertices_reference_initial'][0],
            vertices_reference_final=result['trajectory'][0],vertices_4d_initial=initial['vertices_4d_initial'][0],
            vertices_4d_frozen_field=result['frozen_field_trajectory'],vertices_4d_original_decoder=result['original_decoder_trajectory'],
            frozen_fields=initial['control_grids_initial']['fine_motion'][0],
            original_decoder_fields_at_final_shape=result['original_decoder_control_fields'],
            probe_R_initial=state['pose_initial']['R'],probe_t_initial_mm=state['pose_initial']['center_mm'],
            probe_R_joint_initial=state['pose']['R'],probe_t_joint_initial_mm=state['pose']['center_mm'],
            probe_R_final=result['pose']['R'],probe_t_final_mm=result['pose']['center_mm'])
        arrays={key:value.detach().cpu().numpy() for key,value in arrays.items()}
        arrays.update({f'faces_{c}':f.cpu().numpy() for c,f in self.model.faces.items()})
        arrays.update(chamber_ranges=self.model.chamber_ranges.cpu().numpy(),phases=np.arange(24)/24)
        np.savez_compressed(folder/'reconstruction.npz',**arrays)
        write_json(folder/'enabled_conditions.json',initial['enabled_conditions'])
        if self.config['output']['export_ply']:
            faces=np.concatenate([self.model.faces[c].cpu().numpy()+a for c,(a,b) in zip(CHAMBERS,self.model.chamber_ranges.tolist())])
            for phase,vertices in enumerate(result['trajectory'].detach().cpu().numpy()):
                target=folder/'meshes'/f'phase_{phase:02d}.ply';target.parent.mkdir(exist_ok=True)
                trimesh.Trimesh(vertices,faces,process=False).export(target)
        write_json(folder/'geometry.json',result['geometry']);write_json(folder/'motion.json',result['motion_report'])
        if self.config['output']['visualizations']:
            from .visualization import export_visualizations
            export_visualizations(self.model,patient,state,self.output/'visualizations'/patient.patient_id)
        write_json(folder/'completed.json',dict(identity=self.identity(patient),status=result['status'],arm=result['arm'],
            phases=24,legal4d=result['geometry']['passed'],motion_pass=result['motion_report']['passed']))

    def resource_probe(self,patients):
        """Actual 24-phase decoder and native-loss backward, not a guessed threshold."""
        torch.cuda.reset_peak_memory_stats(self.device);started=time.monotonic()
        initial=self.prior.initialize(collate_patients(patients)['observations'])
        motion=MotionPropagation(self.prior,initial,self.config['motion_limits'])
        z=initial['z_shape_initial'].detach().clone().requires_grad_()
        base={}
        for key in ('R','center_mm'):
            value=torch.tensor(self.calibration['initial_pose'][key],device=self.device,dtype=torch.float32)
            base[key]=value[None].expand(len(patients),*value.shape)
        delta=torch.zeros(len(patients),3,6,device=self.device,requires_grad=True)
        loss=RealTTOLoss(self.model,patients,self.config['loss'],self.config['supervision'])
        trajectory=motion.decode(z)
        trajectory.retain_grad()
        values=loss(trajectory,estimated_pose(delta,base));values['slice'].sum().backward()
        gradient_evidence=dict(shape_gradient_norm=float(z.grad.norm()),
            pose_translation_gradient_norm=float(delta.grad[...,:3].norm()),pose_rotation_gradient_norm=float(delta.grad[...,3:].norm()),
            per_phase_slice_vertex_gradient_norm=trajectory.grad.flatten(2).norm(dim=2).tolist())
        del values,trajectory
        z.grad=None;delta.grad=None
        nonreference=RealTTOLoss(self.model,patients,self.config['loss'],self.config['supervision'],phases=range(1,24))
        noref_values=nonreference(motion.decode(z),estimated_pose(delta,base))
        noref_values['slice'].sum().backward()
        gradient_evidence.update(nonreference_shape_gradient_norm=float(z.grad.norm()),
            nonreference_translation_gradient_norm=float(delta.grad[...,:3].norm()),
            nonreference_rotation_gradient_norm=float(delta.grad[...,3:].norm()),nonreference_cells=len(nonreference.active))
        del noref_values
        # Measure the larger multistart pose workload as well as the joint step.
        starts=self.config['pose']['starts']
        repeated=[patient for patient in patients for _ in range(starts)]
        fixed=initial['vertices_4d_initial'][:,None].expand(-1,starts,-1,-1,-1).reshape(-1,24,10631,3)
        repeated_base={key:value[:,None].expand(-1,starts,*value.shape[1:]).reshape(-1,*value.shape[1:]) for key,value in base.items()}
        pose_delta=torch.zeros(len(repeated),3,6,device=self.device,requires_grad=True)
        pose_loss=RealTTOLoss(self.model,repeated,self.config['loss'],self.config['supervision'])
        pose_values=pose_loss(fixed,estimated_pose(pose_delta,repeated_base))
        (pose_values['slice'].sum()+self.config['loss']['capture_prior_weight']*pose_values['capture_prior'].sum()).backward()
        torch.cuda.synchronize(self.device)
        measurement=dict(peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated(self.device)/2**20,context_mib=0.,
            elapsed_seconds=time.monotonic()-started,patients=[p.patient_id for p in patients],
            batch_size=len(patients),cells=len(loss.active),pose_multistart_cells=len(pose_loss.active),
            **gradient_evidence)
        # CPU geometry adds no GPU tensors; include a real complete audit in timing.
        with torch.no_grad():trajectory=motion.decode(z.detach())[0].cpu().numpy()
        geometry=self.guard.audit(trajectory);measurement['geometry_verified']=geometry['passed']
        measurement['elapsed_with_geometry_seconds']=time.monotonic()-started
        self.prior.assert_frozen()
        return measurement

    def run(self,patient_ids,*,stage='all',resume=False):
        if stage=='synthetic_oracle':
            from .synthetic_oracle import run_oracle
            return run_oracle(self,resume=resume)
        ids=patient_ids or self.config['experiment']['cohort'];completed=[];failures=[]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate patient IDs would alias optimization state')
        if any(Path(p).name!=p or p in ('.','..') for p in ids):raise ValueError('Patient IDs must be single directory names')
        if stage not in ('preflight','resource_probe','initial') and self.config['synthetic']['gate_before_real']:
            external=self.config['synthetic'].get('validation_gate')
            gate=resolve(external) if external else self.output/'synthetic_hidden_gt'/'gate.json'
            if stage=='all' and not gate.exists():
                from .synthetic_oracle import run_oracle
                run_oracle(self,resume=resume)
            if not gate.exists() or not read_json(gate).get('passed'):
                raise RuntimeError('Synthetic hidden-GT recovery gate has not passed; real optimization is not admitted')
            saved_gate=read_json(gate)
            if saved_gate.get('validation_sha256')!=self.protocol['validation_sha256'] or saved_gate['source_sha256']!=self.protocol['source_sha256']:
                raise RuntimeError('Synthetic recovery gate belongs to a different protocol/source')
            if saved_gate.get('seed')!=self.protocol['seed']:raise RuntimeError('Synthetic recovery gate has a different random seed')
        patients=[]
        if stage=='resource_probe' and self.config['runtime'].get('probe_source')=='synthetic':
            from .synthetic_oracle import select_oracle_cases,hidden_patient,validate_source
            dataset,indices,_=select_oracle_cases(self.config,self.calibration,self.config['runtime']['batch_size'])
            validate_source(dataset,self.prior)
            patients=[hidden_patient(dataset[index],f'SYN_probe_{index}')[0] for index in indices]
            return self.resource_probe(patients)
        for patient_id in ids:
            try:patients.append(RealTTODataset(resolve(self.config['data']['root']),[patient_id],config=self.config['data'])[0])
            except Exception as error:failures.append(dict(patient=patient_id,stage='load',error=repr(error)))
        if stage=='resource_probe':
            def render_workload(patient):
                return sum(np.prod(patient.views[r['view']]['masks'].shape[-2:])
                    for r in observation_cells([patient],self.config['supervision']) if r['used'])
            patients.sort(key=render_workload,reverse=True)
            return self.resource_probe(patients[:self.config['runtime']['batch_size']])
        if stage=='preflight':
            report=dict(geometry_backend='PASS' if self.guard.backend_error is None else 'UNVERIFIED',
                geometry_error=self.guard.backend_error,patients=[dict(patient=p.patient_id,input_sha256=p.identity,
                    source_cells=observation_cells([p],self.config['supervision'])) for p in patients],failures=failures)
            write_json(self.output/'preflight.json',report);return report
        size=self.config['runtime']['batch_size'];started=time.monotonic()
        for failure in failures:
            write_json(self.output/'patients'/failure['patient']/'failure.json',failure)
        write_json(self.output/'summary.json',dict(cohort=ids,cohort_denominator=len(ids),completed=completed,failures=failures))
        for offset in range(0,len(patients),size):
            batch=patients[offset:offset+size]
            try:
                self.process(batch,stage=stage,resume=resume);completed.extend(p.patient_id for p in batch)
            except Exception as error:
                failure=dict(patients=[p.patient_id for p in batch],stage=stage,error=repr(error),traceback=traceback.format_exc())
                failures.append(failure)
                for patient in batch:write_json(self.patient_dir(patient)/'failure.json',failure)
                print('JOINT_TTO_FAILURE',failure,flush=True)
            summary=dict(cohort=ids,cohort_denominator=len(ids),completed=completed,failures=failures,
                         elapsed_seconds=time.monotonic()-started)
            write_json(self.output/'summary.json',summary)
        return dict(completed=completed,failures=failures)
