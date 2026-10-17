"""Full-cycle gradient, loss parity, patient isolation and GPU resource contracts."""
import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import trimesh

from data.mask_transform import CHAMBERS
from tto.config import load_config
from tto.dataset import PatientObservations, DEFAULT_VALID, sdf_mm
from tto.gpu import select_gpu, WaitingForGPUResources
from tto.io import ROOT, cpu
from tto.losses import RealTTOLoss, observation_cells
from tto.motion import MotionPropagation
from tto.pose import estimated_pose
from tto.prior import FrozenNeuralFFDPrior
from tto.shape_pose_joint import rollback
from tto.shape_pose_joint import JointShapePoseOptimizer
from tto.io import read_json
from utils.slice_supervision import section_soft_mask, MeshSliceSupervisor


def small_fixture(batch=1):
    mesh=trimesh.creation.icosphere(subdivisions=1,radius=5.)
    vertices=torch.tensor(mesh.vertices,dtype=torch.float32)*torch.tensor([1.,.8,1.2])
    model=SimpleNamespace(template_vertices_mm=torch.cat([vertices]*4),
        chamber_ranges=torch.tensor([[i*len(vertices),(i+1)*len(vertices)] for i in range(4)]),
        faces={c:torch.tensor(mesh.faces) for c in CHAMBERS},norm_to_mm=1.)
    patients=[]
    for owner in range(batch):
        views=[]
        for vi in range(3):
            h,w=24+2*vi+owner,30+2*vi+owner;yy,xx=np.mgrid[:h,:w]
            masks=np.zeros((24,4,h,w),np.uint8)
            for phase in range(24):
                for ci in range(4):
                    if DEFAULT_VALID[vi,ci]:masks[phase,ci]=((xx-w/2-.3)**2+(yy-h/2+.2)**2)<(4+.1*np.sin(phase))**2
            views.append(dict(masks=masks,spacing=np.array([.7,1.1]),
                sources=[['synthetic_observed']*4]*24,weights=np.broadcast_to(DEFAULT_VALID[vi],(24,4)).astype(np.float32),
                annotated=np.broadcast_to(DEFAULT_VALID[vi],(24,4)).copy()))
        mapping=[[dict(nearest_index=t,nearest_native_frame=t,nearest_time_error_ms=0.) for _ in range(3)] for t in range(24)]
        patients.append(PatientObservations(f'patient_{owner}',views,mapping,{},
            dict(contract={'acquisition_status':'synthetic source phase mapping verified'})))
    trajectory=model.template_vertices_mm[None,None].repeat(batch,24,1,1)
    pose=dict(R=torch.eye(3)[None,None].repeat(batch,3,1,1),
              center_mm=torch.tensor([.4,.2,.7])[None,None].repeat(batch,3,1))
    return model,patients,trajectory,pose


class JointTTOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config=load_config(ROOT/'configs/real_tto_joint_v1.yaml')

    def test_all24_native_phases_have_shape_and_pose_gradients(self):
        model,patients,trajectory,base=small_fixture()
        trajectory.requires_grad_();delta=torch.zeros(1,3,6,requires_grad=True)
        supervisor=RealTTOLoss(model,patients,self.config['loss'],self.config['supervision'])
        result=supervisor(trajectory,estimated_pose(delta,base));result['slice'].sum().backward()
        self.assertEqual(len(result['cell_indices']),24*8)
        self.assertEqual({i[1] for i in result['cell_indices']},set(range(24)))
        self.assertTrue((trajectory.grad.flatten(2).norm(dim=2)>0).all())
        self.assertGreater(float(delta.grad[...,:3].norm()),0.)
        self.assertGreater(float(delta.grad[...,3:].norm()),0.)

    def test_original_binary_dice_bce_boundary_parity(self):
        model,patients,trajectory,pose=small_fixture()
        cfg=dict(self.config['loss'],pixel_stride=2)
        result=RealTTOLoss(model,patients,cfg,self.config['supervision'],phases=[0])(trajectory,pose)
        legacy=MeshSliceSupervisor(model.chamber_ranges,model.faces,1.,pixel_stride=2)
        expected={key:[] for key in ('dice','bce','boundary_mm')}
        native=patients[0].native(0)
        for owner,phase,view,ci in result['cell_indices']:
            a,b=model.chamber_ranges[ci].tolist();mask=native['masks'][view][ci]
            rendered=section_soft_mask(trajectory[owner,phase,a:b],model.faces[CHAMBERS[ci]],
                pose['R'][owner,view],pose['center_mm'][owner,view],native['spacing'][view],mask.shape,
                sigma_mm=1.,pixel_stride=2,chunk_size=256)
            truth=torch.tensor(mask[::2,::2].flatten(),dtype=torch.float32)[None]
            field=torch.tensor(native['sdf'][view][ci][::2,::2].flatten())[None]
            terms=legacy._terms(rendered['logits'].flatten()[None],truth,field)
            for key,value in terms.items():expected[key].append(value[0])
        for key,values in expected.items():
            torch.testing.assert_close(result['components'][key][0],torch.stack(values).mean(),atol=2e-6,rtol=2e-6)

    def test_batch_single_and_no_cross_patient_gradients(self):
        model,patients,trajectory,pose=small_fixture(2)
        delta=torch.zeros(2,3,6,requires_grad=True)
        result=RealTTOLoss(model,patients,self.config['loss'],self.config['supervision'],phases=[1,11,23])(
            trajectory,estimated_pose(delta,pose))['slice']
        result[0].backward();self.assertEqual(float(delta.grad[1].abs().max()),0.)
        for owner in range(2):
            single=RealTTOLoss(model,[patients[owner]],self.config['loss'],self.config['supervision'],phases=[1,11,23])(
                trajectory[owner:owner+1],{k:v[owner:owner+1] for k,v in pose.items()})['slice'][0]
            torch.testing.assert_close(result[owner],single,atol=2e-6,rtol=2e-6)

    def test_membership_does_not_use_dice_and_time_failure_is_reported(self):
        _,patients,_,_=small_fixture()
        patients[0].mapping[7][2]['nearest_time_error_ms']=100.
        rows=observation_cells(patients,self.config['supervision'])
        rejected=[r for r in rows if r['phase']==7 and r['view']==2]
        self.assertEqual(len(rejected),4);self.assertTrue(all(not r['used'] for r in rejected))
        self.assertTrue(all(r['used'] for r in rows if r not in rejected))

    def test_real_native_frames_used_once_and_fractional_time_reaches_both_meshes(self):
        model,patients,trajectory,pose=small_fixture()
        patient=patients[0]
        patient.metadata['contract']=dict(cycle_start_s=.15,cycle_duration_s=1.6,anchor_time_s=.4)
        for view in patient.views:
            view.update(times=np.arange(24)*.1,frame_ids=list(range(24)))
        rows=observation_cells(patients,self.config['supervision'])
        keys=[(r['native_frame'],r['view'],r['chamber']) for r in rows]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual({r['native_frame'] for r in rows},set(range(2,18)))
        self.assertEqual(len(rows),16*8)
        for view in patient.views:view['weights'][:]=0
        patient.views[2]['weights'][3,0]=1.
        supervisor=RealTTOLoss(model,patients,self.config['loss'],self.config['supervision'])
        self.assertEqual(len(supervisor.active),1)
        row=supervisor.active[0];self.assertAlmostEqual(row['mesh_alpha'],.5)
        trajectory.requires_grad_();supervisor(trajectory,pose)['slice'].sum().backward()
        self.assertGreater(float(trajectory.grad[0,row['phase']].norm()),0.)
        self.assertGreater(float(trajectory.grad[0,row['mesh_upper']].norm()),0.)
        inactive=set(range(24))-{row['phase'],row['mesh_upper']}
        self.assertEqual(float(trajectory.grad[0,sorted(inactive)].abs().max()),0.)

    def test_rolls_back_shape_pose_and_full_adam_state(self):
        z=torch.nn.Parameter(torch.ones(3));delta=torch.nn.Parameter(torch.ones(3,6))
        opt=torch.optim.Adam([z,delta],lr=.1)
        previous=(z.detach().clone(),delta.detach().clone(),opt.state_dict())
        (z.square().sum()+delta.square().sum()).backward();opt.step()
        self.assertTrue(opt.state)
        rollback(z,delta,opt,previous)
        torch.testing.assert_close(z,previous[0]);torch.testing.assert_close(delta,previous[1]);self.assertEqual(len(opt.state),0)

    def test_gpu_admission_requires_measured_peak_and_respects_headroom(self):
        devices=[dict(physical_index=3,visible_cuda_index=0,uuid='GPU-test',name='fixture',
                      total_mib=10000.,used_mib=7000.,free_mib=3000.,utilization=0.,processes=[{'pid':123}])]
        with self.assertRaises(WaitingForGPUResources):
            select_gpu('auto',lambda d:dict(peak_reserved_mib=2500.,context_mib=200.),discover=lambda:devices)
        result=select_gpu('cuda:0',lambda d:dict(peak_reserved_mib=1000.,context_mib=200.),discover=lambda:devices)
        self.assertEqual(result['selected']['physical_index'],3)
        self.assertEqual(result['torch_device'],'cuda:0')
        self.assertEqual(result['cuda_visible_devices'],'GPU-test')
        self.assertGreater(result['attempts'][0]['required_mib'],1200.)

    def test_checkpoint_dual_motion_forward_and_late_phase_latent_gradient(self):
        cfg=self.config
        prior=FrozenNeuralFFDPrior(ROOT/cfg['prior']['checkpoint'],expected_sha256=cfg['prior']['sha256'])
        zs=torch.zeros(1,64);zm=torch.zeros(1,32)
        with torch.no_grad():out=prior.decode_original(zs,zm,{})
        initial=dict(z_motion_initial=zm,vertices_4d_initial=out['vertices_4d'],
                     control_grids_initial=out['control_grids'],conditions={})
        motion=MotionPropagation(prior,initial,cfg['motion_limits'])
        with torch.no_grad():self.assertTrue(torch.equal(motion.decode(zs),out['vertices_4d']))
        updated=(zs+.01).requires_grad_();locked=motion.decode(updated)
        locked[:,23].square().mean().backward()
        self.assertGreater(float(updated.grad.norm()),0.)
        with torch.no_grad():original=motion.decode(updated.detach(),'original_decoder')
        self.assertGreater(float((locked.detach()-original).abs().max()),0.)
        motion.assert_frozen();prior.assert_frozen()

    def test_actual_joint_solver_rejects_and_restores_complete_patient_state(self):
        cfg=self.config
        prior=FrozenNeuralFFDPrior(ROOT/cfg['prior']['checkpoint'],expected_sha256=cfg['prior']['sha256'])
        _,patients,_,_=small_fixture()
        calibration=read_json(ROOT/cfg['prior']['calibration'])
        pose={key:torch.tensor(calibration['initial_pose'][key],dtype=torch.float32) for key in ('R','center_mm')}
        with torch.no_grad():out=prior.decode_original(torch.zeros(1,64),torch.zeros(1,32),{})
        initial=dict(z_shape_initial=torch.zeros(1,64),z_motion_initial=torch.zeros(1,32),
                     vertices_4d_initial=out['vertices_4d'],control_grids_initial=out['control_grids'],conditions={})
        class Guard:
            calls=0
            def audit(self,*args):
                self.calls+=1
                return dict(passed=self.calls==1,reason='injected rejection after a legal initialization')
        class Evaluator:
            def selection_cells(self,*args):return [dict(exact_hard_dice=.5)]
        guard=Guard();snapshots=[]
        optimizer=JointShapePoseOptimizer(prior,dict(cfg['joint'],updates=1,shape_lr=1e-6,pose_lr=1e-5),
            cfg['loss'],cfg['supervision'],cfg['motion_limits'],guard,Evaluator(),calibration)
        result=optimizer.optimize(patients,initial,[pose],checkpoint=lambda p,s:snapshots.append(deepcopy(cpu(s))))[0]
        self.assertEqual(len(snapshots),1)
        self.assertFalse(result['history'][0]['accepted'])
        self.assertIn('geometry',result['history'][0]['rejected_by'])
        self.assertEqual(result['history'][0]['supervised_phases'],list(range(24)))
        torch.testing.assert_close(snapshots[0]['z_shape'],torch.zeros(64),atol=0,rtol=0)
        torch.testing.assert_close(snapshots[0]['delta'],torch.zeros(3,6),atol=0,rtol=0)
        self.assertEqual(len(snapshots[0]['optimizer']['state']),0)
        self.assertEqual(snapshots[0]['lr_factor'],.5)
        prior.assert_frozen()

    def test_actual_joint_resume_matches_uninterrupted_adam_and_rejects_changed_motion(self):
        cfg=self.config
        prior=FrozenNeuralFFDPrior(ROOT/cfg['prior']['checkpoint'],expected_sha256=cfg['prior']['sha256'])
        with self.assertRaises(ValueError):prior.encode({'pose_gt':{}})
        _,patients,_,_=small_fixture()
        calibration=read_json(ROOT/cfg['prior']['calibration'])
        pose={key:torch.tensor(calibration['initial_pose'][key],dtype=torch.float32) for key in ('R','center_mm')}
        with torch.no_grad():out=prior.decode_original(torch.zeros(1,64),torch.zeros(1,32),{})
        initial=dict(z_shape_initial=torch.zeros(1,64),z_motion_initial=torch.zeros(1,32),
            vertices_4d_initial=out['vertices_4d'],control_grids_initial=out['control_grids'],conditions={})
        class Guard:
            def audit(self,*args):return {'passed':True}
        class Evaluator:
            def selection_cells(self,*args):return [dict(exact_hard_dice=.5,phase=0)]
        optimizer=JointShapePoseOptimizer(prior,dict(cfg['joint'],updates=2,shape_lr=1e-6,pose_lr=1e-5),
            cfg['loss'],cfg['supervision'],cfg['motion_limits'],Guard(),Evaluator(),calibration)
        uninterrupted=[]
        optimizer.optimize(patients,initial,[pose],checkpoint=lambda p,s:uninterrupted.append(deepcopy(cpu(s))))
        first=[]
        def interrupt(patient,state):
            first.append(deepcopy(cpu(state)))
            raise InterruptedError('simulated interruption after atomic checkpoint')
        with self.assertRaises(InterruptedError):optimizer.optimize(patients,initial,[pose],checkpoint=interrupt)
        resumed=[]
        optimizer.optimize(patients,initial,[pose],resumes=[first[0]],checkpoint=lambda p,s:resumed.append(deepcopy(cpu(s))))
        self.assertEqual(resumed[-1]['step'],2)
        self.assertTrue(uninterrupted[-1]['history'][0]['accepted'])
        self.assertTrue(uninterrupted[-1]['optimizer']['state'])
        for key in ('z_shape','delta'):
            torch.testing.assert_close(resumed[-1][key],uninterrupted[-1][key],atol=0,rtol=0)
        for parameter,state in uninterrupted[-1]['optimizer']['state'].items():
            for key,value in state.items():torch.testing.assert_close(resumed[-1]['optimizer']['state'][parameter][key],value,atol=0,rtol=0)
        corrupt=deepcopy(first[0]);corrupt['frozen_fields'][1,0,0,0,0]+=1.
        with self.assertRaises(ValueError):optimizer.optimize(patients,initial,[pose],resumes=[corrupt])


if __name__=='__main__':unittest.main()
