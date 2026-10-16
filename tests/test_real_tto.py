"""CPU contract/renderer/gradient/rollback tests; no patient-specific fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from tto.config import load_config
from tto.dataset import RealTTODataset, collate_patients, DEFAULT_VALID
from tto.evaluator import native_exact
from tto.geometry_guard import GeometryGuard
from tto.io import ROOT, load_state, save_state
from tto.motion_lock import FrozenMotionPropagator, motion_guard
from tto.pose_optimizer import map_pose
from tto.slice_loss import NativeSliceLoss
from tto.synthetic_prior import FrozenSyntheticPrior
from tto.shape_optimizer import LatentShapeOptimizer
from tto.evaluator import ExactNativeEvaluator
from tto.pipeline import RealTTOPipeline
from tto.io import sha256
from data.mask_transform import CHAMBERS, VIEWS
from utils.slice_evaluation import section_mask


def fixture(root, patient, offset=0):
    folder = root/patient;folder.mkdir()
    views = []
    for vi, name in enumerate(VIEWS):
        h, w = 28+2*vi+offset, 36+2*vi+offset
        masks = np.zeros((4, 4, h, w), np.uint8)
        yy, xx = np.mgrid[:h, :w]
        for t in range(4):
            for ci in range(4):
                if DEFAULT_VALID[vi, ci]:
                    masks[t, ci] = (xx-w/2-ci*.5)**2+(yy-h/2)**2 < (4+t*.2)**2
        np.savez(folder/f'{name}.npz', masks=masks)
        np.save(folder/f'{name}.npy', np.zeros((4,h,w), np.uint8))
        views.append(dict(name=name, masks_npz=f'{name}.npz', images_npy=f'{name}.npy',
            spacing_yx_mm=[.7, 1.1], channels=list(CHAMBERS), times_s=[0., .25, .5, .75],
            annotated=np.broadcast_to(DEFAULT_VALID[vi], (4,4)).tolist(),
            sources=[['manual_ed_es' if c else 'not_applicable' for c in DEFAULT_VALID[vi]]]*4))
    manifest = dict(version='real-tto-input-v1', patient_id=patient, cycle_start_s=0.,
        cycle_duration_s=.5, anchor_time_s=0., ED_frame=0, ES_frame=1, views=views)
    (folder/'tto_input.json').write_text(json.dumps(manifest))


class RealTTOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture(self.root, 'new_patient');fixture(self.root, 'other_patient', 2)
        self.config = load_config(ROOT/'configs/real_tto_v1.yaml')['data']
        self.config.update(model_size=32, canonical_fov_mm={v:60. for v in VIEWS})

    def tearDown(self):
        self.temporary.cleanup()

    def test_heterogeneous_native_centres_sources_and_batch_isolation(self):
        dataset = RealTTODataset(self.root, ['new_patient','other_patient'], config=self.config)
        p, q = dataset[0], dataset[1]
        batch = collate_patients([p,q])
        self.assertEqual(tuple(batch['observations']['masks'].shape), (2,24,3,4,32,32))
        self.assertIs(batch['patients'][0], p)
        self.assertNotEqual(p.views[0]['masks'].shape, q.views[0]['masks'].shape)
        self.assertEqual(p.native()['sources'][0][0], 'manual_ed_es')
        self.assertEqual(float(p.native()['weights'][0,0]), 1.)
        self.assertEqual(p.native()['masks'][0].shape[-2:], (28,36))
        for vi, view in enumerate(p.views):
            transform = np.array(p.metadata['model_to_native'])[0, vi]
            centre = transform @ np.array([16.,16.,1.])
            h,w = view['masks'].shape[-2:]
            np.testing.assert_allclose(centre, [w/2,h/2,1.], atol=1e-10)
        self.assertEqual(len(p.images), 3)
        before = q.views[0]['masks'].copy();p.views[0]['masks'][0] = 0
        np.testing.assert_array_equal(before, q.views[0]['masks'])
        with self.assertRaises(ValueError):collate_patients([q,q])

    def test_foreign_patient_and_unknown_annotation_trust_rejected(self):
        path = self.root/'new_patient/tto_input.json';data = json.loads(path.read_text())
        data['patient_id'] = 'other_patient';path.write_text(json.dumps(data))
        dataset = RealTTODataset(self.root,['new_patient'],config=self.config)
        with self.assertRaises(ValueError):dataset[0]
        data['patient_id'] = 'new_patient';data['views'][0]['sources'][0][0] = 'unknown_auto_label'
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):dataset[0]

    def test_native_exact_square_parity_and_rectangular_physical_translation(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=12.)
        r = np.eye(3);spacing = [.7,1.1];center = np.array([.137,-.263,.619])
        for side in (64,128,224):
            actual = native_exact(mesh.vertices,mesh.faces,r,center,spacing,(side,side))
            expected = section_mask(mesh.vertices,mesh.faces,r,center,np.array([side*1.1/2,side*.7/2,1.]),image_size=side)
            np.testing.assert_array_equal(actual,expected)
        a = native_exact(mesh.vertices,mesh.faces,r,center,spacing,(70,94))
        b = native_exact(mesh.vertices+np.array([1.1,0,0]),mesh.faces,r,center,spacing,(70,94))
        c = native_exact(mesh.vertices+np.array([0,.7,0]),mesh.faces,r,center,spacing,(70,94))
        np.testing.assert_array_equal(a[:,:-1],b[:,1:]);np.testing.assert_array_equal(a[:-1],c[1:])

    def test_native_loss_batch_single_and_pose_gradients(self):
        patients = [RealTTODataset(self.root,[p],config=self.config)[0] for p in ('new_patient','other_patient')]
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=5.)
        v = torch.tensor(mesh.vertices, dtype=torch.float32)
        model = SimpleNamespace(template_vertices_mm=v, chamber_ranges=torch.tensor([[i*len(v),(i+1)*len(v)] for i in range(4)]),
            faces={c:torch.tensor(mesh.faces) for c in CHAMBERS})
        vertices = torch.cat([v+torch.tensor([ci*.5,0.,0.]) for ci in range(4)])[None].repeat(2,1,1)
        base = dict(R=torch.eye(3)[None,None].repeat(2,3,1,1), center_mm=torch.tensor([.4,.2,.7])[None,None].repeat(2,3,1))
        delta = torch.nn.Parameter(torch.zeros(2,3,6))
        loss = NativeSliceLoss(model,[p.native() for p in patients],stride=2)(vertices,map_pose(delta,base))[0]
        configured = load_config(ROOT/'configs/real_tto_v1.yaml')['loss']
        explicit = NativeSliceLoss(model,[p.native() for p in patients],stride=2,config=configured)(vertices,map_pose(delta,base))[0]
        torch.testing.assert_close(loss,explicit,atol=0,rtol=0)
        changed = NativeSliceLoss(model,[p.native() for p in patients],stride=2,
            config=dict(configured,sigma_mm=2.))(vertices,map_pose(delta,base))[0]
        self.assertGreater(float((changed-loss).abs().max()),0.)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertTrue((delta.grad[...,:3].flatten(1).norm(dim=-1)>0).all())
        self.assertTrue((delta.grad[...,3:].flatten(1).norm(dim=-1)>0).all())
        for i,p in enumerate(patients):
            single = NativeSliceLoss(model,[p.native()],stride=2)(vertices[i:i+1],{k:x[i:i+1] for k,x in base.items()})[0]
            torch.testing.assert_close(loss.detach()[i],single[0],atol=2e-6,rtol=2e-6)
        # Patient0's objective must have exactly zero derivative to patient1's Pose.
        delta2 = torch.nn.Parameter(torch.zeros(2,3,6))
        independent = NativeSliceLoss(model,[p.native() for p in patients],stride=2)(vertices,map_pose(delta2,base))[0]
        independent[0].backward();self.assertEqual(float(delta2.grad[1].abs().max()),0.)

    def test_empty_native_label_does_not_capture_an_absent_section(self):
        patient = RealTTODataset(self.root,['new_patient'],config=self.config)[0]
        native = patient.native()
        for vi in range(3):native['masks'][vi] = np.zeros_like(native['masks'][vi])
        mesh = trimesh.creation.icosphere(subdivisions=1,radius=5.)
        v = torch.tensor(mesh.vertices,dtype=torch.float32)
        model = SimpleNamespace(template_vertices_mm=v,chamber_ranges=torch.tensor([[i*len(v),(i+1)*len(v)] for i in range(4)]),
            faces={c:torch.tensor(mesh.faces) for c in CHAMBERS})
        pose = dict(R=torch.eye(3)[None,None].repeat(1,3,1,1),center_mm=torch.tensor([0.,0.,100.])[None,None].repeat(1,3,1))
        value,_,missing = NativeSliceLoss(model,[native])(torch.cat([v]*4)[None],pose)
        self.assertEqual(float(value),0.);self.assertEqual(missing,[])

    def test_atomic_resume_rejects_foreign_input(self):
        p = self.root/'state.pth'
        identity = dict(patient='arbitrary', input_sha256='abc')
        save_state(p,dict(identity=identity,parameter=torch.arange(3.),optimizer={'step':1},
            native_time_error=np.float64(.125),geometry=np.array([True,False]),source=np.str_('manual_ed_es')))
        restored=load_state(p,identity)
        self.assertTrue(torch.equal(restored['parameter'],torch.arange(3.)))
        self.assertEqual(restored['native_time_error'],.125)
        self.assertEqual(restored['geometry'],[True,False])
        with self.assertRaises(ValueError):load_state(p,dict(patient='foreign',input_sha256='abc'))

    def test_bad_patient_does_not_block_other_patient_preflight(self):
        config=load_config(ROOT/'configs/real_tto_v1.yaml')
        checkpoint=self.root/'dummy_checkpoint';checkpoint.write_bytes(b'unit-test-only')
        config['prior'].update(checkpoint=str(checkpoint),sha256=sha256(checkpoint))
        config['data']=dict(self.config,root=str(self.root))
        config['output']['root']=str(self.root/'result')
        config['runtime'].update(device='cpu',threads=2)
        pipeline=RealTTOPipeline(config)
        def frozen_stub():
            pipeline.prior=SimpleNamespace(assert_frozen=lambda:None)
            pipeline.guard=SimpleNamespace(backend_error=None)
            pipeline.model=SimpleNamespace(parameters=lambda:iter(()))
        with patch.object(pipeline,'load_frozen_prior',side_effect=frozen_stub), \
                patch.object(pipeline,'identity',side_effect=lambda p:{'patient':p.patient_id}):
            result=pipeline.run(['missing_patient','new_patient'],stage='preflight')
        self.assertEqual(result['completed'],['new_patient'])
        self.assertEqual(len(result['failures']),1)
        self.assertEqual(result['failures'][0]['patients'],['missing_patient'])
        self.assertTrue((self.root/'result/patients/new_patient/preflight.json').exists())

    def test_geometry_unavailable_is_unverified(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        v = torch.tensor(mesh.vertices,dtype=torch.float32)
        model = SimpleNamespace(template_vertices_mm=torch.cat([v]*4),chamber_ranges=torch.tensor([[i*len(v),(i+1)*len(v)] for i in range(4)]),
            faces={c:torch.tensor(mesh.faces) for c in CHAMBERS})
        with patch('tto.geometry_guard.SourceAVCapPolicy',side_effect=ImportError('missing backend')):
            report = GeometryGuard(model).audit(np.tile(model.template_vertices_mm.numpy()[None],(24,1,1)))
        self.assertFalse(report['passed'])
        for key in ('shape','saved_phases','inter_chamber','continuous'):
            self.assertEqual(report[key]['status'],'UNVERIFIED')

    def test_motion_control_mutation_rejected(self):
        fields = torch.zeros(1,24,4,4,4,3)
        model = SimpleNamespace()
        propagator = FrozenMotionPropagator(model,fields)
        propagator.fields[:,1] = 1.
        with self.assertRaises(AssertionError):propagator.propagate(torch.zeros(1,10,3))

    @unittest.skipUnless((ROOT/'outputs/neural_ffd_1k_long_20261010_225918/best_joint.pth').exists(), 'specified checkpoint unavailable')
    def test_real_checkpoint_shape_gradient_frozen_weights_and_motion_replay(self):
        torch.set_num_threads(2)
        cfg = load_config(ROOT/'configs/real_tto_v1.yaml')
        prior = FrozenSyntheticPrior(ROOT/cfg['prior']['checkpoint'],expected_sha256=cfg['prior']['sha256'])
        z = torch.nn.Parameter(torch.zeros(1,64))
        reference = prior.decode_initial_shape(z)
        reference.square().mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertGreater(float(z.grad.norm()),0.)
        with torch.no_grad():
            out = prior.decode_initial_motion(z.detach(),torch.zeros(1,32))
            for key in ('z_shape','z_motion','vertices_ref','vertices_4d','motion_fields'):
                self.assertIn(key,out)
            fields = out['control_grids']['fine_motion']
            replay = FrozenMotionPropagator(prior.model,fields).propagate(out['vertices_ref'])
            self.assertTrue(torch.equal(replay,out['vertices_4d']))
            modified = z.detach()+.01
            locked = prior.reconstruct_4d(modified,torch.zeros(1,32),frozen_fields=fields)
            self.assertEqual(tuple(locked['vertices_4d'].shape),(1,24,10631,3))
            new_fields = prior.export_motion_fields(modified,torch.zeros(1,32))
            self.assertGreater(float((new_fields[:,1:]-fields[:,1:]).abs().max()),0.)
        prior.assert_frozen()

    @unittest.skipUnless((ROOT/'outputs/neural_ffd_1k_long_20261010_225918/best_joint.pth').exists(), 'specified checkpoint unavailable')
    def test_native_geometry_rejects_collapsed_triangle(self):
        torch.set_num_threads(2)
        config=load_config(ROOT/'configs/real_tto_v1.yaml')
        prior=FrozenSyntheticPrior(ROOT/config['prior']['checkpoint'],expected_sha256=config['prior']['sha256'])
        guard=GeometryGuard(prior.model)
        if guard.backend_error:self.skipTest('pinned geometry backend unavailable: '+guard.backend_error)
        with torch.no_grad():points=prior.decode_initial_shape(torch.zeros(1,64))[0].numpy().copy()
        face=prior.model.faces['LV'][0].tolist()
        points[face[0]]=points[face[1]]
        report=guard.discrete(points)
        self.assertFalse(report['passed'])
        self.assertTrue(report['errors'] or any(not r['passed'] for r in report['mesh_records']))

    @unittest.skipUnless((ROOT/'outputs/neural_ffd_1k_long_20261010_225918/best_joint.pth').exists(), 'specified checkpoint unavailable')
    def test_rejected_shape_update_restores_latent_and_adam_state(self):
        torch.set_num_threads(2)
        cfg = load_config(ROOT/'configs/real_tto_v1.yaml')
        prior = FrozenSyntheticPrior(ROOT/cfg['prior']['checkpoint'],expected_sha256=cfg['prior']['sha256'])
        model = prior.model
        patient = RealTTODataset(self.root,['new_patient'],config=self.config)[0]
        calibration = json.loads((ROOT/'configs/real_tto_prior_calibration.json').read_text())
        pose = {k:torch.tensor(calibration['initial_pose'][k],dtype=torch.float32) for k in ('R','center_mm')}
        with torch.no_grad():out = prior.decode_initial_motion(torch.zeros(1,64),torch.zeros(1,32))
        state = dict(z_shape_initial=torch.zeros(64),prior4d=out['vertices_4d'][0],
            fields=out['control_grids']['fine_motion'][0],pose=pose,geometry={'passed':True})
        class RejectGuard:
            def audit(self,*args,**kwargs):return dict(passed=False,status='FAIL',reason='injected geometry failure')
        config = dict(cfg['shape'],updates=1,lr=.00001,stride=4)
        optimizer = LatentShapeOptimizer(model,config,cfg['motion_limits'],RejectGuard(),ExactNativeEvaluator(model),calibration['shape_std'])
        snapshots = []
        result = optimizer.optimize([patient],[state],checkpoint=lambda p,s:snapshots.append(s))[0]
        self.assertEqual(len(snapshots),1)
        self.assertFalse(result['history'][0]['accepted'])
        self.assertEqual(len(result['history'][0]['z_shape_proposed']),64)
        self.assertEqual(result['history'][0]['z_shape_retained'],[0.]*64)
        self.assertGreater(result['history'][0]['gradient_norm'],0.)
        self.assertTrue(torch.equal(result['z_shape'],state['z_shape_initial']))
        self.assertTrue(torch.equal(snapshots[0]['parameter'].detach(),state['z_shape_initial']))
        self.assertEqual(snapshots[0]['optimizer']['state'],{})
        prior.assert_frozen()


if __name__ == '__main__':
    unittest.main()
