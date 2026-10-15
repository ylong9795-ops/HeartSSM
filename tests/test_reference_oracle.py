"""Oracle parameterization, physical limits and GT exclusion contracts."""
import copy
import unittest
from types import SimpleNamespace
import torch
from tests.test_geometry_heartffdnet import fixture_model, observations
from tests.test_staged_heartffdnet import RefinementTests
from net.staged_heartffdnet import StagedHeartFFDNet
from net.geometry_heartffdnet import control_lipschitz_bound
from utils.neural_ffd_losses import NeuralFFDLoss
from utils.slice_supervision import MeshSliceSupervisor
from scripts.run_reference_oracle import ReferenceControls, fit_objective


class OracleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2026)

    def test_initial_identity_bounded_updates_and_frozen_network(self):
        model=fixture_model();model.__class__=StagedHeartFFDNet;model.freeze_reference()
        with torch.no_grad():initial=model.reference_output(observations())
        control=ReferenceControls(model,initial)
        torch.testing.assert_close(control()['vertices_ref'],initial['vertices_ref'],rtol=0,atol=1e-5)
        self.assertEqual(len(list(control.parameters())),3)
        control()['vertices_ref'].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in control.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        with torch.no_grad():
            for p in control.parameters():p.copy_(torch.randn_like(p)*100)
        result=control()
        for i,k in enumerate(('coarse','medium','fine')):
            self.assertLessEqual(float(result['control_grids'][k].abs().max()),model.heads.reference_limit_mm)
            self.assertLessEqual(float(control_lipschitz_bound(result['control_grids'][k],model.control_spacings_mm[i]).max()),.45001)

    def test_A_excludes_GT_B_adds_complete_correspondence(self):
        vertices,obs,ranges,faces=RefinementTests().fixture()
        vertices=vertices.detach().requires_grad_()
        grids={k:torch.zeros(shape,dtype=vertices.dtype) for k,shape in
               [('coarse',(1,4,4,4,3)),('medium',(1,4,4,4,3)),('fine',(1,1,4,4,4,3))]}
        out=dict(vertices_ref=vertices[:,0],vertices_4d=vertices,control_grids=grids)
        class Controls:
            model=SimpleNamespace(control_spacings_mm=torch.ones(3,3,dtype=vertices.dtype))
            def __call__(self):return out
        controls=Controls();obs={k:v for k,v in obs.items() if k!='vertices_gt'}
        anchor=vertices.detach().clone()
        prior=NeuralFFDLoss(ranges,faces,weights=dict(shape=0.)).to(vertices)
        slicer=MeshSliceSupervisor(ranges,faces,2.5,supervision_grid='native_grid',pixel_stride=2,execution_backend='batched')
        _,a=fit_objective(controls,obs,anchor,prior,slicer,2.)
        grad=torch.autograd.grad(a['total'],vertices,retain_graph=True)[0]
        self.assertGreater(float(grad.norm()),0.);self.assertEqual(float(a['gt']),0.)
        poisoned=dict(obs,vertices_gt=torch.full_like(vertices,float('nan')))
        with self.assertRaisesRegex(ValueError,'exclude GT'):
            fit_objective(controls,poisoned,anchor,prior,slicer,2.)
        gt_loss=NeuralFFDLoss(ranges,faces,weights=dict(edge=0.,laplacian=0.,normal=0.,control=0.)).to(vertices)
        _,b=fit_objective(controls,obs,anchor,prior,slicer,2.,gt_criterion=gt_loss,gt=anchor+.5)
        self.assertGreater(float(b['gt']),0.)
        torch.testing.assert_close(b['total']-a['total'],b['gt'])
        torch.testing.assert_close(a['slice'],b['slice']);torch.testing.assert_close(a['prior'],b['prior'])
