"""G1 reference/loss/all-parameter gradient and future freeze contracts."""
import copy
import unittest
from unittest.mock import patch
import torch
from tests.test_geometry_heartffdnet import fixture_model, observations
from net.geometry_heartffdnet import loss_output_adapter
from net.staged_heartffdnet import StagedHeartFFDNet, reference_loss
from utils.neural_ffd_losses import NeuralFFDLoss


class StagedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)

    def test_reference_loss_and_all_gradients_equal_g1(self):
        g1 = fixture_model().eval()
        stage = copy.deepcopy(g1)
        stage.__class__ = StagedHeartFFDNet
        x = observations()
        truth = g1.template_vertices_mm[None,None].expand(1,24,-1,-1)+.3
        criterion = NeuralFFDLoss(g1.chamber_ranges,g1.faces)
        a = g1(x)
        la = criterion(loss_output_adapter(a),truth,
                       control_spacings_mm=g1.control_spacings_mm,stage='shape')
        la['total'].backward()
        with patch.object(stage,'_deform',wraps=stage._deform) as deform:
            b = stage.reference_output(x)
        self.assertEqual(deform.call_count,3)
        torch.testing.assert_close(a['vertices_ref'],b['vertices_ref'],rtol=0,atol=0)
        lb = reference_loss(criterion,b,truth,stage.control_spacings_mm)
        torch.testing.assert_close(la['total'],lb['total'],rtol=0,atol=0)
        for name in ('shape','edge','laplacian','normal','control'):
            torch.testing.assert_close(la['terms'][name],lb['terms'][name],rtol=0,atol=0)
        lb['total'].backward()
        for (name,p),(other,q) in zip(g1.named_parameters(),stage.named_parameters()):
            self.assertEqual(name,other)
            self.assertIsNotNone(p.grad,name)
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-8,msg=name)
        self.assertEqual(float(stage.heads.deform3.conv.weight.grad[3:].abs().max()),0.)
        poisoned = truth.clone();poisoned[:,1:] = float('nan')
        reference_loss(criterion,b,poisoned,stage.control_spacings_mm)

    def test_frozen_reference_and_reserved_motion(self):
        model = fixture_model();model.__class__ = StagedHeartFFDNet
        model.freeze_reference().train()
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        x = observations()
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            v = model(x)
            torch.testing.assert_close(v,model.predict_reference(x),rtol=0,atol=0)
        for name,value in before.items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        with self.assertRaises(NotImplementedError):model.predict_motion(x,v)


if __name__ == '__main__':unittest.main()
