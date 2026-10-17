"""Full-population stages, online gradients and joint optimizer resume."""
import tempfile
from pathlib import Path
import unittest

import torch

from test_joint_ffd_refiner import integration_model
from utils.population_training import PopulationFFDModel, PopulationWorkflow


def fixture():
    joint, obs, pose, _, _ = integration_model()
    obs = dict(obs, pose_gt=pose)
    return PopulationFFDModel(joint.base_model, joint.refiner), obs


class PopulationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_warm_stages_skip_pose_and_joint_optimizes_both_networks(self):
        model, obs = fixture()
        workflow = PopulationWorkflow()
        optimizer = workflow.make_optimizer(model, dict(learning_rate=.001))
        registered = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in registered for p in model.refiner.parameters()))
        for stage in ('shape','motion'):
            workflow.set_stage(model, stage)
            self.assertTrue(all(p.requires_grad for p in model.base_model.parameters()))
            self.assertFalse(any(p.requires_grad for p in model.refiner.parameters()))
            model({k:v for k,v in obs.items() if k!='pose_gt'})
        workflow.set_stage(model,'joint')
        before_base = model.base_model.seed.weight.detach().clone()
        before_head = model.refiner.static_head.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(obs)
            output['vertices_4d'].square().mean().backward()
            self.assertGreater(float(model.base_model.seed.weight.grad.norm()),0.)
            self.assertGreater(float(model.refiner.static_head.weight.grad.norm()),0.)
            optimizer.step()
        self.assertFalse(torch.equal(before_base,model.base_model.seed.weight))
        self.assertFalse(torch.equal(before_head,model.refiner.static_head.weight))
        self.assertGreater(float(model.refiner.chamber_stem.weight.grad.norm()),0.)
        self.assertEqual(registered,{id(p) for p in model.parameters()})

    def test_self_contained_joint_checkpoint_restores_both_models_and_next_adam_step(self):
        workflow = PopulationWorkflow()
        model, obs = fixture()
        model.initialization_metadata = dict(mode='random', inherited_checkpoint=None)
        workflow.set_stage(model,'joint')
        opt = workflow.make_optimizer(model,dict(learning_rate=.001))
        opt.zero_grad(); model(obs)['vertices_4d'].square().mean().backward(); opt.step()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pth'
            workflow.save_checkpoint(path,model,opt,config=dict(training=dict(seed=2026)),identity={},
                epoch=61,sample_offset=4,step=12001,runtime=dict(max_cases=4))
            other,saved=workflow.load_checkpoint(path)
            self.assertEqual(other.active_stage,'joint')
            self.assertEqual((saved['epoch'],saved['sample_offset'],saved['step']),(61,4,12001))
            self.assertTrue(all(p.requires_grad for p in other.parameters()))
            other_opt=workflow.make_optimizer(other,dict(learning_rate=.001))
            other_opt.load_state_dict(saved['optimizer'])
            for net,optimizer in ((model,opt),(other,other_opt)):
                optimizer.zero_grad(); net(obs)['vertices_4d'].square().mean().backward(); optimizer.step()
            for a,b in zip(model.parameters(),other.parameters()):
                torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_warm_checkpoint_keeps_refiner_in_optimizer_for_later_joint_activation(self):
        workflow=PopulationWorkflow()
        model,_=fixture()
        model.initialization_metadata=dict(mode='random')
        workflow.set_stage(model,'shape')
        opt=workflow.make_optimizer(model,dict(learning_rate=.001))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'warm.pth'
            workflow.save_checkpoint(path,model,opt,config={},identity={})
            restored,saved=workflow.load_checkpoint(path)
            workflow.set_stage(restored,'joint')
            newopt=workflow.make_optimizer(restored,dict(learning_rate=.001))
            newopt.load_state_dict(saved['optimizer'])
            self.assertEqual(len(newopt.param_groups[0]['params']),len(list(restored.parameters())))

    def test_tail_checkpoint_prefers_fewer_failed_cells_before_minimum_dice(self):
        workflow=PopulationWorkflow()
        candidates=workflow.best_checkpoint_candidates(dict(complete=True,stage='joint',validation=dict(
            reference_vertex_rmse_mm=1.,relative_motion_rmse_mm=1.,mean_exact_dice=.98,
            native_dice_fail_cells=123,minimum_exact_dice=.93)))
        self.assertIn('best_tail',candidates)
        self.assertAlmostEqual(candidates['best_tail']['score'],123.07)


if __name__=='__main__':
    unittest.main()
