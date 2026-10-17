"""Nonzero shared Joint checkpoints, frozen caches and residual objective wiring."""
import tempfile
import unittest
from pathlib import Path

import torch

from test_joint_ffd_refiner import integration_model
from train_neural_ffd import save_checkpoint
from utils.joint_refiner_training import (configure_refiner_training, load_refiner_checkpoint,
    refiner_objective, sampled_composed_jacobians, save_refiner_checkpoint, tail_summary)


class RefinerTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_cached_path_matches_online_values_and_gradients_after_learning(self):
        model, obs, pose, _, _ = integration_model()
        configure_refiner_training(model)
        self.assertFalse(any(p.requires_grad for p in model.base_model.parameters()))
        with torch.no_grad():
            model.refiner.static_head.weight.normal_(0, .1)
            model.refiner.dynamic_head.weight.normal_(0, .1)
            base = model.base_model(obs)
            features = model.observation_features(base, obs, pose=pose)
        values, gradients = [], []
        for online in (True, False):
            model.zero_grad(set_to_none=True)
            out = model(obs, pose=pose) if online else model.refine_base(base, features, phases=obs['phases'])
            out['vertices_4d'].square().mean().backward()
            values.append(out['vertices_4d'].detach())
            gradients.append(model.refiner.chamber_stem.weight.grad.clone())
        torch.testing.assert_close(*values, atol=0, rtol=0)
        torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-9)
        self.assertTrue(all(p.grad is None for p in model.base_model.parameters()))

    def test_trained_checkpoint_restores_optimizer_predictions_and_next_update(self):
        model, obs, pose, _, _ = integration_model()
        configure_refiner_training(model)
        optimizer = torch.optim.Adam(model.refiner.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            model(obs, pose=pose)['vertices_4d'].square().mean().backward()
            optimizer.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            save_checkpoint(path/'base.pth', model.base_model, None, config={}, identity={}, epoch=75, step=60000)
            source = torch.load(path/'base.pth', weights_only=True)
            save_refiner_checkpoint(path/'joint.pth', model, optimizer, source_checkpoint=source,
                experiment=dict(freeze='base_frozen_refiner_only'), step=2,
                sampler_state=torch.Generator().manual_seed(99).get_state())
            other, saved = load_refiner_checkpoint(path/'joint.pth')
            configure_refiner_training(other)
            other_optimizer = torch.optim.Adam(other.refiner.parameters(), lr=.001)
            other_optimizer.load_state_dict(saved['optimizer'])
            self.assertEqual(saved['step'], 2)
            self.assertEqual(saved['base_checkpoint']['step'], 60000)
            for network, opt in ((model, optimizer), (other, other_optimizer)):
                opt.zero_grad()
                network(obs, pose=pose)['vertices_4d'].square().mean().backward()
                opt.step()
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_residual_regularizers_are_weighted_in_total_with_nonzero_gradients(self):
        model, _, _, _, _ = integration_model()
        control = torch.tensor(2., requires_grad=True)
        output = dict(vertices_4d=torch.zeros(1), control_regularization=dict(
            spatial_gradient_l2=control.square(), cyclic_velocity_l2=control*3,
            cyclic_acceleration_l2=control*4))
        class Criterion:
            def __call__(self, *args, **kwargs):
                return dict(total=control*0+5, terms={})
        loss = refiner_objective(model, output, dict(vertices_gt=torch.zeros(1)), Criterion(), lambda *a: {},
            dict(spatial_gradient_l2=.1, cyclic_velocity_l2=.2, cyclic_acceleration_l2=.3))
        self.assertAlmostEqual(float(loss['total']), 9.)
        loss['total'].backward()
        self.assertAlmostEqual(float(control.grad), 2.2, places=6)

    def test_composed_jacobians_replay_actual_base_and_residual_mapping(self):
        model, obs, pose, _, _ = integration_model()
        with torch.no_grad():
            output = model(obs, pose=pose)
            base = {k: v[:1] if isinstance(v, torch.Tensor) else
                    {a: b[:1] if b is not None else None for a, b in v.items()}
                    for k, v in output['base_output'].items() if k in ('vertices_4d', 'control_grids')}
            result = sampled_composed_jacobians(model.base_model, base,
                output['refinement']['controls_res_mm'][:1], per_chamber=4)
        self.assertEqual(result['base'], result['refiner'])
        self.assertEqual(result['base']['nonpositive'], 0)
        self.assertLess(result['maximum_base_replay_error_mm'], .002)

    def test_tail_summary_preserves_right_heart_and_worst_identities(self):
        rows = [dict(case_id='a', frame=0, view='A4C', chamber=c, exact_dice=d,
                     boundary_assd_mm=.1, boundary_hd95_mm=.2)
                for c, d in (('LV', .999), ('LA', .99), ('RV', .96), ('RA', .89))]
        result = tail_summary(rows)
        self.assertEqual(result['all']['below_098'], 2)
        self.assertEqual(result['right_heart']['below_095'], 1)
        self.assertEqual(result['all']['cases_all_cells_ge_098'], 0)
        self.assertEqual(result['all']['worst_cells'][0]['chamber'], 'RA')


if __name__ == '__main__':
    unittest.main()
