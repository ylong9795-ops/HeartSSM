"""Shared population refinement, continuous controls and original FFD contracts."""
import io
import unittest

import torch
from torch import nn

from net.joint_ffd_refiner import (
    JointFFDRefiner, ObservationFFDRefiner, apply_residual_ffd,
    evaluate_residual_controls, residual_control_regularization, smooth_control_limit,
)
from net.neural_ffdnet import NeuralFFDNet
from test_observation_residual import fixture
from utils.ffd_physical import bspline_ffd_physical, control_geometry


def inputs(batch=2, frames=5, shape=(4, 5, 6), dtype=torch.float64):
    generator = torch.Generator().manual_seed(714)
    x = torch.randn(batch, frames, 16, *shape, generator=generator, dtype=dtype)
    x[:, :, 3::4] = torch.rand(batch, frames, 4, *shape, generator=generator, dtype=dtype)
    z_shape = torch.randn(batch, 64, generator=generator, dtype=dtype)
    z_motion = torch.randn(batch, 32, generator=generator, dtype=dtype)
    return x, z_shape, z_motion


def small_refiner(activated=False, **options):
    torch.manual_seed(91)
    settings = dict(hidden_channels=8, spatial_blocks=1, temporal_node_chunk_size=37, spatial_frame_chunk_size=2)
    settings.update(options)
    model = ObservationFFDRefiner(**settings).double()
    if activated:
        with torch.no_grad():
            for head in (model.static_head, model.dynamic_head):
                nn.init.normal_(head.weight, 0., .15)
                nn.init.normal_(head.bias, 0., .1)
    return model


class SharedRefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_24_phase_fine_grid_and_lightweight_shared_parameters(self):
        model = ObservationFFDRefiner()
        self.assertLess(sum(p.numel() for p in model.parameters()), 100000)
        self.assertEqual(model.chamber_stem.groups, 4)
        self.assertEqual(model.temporal[0].padding_mode, 'circular')
        self.assertFalse(any(isinstance(layer, nn.Embedding) for layer in model.modules()))
        x = torch.zeros(1, 24, 16, 24, 24, 32)
        x[:, :, 3] = 1.
        with torch.no_grad():
            result = model(x, torch.zeros(1, 64), torch.zeros(1, 32))
        self.assertEqual(result['controls_res_mm'].shape, (1, 24, 24, 24, 32, 3))
        self.assertEqual(float(result['controls_res_mm'].abs().max()), 0.)

    def test_zero_heads_preserve_base_vertices_exactly_and_can_start_learning(self):
        model = small_refiner()
        result = model(*inputs(frames=4))
        for key in ('controls_static_mm', 'controls_dynamic_mm', 'controls_res_mm'):
            self.assertEqual(float(result[key].abs().max()), 0.)
        origin, spacing = control_geometry([-10., -12., -8.], [10., 12., 8.], (4, 5, 6))
        vertices = torch.rand(2, 4, 11, 3, dtype=torch.float64)*4-2
        final = apply_residual_ffd(vertices, result['controls_res_mm'], origin, spacing)
        self.assertTrue(torch.equal(vertices, final))
        target = torch.zeros_like(final)
        target[..., 0] = .2
        target[:, 1:, :, 1] = -.1
        (final-(vertices+target)).square().mean().backward()
        self.assertGreater(float(model.static_head.weight.grad.norm()), 0.)
        self.assertGreater(float(model.dynamic_head.weight.grad.norm()), 0.)
        self.assertTrue(torch.isfinite(model.static_head.weight.grad).all())

    def test_static_dynamic_reference_split_and_total_norm_bound(self):
        model = small_refiner(activated=True)
        x, zs, zm = inputs()
        with torch.no_grad():
            model.static_head.weight.mul_(500.)
            model.dynamic_head.weight.mul_(1000.)
        result = model(x, zs, zm, query_phases=torch.linspace(-1., 2., 97, dtype=x.dtype))
        total, static, dynamic = (result[key] for key in ('controls_res_mm', 'controls_static_mm', 'controls_dynamic_mm'))
        torch.testing.assert_close(total, static[:, None]+dynamic)
        self.assertLessEqual(float(total.norm(dim=-1).max()), 3.+1e-12)
        for slot in (0, 32, 64, 96):
            self.assertEqual(float(dynamic[:, slot].abs().max()), 0.)
            self.assertTrue(torch.equal(total[:, slot], static))

    def test_zero_limit_has_identity_derivative_and_no_nan(self):
        vector = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        limited = smooth_control_limit(vector)
        gradient, = torch.autograd.grad(limited.sum(), vector)
        torch.testing.assert_close(gradient, torch.ones_like(vector), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_static_is_independent_of_motion_latent_dynamic_uses_both_conditions(self):
        model = small_refiner(activated=True)
        x, zs, zm = inputs()
        one, two = model(x, zs, zm), model(x, zs, zm+1.7)
        self.assertTrue(torch.equal(one['controls_static_mm'], two['controls_static_mm']))
        self.assertGreater(float((one['controls_dynamic_mm']-two['controls_dynamic_mm']).norm()), 1e-6)
        zs, zm = zs.requires_grad_(), zm.requires_grad_()
        result = model(x, zs, zm)
        gradients = torch.autograd.grad(result['controls_res_mm'].square().sum(), (zs, zm))
        self.assertTrue(all(float(g.norm()) > 0 and torch.isfinite(g).all() for g in gradients))

    def test_no_support_is_neutral_after_training_and_unsupported_nan_hints_are_ignored(self):
        model = small_refiner(activated=True)
        x, zs, zm = inputs()
        x[0, :, 3::4] = 0
        # Vector hints with no support are placeholders, not measurements.
        for c in range(4):
            x[0, :, c*4:c*4+3] = float('nan')
        result = model(x, zs, zm)
        self.assertEqual(result['has_observation_evidence'].tolist(), [False, True])
        self.assertEqual(float(result['controls_res_mm'][0].abs().max()), 0.)
        self.assertGreater(float(result['controls_res_mm'][1].norm()), 0.)

    def test_chambers_remain_distinct_before_learned_fusion(self):
        model = small_refiner(activated=True)
        x, zs, zm = inputs(batch=1)
        one = model(x, zs, zm)['controls_res_mm']
        # Swap LV/RV blocks but keep Shape/Motion codes unchanged. An
        # unweighted pre-average would erase this change entirely.
        swapped = x.clone()
        swapped[:, :, :4], swapped[:, :, 8:12] = x[:, :, 8:12], x[:, :, :4]
        two = model(swapped, zs, zm)['controls_res_mm']
        self.assertGreater(float((one-two).norm()), 1e-5)
        isolated = torch.zeros(1, 16, 4, 5, 6, dtype=x.dtype)
        isolated[:, :4] = 1.
        stem = model.chamber_stem(isolated)
        self.assertEqual(float(stem[:, 2:].abs().sum()), 0.)

    def test_batch_matches_separate_patients_and_gradients_do_not_cross(self):
        model = small_refiner(activated=True)
        x, zs, zm = (v.requires_grad_() for v in inputs())
        before = tuple((key, parameter.shape) for key, parameter in model.named_parameters())
        full = model(x, zs, zm)['controls_res_mm']
        singles = torch.cat([model(x[i:i+1], zs[i:i+1], zm[i:i+1])['controls_res_mm'] for i in range(2)])
        torch.testing.assert_close(full, singles, rtol=1e-11, atol=1e-12)
        gradients = torch.autograd.grad(full[0].square().sum(), (x, zs, zm))
        for gradient in gradients:
            self.assertGreater(float(gradient[0].norm()), 0.)
            self.assertEqual(float(gradient[1].abs().max()), 0.)
        after = tuple((key, parameter.shape) for key, parameter in model.named_parameters())
        self.assertEqual(before, after)

    def test_periodic_temporal_path_connects_last_phase_to_first_locally(self):
        model = small_refiner(activated=True)
        x, zs, zm = inputs(batch=1, frames=5)
        before = model(x, zs, zm)['dynamic_coefficients_mm']
        changed = x.clone()
        changed[:, 0, 0] += 2.
        after = model(changed, zs, zm)['dynamic_coefficients_mm']
        self.assertGreater(float((after[:, -1]-before[:, -1]).norm()), 1e-7)
        self.assertTrue(torch.equal(after[:, 2], before[:, 2]))

    def test_spatial_and_temporal_chunking_preserve_values_and_gradients(self):
        model = small_refiner(activated=True)
        other = small_refiner(activated=True, spatial_frame_chunk_size=100, temporal_node_chunk_size=10000)
        other.load_state_dict(model.state_dict())
        x, zs, zm = inputs(batch=1)
        values, gradients = [], []
        for network in (model, other):
            xx = x.clone().requires_grad_()
            result = network(xx, zs, zm)['controls_res_mm']
            values.append(result)
            gradients.append(torch.autograd.grad(result.square().sum(), xx)[0])
        torch.testing.assert_close(*values, atol=1e-12, rtol=1e-11)
        torch.testing.assert_close(*gradients, atol=1e-12, rtol=1e-10)

    def test_shared_parameter_update_improves_mesh_loss_and_reaches_observation_stem(self):
        model = small_refiner()
        x, zs, zm = inputs(frames=4)
        origin, spacing = control_geometry([-10., -10., -10.], [10., 10., 10.], (4, 5, 6))
        vertices = torch.linspace(-3., 3., 2*4*9*3, dtype=x.dtype).reshape(2, 4, 9, 3)
        target = vertices+vertices.new_tensor([.12, -.07, .04])
        target[:, 1:, :, 1] -= .05
        optimiser = torch.optim.SGD(model.parameters(), lr=.01)
        def loss():
            controls = model(x, zs, zm)['controls_res_mm']
            final = apply_residual_ffd(vertices, controls, origin, spacing)
            return (final-target).square().mean()
        before = loss()
        before.backward()
        self.assertGreater(float(model.static_head.weight.grad.norm()), 0.)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        after = loss()
        self.assertLess(float(after), float(before))
        after.backward()
        self.assertGreater(float(model.chamber_stem.weight.grad.norm()), 0.)
        self.assertGreater(float(model.temporal[0].weight.grad.norm()), 0.)

    def test_single_phase_is_static_only_and_checkpoint_round_trip(self):
        model = small_refiner(activated=True)
        x, zs, zm = inputs(batch=1, frames=1)
        result = model(x, zs, zm, query_phases=x.new_tensor([0., .37, 1.]))
        self.assertEqual(float(result['controls_dynamic_mm'].abs().max()), 0.)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = small_refiner()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(result['controls_res_mm'], restored(x, zs, zm, query_phases=x.new_tensor([0., .37, 1.]))['controls_res_mm'])

    def test_invalid_observation_units_shapes_support_and_nonuniform_phases_rejected(self):
        model = small_refiner()
        x, zs, zm = inputs()
        for bad in (x[:, :, :12], x.float()):
            with self.assertRaises(ValueError):
                model(bad, zs, zm)
        for value in (-1., float('nan')):
            bad = x.clone()
            bad[0, 0, 3, 0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, 'support'):
                model(bad, zs, zm)
        bad_phases = torch.arange(5, dtype=x.dtype)/5
        bad_phases[2] += .02
        with self.assertRaisesRegex(ValueError, 'uniform'):
            model(x, zs, zm, phases=bad_phases)


class ContinuousFFDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_periodic_field_matches_value_velocity_and_acceleration_at_cycle_boundary(self):
        torch.manual_seed(44)
        static = torch.randn(1, 4, 4, 4, 3, dtype=torch.float64)
        dynamic = torch.randn(1, 6, 4, 4, 4, 3, dtype=static.dtype)
        phase = static.new_tensor([0., 1.], requires_grad=True)
        result = evaluate_residual_controls(static, dynamic, query_phases=phase)
        total = result['controls_res_mm']
        self.assertTrue(torch.equal(total[:, 0], total[:, 1]))
        derivatives, accelerations = [], []
        for i in range(2):
            derivative, = torch.autograd.grad(total[:, i].square().sum(), phase, create_graph=True, retain_graph=True)
            acceleration, = torch.autograd.grad(derivative[i], phase, retain_graph=True)
            derivatives.append(derivative[i])
            accelerations.append(acceleration[i])
        torch.testing.assert_close(*derivatives, atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(*accelerations, atol=1e-9, rtol=1e-12)
        self.assertEqual(float(result['controls_dynamic_mm'].abs().max()), 0.)

    def test_query_time_gradient_matches_finite_difference(self):
        torch.manual_seed(76)
        static = torch.randn(1, 4, 4, 4, 3, dtype=torch.float64)*.2
        dynamic = torch.randn(1, 5, 4, 4, 4, 3, dtype=static.dtype)*.3
        phase = static.new_tensor([.317], requires_grad=True)
        def value(t):
            return evaluate_residual_controls(static, dynamic, query_phases=t)['controls_res_mm'].square().sum()
        derivative, = torch.autograd.grad(value(phase), phase)
        eps = 1e-6
        numerical = (value(phase.detach()+eps)-value(phase.detach()-eps))/(2*eps)
        torch.testing.assert_close(derivative[0], numerical, atol=1e-7, rtol=1e-6)

    def test_application_uses_current_vertices_and_original_physical_ffd(self):
        torch.manual_seed(45)
        c = torch.randn(2, 3, 4, 5, 6, 3, dtype=torch.float64)*.1
        v = (torch.rand(2, 3, 12, 3, dtype=c.dtype)*4-2).requires_grad_()
        c.requires_grad_()
        origin, spacing = control_geometry([-10., -12., -9.], [10., 12., 9.], (4, 5, 6))
        actual = apply_residual_ffd(v, c, origin, spacing, frame_chunk_size=2)
        expected = torch.stack([v[:, t]+bspline_ffd_physical(v[:, t], c[:, t], origin, spacing) for t in range(3)], 1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        grad = torch.autograd.grad(actual.square().sum(), (v, c))
        self.assertTrue(all(torch.isfinite(g).all() and float(g.norm()) > 0 for g in grad))
        with self.assertRaisesRegex(ValueError, 'outside'):
            apply_residual_ffd(v+100, c, origin, spacing)

    def test_physical_regularization_respects_spacing_and_cyclic_last_first_edge(self):
        constant = torch.ones(1, 4, 4, 5, 6, 3, dtype=torch.float64)
        for term in residual_control_regularization(constant, [1., 2., 3.]).values():
            self.assertEqual(float(term), 0.)
        varying = constant.clone()
        varying[:, 0] = 0.
        result = residual_control_regularization(varying, [1., 2., 3.])
        self.assertAlmostEqual(float(result['cyclic_velocity_l2']), 8.)
        ramp = torch.arange(4, dtype=constant.dtype)[None, None, :, None, None, None].expand_as(constant)
        one = residual_control_regularization(ramp, [1., 2., 3.])['spatial_gradient_l2']
        two = residual_control_regularization(ramp, [2., 2., 3.])['spatial_gradient_l2']
        torch.testing.assert_close(one, two*4)


def integration_model(missing=False):
    gt, observations, ranges, faces, pose, rotation, _ = fixture()
    observations['phases'] = torch.tensor([[0., .5], [0., .5]], dtype=torch.float64)
    template = gt[0, 0].clone()
    if missing:
        template[:8] += rotation[:, 2]*10
    base = NeuralFFDNet(template, ranges, faces, template_sha256='test', asset_sha256='test', norm_to_mm=10.,
        encoder_config=dict(width=16, heads=4, temporal_layers=1, spatial_sides=(2, 2), query_slots=1,
                            cnn_chunk_size=4, checkpoint_cnn=False),
        decoder_config=dict(frames=2, channels=8, seed_shape=(4, 4, 4), regional_shape=False,
                            lower_mm=(-20., -25., -20.), upper_mm=(25., 25., 25.)))
    base.eval()
    refiner = ObservationFFDRefiner(hidden_channels=8, spatial_blocks=1)
    return JointFFDRefiner(base, refiner), observations, pose, ranges, faces


class JointPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_base_to_observation_to_splat_to_refiner_preserves_initial_mesh_and_gradients(self):
        model, observations, pose, _, faces = integration_model()
        output = model(observations, pose=pose)
        self.assertEqual(output['observation_splat']['features'].shape, (2, 2, 16, 16, 16, 16))
        self.assertEqual(output['refinement']['controls_res_mm'].shape, (2, 2, 16, 16, 16, 3))
        self.assertTrue(torch.equal(output['vertices_4d'], output['vertices_base_4d']))
        self.assertTrue(output['refinement']['has_observation_evidence'].all())
        self.assertIs(output['control_grids'], output['base_output']['control_grids'])
        for chamber in faces:
            self.assertTrue(torch.equal(model.base_model.faces[chamber], faces[chamber]))
        loss = output['vertices_4d'].square().mean()
        loss.backward()
        self.assertGreater(float(model.base_model.seed.weight.grad.norm()), 0.)
        self.assertGreater(float(model.refiner.static_head.weight.grad.norm()), 0.)
        self.assertIsNotNone(model.base_model.encoder.shape_head[-1].weight.grad)

    def test_explicit_pose_and_no_gt_vertex_or_case_id_dependency(self):
        model, observations, pose, _, _ = integration_model()
        with torch.no_grad():
            nn.init.normal_(model.refiner.static_head.weight, 0., .1)
            one = model(observations, pose=pose)
            poisoned = dict(observations, vertices_gt=torch.full((1,), float('nan')),
                            pose_gt={'probe_R': torch.full((1,), float('nan'))}, case_id=['new_a', 'new_b'])
            two = model(poisoned, pose=pose)
        torch.testing.assert_close(one['vertices_4d'], two['vertices_4d'], atol=0, rtol=0)
        with self.assertRaises(TypeError):
            model(observations)

    def test_missing_section_recovery_is_used_by_same_shared_refiner(self):
        model, observations, pose, _, _ = integration_model(missing=True)
        with torch.no_grad():
            result = model(observations, pose=pose)
        splat = result['observation_splat']
        self.assertTrue(splat['cell_needs_recovery'][0])
        self.assertTrue(splat['cell_recovery_available'][0])
        self.assertGreater(float(splat['features'][0, 0, 3].sum()), 0.)
        self.assertTrue(torch.equal(result['vertices_4d'], result['vertices_base_4d']))


if __name__ == '__main__':
    unittest.main()
