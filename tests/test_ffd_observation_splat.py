"""FFD/splat duality, physical support, channel isolation and mesh recovery."""
import unittest

import torch

from test_observation_residual import fixture
from utils.ffd_observation_splat import (
    build_missing_section_recovery, splat_observation_residuals, splat_to_fine_controls,
)
from utils.ffd_physical import bspline_ffd_physical, bspline_support_physical, control_geometry
from utils.observation_residual import ObservationResidualExtractor


def geometry(dtype=torch.float64):
    shape = (6, 7, 8)
    origin = torch.tensor([-4., 1., -8.], dtype=dtype)
    spacing = torch.tensor([.7, 1.3, 2.1], dtype=dtype)
    q = origin.new_tensor([[1., 1., 1.], [4., 5., 6.], [2.23, 3.41, 4.17]])
    return origin+q*spacing, origin, spacing, shape


def packed(points, displacement, indices, cells=None, has_section=None, has_boundary=None):
    if cells is None:
        cells = indices.unique(dim=0)
    count = len(cells)
    return dict(points_world_mm=points, displacement_world_mm=displacement, point_indices=indices,
                cell_indices=cells, confidence=points.new_ones(len(points)),
                weights_mm=points.new_ones(len(points)), valid=torch.ones(len(points), dtype=torch.bool),
                cell_has_section=torch.ones(count, dtype=torch.bool) if has_section is None else has_section,
                cell_has_target_boundary=torch.ones(count, dtype=torch.bool) if has_boundary is None else has_boundary)


class PhysicalSplatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_partition_of_unity_and_linear_precision_on_closed_domain(self):
        p, origin, spacing, shape = geometry()
        index, weights, valid = bspline_support_physical(p, origin, spacing, shape)
        self.assertTrue(valid.all())
        self.assertEqual(index.shape, (3, 64))
        self.assertGreaterEqual(int(index.min()), 0)
        self.assertLess(int(index.max()), 6*7*8)
        torch.testing.assert_close(weights.sum(-1), p.new_ones(3), rtol=1e-13, atol=1e-13)
        nodes = torch.stack(torch.meshgrid(*(torch.arange(s, dtype=p.dtype) for s in shape), indexing='ij'), -1)
        node_world = origin+nodes*spacing
        matrix = p.new_tensor([[.2, -.3, .4], [1.1, .7, -.5], [-.6, .9, .1]])
        bias = p.new_tensor([3., -2., 1.])
        control = (node_world@matrix+bias)[None]
        actual = bspline_ffd_physical(p[None], control, origin, spacing)[0]
        torch.testing.assert_close(actual, p@matrix+bias, atol=1e-12, rtol=1e-12)

    def test_scatter_is_the_weighted_transpose_of_original_ffd_evaluation(self):
        p, origin, spacing, shape = geometry()
        r = p.new_tensor([[1., -2., .3], [-.4, .7, 1.1], [.8, .2, -.9]])
        confidence, arc = p.new_tensor([.3, .8, 1.]), p.new_tensor([.2, .7, 1.3])
        controls = p.new_zeros(1, *shape, 3).requires_grad_()
        objective = (bspline_ffd_physical(p[None], controls, origin, spacing)[0]*r
                     *(confidence*arc)[:, None]).sum()
        transpose, = torch.autograd.grad(objective, controls)
        feature = splat_to_fine_controls(p, r, origin, spacing, shape, confidence=confidence, sample_weights=arc)
        numerator = feature[:3]*(feature[3:]+1e-8)
        torch.testing.assert_close(numerator.movedim(0, -1), transpose[0], atol=1e-13, rtol=1e-12)
        self.assertAlmostEqual(float(feature[3].sum()), float((confidence*arc).sum()), places=12)

    def test_confidence_weighted_average_clipping_and_zero_evidence(self):
        p, origin, spacing, shape = geometry()
        p = p[2:].repeat(2, 1)
        r = p.new_tensor([[3., 4., 0.], [-3., -4., 0.]])
        feature = splat_to_fine_controls(p, r, origin, spacing, shape, confidence=p.new_tensor([.5, 1.]),
                                         sample_weights=p.new_tensor([4., 1.]), max_displacement_mm=2.)
        _, basis, _ = bspline_support_physical(p[:1], origin, spacing, shape)
        index, _, _ = bspline_support_physical(p[:1], origin, spacing, shape)
        support = feature[3].flatten()[index[0]]
        torch.testing.assert_close(support, 3*basis[0])
        expected = p.new_tensor([.4, 1.6/3, 0.])[:, None]*support/(support+1e-8)
        torch.testing.assert_close(feature[:3].flatten(1)[:, index[0]], expected)
        self.assertLessEqual(float(feature[:3].norm(dim=0).max()), 2.)
        empty = splat_to_fine_controls(p, r, origin, spacing, shape, confidence=p.new_zeros(2))
        self.assertEqual(float(empty.abs().max()), 0.)

    def test_arc_length_support_is_invariant_to_duplicate_half_weight_samples(self):
        p, origin, spacing, shape = geometry()
        r = p.new_tensor([[1., .2, -.4]]).expand(3, -1)
        arc = p.new_tensor([.3, .7, 1.4])
        one = splat_to_fine_controls(p, r, origin, spacing, shape, sample_weights=arc)
        split = splat_to_fine_controls(p.repeat_interleave(2, 0), r.repeat_interleave(2, 0), origin, spacing,
                                       shape, sample_weights=arc.repeat_interleave(2)/2)
        torch.testing.assert_close(one, split, rtol=1e-12, atol=1e-12)

    def test_fov_validity_nonfinite_and_unsupported_samples_cannot_poison_grid(self):
        p, origin, spacing, shape = geometry()
        good = p[2:]
        bad = torch.cat((good, good, good, good, good))
        bad[1, 0] = float('nan')
        bad[2] += 1000
        r = p.new_ones(5, 3)
        r[3, 1] = float('inf')
        valid = torch.tensor([True, True, True, True, False])
        feature = splat_to_fine_controls(bad, r, origin, spacing, shape, valid=valid)
        expected = splat_to_fine_controls(good, p.new_ones(1, 3), origin, spacing, shape)
        torch.testing.assert_close(feature, expected)
        confidence = p.new_tensor([1., float('inf'), 0., 0., 0.])
        torch.testing.assert_close(splat_to_fine_controls(bad, r, origin, spacing, shape, confidence=confidence), expected)
        with self.assertRaisesRegex(ValueError, 'outside'):
            bspline_support_physical(bad[2:3], origin, spacing, shape)
        _, weight, supported = bspline_support_physical(bad[2:3], origin, spacing, shape, strict=False)
        self.assertFalse(supported.any())
        self.assertEqual(float(weight.sum()), 0.)

    def test_float32_boundary_roundoff_matches_ffd_acceptance(self):
        p, origin, spacing, shape = geometry(torch.float32)
        upper = origin+(torch.tensor(shape)-2)*spacing
        p = torch.nextafter(upper, torch.full_like(upper, float('inf')))[None]
        _, weight, supported = bspline_support_physical(p, origin, spacing, shape)
        self.assertTrue(supported.all())
        controls = p.new_ones(1, *shape, 3)
        torch.testing.assert_close(bspline_ffd_physical(p[None], controls, origin, spacing), p.new_ones(1, 1, 3))
        feature = splat_to_fine_controls(p, p.new_ones(1, 3), origin, spacing, shape)
        torch.testing.assert_close(feature[3].sum(), weight.sum())

    def test_chunking_values_and_gradients_and_all_empty_backward(self):
        p, origin, spacing, shape = geometry()
        p = p[2:].repeat(3, 1)+p.new_tensor([[0., 0., 0.], [.11, .07, .12], [-.09, .03, .04]])
        r = p.new_tensor([[1., .3, -.4], [-.2, .4, .8], [.6, -.5, .2]])
        results, gradients = [], []
        for chunk in (1, 8192):
            pp, rr = p.clone().requires_grad_(), r.clone().requires_grad_()
            feature = splat_to_fine_controls(pp, rr, origin, spacing, shape, point_chunk_size=chunk)
            results.append(feature)
            gradients.append(torch.autograd.grad(feature.square().sum(), (pp, rr)))
        torch.testing.assert_close(*results)
        for one, many in zip(*gradients):
            torch.testing.assert_close(one, many)
        empty = p.new_empty(0, 3).requires_grad_()
        feature = splat_to_fine_controls(empty, empty, origin, spacing, shape)
        feature.sum().backward()
        self.assertEqual(float(feature.abs().max()), 0.)
        self.assertEqual(empty.grad.shape, (0, 3))

    def test_gradcheck_coordinates_vectors_confidence_and_arc_weights(self):
        p, origin, spacing, shape = geometry()
        p = p[2:].clone().requires_grad_()
        r = p.new_tensor([[.7, -.3, .9]], requires_grad=True)
        confidence = p.new_tensor([.6], requires_grad=True)
        arc = p.new_tensor([.7], requires_grad=True)
        probe = torch.linspace(-.7, 1.3, 4*6*7*8, dtype=p.dtype).reshape(4, *shape)
        def value(pp, rr, cc, ww):
            return (splat_to_fine_controls(pp, rr, origin, spacing, shape, confidence=cc, sample_weights=ww)*probe).sum()
        self.assertTrue(torch.autograd.gradcheck(value, (p, r, confidence, arc), atol=1e-5, rtol=1e-4))

    def test_invalid_geometry_and_weights_are_rejected(self):
        p, origin, spacing, shape = geometry()
        r = torch.ones_like(p)
        for options in (dict(confidence=p.new_tensor([1., -.1, 1.])), dict(sample_weights=-p.new_ones(3)),
                        dict(max_displacement_mm=0.), dict(eps=0.), dict(point_chunk_size=0)):
            with self.assertRaises(ValueError):
                splat_to_fine_controls(p, r, origin, spacing, shape, **options)
        for bad_shape in ((3, 7, 8), (6.5, 7, 8)):
            with self.assertRaises(ValueError):
                splat_to_fine_controls(p, r, origin, spacing, bad_shape)


class PackedSplatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_patient_time_chamber_isolation_and_same_chamber_view_fusion(self):
        p, origin, spacing, shape = geometry()
        points = p[2:].repeat(5, 1)
        indices = torch.tensor([[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 2, 2], [1, 2, 1, 0], [0, 1, 2, 3]])
        delta = p.new_tensor([[2., 0., 0.], [0., 2., 0.], [-2., 0., 0.], [0., 0., 3.], [0., -3., 0.]])
        residual = packed(points, delta, indices)
        result = splat_observation_residuals(residual, origin, spacing, shape, batch_size=2, frames=3)
        features = result['features']
        self.assertEqual(features.shape, (2, 3, 16, *shape))
        self.assertEqual(result['channel_names'][8:12], ('RV_dx_mm', 'RV_dy_mm', 'RV_dz_mm', 'RV_support'))
        torch.testing.assert_close(features[0, 0, 3].sum(), p.new_tensor(2.))
        torch.testing.assert_close(features[0, 0, 11].sum(), p.new_tensor(1.))
        for b, t, c, expected in ((0, 0, 0, [1., 1., 0.]), (0, 0, 2, [-2., 0., 0.]),
                                   (1, 2, 0, [0., 0., 3.]), (0, 1, 3, [0., -3., 0.])):
            support = features[b, t, c*4+3]
            torch.testing.assert_close(features[b, t, c*4:c*4+3],
                p.new_tensor(expected)[:, None, None, None]*support/(support+1e-8))
        self.assertEqual(float(features[1, :2].abs().sum()), 0.)
        self.assertEqual(float(features[0, 0, 4:8].abs().sum()), 0.)

    def test_default_fine_grid_preserves_24_phases_with_no_observations(self):
        p, origin, spacing, _ = geometry(torch.float32)
        residual = packed(p[:0], p[:0], torch.empty(0, 4, dtype=torch.long))
        result = splat_observation_residuals(residual, origin, spacing, batch_size=1, frames=24)
        self.assertEqual(result['features'].shape, (1, 24, 16, 24, 24, 32))
        self.assertEqual(float(result['features'].abs().max()), 0.)

    def test_missing_section_has_observation_only_normal_capture_on_predicted_surface(self):
        gt, obs, ranges, faces, pose, rotation, translation = fixture()
        normal = rotation[:, 2]
        predicted = gt.clone()
        predicted[0, 0, :8] += 10*normal
        predicted.requires_grad_()
        residual = ObservationResidualExtractor(ranges, faces, 10.)(predicted, obs, pose=pose)
        origin, spacing = control_geometry(gt.new_tensor([-20., -25., -20.]), gt.new_tensor([25., 25., 25.]), (6, 7, 8))
        with self.assertRaisesRegex(ValueError, 'recovery'):
            splat_observation_residuals(residual, origin, spacing, (6, 7, 8), batch_size=2, frames=2)
        recovery = build_missing_section_recovery(predicted, obs, residual, pose=pose, chamber_ranges=ranges,
                                                   faces=faces, norm_to_mm=10.)
        self.assertGreater(len(recovery['point_indices']), 0)
        self.assertTrue((recovery['point_indices'] == torch.tensor([0, 0, 0, 0])).all())
        self.assertFalse(recovery['points_world_mm'].requires_grad)
        target_local = (recovery['target_points_world_mm']-translation)@rotation
        torch.testing.assert_close(target_local[:, 2], torch.zeros_like(target_local[:, 2]))
        self.assertTrue((recovery['displacement_world_mm']@normal < 0).all())
        # Closest surface anchor lies on the facing box wall, not on the plane
        # or at a nearest vertex; queries span wall interiors.
        local_anchor = (recovery['points_world_mm']-translation)@rotation
        torch.testing.assert_close(local_anchor[:, 2], local_anchor.new_full((len(local_anchor),), 8.31), atol=1e-6, rtol=0)
        result = splat_observation_residuals(residual, origin, spacing, (6, 7, 8), batch_size=2, frames=2, recovery=recovery)
        self.assertEqual(result['cell_needs_recovery'].tolist(), [True, False, False, False])
        self.assertEqual(result['cell_recovery_available'].tolist(), [True, False, False, False])
        self.assertTrue(result['recovery_point_used'].all())
        controls = result['features'][0, 0, :3].movedim(0, -1)[None]
        movement = bspline_ffd_physical(recovery['points_world_mm'][None], controls, origin, spacing)[0]
        self.assertTrue((movement@normal < 0).all())
        self.assertEqual(float(result['features'][0, 0, 4:].abs().sum()), 0.)

    def test_recovery_no_points_for_empty_targets_and_rejects_cross_chamber_injection(self):
        gt, obs, ranges, faces, pose, _, _ = fixture()
        residual = ObservationResidualExtractor(ranges, faces, 10.)(gt, obs, pose=pose)
        recovery = build_missing_section_recovery(gt, obs, residual, pose=pose, chamber_ranges=ranges, faces=faces, norm_to_mm=10.)
        self.assertEqual(recovery['points_world_mm'].shape, (0, 3))
        p, origin, spacing, shape = geometry()
        main = packed(p[:1], p.new_ones(1, 3), torch.tensor([[0, 0, 0, 0]]))
        injected = packed(p[:1], p.new_ones(1, 3), torch.tensor([[0, 0, 0, 1]]))
        with self.assertRaisesRegex(ValueError, 'Recovery points'):
            splat_observation_residuals(main, origin, spacing, shape, batch_size=1, frames=1, recovery=injected)

    def test_unusable_recovery_has_explicit_availability_and_indices_are_validated(self):
        p, origin, spacing, shape = geometry()
        cells = torch.tensor([[0, 0, 0, 0]])
        main = packed(p[:0], p[:0], cells[:0], cells=cells, has_section=torch.tensor([False]))
        recovery = packed(p[:1]+1000, p.new_ones(1, 3), cells)
        result = splat_observation_residuals(main, origin, spacing, shape, batch_size=1, frames=1, recovery=recovery)
        self.assertTrue(result['cell_needs_recovery'][0])
        self.assertFalse(result['cell_recovery_available'][0])
        self.assertEqual(float(result['features'].abs().max()), 0.)
        main['cell_indices'] = torch.tensor([[0, 0, 0, 4]])
        with self.assertRaisesRegex(ValueError, 'index outside'):
            splat_observation_residuals(main, origin, spacing, shape, batch_size=1, frames=1)


if __name__ == '__main__':
    unittest.main()
