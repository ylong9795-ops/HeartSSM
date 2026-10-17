"""Physical SDF hints, independent GT rerendering and coordinate contracts."""
import copy
import unittest

import numpy as np
import torch
import trimesh
from torch.nn import functional as F

from data.mask_transform import CHAMBERS, VIEWS, MaskTransform
from utils.observation_residual import ObservationResidualExtractor, sample_sdf_residual
from utils.slice_supervision import exact_section_mask


def fixture():
    """Oblique translated plane, anisotropic native/model grids, sparse cells."""
    dtype = torch.float64
    mesh = trimesh.creation.box(extents=[10., 12., 4.])
    local = torch.tensor(mesh.vertices.copy(), dtype=dtype)+torch.tensor([.37, -.23, .31])
    # Cyclic permutation: local x -> world y, y -> world z, normal -> world x.
    rotation = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], dtype=dtype)
    translation = torch.tensor([3., -5., 7.], dtype=dtype)
    world = local@rotation.T+translation
    gt = world.repeat(4, 1)[None, None].repeat(2, 2, 1, 1)
    faces = {c: torch.tensor(mesh.faces.copy()) for c in CHAMBERS}
    ranges = torch.tensor([[8*c, 8*(c+1)] for c in range(4)])
    annotated = np.zeros((2, 2, 3, 4), bool)
    for cell in ((0, 0, 0, 0), (0, 1, 2, 2), (1, 0, 1, 1), (1, 1, 2, 3)):
        annotated[cell] = True
    native_spacing = np.array([.4, .6])
    mask = exact_section_mask(world.numpy(), mesh.faces, rotation.numpy(), translation.numpy(),
                              native_spacing, (64, 80))
    native = np.broadcast_to(mask, (2, 3, 4, 64, 80)).copy()
    # A different fixed FOV + output resolution. Always sample the resulting
    # model SDF with its OWN spacing, never with native_pixel_spacing.
    transform = MaskTransform(target_size=(56, 72), canonical_fov_mm={v: (28., 50.4) for v in VIEWS})
    transformed = [transform(native, native_spacing, view_present=np.ones(3, bool), annotated=a)
                   for a in annotated]
    obs = {key: torch.from_numpy(np.stack([x[key] for x in transformed])) for key in transformed[0]}
    obs.update(sdf_units=['mm', 'mm'], annotated=torch.from_numpy(annotated),
               view_present=torch.ones(2, 3, dtype=torch.bool),
               phases=torch.tensor([[.13, .61], [.27, .89]], dtype=dtype),
               frame_indices=torch.tensor([[3, 15], [6, 21]]), case_id=['alpha', 'beta'])
    # No vertices_gt or pose_gt in observations: only the explicitly supplied
    # pose is used. Nonunit scale catches both omitted and double conversion.
    pose = dict(probe_R=rotation.expand(2, 2, 3, 3, 3).clone(),
                probe_t=(translation/10).expand(2, 2, 3, 3).clone())
    return gt, obs, ranges, faces, pose, rotation, translation


class SDFResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_anisotropic_plane_sdf_sign_magnitude_clipping_and_fov(self):
        spacing = torch.tensor([.4, 1.3], dtype=torch.float64)
        yy, xx = torch.meshgrid((torch.arange(32)-16)*spacing[0],
                                (torch.arange(40)-20)*spacing[1], indexing='ij')
        # Unit normal (3/5,4/5); linear field makes interpolation exact.
        sdf = .6*xx+.8*yy
        xy = sdf.new_tensor([[3., 4.], [-3., -4.], [0., 0.], [100., 0.]])
        result = sample_sdf_residual(sdf, xy, spacing, max_distance_mm=2.)
        torch.testing.assert_close(result['signed_distance_mm'][:3], sdf.new_tensor([5., -5., 0.]))
        torch.testing.assert_close(result['gradient_xy'], sdf.new_tensor([[.6, .8]]).expand(4, -1))
        torch.testing.assert_close(result['displacement_xy_mm'],
                                   sdf.new_tensor([[-1.2, -1.6], [1.2, 1.6], [0., 0.], [0., 0.]]))
        self.assertEqual(result['valid'].tolist(), [True, True, True, False])
        self.assertEqual(result['confidence'].tolist(), [True, True, True, False])

    def test_pixel_centre_fov_bounds_and_empty_points(self):
        sdf = torch.arange(30, dtype=torch.float64).reshape(5, 6)
        xy = sdf.new_tensor([[-3., -2.5], [2., 1.5], [-3.01, 0.], [2.01, 0.]])
        result = sample_sdf_residual(sdf, xy, [1., 1.])
        self.assertEqual(result['valid'].tolist(), [True, True, False, False])
        torch.testing.assert_close(result['signed_distance_mm'][:2], sdf.new_tensor([0., 29.]))
        empty = sample_sdf_residual(sdf, xy[:0], [1., 1.])
        self.assertEqual(empty['displacement_xy_mm'].shape, (0, 2))

    def test_flat_saturated_and_ambiguous_gradients_have_no_correction(self):
        yy, xx = torch.meshgrid(torch.arange(20, dtype=torch.float64)-10,
                                torch.arange(20, dtype=torch.float64)-10, indexing='ij')
        xy = xx.new_tensor([[0., 0.], [5., 0.], [-5., 0.]])
        for sdf in (torch.full_like(xx, 15.), torch.full_like(xx, -15.)):
            result = sample_sdf_residual(sdf, xy, [1., 1.])
            self.assertFalse(result['confidence'].any())
            self.assertEqual(float(result['displacement_xy_mm'].abs().sum()), 0.)
        saturated = sample_sdf_residual(xx.clamp(-2, 2), xy, [1., 1.])
        self.assertEqual(saturated['confidence'].tolist(), [True, False, False])
        centre = sample_sdf_residual((xx.square()+yy.square()).sqrt()-3., xy[:1], [1., 1.])
        self.assertFalse(centre['confidence'].any())
        self.assertEqual(float(centre['signed_distance_mm'][0]), -3.)

    def test_sample_coordinate_gradient_matches_finite_difference(self):
        yy, xx = torch.meshgrid(torch.arange(24, dtype=torch.float64)-12,
                                torch.arange(28, dtype=torch.float64)-14, indexing='ij')
        sdf = .6*xx+.8*yy
        xy = sdf.new_tensor([[1.31, -.43]], requires_grad=True)
        def value(q):
            return sample_sdf_residual(sdf, q, [1., 1.])['displacement_xy_mm'].square().sum()
        derivative, = torch.autograd.grad(value(xy), xy)
        eps = 1e-5
        for axis in range(2):
            step = torch.zeros_like(xy)
            step[0, axis] = eps
            numerical = (value(xy.detach()+step)-value(xy.detach()-step))/(2*eps)
            torch.testing.assert_close(derivative[0, axis], numerical, atol=1e-8, rtol=1e-6)

    def test_invalid_physical_inputs_are_rejected(self):
        sdf, xy = torch.ones(4, 5), torch.zeros(2, 2)
        for spacing in ([0., 1.], [float('nan'), 1.]):
            with self.assertRaises(ValueError):
                sample_sdf_residual(sdf, xy, spacing)
        with self.assertRaises(ValueError):
            sample_sdf_residual(sdf[:1], xy, [1., 1.])


class ObservationResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gt_rerender_pixel_bias_and_known_physical_offsets(self):
        gt, obs, ranges, faces, pose, rotation, _ = fixture()
        extractor = ObservationResidualExtractor(ranges, faces, 10.)
        baseline = extractor(gt, obs, pose=pose)
        # Independent binary rendering + EDT has finite raster bias at exact GT.
        # Do not replace this with an artificial exact-zero GT correction.
        pixel_diagonal = obs['pixel_spacing'][0, 0, 0].norm()
        self.assertLess(float(baseline['signed_distance_mm'].abs().max()), 2*float(pixel_diagonal))
        for local_shift in ([2.8, 0., 0.], [-2.8, 0., 0.], [0., 2.8, 0.], [0., -2.8, 0.]):
            shift = gt.new_tensor(local_shift)@rotation.T
            predicted = extractor(gt+shift, obs, pose=pose)
            usable = predicted['confidence']
            weighted_dot = ((predicted['displacement_world_mm'][usable]*shift).sum(-1)
                            *predicted['weights_mm'][usable]).sum()
            self.assertLess(float(weighted_dot), 0.)
            # Independent test that a small physical correction approaches the
            # observed zero level set, rather than just having the right sign.
            indices = predicted['point_indices']
            for cell in range(len(predicted['cell_indices'])):
                start, end = predicted['cell_offsets'][cell:cell+2]
                b, t, v, c = indices[start].tolist()
                corrected = predicted['points_xy_mm'][start:end]+.2*predicted['displacement_xy_mm'][start:end]
                after = sample_sdf_residual(obs['sdf'][b, t, v, c].to(gt), corrected,
                                            obs['pixel_spacing'][b, t, v])
                self.assertLess(float(after['signed_distance_mm'].abs().mean()),
                                float(predicted['signed_distance_mm'][start:end].abs().mean()))

    def test_world_lift_metadata_and_arc_length(self):
        gt, obs, ranges, faces, pose, rotation, translation = fixture()
        result = ObservationResidualExtractor(ranges, faces, 10.)(gt, obs, pose=pose)
        torch.testing.assert_close(result['cell_indices'], obs['annotated'].nonzero())
        torch.testing.assert_close((result['points_world_mm']-translation)@rotation,
                                   F.pad(result['points_xy_mm'], (0, 1)))
        local_delta = result['displacement_world_mm']@rotation
        torch.testing.assert_close(local_delta[:, :2], result['displacement_xy_mm'])
        torch.testing.assert_close(local_delta[:, 2], torch.zeros_like(local_delta[:, 2]))
        owners = result['point_cell_ids']
        torch.testing.assert_close(result['point_indices'], result['cell_indices'][owners])
        cells = result['cell_indices']
        torch.testing.assert_close(result['point_phases'], obs['phases'][cells[owners, 0], cells[owners, 1]])
        torch.testing.assert_close(result['cell_frame_indices'], obs['frame_indices'][cells[:, 0], cells[:, 1]])
        self.assertEqual(result['cell_case_ids'], ['alpha', 'alpha', 'beta', 'beta'])
        for start, end in zip(result['cell_offsets'][:-1], result['cell_offsets'][1:]):
            self.assertAlmostEqual(float(result['weights_mm'][start:end].sum()), 44., places=6)
        self.assertEqual(sum(p.numel() for p in ObservationResidualExtractor(ranges, faces, 10.).parameters()), 0)

    def test_serial_batched_values_and_gradients_agree_and_are_cell_local(self):
        gt, obs, ranges, faces, pose, rotation, _ = fixture()
        # Same chamber in two different patients, one with a missing section:
        # exercise real multi-cell batching and padded segment validity.
        obs['annotated'][1, 1, 2, 0] = True
        for key in ('masks', 'sdf'):
            obs[key][1, 1, 2, 0] = obs[key][0, 0, 0, 0]
        gt[1, 1, :8] += gt.new_tensor([0., 0., 10.])@rotation.T
        gradients, results = [], []
        for backend in ('serial', 'batched'):
            v = (gt+gt.new_tensor([.71, -.43, 0.])@rotation.T).requires_grad_()
            result = ObservationResidualExtractor(ranges, faces, 10., execution_backend=backend,
                                                   cell_chunk_size=3)(v, obs, pose=pose)
            end = result['cell_offsets'][1]
            value = (result['displacement_world_mm'][:end].square().sum(-1)*result['weights_mm'][:end]).sum()
            gradient, = torch.autograd.grad(value, v)
            self.assertGreater(float(gradient[0, 0, :8].norm()), 0.)
            self.assertEqual(float(gradient[1].abs().max()), 0.)
            self.assertEqual(float(gradient[0, 1].abs().max()), 0.)
            self.assertEqual(float(gradient[0, 0, 8:].abs().max()), 0.)
            gradients.append(gradient)
            results.append(result)
        for key in ('points_world_mm', 'displacement_world_mm', 'signed_distance_mm', 'weights_mm'):
            torch.testing.assert_close(results[0][key], results[1][key])
        torch.testing.assert_close(*gradients)

    def test_mesh_to_residual_gradient_matches_finite_difference(self):
        gt, obs, ranges, faces, pose, rotation, _ = fixture()
        extractor = ObservationResidualExtractor(ranges, faces, 10., sample_step_mm=.3)
        offset = gt.new_tensor(.71, requires_grad=True)
        def value(x):
            shift = torch.stack((x, x*0-.43, x*0))@rotation.T
            result = extractor(gt+shift, obs, pose=pose)
            return (result['displacement_world_mm'].square().sum(-1)*result['weights_mm']).sum()
        derivative, = torch.autograd.grad(value(offset), offset)
        eps = 1e-5
        numerical = (value(offset.detach()+eps)-value(offset.detach()-eps))/(2*eps)
        torch.testing.assert_close(derivative, numerical, atol=1e-6, rtol=1e-5)

    def test_missing_section_empty_full_target_and_unavailable_cells(self):
        gt, obs, ranges, faces, pose, rotation, _ = fixture()
        gt[0, 0, :8] += gt.new_tensor([0., 0., 10.])@rotation.T
        obs['masks'][0, 1, 2, 2] = 0
        obs['sdf'][0, 1, 2, 2] = 15
        obs['masks'][1, 0, 1, 1] = 1
        obs['sdf'][1, 0, 1, 1] = -15
        # Unavailable placeholders may be nonfinite and must not be sampled.
        obs['sdf'][~obs['annotated']] = float('nan')
        extractor = ObservationResidualExtractor(ranges, faces, 10.)
        result = extractor(gt, obs, pose=pose)
        self.assertEqual(result['cell_has_section'].tolist(), [False, True, True, True])
        self.assertEqual(result['cell_has_target_boundary'].tolist(), [True, False, False, True])
        self.assertEqual(result['cell_offsets'][1], 0)
        self.assertFalse(result['confidence'][result['point_cell_ids'] < 3].any())
        self.assertTrue(torch.isfinite(result['displacement_world_mm']).all())
        obs['annotated'].zero_()
        obs['view_present'].zero_()
        empty = extractor(gt, obs, pose=pose)
        self.assertEqual(empty['points_world_mm'].shape, (0, 3))
        self.assertEqual(empty['point_indices'].shape, (0, 4))
        self.assertEqual(empty['cell_offsets'].tolist(), [0])

    def test_static_validity_float32_and_explicit_unit_pose_checks(self):
        gt, obs, ranges, faces, pose, _, _ = fixture()
        extractor = ObservationResidualExtractor(ranges, faces, 10.)
        obs['annotated'] = obs['annotated'].any(1)
        result = extractor(gt.float(), obs, pose=pose)
        self.assertEqual(result['points_world_mm'].dtype, torch.float32)
        self.assertEqual(len(result['cell_indices']), 8)
        for units in ('dimensionless', None):
            bad = dict(obs, sdf_units=units)
            with self.assertRaisesRegex(ValueError, 'sdf_units'):
                extractor(gt, bad, pose=pose)
        bad_pose = copy.deepcopy(pose)
        bad_pose['probe_R'][0, 0, 0, 0, 0] = 2.
        with self.assertRaisesRegex(ValueError, 'rotations'):
            extractor(gt, obs, pose=bad_pose)
        with self.assertRaises(TypeError):
            extractor(gt, obs)


if __name__ == '__main__':
    unittest.main()
