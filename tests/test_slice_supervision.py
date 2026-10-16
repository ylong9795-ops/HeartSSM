"""Physical boundary directions and unreduced synthetic slice contracts."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from data.mask_transform import CHAMBERS
from train_neural_ffd import training_collate
from utils.neural_ffd_losses import NeuralFFDLoss
from utils.slice_supervision import (
    MeshSliceSupervisor, exact_section_mask, paired_boundary_distance,
    plane_segments, surface_capture_distance, empty_section_clearance, OBSERVATION_KINDS, section_quadrature,
)


def fixture(batch=2, frames=2, empty=False):
    mesh = trimesh.creation.box(extents=[6., 8., 4.])
    mesh.apply_translation([.137, -.253, 4.31 if empty else .31])
    vertices = torch.tensor(mesh.vertices, dtype=torch.float64)
    faces = {c: torch.tensor(mesh.faces.copy()) for c in CHAMBERS}
    n = len(vertices)
    ranges = torch.tensor([[i*n, (i+1)*n] for i in range(4)])
    gt = vertices.repeat(4, 1)[None, None].repeat(batch, frames, 1, 1)
    rotations = torch.eye(3, dtype=torch.float64).expand(batch, frames, 3, 3, 3).clone()
    translations = torch.zeros(batch, frames, 3, 3, dtype=torch.float64)
    spacing = torch.tensor([.65, 1.15], dtype=torch.float64).expand(batch, frames, 3, 2).clone()
    native = torch.zeros(batch, frames, 3, 4, 22, 34, dtype=torch.float64)
    annotated = torch.zeros(batch, frames, 3, 4, dtype=torch.bool)
    cells = [[0, 0, 0, 0], [0, frames-1, 2, 3], [batch-1, 0, 1, 2], [batch-1, frames-1, 2, 0]]
    for b, t, v, c in cells:
        annotated[b, t, v, c] = True
        native[b, t, v, c] = torch.from_numpy(exact_section_mask(
            vertices.numpy(), mesh.faces, np.eye(3), np.zeros(3), [.65, 1.15], (22, 34)))
    observations = dict(
        masks=torch.zeros(batch, frames, 3, 4, 12, 14, dtype=torch.float64),
        sdf=torch.zeros(batch, frames, 3, 4, 12, 14, dtype=torch.float64),
        pixel_spacing=torch.full_like(spacing, 2.), native_masks=native,
        native_pixel_spacing=spacing, vertices_gt=gt, pose_gt=dict(probe_R=rotations, probe_t=translations),
        view_present=torch.ones(batch, frames, 3, dtype=torch.bool), annotated=annotated,
        phases=torch.arange(frames, dtype=torch.float64)[None].expand(batch, -1)/frames,
        frame_indices=torch.arange(frames)[None].expand(batch, -1)*2,
        case_id=[f'patient_{i}' for i in range(batch)])
    return gt, observations, ranges, faces


def supervisor(ranges, faces, backend='batched', **options):
    config = dict(target_mode='paired_soft_geometry_v1', supervision_grid='native_grid',
                  boundary_objective='paired_boundary_distance_v1', pixel_stride=3,
                  chunk_size=64, cell_chunk_size=2, execution_backend=backend)
    config.update(options)
    return MeshSliceSupervisor(ranges, faces, 1., **config)


class SliceBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_gt_is_stationary_despite_native_rasterisation(self):
        gt, obs, ranges, faces = fixture()
        obs['vertices_gt'] = gt.clone().requires_grad_()
        predicted = gt.clone().requires_grad_()
        result = supervisor(ranges, faces)(predicted, obs)
        self.assertLess(float(result['cell_losses'].abs().max()), 1e-10)
        result['total'].backward()
        self.assertLess(float(predicted.grad.abs().max()), 1e-10)
        self.assertIsNone(obs['vertices_gt'].grad)
        self.assertEqual(result['diagnostics']['recovery_cells'], 0)

    def test_aggregated_observation_is_stationary_and_improves_finite_geometry(self):
        gt, obs, ranges, faces = fixture()
        weights = {key: 0. for key in ('shape', 'trajectory', 'motion', 'velocity', 'volume',
                                     'edge', 'laplacian', 'normal', 'control')}
        weights['slice'] = 1.
        objective = NeuralFFDLoss(ranges, faces, objective_mode='grouped_gt_boundary_v1', weights=weights)
        controls = dict(coarse=gt.new_zeros(1, 2, 2, 2, 3), medium=gt.new_zeros(1, 2, 2, 2, 3),
                        fine_motion=gt.new_zeros(1, 2, 2, 2, 2, 3), regions=None, regional_motion=None)
        slicer = supervisor(ranges, faces)
        def loss(vertices):
            return objective(dict(vertices_4d=vertices, control_grids=controls), gt,
                             slice_result=slicer(vertices, obs))['L_O']
        exact = gt.clone().requires_grad_()
        value = loss(exact)
        value.backward()
        self.assertLess(float(value.abs()), 1e-10)
        self.assertLess(float(exact.grad.abs().max()), 1e-10)
        offset = torch.nn.Parameter(gt.new_tensor([.7, -.4, .15]))
        before = loss(gt+offset)
        gradient, = torch.autograd.grad(before, offset)
        updated = offset.detach()-.3*gradient
        self.assertLess(float(loss(gt+updated)), float(before))
        # Independent coordinate error of this translated box's section edges.
        self.assertLess(float(updated[:2].abs().sum()), float(offset[:2].abs().sum()))

    def test_cells_remain_differentiable_identified_and_isolated(self):
        gt, obs, ranges, faces = fixture()
        predicted = (gt+torch.tensor([.8, -.4, 0.])).requires_grad_()
        result = supervisor(ranges, faces)(predicted, obs)
        expected = obs['annotated'].nonzero()
        torch.testing.assert_close(result['cell_indices'], expected)
        self.assertEqual(result['cell_case_ids'], ['patient_0', 'patient_0', 'patient_1', 'patient_1'])
        torch.testing.assert_close(result['cell_phases'], obs['phases'][expected[:, 0], expected[:, 1]])
        torch.testing.assert_close(result['cell_frame_indices'], obs['frame_indices'][expected[:, 0], expected[:, 1]])
        self.assertTrue(result['cell_losses'].requires_grad)
        grad, = torch.autograd.grad(result['cell_terms']['boundary_mm'][0], predicted)
        self.assertGreater(float(grad[0, 0, :8].norm()), 0.)
        self.assertEqual(float(grad[1].abs().max()), 0.)
        self.assertEqual(float(grad[0, 1:].abs().max()), 0.)
        self.assertEqual(float(grad[0, 0, 8:].abs().max()), 0.)

    def test_serial_batched_values_gradients_and_chamber_reduction_agree(self):
        gt, obs, ranges, faces = fixture()
        vertices = [(gt+torch.tensor([.6, -.3, .1])).requires_grad_() for _ in range(2)]
        results = [supervisor(ranges, faces, mode, reduction='chamber_mean')(v, obs)
                   for mode, v in zip(('serial', 'batched'), vertices)]
        for result in results:
            result['total'].backward()
            cells, terms = result['cell_indices'][:, 3], result['cell_losses']
            expected = torch.stack([terms[cells == c].mean() for c in cells.unique()]).mean()
            torch.testing.assert_close(result['total'], expected)
        torch.testing.assert_close(results[0]['cell_losses'], results[1]['cell_losses'])
        torch.testing.assert_close(vertices[0].grad, vertices[1].grad)

    def test_shared_parameter_step_improves_independent_section_geometry(self):
        gt, obs, ranges, faces = fixture()
        shift = torch.nn.Parameter(torch.tensor([.7, -.4, .15], dtype=torch.float64))
        slicer = supervisor(ranges, faces)
        before = slicer(gt+shift, obs)
        gradient, = torch.autograd.grad(before['cell_terms']['boundary_mm'].sum(), shift)
        self.assertGreater(float(gradient[0]), 0.)
        self.assertLess(float(gradient[1]), 0.)
        updated = shift.detach()-.15*gradient
        after = slicer(gt+updated, obs)
        self.assertTrue((after['cell_terms']['boundary_mm'] < before['cell_terms']['boundary_mm']).all())
        # Independent physical bbox error of the actual plane intersections.
        def section_error(offset):
            segment = plane_segments(gt[0, 0, :8]+offset, faces['LV'], torch.eye(3, dtype=gt.dtype), gt.new_zeros(3))
            curve = segment.flatten(0, 1)
            target = gt[0, 0, :8, :2]
            return (curve.amin(0)-target.amin(0)).abs().sum()+(curve.amax(0)-target.amax(0)).abs().sum()
        self.assertLess(float(section_error(updated)), float(section_error(shift.detach())))

    def test_missing_section_has_surface_capture_and_finite_step_progress(self):
        gt, obs, ranges, faces = fixture()
        shift = torch.nn.Parameter(torch.tensor([1.7, -.9, 4.2], dtype=torch.float64))
        slicer = supervisor(ranges, faces)
        before = slicer(gt+shift, obs)
        self.assertTrue(before['cell_missing_sections'].all())
        self.assertEqual(before['diagnostics']['recovery_cells'], 4)
        self.assertTrue((before['cell_terms']['recovery_mm'] > 0).all())
        gradient, = torch.autograd.grad(before['cell_losses'].sum(), shift)
        self.assertGreater(float(gradient[2]), 0.)
        self.assertGreater(float(gradient[0]), 0.)
        self.assertLess(float(gradient[1]), 0.)
        after = slicer(gt+shift.detach()-.1*gradient, obs)
        self.assertTrue((after['cell_terms']['recovery_mm'] < before['cell_terms']['recovery_mm']).all())

    def test_capture_derivative_matches_finite_difference(self):
        gt, _, _, faces = fixture()
        shift = torch.tensor(4.2, dtype=gt.dtype, requires_grad=True)
        points = torch.tensor([[.137, -.253, 0.], [1., 1., 0.]], dtype=gt.dtype)
        def loss(z):
            return surface_capture_distance(points, gt[0, 0, :8]+torch.stack((z*0, z*0, z)), faces['LV'])
        derivative, = torch.autograd.grad(loss(shift), shift)
        eps = 1e-5
        numerical = (loss(shift.detach()+eps)-loss(shift.detach()-eps))/(2*eps)
        torch.testing.assert_close(derivative, numerical, rtol=1e-6, atol=1e-8)

    def test_boundary_direction_matches_finite_difference_and_vanishes_near_gt(self):
        gt, _, _, faces = fixture()
        teacher = plane_segments(gt[0, 0, :8], faces['LV'], torch.eye(3, dtype=gt.dtype), gt.new_zeros(3))
        offset = torch.tensor(.73, dtype=gt.dtype, requires_grad=True)
        spacing = gt.new_tensor([.65, 1.15])
        def objective(x):
            return paired_boundary_distance(teacher+torch.stack((x, x*0-.29)), teacher, spacing, (22, 34))
        derivative, = torch.autograd.grad(objective(offset), offset)
        eps = 1e-5
        numerical = (objective(offset.detach()+eps)-objective(offset.detach()-eps))/(2*eps)
        torch.testing.assert_close(derivative, numerical, rtol=1e-5, atol=1e-8)
        def gradient(x):
            shift = torch.tensor(x, dtype=gt.dtype, requires_grad=True)
            value = paired_boundary_distance(teacher+torch.stack((shift, shift*0)), teacher, spacing, (22, 34))
            return torch.autograd.grad(value, shift)[0]
        self.assertLess(float(gradient(1e-6).abs()), float(gradient(.1).abs())*1e-3)

    def test_nonrigid_curve_gradient_includes_arc_length_weights(self):
        gt, _, _, faces = fixture()
        teacher = plane_segments(gt[0, 0, :8], faces['LV'], torch.eye(3, dtype=gt.dtype), gt.new_zeros(3))
        scale = torch.tensor(1.073, dtype=gt.dtype, requires_grad=True)
        def objective(x):
            return paired_boundary_distance(teacher*torch.stack((x, x*0+.94))+teacher.new_tensor([.43, -.27]),
                                            teacher, gt.new_tensor([.65, 1.15]), (22, 34))
        derivative, = torch.autograd.grad(objective(scale), scale)
        eps = 1e-6
        numerical = (objective(scale.detach()+eps)-objective(scale.detach()-eps))/(2*eps)
        torch.testing.assert_close(derivative, numerical, rtol=1e-5, atol=1e-8)

    def test_mixed_present_missing_and_empty_cells_agree_across_backends(self):
        gt, obs, ranges, faces = fixture()
        obs['vertices_gt'] = gt.clone()
        # Empty GT/prediction for patient 1, phase 0, chamber RV.
        obs['vertices_gt'][1, 0, 16:24, 2] += 4.
        obs['native_masks'][1, 0, 1, 2] = 0
        perturbed = obs['vertices_gt'].clone()+torch.tensor([.4, -.2, 0.])
        # Missing predicted section for patient 0, phase 1, chamber RA.
        perturbed[0, 1, 24:32, 2] += 4.2
        inputs = [perturbed.clone().requires_grad_() for _ in range(2)]
        results = [supervisor(ranges, faces, mode)(v, obs)
                   for mode, v in zip(('serial', 'batched'), inputs)]
        for result in results:
            result['total'].backward()
            self.assertEqual(result['diagnostics']['matched_empty_sections'], 1)
            self.assertEqual(result['diagnostics']['recovery_cells'], 1)
        torch.testing.assert_close(results[0]['cell_missing_sections'], torch.tensor([False, True, True, False]))
        torch.testing.assert_close(results[0]['cell_losses'], results[1]['cell_losses'])
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad)

    def test_empty_cells_are_retained_and_spurious_sections_are_penalised(self):
        gt, obs, ranges, faces = fixture(empty=True)
        predicted = gt.clone().requires_grad_()
        slicer = supervisor(ranges, faces)
        empty = slicer(predicted, obs)
        self.assertEqual(len(empty['cell_losses']), 4)
        self.assertEqual(empty['diagnostics']['matched_empty_sections'], 4)
        self.assertEqual(float(empty['total']), 0.)
        empty['total'].backward()
        self.assertEqual(float(predicted.grad.abs().max()), 0.)
        spurious = (gt-torch.tensor([0., 0., 4.])).requires_grad_()
        result = slicer(spurious, obs)
        self.assertGreater(float(result['total']), 0.)
        result['total'].backward()
        self.assertGreater(float(spurious.grad.norm()), 0.)

    def test_empty_target_has_physical_suppression_even_when_stride_misses_curve(self):
        gt, obs, ranges, faces = fixture(empty=True)
        offset = torch.tensor(-4., dtype=gt.dtype, requires_grad=True)
        slicer = supervisor(ranges, faces, pixel_stride=32)
        def render(z):
            return slicer(gt+torch.stack((z*0, z*0, z)), obs)
        result = render(offset)
        kind = OBSERVATION_KINDS.index('empty_target')
        self.assertTrue((result['cell_observation_kind'] == kind).all())
        self.assertTrue((result['cell_terms']['empty_target_mm'] > 0).all())
        torch.testing.assert_close(result['cell_terms']['boundary_mm'], result['cell_terms']['empty_target_mm'])
        gradient, = torch.autograd.grad(result['cell_terms']['boundary_mm'].mean(), offset)
        self.assertLess(float(gradient), 0.)  # Return to GT's positive half-space.
        after = render(offset.detach()-.5*gradient)
        self.assertLess(float(after['cell_terms']['boundary_mm'].mean()), float(result['cell_terms']['boundary_mm'].mean()))
        eps = 1e-5
        numerical = (render(offset.detach()+eps)['cell_terms']['boundary_mm'].mean()-
                     render(offset.detach()-eps)['cell_terms']['boundary_mm'].mean())/(2*eps)
        torch.testing.assert_close(gradient, numerical, rtol=1e-6, atol=1e-8)
        vertices = gt[0, 0, :8].clone().requires_grad_()
        stationary = empty_section_clearance(vertices, gt[0, 0, :8], torch.eye(3, dtype=gt.dtype),
                                             gt.new_zeros(3), gt.new_tensor([.65, 1.15]))
        stationary.backward()
        self.assertEqual(float(stationary), 0.)
        self.assertEqual(float(vertices.grad.abs().max()), 0.)

    def test_residual_kinds_keep_capture_and_empty_rules_explicit(self):
        gt, obs, ranges, faces = fixture()
        predicted = gt.clone()
        predicted[0, 1, 24:32, 2] += 4.2
        obs['vertices_gt'][1, 0, 16:24, 2] += 4.
        predicted[1, 0, 16:24, 2] += 4.
        obs['native_masks'][1, 0, 1, 2] = 0
        result = supervisor(ranges, faces)(predicted, obs)
        names = result['cell_observation_kind_names']
        kinds = [names[i] for i in result['cell_observation_kind'].tolist()]
        self.assertEqual(kinds, ['paired_boundary', 'recovery', 'matched_empty', 'paired_boundary'])
        terms = result['cell_terms']
        torch.testing.assert_close(terms['boundary_mm'], terms['curve_mm']+terms['recovery_mm']+terms['empty_target_mm'])

    def test_boundary_survives_stride_missing_all_foreground_pixels(self):
        gt, obs, ranges, faces = fixture()
        result = supervisor(ranges, faces, pixel_stride=32)(gt+torch.tensor([.7, 0., 0.]), obs)
        self.assertFalse(obs['native_masks'][..., ::32, ::32].any())
        self.assertTrue((result['cell_terms']['boundary_mm'] > 0).all())

    def test_submillimetre_curve_schedule_is_independent_of_soft_pixel_stride(self):
        gt, obs, ranges, faces = fixture()
        slicer = supervisor(ranges, faces, pixel_stride=4, boundary_sample_step_mm=.5,
                            boundary_fine_sample_step_mm=.25, boundary_fine_start_fraction=.75)
        slicer.set_joint_progress(.749)
        self.assertEqual(slicer.boundary_step_cap_mm, .5)
        slicer.set_joint_progress(.75)
        self.assertEqual(slicer.boundary_step_cap_mm, .25)
        with patch('utils.slice_supervision.section_quadrature', wraps=section_quadrature) as quadrature:
            result = slicer(gt+gt.new_tensor([.7, -.4, 0.]), obs)
            self.assertTrue(quadrature.called)
            self.assertTrue(all(float(call.args[1]) <= .25 for call in quadrature.call_args_list))
        self.assertEqual(result['diagnostics']['pixel_stride'], 4)
        self.assertEqual(result['diagnostics']['boundary_sample_step_cap_mm'], .25)
        predicted = gt.clone().requires_grad_()
        matched = slicer(predicted, obs)['cell_terms']['boundary_mm'].sum()
        matched.backward()
        self.assertLess(float(matched.abs()), 1e-10)
        self.assertLess(float(predicted.grad.abs().max()), 1e-10)

    def test_boundary_far_outside_native_fov_still_has_a_return_direction(self):
        gt, obs, ranges, faces = fixture()
        offset = torch.tensor(50., dtype=gt.dtype, requires_grad=True)
        slicer = supervisor(ranges, faces)
        def objective(x):
            return slicer(gt+torch.stack((x, x*0, x*0)), obs)['cell_terms']['boundary_mm']
        before = objective(offset)
        gradient, = torch.autograd.grad(before.mean(), offset)
        self.assertGreater(float(gradient), .9)
        after = objective(offset.detach()-.5*gradient)
        self.assertTrue((after < before).all())

    def test_boundary_distance_scales_in_mm_on_anisotropic_rectangular_grid(self):
        gt, _, _, faces = fixture()
        rotation, translation = torch.eye(3, dtype=gt.dtype), gt.new_zeros(3)
        teacher = plane_segments(gt[0, 0, :8], faces['LV'], rotation, translation)
        prediction = teacher+teacher.new_tensor([.7, -.4])
        spacing = gt.new_tensor([.65, 1.15])
        value = paired_boundary_distance(prediction, teacher, spacing, (22, 34))
        scaled = paired_boundary_distance(prediction*3, teacher*3, spacing*3, (22, 34), beta_mm=3.)
        torch.testing.assert_close(scaled, value*3, rtol=1e-9, atol=1e-10)

    def test_joint_collation_preserves_native_inputs_only_when_needed(self):
        gt, obs, _, _ = fixture()
        sample = {key: value[0] for key, value in obs.items() if isinstance(value, torch.Tensor)}
        sample['pose_gt'] = {key: value[0] for key, value in obs['pose_gt'].items()}
        joint = training_collate([sample], 'joint')
        self.assertIn('native_masks', joint)
        self.assertIn('native_pixel_spacing', joint)
        shape = training_collate([sample], 'shape')
        self.assertNotIn('native_masks', shape)
        self.assertNotIn('pose_gt', shape)

    def test_historical_modes_also_expose_consistent_unreduced_cells(self):
        gt, obs, ranges, faces = fixture()
        obs.update(masks=obs['native_masks'], sdf=torch.zeros_like(obs['native_masks']),
                   pixel_spacing=obs['native_pixel_spacing'])
        for target_mode in ('paired_soft_geometry_v1', 'binary_native_v1'):
            results = []
            vertices = []
            for backend in ('serial', 'batched'):
                slicer = MeshSliceSupervisor(ranges, faces, 1., target_mode=target_mode, execution_backend=backend)
                prediction = (gt+torch.tensor([.6, -.3, 0.])).requires_grad_()
                result = slicer(prediction, obs)
                result['total'].backward()
                torch.testing.assert_close(result['total'], result['cell_losses'].mean())
                torch.testing.assert_close(result['cell_indices'], obs['annotated'].nonzero())
                results.append(result)
                vertices.append(prediction)
            torch.testing.assert_close(results[0]['cell_losses'], results[1]['cell_losses'])
            torch.testing.assert_close(vertices[0].grad, vertices[1].grad)


if __name__ == '__main__':
    unittest.main()
