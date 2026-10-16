"""GT objective decomposition, control ownership and chamber boundary tails."""
import json
import unittest

import torch
import trimesh

from data.mask_transform import CHAMBERS
from train_neural_ffd import compute_objective
from utils.neural_ffd_losses import NeuralFFDLoss
from utils.slice_supervision import MeshSliceSupervisor, exact_section_mask
from utils.slice_supervision import OBSERVATION_KINDS


COMPONENTS = ('shape', 'trajectory', 'motion', 'velocity', 'volume', 'edge',
              'laplacian', 'normal', 'control', 'slice')


def fixture(frames=3):
    mesh = trimesh.creation.box(extents=[2., 4., 6.])
    vertices = torch.tensor(mesh.vertices, dtype=torch.float64)
    gt = torch.cat([vertices+vertices.new_tensor([8.*c, 0., 0.]) for c in range(4)])[None, None]
    displacement = torch.arange(frames, dtype=vertices.dtype)[None, :, None, None]*vertices.new_tensor([.2, -.1, .3])
    gt = gt+displacement
    ranges = torch.tensor([[c*8, (c+1)*8] for c in range(4)])
    faces = {c: torch.tensor(mesh.faces.copy()) for c in CHAMBERS}
    controls = dict(coarse=gt.new_zeros(1, 2, 2, 2, 3), medium=gt.new_zeros(1, 2, 2, 2, 3),
                    fine_motion=gt.new_zeros(1, frames, 2, 2, 2, 3), regions=None, regional_motion=None)
    return gt, ranges, faces, controls


def criterion(ranges, faces, **weights):
    config = {key: 0. for key in COMPONENTS}
    config.update(weights or dict(shape=1., trajectory=1., motion=1., velocity=.2, volume=.01,
                                 edge=.05, laplacian=.05, normal=.01))
    return NeuralFFDLoss(ranges, faces, objective_mode='grouped_gt_boundary_v1', weights=config)


def slices(boundary):
    # Unequal chamber counts: LV4 / LA2 / RV1 / RA1.
    indices = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 2, 0, 0],
                            [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 2, 2], [0, 1, 2, 3]])
    return dict(cell_indices=indices, cell_terms=dict(boundary_mm=boundary),
                cell_losses=boundary*0+100., total=boundary.sum()*0+999.,
                diagnostics=dict(boundary_objective='paired_boundary_distance_v1', target_mode='paired_soft_geometry_v1'))


class GTObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_groups_have_an_exact_single_total_and_no_control_double_count(self):
        gt, ranges, faces, controls = fixture()
        p = (gt+torch.randn_like(gt)*.03).requires_grad_()
        residuals = torch.arange(8, dtype=gt.dtype, requires_grad=True)
        loss = criterion(ranges, faces, shape=1., trajectory=1., motion=1., control=.1, slice=.5)(
            dict(vertices_4d=p, control_grids=controls), gt, slice_result=slices(residuals),
            control_spacings_mm=torch.ones(3, 3))
        torch.testing.assert_close(loss['total'], loss['L_S']+loss['L_M']+.5*loss['L_O'])
        weighted_terms = sum(loss['active_weights'][key]*value for key, value in loss['terms'].items())
        torch.testing.assert_close(loss['total'], weighted_terms)
        self.assertEqual(loss['active_weights']['control'], 0.)
        loss['total'].backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.norm()), 0.)
        self.assertGreater(float(residuals.grad.norm()), 0.)

    def test_all_phase_positions_are_supervised_even_with_perfect_reference(self):
        gt, ranges, faces, controls = fixture()
        p = gt.clone()
        p[:, 1:] += p.new_tensor([.6, -.3, .2])
        p.requires_grad_()
        loss = criterion(ranges, faces, shape=1., trajectory=1.)(dict(vertices_4d=p, control_grids=controls), gt)
        self.assertEqual(float(loss['terms']['shape']), 0.)
        self.assertGreater(float(loss['terms']['trajectory']), 0.)
        loss['L_S'].backward()
        self.assertGreater(float(p.grad[:, 1:].norm()), 0.)

    def test_unobserved_vertex_has_gt_gradient_and_no_template_anchor(self):
        gt, ranges, faces, controls = fixture()
        teacher = gt.clone().requires_grad_()
        p = gt.clone()
        p[0, 2, 31, 1] += .4  # No observation or visibility mask is supplied.
        p.requires_grad_()
        loss = criterion(ranges, faces, trajectory=1.)(dict(vertices_4d=p, control_grids=controls), teacher)
        loss['total'].backward()
        self.assertGreater(float(p.grad[0, 2, 31, 1]), 0.)
        self.assertIsNone(teacher.grad)
        self.assertFalse(any('template' in key or 'unknown' in key for key in loss['terms']))

    def test_shape_translation_does_not_create_gt_motion_error(self):
        gt, ranges, faces, controls = fixture()
        p = (gt+gt.new_tensor([1., -2., .5])).requires_grad_()
        loss = criterion(ranges, faces)(dict(vertices_4d=p, control_grids=controls), gt)
        self.assertGreater(float(loss['L_S']), 0.)
        self.assertLess(abs(float(loss['L_M'])), 1e-10)

    def test_gt_motion_amplitude_and_direction_are_independent_diagnostics(self):
        gt, ranges, faces, controls = fixture()
        objective = criterion(ranges, faces)
        for factor, ratio, direction in ((.5, .5, 1.), (-1., 1., -1.)):
            p = gt[:, :1]+factor*(gt-gt[:, :1])
            loss = objective(dict(vertices_4d=p, control_grids=controls), gt)
            diagnostics = loss['motion_diagnostics']
            torch.testing.assert_close(diagnostics['amplitude_ratio'], gt.new_full((4,), ratio))
            torch.testing.assert_close(diagnostics['direction_cosine'], gt.new_full((4,), direction))
            self.assertTrue(diagnostics['amplitude_defined'].all())
            self.assertTrue(diagnostics['direction_defined'].all())
            self.assertGreater(float(loss['terms']['motion']), 0.)
            self.assertFalse(diagnostics['amplitude_ratio'].requires_grad)

    def test_velocity_supervision_includes_last_to_first_step(self):
        gt, ranges, faces, controls = fixture()
        p = gt.clone()
        p[:, -1, :, 0] += .6
        loss = criterion(ranges, faces, velocity=1.)(dict(vertices_4d=p, control_grids=controls), gt)
        # x-velocity residuals [0,.6,-.6], Huber beta1, 3 XYZ components.
        self.assertAlmostEqual(float(loss['L_M']), .04, places=12)

    def test_observation_balances_chambers_and_increases_tail_gradient(self):
        gt, ranges, faces, controls = fixture()
        boundary = torch.tensor([1., 2., 3., 10., 2., 6., 4., 8.], dtype=gt.dtype, requires_grad=True)
        loss = criterion(ranges, faces, slice=1.)(dict(vertices_4d=gt, control_grids=controls), gt,
                                                  slice_result=slices(boundary))
        # Means [4,4,4,8], top quartile [10,6,4,8]: equal-chamber mean=12.
        self.assertEqual(float(loss['L_O']), 12.)
        torch.testing.assert_close(loss['observation_summary']['cell_counts'], torch.tensor([4, 2, 1, 1]))
        loss['total'].backward()
        expected = gt.new_tensor([.0625, .0625, .0625, .3125, .125, .375, .5, .5])
        torch.testing.assert_close(boundary.grad, expected)

    def test_absent_observation_chambers_are_reported_without_zero_dilution(self):
        gt, ranges, faces, controls = fixture()
        boundary = gt.new_tensor([1., 2., 3., 10., 2., 6., 4., 8.])
        observed = slices(boundary)
        observed['cell_indices'] = observed['cell_indices'][:4]
        observed['cell_terms']['boundary_mm'] = boundary[:4]
        loss = criterion(ranges, faces, slice=1.)(dict(vertices_4d=gt, control_grids=controls), gt, slice_result=observed)
        self.assertEqual(float(loss['L_O']), 14.)
        torch.testing.assert_close(loss['observation_summary']['present'], torch.tensor([True, False, False, False]))

    def test_proxy_status_never_claims_pass_fail_without_train_calibration(self):
        gt, ranges, faces, controls = fixture()
        boundary = gt.new_tensor([.1, .3, .05, 0., 0., .2, 0., 0.], requires_grad=True)
        observed = slices(boundary)
        observed['cell_observation_kind'] = torch.tensor([0, 0, 1, 2, 3, 0, 2, 4])
        observed['cell_observation_kind_names'] = OBSERVATION_KINDS
        objective = NeuralFFDLoss(ranges, faces, objective_mode='grouped_gt_boundary_v1',
                                 weights=dict(control=0., slice=1.))
        loss = objective(dict(vertices_4d=gt, control_grids=controls), gt, slice_result=observed)
        summary = loss['observation_summary']
        self.assertEqual(summary['boundary_threshold_state'], 'DISABLED_UNCALIBRATED')
        self.assertIsNone(summary['proxy_boundary_pass_counts'])
        self.assertIsNone(summary['proxy_boundary_exceedance_counts'])
        self.assertIsNone(summary['proxy_failure_counts'])
        torch.testing.assert_close(summary['recovery_counts'], torch.tensor([1, 0, 0, 0]))
        torch.testing.assert_close(summary['empty_target_counts'], torch.tensor([0, 1, 0, 0]))
        without_kinds = objective(dict(vertices_4d=gt, control_grids=controls), gt, slice_result=slices(boundary))
        torch.testing.assert_close(loss['L_O'], without_kinds['L_O'])
        grad_typed, = torch.autograd.grad(loss['L_O'], boundary, retain_graph=True)
        grad_untyped, = torch.autograd.grad(without_kinds['L_O'], boundary)
        torch.testing.assert_close(grad_typed, grad_untyped)
        serialised = {key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                      for key, value in summary.items()}
        json.dumps(serialised, allow_nan=False)

    def test_uncalibrated_numeric_proxy_thresholds_are_rejected(self):
        gt, ranges, faces, controls = fixture()
        for threshold in (0., -1., float('nan'), float('inf'), .2, .5):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, 'TRAIN-only'):
                NeuralFFDLoss(ranges, faces, observation_boundary_threshold_mm=threshold)
        observed = slices(gt.new_zeros(8))
        observed['cell_observation_kind'] = torch.full((8,), 99, dtype=torch.long)
        observed['cell_observation_kind_names'] = OBSERVATION_KINDS
        with self.assertRaisesRegex(ValueError, 'observation kinds'):
            criterion(ranges, faces, slice=1.)(dict(vertices_4d=gt, control_grids=controls), gt, slice_result=observed)

    def test_invalid_or_legacy_boundary_inputs_cannot_be_silently_reused(self):
        gt, ranges, faces, controls = fixture()
        objective = criterion(ranges, faces, slice=1.)
        for kind in ('legacy', 'duplicate', 'negative', 'out_of_bounds'):
            observed = slices(gt.new_ones(8))
            if kind == 'legacy': observed['diagnostics']['boundary_objective'] = 'legacy_probability_v1'
            if kind == 'duplicate': observed['cell_indices'][1] = observed['cell_indices'][0]
            if kind == 'negative': observed['cell_terms']['boundary_mm'][0] = -1.
            if kind == 'out_of_bounds': observed['cell_indices'][0, 1] = 3
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                objective(dict(vertices_4d=gt, control_grids=controls), gt, slice_result=observed)

    def test_control_partition_preserves_scale_and_reference_motion_ownership(self):
        gt, ranges, faces, controls = fixture()
        xyz = torch.meshgrid(*(torch.arange(2, dtype=gt.dtype) for _ in range(3)), indexing='ij')
        def grid(axis, scale):
            return (xyz[axis][..., None].expand(2, 2, 2, 3)*scale)[None].clone()
        controls.update(coarse=grid(0, 1).requires_grad_(), medium=grid(1, 2).requires_grad_(),
                        fine_motion=torch.stack([grid(2, scale) for scale in (3, 6, 9)], 1).requires_grad_())
        loss = criterion(ranges, faces, control=1.)(dict(vertices_4d=gt, control_grids=controls), gt,
                                                   control_spacings_mm=torch.ones(3, 3))
        self.assertAlmostEqual(float(loss['L_S']), 8/9)
        self.assertAlmostEqual(float(loss['L_M']), 39/9)
        self.assertAlmostEqual(float(loss['total']), 47/9)
        torch.testing.assert_close(loss['terms']['control'], loss['terms']['control_shape']+loss['terms']['control_motion'])
        loss['L_S'].backward(retain_graph=True)
        fine = controls['fine_motion']
        self.assertGreater(float(fine.grad[:, :1].norm()), 0.)
        self.assertEqual(float(fine.grad[:, 1:].abs().max()), 0.)
        for value in controls.values():
            if isinstance(value, torch.Tensor): value.grad = None
        loss['L_M'].backward()
        self.assertEqual(float(fine.grad[:, :1].abs().max()), 0.)
        self.assertGreater(float(fine.grad[:, 1:].norm()), 0.)
        self.assertIsNone(controls['coarse'].grad)

        controls['regions'] = grid(0, 4)[:, None].expand(-1, 4, -1, -1, -1, -1).clone()
        regional_motion = grid(0, 2)[:, None, None].expand(-1, 3, 4, -1, -1, -1, -1).clone()
        regional_motion[:, 0] = 0
        controls['regional_motion'] = regional_motion
        loss = criterion(ranges, faces, control=1.)(dict(vertices_4d=gt, control_grids=controls), gt,
            control_spacings_mm=torch.ones(3, 3), regional_spacing_mm=torch.ones(3),
            regional_motion_spacings_mm=torch.ones(4, 3))
        self.assertAlmostEqual(float(loss['L_S']), 2.)
        self.assertAlmostEqual(float(loss['L_M']), 55/12)
        self.assertAlmostEqual(float(loss['total']), 79/12)

    def test_legacy_total_and_reference_pretraining_remain_compatible(self):
        gt, ranges, faces, controls = fixture()
        p = gt.clone()
        p[:, 1:] += .3
        for mode in ('legacy_v1', 'grouped_gt_boundary_v1'):
            objective = NeuralFFDLoss(ranges, faces, objective_mode=mode)
            stages = ('joint', 'shape', 'motion') if mode == 'legacy_v1' else ('shape', 'motion')
            for stage in stages:
                observed = dict(total=gt.new_tensor(.7)) if stage == 'joint' else None
                loss = objective(dict(vertices_4d=p, control_grids=controls), gt, stage=stage,
                                 slice_result=observed, control_spacings_mm=torch.ones(3, 3))
                expected = sum(loss['active_weights'][key]*value for key, value in loss['terms'].items())
                torch.testing.assert_close(loss['total'], expected)
                self.assertEqual(loss['active_weights']['trajectory'], 0.)
                if stage == 'shape':
                    self.assertLess(abs(float(loss['L_S'])), 1e-10)
                    self.assertEqual(float(loss['L_M']), 0.)
                    self.assertEqual(float(loss['L_O']), 0.)

    def test_single_frame_diagnostics_do_not_return_nan(self):
        gt, ranges, faces, controls = fixture(frames=1)
        loss = criterion(ranges, faces)(dict(vertices_4d=gt, control_grids=controls), gt)
        for key, value in loss['motion_diagnostics'].items():
            if isinstance(value, torch.Tensor): self.assertTrue(torch.isfinite(value).all(), key)
        self.assertFalse(loss['motion_diagnostics']['amplitude_defined'].any())
        self.assertFalse(loss['motion_diagnostics']['direction_defined'].any())

    def test_joint_pipeline_uses_one_forward_one_backward_and_one_step(self):
        gt, ranges, faces, controls = fixture()
        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(gt.new_tensor([.4, -.2, .1]))
                self.chamber_ranges, self.faces = ranges, faces
                self.control_spacings_mm = gt.new_ones(3, 3)
                self.regional_shape = self.regional_motion = False
                self.calls = 0
                self.backward_calls = 0
                def backward_hook(gradient):
                    self.backward_calls += 1
                self.offset.register_hook(backward_hook)

            def forward(self, batch):
                self.calls += 1
                phase = 1+torch.arange(3, dtype=gt.dtype)[None, :, None, None]*.1
                return dict(vertices_4d=gt+self.offset*phase, control_grids=controls)

        native = gt.new_zeros(1, 3, 3, 4, 48, 80)
        for phase in range(3):
            for chamber, (left, right) in enumerate(ranges.tolist()):
                rendered = exact_section_mask(gt[0, phase, left:right].numpy(), faces[CHAMBERS[chamber]].numpy(),
                    torch.eye(3).numpy(), torch.zeros(3).numpy(), [1., 1.], (48, 80))
                native[0, phase, :, chamber] = torch.from_numpy(rendered)
        batch = dict(vertices_gt=gt, native_masks=native, native_pixel_spacing=gt.new_ones(1, 3, 3, 2),
                     view_present=torch.ones(1, 3, dtype=torch.bool),
                     annotated=torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]], dtype=torch.bool),
                     pose_gt=dict(probe_R=torch.eye(3, dtype=gt.dtype).expand(1, 3, 3, 3, 3), probe_t=gt.new_zeros(1, 3, 3, 3)))
        slicer = MeshSliceSupervisor(ranges, faces, 1., target_mode='paired_soft_geometry_v1',
            supervision_grid='native_grid', boundary_objective='paired_boundary_distance_v1', execution_backend='batched')
        model = CountingModel()
        objective = criterion(ranges, faces, shape=1., trajectory=1., motion=1., velocity=.2, slice=.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = model.offset.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        _, losses, observed = compute_objective(model, batch, objective, slicer, 'joint')
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.backward_calls, 1)
        self.assertFalse(torch.equal(before, model.offset.detach()))
        self.assertEqual(len(observed['cell_losses']), 24)
        self.assertTrue((losses['observation_summary']['cell_counts'] > 0).all())


if __name__ == '__main__':
    unittest.main()
