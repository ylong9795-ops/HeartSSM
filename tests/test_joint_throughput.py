"""Throughput execution must preserve the active geometric objective/gradients."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from test_slice_supervision import fixture, supervisor
from utils.slice_supervision import batched_paired_boundary_distance, paired_boundary_distance, plane_segments
from utils.training_execution import configure_execution, StepProfiler


class JointThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_boundary_only_preserves_all_cells_geometric_losses_and_vertex_gradients(self):
        gt, obs, ranges, faces = fixture()
        values, gradients = [], []
        for fast in (False, True):
            v = (gt+gt.new_tensor([.71, -.37, .13])).requires_grad_()
            result = supervisor(ranges, faces, boundary_only=fast, cell_chunk_size=8)(v, obs)
            values.append(result)
            gradients.append(torch.autograd.grad(result['cell_terms']['boundary_mm'].sum(), v)[0])
        for key in ('cell_indices', 'cell_observation_kind', 'cell_missing_sections'):
            torch.testing.assert_close(values[0][key], values[1][key])
        torch.testing.assert_close(values[0]['cell_terms']['boundary_mm'], values[1]['cell_terms']['boundary_mm'], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-9)
        self.assertFalse(values[1]['diagnostics']['probability_terms_evaluated'])

    def test_empty_recovery_and_paired_branches_keep_same_semantics(self):
        gt, obs, ranges, faces = fixture()
        obs['vertices_gt'] = gt.clone()
        obs['vertices_gt'][1, 0, 16:24, 2] += 4.
        obs['native_masks'][1, 0, 1, 2] = 0
        pred = obs['vertices_gt'].clone()+gt.new_tensor([.4, -.2, 0.])
        pred[0, -1, 24:32, 2] += 4.2
        results, gradients = [], []
        for fast in (False, True):
            v = pred.clone().requires_grad_()
            result = supervisor(ranges, faces, boundary_only=fast)(v, obs)
            results.append(result)
            gradients.append(torch.autograd.grad(result['cell_terms']['boundary_mm'].sum(), v)[0])
        for key in ('cell_observation_kind', 'cell_missing_sections'):
            torch.testing.assert_close(results[0][key], results[1][key])
        torch.testing.assert_close(results[0]['cell_terms']['boundary_mm'], results[1]['cell_terms']['boundary_mm'])
        torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-9)

    def test_batched_quadrature_nonrigid_fov_clipping_and_gradients_match_serial(self):
        gt, _, _, faces = fixture()
        base = plane_segments(gt[0, 0, :8], faces['LV'], torch.eye(3, dtype=gt.dtype), gt.new_zeros(3))
        values, gradients = [], []
        for fast in (False, True):
            scale = gt.new_tensor(1.13, requires_grad=True)
            curves = base*torch.stack((scale, scale*0+.91))+gt.new_tensor([2.3, -.23])
            curves = torch.stack((curves, curves+gt.new_tensor([30., 0.])))
            teacher = torch.stack((base, base))
            spacing = gt.new_tensor([[.4, .7], [.65, 1.1]])
            if fast:
                present = torch.ones(curves.shape[:2], dtype=torch.bool)
                value = batched_paired_boundary_distance(curves, present, teacher, present, spacing, (18, 22), sample_step_mm=.27)
            else:
                value = torch.stack([paired_boundary_distance(curves[i], teacher[i], spacing[i], (18, 22), sample_step_mm=.27) for i in range(2)])
            values.append(value)
            gradients.append(torch.autograd.grad(value.sum(), scale)[0])
        torch.testing.assert_close(*values, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-9)

    def test_exact_gt_remains_stationary_on_fast_path(self):
        gt, obs, ranges, faces = fixture()
        v = gt.clone().requires_grad_()
        result = supervisor(ranges, faces, boundary_only=True)(v, obs)
        loss = result['cell_terms']['boundary_mm'].sum()
        loss.backward()
        self.assertLess(float(loss.abs()), 1e-10)
        self.assertLess(float(v.grad.abs().max()), 1e-10)

    def test_throughput_spends_memory_without_mutating_checkpoint_specification(self):
        _, _, ranges, faces = fixture()
        encoder = SimpleNamespace(cnn_chunk_size=6, checkpoint_cnn=True)
        model = SimpleNamespace(encoder=encoder, encoder_config=dict(cnn_chunk_size=6, checkpoint_cnn=True))
        spec = copy.deepcopy(model.encoder_config)
        slicer = supervisor(ranges, faces)
        settings = configure_execution(model, slicer, objective_mode='grouped_gt_boundary_v1', memory_profile='throughput')
        self.assertEqual(encoder.cnn_chunk_size, 72)
        self.assertFalse(encoder.checkpoint_cnn)
        self.assertEqual(slicer.cell_chunk_size, 64)
        self.assertEqual(slicer.chunk_size, 1024)
        self.assertTrue(slicer.boundary_only)
        self.assertEqual(model.encoder_config, spec)
        self.assertTrue(settings['objective_unchanged'])
        with self.assertRaises(ValueError):
            configure_execution(model, slicer, objective_mode='legacy_v1', memory_profile='throughput')

    def test_profiler_closes_at_budget_and_exports_reviewable_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            with StepProfiler(1, folder, 'cpu') as profiler:
                with torch.profiler.record_function('test_forward'):
                    torch.ones(4).square().sum()
                profiler.step()
                self.assertIsNone(profiler.profiler)
            self.assertTrue((Path(folder)/'training_trace.json').is_file())
            self.assertTrue((Path(folder)/'training_profile.txt').is_file())


if __name__ == '__main__':
    unittest.main()
