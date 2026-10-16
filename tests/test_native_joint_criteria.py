"""Frozen Native evaluation goals remain separate from training proxies."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import trimesh

from data.mask_transform import CHAMBERS
from train_neural_ffd import native_cell_criteria, exact_metrics, validation_summary
from utils.slice_evaluation import render_source_slice


class NativeJointCriteriaTests(unittest.TestCase):
    def test_inclusive_native_goals_and_independent_boundary_diagnostics(self):
        boundary = dict(boundary_assd_mm=.5, boundary_hd95_mm=1., empty_boundary=None)
        result = native_cell_criteria(.98, boundary)
        self.assertTrue(result['native_dice_pass'])
        self.assertTrue(result['native_boundary_diagnostic_pass'])
        result = native_cell_criteria(.99, dict(boundary, boundary_assd_mm=.5001))
        self.assertTrue(result['native_dice_pass'])
        self.assertFalse(result['native_boundary_diagnostic_pass'])
        result = native_cell_criteria(.9799, boundary)
        self.assertFalse(result['native_dice_pass'])
        self.assertTrue(result['native_boundary_diagnostic_pass'])

    def test_empty_boundaries_are_explicit_not_zero_distance_successes(self):
        both = native_cell_criteria(1., dict(boundary_assd_mm=0., boundary_hd95_mm=0., empty_boundary='both'))
        self.assertTrue(both['native_dice_pass'])
        self.assertIsNone(both['native_boundary_diagnostic_pass'])
        one = native_cell_criteria(0., dict(boundary_assd_mm=None, boundary_hd95_mm=None, empty_boundary='one'))
        self.assertFalse(one['native_dice_pass'])
        self.assertFalse(one['native_boundary_diagnostic_pass'])

    def test_cohort_average_cannot_hide_one_failed_cell_or_unrun_exact(self):
        metrics = {c: dict(reference_vertex_rmse_mm=0., trajectory_vertex_rmse_mm=0.,
                          relative_motion_rmse_mm=0., volume_mae_ml=0.) for c in CHAMBERS}
        scores = [1.]*99+[.979]
        exact = dict(summary=dict(cells=100, mean_exact_dice=float(np.mean(scores))),
                     cells=[dict(exact_dice=value, frame=0) for value in scores])
        summary = validation_summary(dict(cases=[dict(metrics=metrics, exact=exact)], case_count=1))
        self.assertGreater(summary['mean_exact_dice'], .98)
        self.assertFalse(summary['all_native_dice_pass'])
        self.assertEqual(summary['native_dice_fail_cells'], 1)
        self.assertEqual(summary['minimum_exact_dice'], .979)
        unrun = validation_summary(dict(cases=[dict(metrics=metrics)], case_count=1))
        self.assertIsNone(unrun['all_native_dice_pass'])
        self.assertIsNone(unrun['native_dice_fail_cells'])

    def test_exact_report_scores_every_valid_cell_using_frozen_renderer(self):
        mesh = trimesh.creation.box(extents=[6., 8., 4.]);mesh.apply_translation([.137, -.253, .31])
        vertices = torch.tensor(mesh.vertices, dtype=torch.float64)
        gt = vertices.repeat(4, 1)[None, None]
        faces = {c: torch.tensor(mesh.faces.copy()) for c in CHAMBERS}
        ranges = torch.tensor([[8*c, 8*(c+1)] for c in range(4)])
        mask = render_source_slice(mesh.vertices, mesh.faces, np.eye(3), np.zeros(3), np.array([16., 16., 1.]), 1., (32, 32))
        native = torch.from_numpy(mask).expand(1, 1, 3, 4, 32, 32).clone()
        batch = dict(vertices_gt=gt, native_masks=native, native_pixel_spacing=torch.ones(1, 1, 3, 2),
            view_present=torch.ones(1, 3, dtype=torch.bool),
            annotated=torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]], dtype=torch.bool),
            pose_gt=dict(probe_R=torch.eye(3).expand(1, 1, 3, 3, 3), probe_t=torch.zeros(1, 1, 3, 3),
                         probe_s=torch.tensor([16., 16., 1.]).expand(1, 1, 3, 3)))
        model = SimpleNamespace(chamber_ranges=ranges, faces=faces, norm_to_mm=1.)
        report = exact_metrics(model, gt, batch)
        self.assertEqual(len(report['cells']), 8)
        self.assertEqual(report['summary']['native_dice_pass_cells'], 8)
        self.assertTrue(report['summary']['all_native_dice_pass'])
        self.assertEqual(report['summary']['native_boundary_diagnostic_pass_cells'], 8)
        self.assertEqual(report['criteria']['boundary_role'], 'diagnostic_only')
        self.assertEqual(report['criteria']['proxy_threshold'], 'disabled_pending_TRAIN_calibration')
        predicted = gt.clone();predicted[:, :, :8, 2] += 10.
        missing = exact_metrics(model, predicted, batch)
        self.assertEqual(missing['summary']['native_dice_fail_cells'], 3)
        self.assertFalse(missing['summary']['all_native_dice_pass'])
        self.assertEqual(missing['summary']['native_boundary_diagnostic_fail_cells'], 3)
        for row in missing['cells']:
            self.assertIn('boundary_assd_mm', row)
            self.assertIn('boundary_hd95_mm', row)


if __name__ == '__main__':
    unittest.main()
