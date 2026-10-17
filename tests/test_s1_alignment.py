"""Known physical shifts, empty-slice capture, robust balance and pose gauge."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

from tto.alignment.centres import balanced_correction, centroid, measure
from tto.alignment.global_alignment import NativePlaneSearch, canonical_pose, shift_centre, score_cells
from tto.alignment.config import validate_pose_config
from tto.config import load_config
from tto.evaluator import native_exact
from tto.state import TTOState
from test_real_tto_joint import small_fixture


class S1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_oblique_anisotropic_translation_sign_is_positive_plane_shift(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=12.)
        rotation = Rotation.from_euler('xyz', [20., -30., 15.], degrees=True).as_matrix()
        base = dict(R=rotation, center_mm=np.zeros(3))
        true = shift_centre(base, np.array([6., -3.5]))
        spacing, shape = [.5, 1.2], (100, 86)
        obs = native_exact(mesh.vertices, mesh.faces, **dict(rotation=rotation, center_mm=true['center_mm'],
            spacing_yx=spacing, shape=shape))
        pred = native_exact(mesh.vertices, mesh.faces, rotation, base['center_mm'], spacing, shape)
        diagnostic = measure(pred, obs, spacing)
        np.testing.assert_allclose(diagnostic['centroid_error_xy_mm'], [6., -3.5], atol=.08)
        corrected = shift_centre(base, diagnostic['centroid_error_xy_mm'])
        revised = native_exact(mesh.vertices, mesh.faces, rotation, corrected['center_mm'], spacing, shape)
        self.assertGreater(measure(revised, obs, spacing)['exact_hard_dice'], .98)
        self.assertLess(measure(revised, obs, spacing)['centroid_error_norm_mm'], .1)

    def test_empty_centroids_are_undefined_and_area_is_independent(self):
        empty = np.zeros((32, 40), bool); obs = empty.copy(); obs[10:16, 12:18] = True
        self.assertIsNone(centroid(empty))
        missing = measure(empty, obs, [.7, 1.2])
        self.assertIsNone(missing['centroid_error_xy_mm'])
        self.assertEqual(missing['area_ratio'], 0.)
        large = empty.copy(); large[7:19, 9:21] = True
        diagnostic = measure(large, obs, [.7, 1.2])
        self.assertEqual(diagnostic['area_ratio'], 4.)
        self.assertEqual(diagnostic['centroid_error_norm_mm'], 0.)

    def test_temporal_then_chamber_balance_has_no_lv_area_or_count_dominance(self):
        rows = [dict(chamber=0, fit_weight=1., centroid_error_xy_mm=np.array([0., 0.]), prediction_area_mm2=10000.)]*100
        rows += [dict(chamber=2, fit_weight=1., centroid_error_xy_mm=np.array([10., 0.]), prediction_area_mm2=10.)]*2
        result = balanced_correction(rows, delta_mm=20.)
        np.testing.assert_allclose(result['correction_xy_mm'], [5., 0.])
        # Source outlier robustness within a chamber.
        outliers = [dict(chamber=0, fit_weight=1., centroid_error_xy_mm=np.array([2., 1.])) for _ in range(20)]
        outliers += [dict(chamber=0, fit_weight=.1, centroid_error_xy_mm=np.array([100., 100.]))]
        np.testing.assert_allclose(balanced_correction(outliers)['correction_xy_mm'], [2., 1.], atol=.02)

    def test_global_rigid_gauge_conversion_preserves_native_render(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=8.)
        gr = Rotation.from_euler('xyz', [30., -15., 20.], degrees=True).as_matrix()
        gt = np.array([13., -9., 5.])
        rotation = Rotation.from_euler('xyz', [5., 12., -3.], degrees=True).as_matrix()
        center = np.array([.23, -.19, 1.17])
        world = dict(R=(gr @ rotation)[None], center_mm=(center @ gr.T+gt)[None])
        recovered = canonical_pose(dict(R=gr, center_mm=gt), world)
        a = native_exact(mesh.vertices @ gr.T+gt, mesh.faces, world['R'][0], world['center_mm'][0], [.7, 1.1], (50, 70))
        b = native_exact(mesh.vertices, mesh.faces, recovered['R'][0], recovered['center_mm'][0], [.7, 1.1], (50, 70))
        np.testing.assert_array_equal(a, b)
        model, patients, trajectory, _ = small_fixture()
        state = TTOState(patient_id='patient_0', identity=dict(patient_id='patient_0'),
            z_shape=torch.zeros(1,64), z_motion=torch.zeros(1,32), conditions={},
            vertices_reference=trajectory[0,0], trajectory=trajectory[0],
            global_transform=dict(R=torch.eye(3), center_mm=torch.tensor([1.,0.,0.])),
            diagnostics=dict(pose_gauge='canonical_identity_global'))
        with self.assertRaisesRegex(ValueError, 'identity global'):
            state.validate()

    def test_broad_capture_recovers_an_empty_33mm_offset_without_mesh_updates(self):
        model, patients, trajectory, _ = small_fixture()
        patient = patients[0]
        patient.metadata.update(input_mode='aligned24')
        patient.metadata['contract'].update(input_mode='aligned24', phase_alignment=dict(status='VERIFIED', phase_grid_status='VERIFIED'))
        for vi, view in enumerate(patient.views):
            view['annotated'][:] = False
            if vi == 0:
                for t in (0, 12):
                    view['annotated'][t, [0, 1]] = True
                    for ci in (0, 1):
                        a, b = model.chamber_ranges[ci].tolist()
                        view['masks'][t, ci] = native_exact(trajectory[0, t, a:b].numpy(), model.faces['LV'].numpy(),
                            np.eye(3), np.zeros(3), view['spacing'], view['masks'].shape[-2:])
        config = load_config('configs/tto.yaml')
        options = deepcopy(config['alignment']['pose'])
        options.update(coarse_tilt_deg=10., coarse_roll_deg=10., capture_quantiles=[.5],
            beam_width=2, retain_candidates=2, max_seconds_per_view=60.)
        engine = NativePlaneSearch(model, patient, trajectory[0].numpy(), config['supervision'], options)
        original = trajectory.clone()
        result = engine.search_view(0, dict(R=np.eye(3), center_mm=np.array([0., 0., 33.])))
        self.assertTrue(all(r['centroid_error_xy_mm'] is None for r in result['baseline_cells']))
        self.assertGreater(result['candidates'][0]['metrics']['trusted_macro_dice'], .97)
        self.assertGreaterEqual(len(result['candidates']), 2)
        self.assertTrue(any(r['stage'] == 'independent_mesh_capture' for r in result['search_records']))
        torch.testing.assert_close(trajectory, original, rtol=0, atol=0)
        self.assertEqual(engine.search_view(1, dict(R=np.eye(3), center_mm=np.zeros(3)))['status'], 'NO_USABLE_OBSERVATIONS')

    def test_search_policy_rejects_15mm_local_only_capture(self):
        config = load_config('configs/tto.yaml')['alignment']['pose']
        config['coarse_depth_span_mm'] = 15.
        with self.assertRaisesRegex(ValueError, '40 mm'):
            validate_pose_config(config)


if __name__ == '__main__':
    unittest.main()
