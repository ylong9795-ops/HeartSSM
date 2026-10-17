"""Scaffold boundaries, real decoder initialization and persisted identity checks."""
from copy import deepcopy
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_real_tto import fixture
from test_real_tto_joint import small_fixture
from tto.config import load_config, validate_config
from tto.dataset import RealTTODataset
from tto.geometry.motion_metrics import motion_metrics
from tto.geometry.pose_math import so3, transform_points
from tto.io import ROOT, load_state, save_state
from tto.pipeline import TTOPipeline
from tto.run import main
from tto.state import TTOState
from tto.supervision.evaluation import NativeEvaluator
from tto.supervision.losses import NativeSupervision
from tto.synthetic_oracle import evaluate_hidden, hidden_patient


class ScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root, 'patient')
        self.config = load_config('configs/tto.yaml')
        self.config['data'].update(root=str(self.root), input_mode='raw_rr', model_size=32,
            canonical_fov_mm={v: 60. for v in ('A2C', 'A3C', 'A4C')})
        self.config['output']['root'] = str(self.root / 'result')
        self.config['runtime']['threads'] = 2

    def test_configuration_rejects_legacy_solver_and_final_chamber_corrections(self):
        for name, value in [('version', 'real-tto-joint-v1'), ('motion_limits', {}), ('pose', {})]:
            config = deepcopy(self.config)
            config[name] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config = deepcopy(self.config)
        config['alignment']['chamber_mode'] = 'final_static_offsets'
        with self.assertRaises(ValueError):
            validate_config(config)
        config = deepcopy(self.config)
        config['loss']['capture_prior_weight'] = .02
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_missing_foreground_is_reported_without_capture_gradient(self):
        model, patients, trajectory, pose = small_fixture()
        trajectory.requires_grad_()
        pose['center_mm'] = (pose['center_mm'] + torch.tensor([0., 0., 100.])).requires_grad_()
        result = NativeSupervision(model, patients, self.config['loss'],
            self.config['supervision'], phases=[0])(trajectory, pose)
        self.assertTrue(all(result['missing_sections']))
        self.assertGreater(float(result['slice'][0]), 0.)
        result['slice'].sum().backward()
        self.assertEqual(float(trajectory.grad.abs().max()), 0.)
        self.assertEqual(float(pose['center_mm'].grad.abs().max()), 0.)
        self.assertEqual(set(result['components']), {'dice', 'bce', 'boundary_mm'})
        for view in patients[0].views:
            view['masks'][:] = 0
        patients[0]._native_cache.clear()
        empty = NativeSupervision(model, patients, self.config['loss'],
            self.config['supervision'], phases=[0])(trajectory.detach(), pose)
        self.assertEqual(float(empty['slice'][0]), 0.)

    def test_unimplemented_stage_fails_before_model_or_output_and_cli_is_nonzero(self):
        pipeline = TTOPipeline(self.config)
        with patch.object(pipeline, 'load_prior', side_effect=AssertionError('unexpected model load')):
            for stage in ('global_alignment', 'chamber_candidates', 'shape', 'motion', 'joint'):
                with self.assertRaises(NotImplementedError):
                    pipeline.run(['patient'], stage=stage)
        self.assertFalse((self.root / 'result').exists())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--stage', 'joint'])
        self.assertEqual(error.exception.code, 2)

    def test_preflight_never_loads_prior(self):
        pipeline = TTOPipeline(self.config)
        with patch.object(pipeline, 'load_prior', side_effect=AssertionError('unexpected model load')):
            report = pipeline.run(['missing', 'patient'], stage='preflight')
        self.assertEqual(report['completed'], ['patient'])
        self.assertFalse(report['optimization_performed'])
        self.assertEqual(len(report['failures']), 1)
        self.assertEqual(report['failures'][0]['patient_id'], 'missing')

    def test_motion_diagnostics_do_not_constrain_search_or_mutate_trajectories(self):
        model, _, trajectory, _ = small_fixture()
        candidate = trajectory[0].clone()
        candidate[:, :, 0] += torch.sin(torch.arange(24) * (2*np.pi/24))[:, None]
        before = candidate.clone()
        report = motion_metrics(model, candidate, trajectory[0])
        self.assertGreater(report['chambers']['LV']['relative_motion_rmse_mm'], .1)
        torch.testing.assert_close(before, candidate, rtol=0, atol=0)
        self.assertNotIn('passed', report)
        with self.assertRaises(ValueError):
            motion_metrics(model, trajectory.repeat(2, 1, 1, 1), trajectory)

    def test_hidden_patient_excludes_truth_and_evaluator_requires_committed_endpoint(self):
        model, patients, trajectory, _ = small_fixture()
        patient = patients[0]
        # Same native grid for this synthetic boundary fixture.
        masks = np.stack([np.repeat(v['masks'][0:1, :, :24, :30], 24, 0) for v in patient.views], 1)
        sample = dict(native_masks=torch.from_numpy(masks), native_pixel_spacing=torch.ones(24, 3, 2),
            annotated=torch.ones(24, 3, 4, dtype=torch.bool), view_present=torch.ones(24, 3, dtype=torch.bool),
            masks=torch.from_numpy(masks), sdf=torch.zeros_like(torch.from_numpy(masks), dtype=torch.float32),
            phases=torch.arange(24)/24, vertices_gt=trajectory[0], norm_to_mm=1.,
            pose_gt=dict(probe_R=torch.eye(3).expand(24, 3, 3, 3), probe_t=torch.zeros(24, 3, 3)),
            case_id='case', sample_sha256='sample', source_manifest_sha256='manifest')
        observed, truth = hidden_patient(sample, 'synthetic')
        self.assertFalse({'vertices_gt', 'pose_gt'} & observed.encoder.keys())
        self.assertNotIn('vertices_gt', observed.metadata)
        self.assertIn('vertices_gt', truth)
        with self.assertRaises(FileNotFoundError):
            evaluate_hidden(None, self.root / 'not_saved.pth', truth)
        state = TTOState(patient_id='synthetic', identity=dict(patient_id='synthetic',
            case_id='case', sample_sha256='sample'), z_shape=torch.zeros(1, 64),
            z_motion=torch.zeros(1, 32), conditions={}, vertices_reference=trajectory[0, 0],
            trajectory=trajectory[0], global_transform=dict(R=torch.eye(3), center_mm=torch.zeros(3)))
        path = self.root / 'committed.pth'
        save_state(path, state.to_payload())
        with patch('tto.synthetic_oracle.surface_distances', return_value=dict(ASSD_mm=0., HD95_mm=0.)):
            report = evaluate_hidden(model, path, truth)
        self.assertEqual(report['trajectory_vertex_rmse_mm'], 0.)
        self.assertEqual(report['relative_motion_rmse_mm'], 0.)
        self.assertEqual(report['recovery_gate'], 'NOT_IMPLEMENTED')
        with self.assertRaises(ValueError):
            evaluate_hidden(model, path, dict(truth, sample_sha256='foreign'))

    def test_rigid_state_mapping_preserves_native_evaluation(self):
        model, patients, trajectory, pose = small_fixture()
        patient = patients[0]
        evaluator = NativeEvaluator(model, self.config['supervision'])
        base_pose = {key: value[0] for key, value in pose.items()}
        rows = evaluator.cells(trajectory[0].numpy(),
            {key: value.numpy() for key, value in base_pose.items()}, patient, 'initialized')
        transform = dict(R=so3(torch.tensor([20., 30., 40.])), center_mm=torch.tensor([13., -8., 4.]))
        # Exercise storage of a differentiable decoder result as well.
        vertices = trajectory[0].clone().requires_grad_() * 1.
        state = TTOState(patient_id=patient.patient_id,
            identity=dict(patient_id=patient.patient_id, observations_sha256=patient.identity),
            z_shape=torch.zeros(1, 64), z_motion=torch.zeros(1, 32), conditions={},
            vertices_reference=vertices[0], trajectory=vertices, global_transform=transform,
            observation_pose=dict(R=transform['R'] @ base_pose['R'],
                center_mm=transform_points(base_pose['center_mm'], transform)))
        path = self.root / 'transformed.pth'
        save_state(path, state.to_payload())
        restored = TTOState.from_payload(load_state(path))
        summary, transformed = evaluator.evaluate_state(restored, patient)
        np.testing.assert_allclose([r['exact_hard_dice'] for r in rows],
                                   [r['exact_hard_dice'] for r in transformed], atol=0, rtol=0)
        self.assertEqual(summary['all_annotated']['cells'], 24*8)
        self.assertTrue(all(r['renderer_error'] is None for r in transformed))

    @unittest.skipUnless((ROOT / 'outputs/neural_ffd_1k_long_20261010_225918/best_joint.pth').is_file(),
                         'Pinned pretrained checkpoint unavailable')
    def test_real_initialization_motion_gradient_resume_and_state_boundaries(self):
        pipeline = TTOPipeline(self.config)
        patient = RealTTODataset(self.root, ['patient'], config=self.config['data'])[0]
        state = pipeline.initialize(patient)
        self.assertEqual(state.stage.value, 'initialized')
        self.assertIsNone(state.observation_pose)
        self.assertTrue(all(v == 0 for v in state.diagnostics['forward_consistency'].values()))
        with self.assertRaises(ValueError):
            pipeline.prior.encode({'pose_gt': {}})
        zs = state.z_shape.clone().requires_grad_()
        zm = state.z_motion.clone().requires_grad_()
        decoded = pipeline.prior.decode_original(zs, zm, state.conditions)
        decoded['vertices_4d'][:, 23].square().mean().backward()
        for latent in (zs, zm):
            self.assertTrue(torch.isfinite(latent.grad).all())
            self.assertGreater(float(latent.grad.norm()), 0.)
        pipeline.prior.assert_frozen()
        path = self.root / 'result/patients/patient/initialized.pth'
        restored = pipeline.initialize(patient, resume=True)
        torch.testing.assert_close(restored.trajectory, state.trajectory, rtol=0, atol=0)
        with self.assertRaises(FileExistsError):
            pipeline.initialize(patient)
        evaluator = NativeEvaluator(pipeline.prior.model, self.config['supervision'])
        with self.assertRaisesRegex(ValueError, 'registered'):
            evaluator.evaluate_state(restored, patient)
        payload = load_state(path)
        payload['chamber_offsets'] = torch.zeros(4, 3)
        with self.assertRaises((ValueError, TypeError)):
            TTOState.from_payload(payload)
        payload.pop('chamber_offsets')
        payload['identity']['patient_id'] = 'foreign'
        with self.assertRaises(ValueError):
            TTOState.from_payload(payload)
        with self.assertRaises(ValueError):
            TTOState.from_payload(dict(version='real-tto-joint-v1'))
        changed = deepcopy(self.config)
        changed['loss']['sigma_mm'] = 2.
        with self.assertRaises(ValueError):
            TTOPipeline(changed).initialize(patient, resume=True)


if __name__ == '__main__':
    unittest.main()
