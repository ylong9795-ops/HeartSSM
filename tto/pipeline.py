"""One resumable patient-isolated route from native observations to exported4D."""
from pathlib import Path
import traceback
import time

import numpy as np
import torch

from .config import resolve
from .dataset import RealTTODataset, collate_patients
from .evaluator import ExactNativeEvaluator, native_exact
from .geometry_guard import GeometryGuard
from .io import ROOT, cpu, device_tree, digest_json, load_state, read_json, save_state, sha256, write_json
from .joint_optimizer import GuardedJointOptimizer
from .motion_lock import FrozenMotionPropagator
from .pose_optimizer import MultiViewPoseOptimizer
from .shape_optimizer import LatentShapeOptimizer
from .synthetic_prior import FrozenSyntheticPrior

STAGES = ('initial', 'pose', 'shape', 'motion_lock', 'joint', 'evaluate', 'export')


class RealTTOPipeline:
    def __init__(self, config):
        self.config = config
        self.output = resolve(config['output']['root'])
        self.output.mkdir(parents=True, exist_ok=True)
        runtime = config['runtime']
        torch.set_num_threads(runtime['threads'])
        torch.manual_seed(runtime['seed'])
        np.random.seed(runtime['seed'])
        self.device = torch.device(runtime['device'])
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        self.prior = None
        self.source_identity = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'tto').glob('*.py'))}
        # Freeze identities of the reused implementations as well as this adapter.
        reused = ('train_neural_ffd.py', 'net/neural_ffdnet.py', 'net/multiview_encoder.py',
            'data/mask_transform.py', 'utils/ffd_physical.py', 'utils/slice_supervision.py',
            'utils/slice_evaluation.py', 'utils/neural_ffd_losses.py', 'utils/geometry_pair_policy.py',
            'utils/geometry_v2_checks.py')
        self.source_identity.update({p: sha256(ROOT/p) for p in reused})

    def load_frozen_prior(self):
        if self.prior is None:
            cfg = self.config['prior']
            self.prior = FrozenSyntheticPrior(resolve(cfg['checkpoint']), device=self.device, expected_sha256=cfg['sha256'])
            self.model = self.prior.model
            self.calibration = read_json(resolve(cfg['calibration']))
            if self.calibration['checkpoint_sha256'] != self.prior.checkpoint_sha256:
                raise ValueError('Synthetic calibration belongs to another checkpoint')
            self.guard = GeometryGuard(self.model)
            self.evaluator = ExactNativeEvaluator(self.model)
            self.pose_optimizer = MultiViewPoseOptimizer(self.model, self.config['pose'], self.evaluator,
                self.config['runtime']['seed'], self.config['loss'])
            self.shape_optimizer = LatentShapeOptimizer(self.model, self.config['shape'], self.config['motion_limits'],
                self.guard, self.evaluator, self.calibration['shape_std'], self.config['loss'])
            self.joint_optimizer = GuardedJointOptimizer(self.model, self.config['joint'], self.shape_optimizer,
                self.guard, self.evaluator, self.config['motion_limits'])
        return self.prior

    def load_patient(self, patient_ids):
        dataset = RealTTODataset(resolve(self.config['data']['root']), patient_ids, config=self.config['data'])
        return [dataset[i] for i in range(len(dataset))]

    def identity(self, patient):
        scientific = {k: v for k, v in self.config.items() if k not in ('runtime', 'output')}
        return dict(patient=patient.patient_id, input_sha256=patient.identity, config_sha256=digest_json(scientific),
            seed=self.config['runtime']['seed'], checkpoint_sha256=self.prior.checkpoint_sha256,
            calibration_sha256=sha256(resolve(self.config['prior']['calibration'])),
            source_sha256=digest_json(self.source_identity))

    def patient_dir(self, patient):
        return self.output/'patients'/patient.patient_id

    def stage_path(self, patient, stage):
        return self.patient_dir(patient)/'states'/(stage+'.pth')

    def checkpoint_callback(self, stage):
        def callback(patient, progress):
            save_state(self.patient_dir(patient)/'checkpoints'/(stage+'.pth'),
                dict(identity=self.identity(patient), stage=stage, progress=progress))
        return callback

    def optimization_resumes(self, patients, stage, resume):
        values = []
        for patient in patients:
            path = self.patient_dir(patient)/'checkpoints'/(stage+'.pth')
            values.append(load_state(path, self.identity(patient), self.device)['progress'] if resume and path.exists() else None)
        return values

    def initialize_mesh(self, patients):
        batch = collate_patients(patients)['observations']
        latents = self.prior.encode_real_observations(batch)
        with torch.no_grad():
            output = self.prior.decode_initial_motion(latents['z_shape'], latents['z_motion'])
            fields = output['control_grids']['fine_motion'].detach()
            replay = FrozenMotionPropagator(self.model, fields).propagate(output['vertices_ref'])
        if not torch.equal(replay, output['vertices_4d']):
            raise AssertionError('Original decoder and Motion field recurrence differ')
        states = []
        for i, patient in enumerate(patients):
            contract = patient.metadata['contract']
            pose = contract.get('initial_pose', self.calibration['initial_pose'])
            pose = {k: torch.tensor(pose[k], dtype=torch.float32, device=self.device) for k in ('R', 'center_mm')}
            if pose['R'].shape != (3, 3, 3) or pose['center_mm'].shape != (3, 3):
                raise ValueError('Initial Pose requires R[3,3,3], center_mm[3,3]')
            if not torch.isfinite(pose['R']).all() or not torch.isfinite(pose['center_mm']).all():
                raise ValueError('Nonfinite initial Pose')
            eye = torch.eye(3, device=self.device)
            if not torch.allclose(pose['R'].transpose(-1,-2)@pose['R'], eye.expand(3,3,3), atol=1e-4, rtol=0) or (torch.linalg.det(pose['R'])<=0).any():
                raise ValueError('Initial Pose must be SO(3)')
            geometry = self.guard.audit(replay[i].cpu().numpy())
            states.append(dict(patient=patient.patient_id, z_shape_initial=latents['z_shape'][i],
                z_shape=latents['z_shape'][i], z_motion=latents['z_motion'][i], fields=fields[i],
                prior4d=replay[i], trajectory=replay[i], pose=pose, initial_pose=pose,
                geometry=geometry, original_geometry=geometry, status='INITIAL',
                pose_origin='explicit input estimate' if 'initial_pose' in contract else 'fixed synthetic TRAIN Pose prior; unverified real Pose',
                motion_replay_bitwise=True))
            write_json(self.patient_dir(patient)/'input_manifest.json', patient.metadata)
        return states

    def optimize_pose(self, patients, states, resume=False):
        results = self.pose_optimizer.optimize(patients, torch.stack([s['prior4d'][0] for s in states]),
            [s['pose'] for s in states], resumes=self.optimization_resumes(patients, 'pose', resume),
            checkpoint=self.checkpoint_callback('pose'))
        for state, result in zip(states, results):
            state['pose'] = device_tree(result['pose'], self.device)
            state['pose_result'] = result
        return states

    def optimize_shape(self, patients, states, resume=False):
        results = self.shape_optimizer.optimize(patients, states,
            resumes=self.optimization_resumes(patients, 'shape', resume), checkpoint=self.checkpoint_callback('shape'))
        for state, result in zip(states, results):
            state.update(z_shape=result['z_shape'], trajectory=result['trajectory'], geometry=result['geometry'],
                         shape_status=result['status'], shape_result=result)
        return states

    def apply_motion_lock(self, patients, states):
        with torch.no_grad():
            z = torch.stack([s['z_shape'] for s in states])
            motion = FrozenMotionPropagator(self.model, torch.stack([s['fields'] for s in states]),
                torch.stack([s['prior4d'] for s in states]), self.config['motion_limits'])
            trajectories = motion.propagate(self.model.decode_shape(z))
            reports = motion.check(trajectories)
        for i, state in enumerate(states):
            if not reports[i]['passed']:
                state['motion_rejection'] = reports[i]
                state['z_shape'] = state['z_shape_initial'].clone()
                state['trajectory'] = state['prior4d'].clone()
                state['shape_status'] = 'ROLLED_BACK'
                state['geometry'] = state['original_geometry']
                reports[i] = FrozenMotionPropagator(self.model, state['fields'][None], state['prior4d'][None],
                    self.config['motion_limits']).check(state['trajectory'][None])[0]
            else:
                state['trajectory'] = trajectories[i]
            state['motion_report'] = reports[i]
        return states

    def guarded_joint(self, patients, states, resume=False):
        results = self.joint_optimizer.optimize(patients, states,
            resumes=self.optimization_resumes(patients, 'joint', resume), checkpoint=self.checkpoint_callback('joint'))
        for state, result in zip(states, results):
            state.update(z_shape=result['z_shape'], pose=device_tree(result['pose'], self.device),
                trajectory=result['trajectory'], geometry=result['geometry'], joint_result=result)
            state['motion_report'] = FrozenMotionPropagator(self.model, state['fields'][None], state['prior4d'][None],
                self.config['motion_limits']).check(state['trajectory'][None])[0]
            state['status'] = 'ACCEPTED' if state['geometry']['passed'] and state['motion_report']['passed'] else 'REJECTED'
        return states

    def evaluate(self, patients, states):
        for patient, state in zip(patients, states):
            pose = {k: v.cpu().numpy() for k, v in state['pose'].items()}
            before = self.evaluator.cells(state['prior4d'].cpu().numpy(),
                {k:v.cpu().numpy() for k,v in state['pose_result']['pose'].items()}, patient, 'before_shape')
            fixed_pose = {k:v.cpu().numpy() for k,v in state['pose_result']['pose'].items()}
            shape_report, _ = self.evaluator.evaluate(state['shape_result']['trajectory'].cpu().numpy(),
                fixed_pose, patient, stage='shape_after', before=before,
                output=self.patient_dir(patient)/'evaluation'/'shape_only')
            report, _ = self.evaluator.evaluate(state['trajectory'].cpu().numpy(), pose, patient, stage='final',
                before=before, motion=state['motion_report'], geometry=state['geometry'], output=self.patient_dir(patient)/'evaluation')
            report['shape_only_at_fixed_pose'] = shape_report
            write_json(self.patient_dir(patient)/'evaluation'/'metrics.json', report)
            state['evaluation'] = report
        return states

    def export(self, patients, states):
        import trimesh
        from data.mask_transform import CHAMBERS, VIEWS
        for patient, state in zip(patients, states):
            folder = self.patient_dir(patient)/'export';folder.mkdir(parents=True, exist_ok=True)
            arrays = {k: state[k].detach().cpu().numpy() for k in ('z_shape', 'z_motion', 'fields', 'trajectory', 'prior4d')}
            arrays.update({f'faces_{c}':f.cpu().numpy() for c, f in self.model.faces.items()})
            arrays.update(chamber_ranges=self.model.chamber_ranges.cpu().numpy(),
                          probe_R=state['pose']['R'].cpu().numpy(), center_mm=state['pose']['center_mm'].cpu().numpy())
            np.savez_compressed(folder/'reconstruction.npz', **arrays)
            write_json(folder/'geometry.json', state['geometry']);write_json(folder/'motion.json', state['motion_report'])
            write_json(folder/'pose.json', state['pose'])
            if self.config['output']['export_ply']:
                faces = np.concatenate([self.model.faces[c].cpu().numpy()+a for c, (a, b) in zip(CHAMBERS, self.model.chamber_ranges.tolist())])
                mesh_dir = folder/'meshes';mesh_dir.mkdir(exist_ok=True)
                for phase, points in enumerate(arrays['trajectory']):
                    trimesh.Trimesh(points, faces, process=False).export(mesh_dir/f'phase_{phase:02d}.ply')
            if self.config['output']['visualizations']:
                self._visualize(patient, state, folder)
            write_json(folder/'completed.json', dict(patient=patient.patient_id, status=state['status'],
                identity=self.identity(patient), mesh_phases=24, legal4d=state['geometry']['passed'],
                motion_guard_pass=state['motion_report']['passed'], source_identity=self.source_identity))
        return states

    def _visualize(self, patient, state, folder):
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from data.mask_transform import CHAMBERS, VIEWS
        native = patient.native(0);pose = state['pose'];v = state['trajectory'][0].cpu().numpy()
        for vi, view in enumerate(VIEWS):
            h, w = native['masks'][vi].shape[-2:]
            raw = patient.images[vi][0] if patient.images else np.zeros((h, w, 3), np.uint8)
            if raw.ndim == 2:raw = np.repeat(raw[..., None], 3, axis=-1)
            canvas = raw[..., :3].astype(np.uint8).copy()
            for ci, c in enumerate(CHAMBERS):
                if not native['annotated'][vi, ci]:continue
                a, b = self.model.chamber_ranges[ci].tolist()
                prediction = native_exact(v[a:b], self.model.faces[c].cpu().numpy(), pose['R'][vi].cpu().numpy(),
                    pose['center_mm'][vi].cpu().numpy(), native['spacing'][vi], (h, w))
                for mask, color in ((native['masks'][vi][ci], (0,255,0)), (prediction, (255,0,255))):
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, color, 1)
            cv2.imwrite(str(folder/f'{view}_native_overlay.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        for ax, c in zip(axes.flat, CHAMBERS):
            m = state['motion_report']['metrics']['chambers'][c]
            ax.plot(np.arange(24)/24, m['prior_volume_curve_ml'], label='Frozen prior')
            ax.plot(np.arange(24)/24, m['volume_curve_ml'], label='TTO')
            ax.set(title=c, xlabel='Cycle phase', ylabel='Volume (ml)');ax.legend()
        fig.tight_layout();fig.savefig(folder/'motion_volumes.png');plt.close(fig)

    def run(self, patient_ids, *, stage='all', resume=False):
        if stage not in ('all', 'preflight', *STAGES):raise ValueError(stage)
        if len(set(patient_ids)) != len(patient_ids):raise ValueError('Duplicate patient IDs')
        if any(Path(p).name != p or p in ('.', '..') for p in patient_ids):
            raise ValueError('Patient IDs must be single directory names')
        write_json(self.output/'run_config.json', self.config)
        requested = STAGES if stage == 'all' else STAGES[:STAGES.index(stage)+1] if stage != 'preflight' else ()
        completed, failures = [], []
        size = self.config['runtime']['batch_size']
        def record_failure(ids, failed_stage, error):
            record = dict(patients=ids, stage=failed_stage, error=repr(error), traceback=traceback.format_exc(),
                          retained_in_cohort=True, time_ns=time.time_ns())
            failures.append(record)
            for patient_id in ids:
                directory = self.output/'patients'/patient_id
                write_json(directory/'failure.json', record)
                write_json(directory/'failures'/f'{record["time_ns"]}.json', record)
            print('TTO_FAILURE', ids, repr(error), flush=True)
        for start in range(0, len(patient_ids), size):
            ids = patient_ids[start:start+size]
            patients = []
            for patient_id in ids:
                try:patients.extend(self.load_patient([patient_id]))
                except Exception as error:record_failure([patient_id], 'load_patient', error)
            if not patients:continue
            ids = [p.patient_id for p in patients]
            try:
                self.load_frozen_prior()
                if stage == 'preflight':
                    for patient in patients:
                        write_json(self.patient_dir(patient)/'preflight.json', dict(identity=self.identity(patient),
                            native_shapes=[list(v['masks'].shape) for v in patient.views], image_status=patient.metadata['image_status'],
                            model_shape=list(patient.encoder['masks'].shape), geometry_backend_error=self.guard.backend_error,
                            network_trainable_parameters=sum(p.requires_grad for p in self.model.parameters()),
                            geometry_backend_status='PASS' if self.guard.backend_error is None else 'UNVERIFIED'))
                    if self.guard.backend_error is not None:
                        raise RuntimeError('Geometry preflight UNVERIFIED: '+self.guard.backend_error)
                    completed.extend(ids);continue
                states = [None]*len(patients)
                for current in requested:
                    pending = []
                    for i, patient in enumerate(patients):
                        path = self.stage_path(patient, current)
                        if path.exists():
                            if not resume:raise FileExistsError(f'Use --resume or a fresh output directory: {path}')
                            states[i] = load_state(path, self.identity(patient), self.device)['state']
                        else:pending.append(i)
                    if not pending:continue
                    subset = [patients[i] for i in pending]
                    inputs = [states[i] for i in pending]
                    print('TTO_STAGE', current, [p.patient_id for p in subset], flush=True)
                    if current == 'initial':results = self.initialize_mesh(subset)
                    elif current == 'pose':results = self.optimize_pose(subset, inputs, resume)
                    elif current == 'shape':results = self.optimize_shape(subset, inputs, resume)
                    elif current == 'motion_lock':results = self.apply_motion_lock(subset, inputs)
                    elif current == 'joint':results = self.guarded_joint(subset, inputs, resume)
                    elif current == 'evaluate':results = self.evaluate(subset, inputs)
                    else:results = self.export(subset, inputs)
                    self.prior.assert_frozen()
                    for i, result in zip(pending, results):
                        states[i] = result
                        save_state(self.stage_path(patients[i], current), dict(identity=self.identity(patients[i]), stage=current, state=result))
                        write_json(self.patient_dir(patients[i])/(current+'_status.json'), dict(stage=current, completed=True,
                            status=result.get('status'), shape_status=result.get('shape_status'), geometry_pass=result.get('geometry', {}).get('passed')))
                completed.extend(ids)
                for patient in patients:
                    previous_failure = self.patient_dir(patient)/'failure.json'
                    if previous_failure.exists():
                        archive = self.patient_dir(patient)/'failures'/f'recovered_{time.time_ns()}.json'
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        previous_failure.replace(archive)
            except Exception as error:
                record_failure(ids, stage, error)
        if self.prior is not None:self.prior.assert_frozen()
        report = dict(requested=list(patient_ids), completed=completed, failures=failures, stage=stage,
            checkpoint_unchanged=sha256(resolve(self.config['prior']['checkpoint'])) == self.config['prior']['sha256'],
            network_unchanged=True if self.prior is not None else None, source_identity=self.source_identity)
        write_json(self.output/('preflight.json' if stage == 'preflight' else 'run_status.json'), report)
        return report
