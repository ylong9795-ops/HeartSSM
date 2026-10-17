"""Stage controller: exact initialization and frozen-mesh S1 pose candidates."""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import torch

from .config import resolve, validate_config
from .dataset import RealTTODataset, collate_patients
from .experiments import CAPABILITIES
from .io import cpu, digest_json, jsonable, load_state, read_json, save_state, sha256, write_json
from .prior import FrozenNeuralFFDPrior
from .state import TTOState
from .supervision.observations import observation_cells


class TTOPipeline:
    def __init__(self, config):
        self.config = validate_config(config)
        self.output = resolve(config['output']['root'])
        self.prior = None

    def load_prior(self):
        if self.prior is None:
            runtime = self.config['runtime']
            torch.set_num_threads(runtime['threads'])
            torch.manual_seed(runtime['seed'])
            self.prior = FrozenNeuralFFDPrior(resolve(self.config['prior']['checkpoint']),
                expected_sha256=self.config['prior']['sha256'], device=runtime['device'])
        return self.prior

    def identity(self, patient):
        source = {str(p.relative_to(Path(__file__).parent)): sha256(p)
                  for p in sorted(Path(__file__).parent.rglob('*.py'))}
        identity = dict(patient_id=patient.patient_id, observations_sha256=patient.identity,
                    checkpoint_sha256=self.config['prior']['sha256'],
                    config_sha256=digest_json(self.config), source_sha256=digest_json(source))
        contract = patient.metadata['contract']
        for key in ('case_id', 'sample_sha256'):
            if key in contract:
                identity[key] = contract[key]
        return identity

    def preflight(self, patient):
        batch = collate_patients([patient])['observations']
        rows = observation_cells([patient], self.config['supervision'])
        used = [row for row in rows if row['used']]
        if not used:
            raise ValueError('No usable native observations inside selected cycle')
        return dict(patient_id=patient.patient_id, status='PREFLIGHT_COMPLETE',
                    model_loaded=False, optimization_performed=False,
                    native_shapes=[list(v['masks'].shape) for v in patient.views],
                    encoder_shape=list(batch['masks'].shape),
                    annotated_cells=len(rows), usable_cells=len(used),
                    observation_manifest=rows, metadata=patient.metadata,
                    input_mode=self.config['data']['input_mode'],
                    s0_input_audit=self.input_audit(patient),
                    chamber_mode=self.config['alignment']['chamber_mode'])

    def input_audit(self, patient):
        aligned = patient.metadata.get('input_mode') == 'aligned24'
        alignment = patient.metadata.get('phase_alignment', {})
        verified = aligned and all(alignment.get(k) == 'VERIFIED' for k in ('status', 'phase_grid_status'))
        return dict(engineering_contract='PASS' if aligned else 'LEGACY_RAW_RR',
            phase0_status=alignment.get('status', 'UNVERIFIED'),
            phase_grid_status=alignment.get('phase_grid_status', 'UNVERIFIED'),
            phase_alignment_evidence=alignment,
            ready_for_shape_tto=verified,
            temporal_resampling_performed=patient.metadata.get('temporal_resampling_performed', True),
            phase_order_changed=patient.metadata.get('phase_order_changed', 'raw RR construction'))

    def initialize(self, patient, *, resume=False):
        preflight = self.preflight(patient)
        identity = self.identity(patient)
        path = self.output / 'patients' / patient.patient_id / 'initialized.pth'
        if path.exists():
            if not resume:
                raise FileExistsError(f'Existing state; use --resume or a different output: {path}')
            return TTOState.from_payload(load_state(path, identity=identity))
        if resume:
            raise FileNotFoundError(f'No initialization state to resume: {path}')
        prior = self.load_prior()
        preprocessing = (prior.audit_preprocessing(self.config['data'])
                         if self.config['data']['input_mode'] == 'aligned24' else None)
        initial = prior.initialize(collate_patients([patient])['observations'])
        state = TTOState(patient_id=patient.patient_id, identity=identity,
            z_shape=initial['z_shape_initial'], z_motion=initial['z_motion_initial'],
            conditions=initial['conditions'], vertices_reference=initial['vertices_reference_initial'][0],
            trajectory=initial['vertices_4d_initial'][0],
            global_transform=dict(R=torch.eye(3, device=prior.device), center_mm=torch.zeros(3, device=prior.device)),
            diagnostics=dict(forward_consistency=initial['forward_consistency'],
                             enabled_conditions=initial['enabled_conditions'],
                             s0_input_audit=self.input_audit(patient),
                             s0_status=('PASS' if self.input_audit(patient)['ready_for_shape_tto'] else
                                        'ENGINEERING_PASS_PHASE_ALIGNMENT_UNVERIFIED'),
                             checkpoint_sha256=prior.checkpoint_sha256,
                             pretrained_preprocessing=preprocessing,
                             initial_shapes=dict(shape=list(initial['z_shape_initial'].shape),
                                 motion=list(initial['z_motion_initial'].shape),
                                 trajectory=list(initial['vertices_4d_initial'].shape)),
                             alignment_status='UNREGISTERED', geometry_status='NOT_EVALUATED',
                             optimization_performed=False))
        prior.assert_frozen()
        save_state(path, state.to_payload())
        write_json(path.parent / 'preflight.json', preflight)
        return state

    def align(self, patient, *, initialization_root=None, resume=False):
        from .alignment.global_alignment import align_patient
        path = self.output / 'patients' / patient.patient_id / 'aligned.pth'
        identity = self.identity(patient)
        if path.exists():
            if not resume: raise FileExistsError(f'Existing S1 state: {path}')
            return TTOState.from_payload(load_state(path, identity=identity))
        if resume: raise FileNotFoundError(path)
        root = resolve(initialization_root) if initialization_root else self.output
        source = root / 'patients' / patient.patient_id / 'initialized.pth'
        if not source.exists() and initialization_root is None:
            self.initialize(patient)
        initial = TTOState.from_payload(load_state(source))
        # Explicit S0->S1 lineage: new source/config identity is expected, but
        # patient, input, original checkpoint and original decoded mesh must agree.
        for key in ('patient_id', 'observations_sha256', 'checkpoint_sha256'):
            if initial.identity.get(key) != identity.get(key):
                raise ValueError(f'S0/S1 {key} mismatch')
        prior = self.load_prior()
        with torch.no_grad():
            decoded = prior.decode_original(initial.z_shape.to(prior.device), initial.z_motion.to(prior.device),
                                            initial.conditions)
        if not torch.equal(decoded['vertices_4d'][0].cpu(), initial.trajectory.cpu()):
            raise ValueError('S0 mesh differs from the pinned original Decoder')
        options = self.config['alignment']['pose']
        seed_path = resolve(options['initial_pose_file'])
        if sha256(seed_path) != options['initial_pose_sha256']:
            raise ValueError('Initial pose calibration SHA256 mismatch')
        calibration = read_json(seed_path)
        if calibration['checkpoint_sha256'] != prior.checkpoint_sha256:
            raise ValueError('Initial pose and pretrained checkpoint identity differ')
        TTOState._validate_pose({k: torch.as_tensor(v, dtype=torch.float64)
                                for k, v in calibration['initial_pose'].items()}, (3,))
        state, results = align_patient(prior.model, patient, initial, config=options,
            policy=self.config['supervision'], initial_pose=calibration['initial_pose'])
        state.identity = identity
        state.diagnostics['s1']['initialization_source'] = dict(path=str(source), sha256=sha256(source),
            identity=initial.identity, original_decoder_replay_max_abs_mm=0.)
        for key in ('z_shape', 'z_motion', 'trajectory', 'vertices_reference'):
            if not torch.equal(getattr(initial, key), getattr(state, key)):
                raise AssertionError(f'S1 changed frozen {key}')
        prior.assert_frozen()
        save_state(path, state.to_payload())
        write_json(path.parent / 'alignment.json', dict(patient_id=patient.patient_id,
            identity=state.identity, diagnostics=state.diagnostics))
        for vi, result in enumerate(results):
            view_path = path.parent / result['view']
            write_json(view_path / 'search.json', {k: v for k, v in result.items()
                if k not in ('baseline_cells', 'centre_cells', 'selected_cells', 'candidates')})
            write_json(view_path / 'candidates.json', [{k: v for k, v in c.items() if k != 'cells'}
                                                     for c in result['candidates']])
            stages = [('baseline', result['baseline_cells']), ('centre_only', result['centre_cells'])]
            stages += [(f'candidate_{i}', c['cells']) for i, c in enumerate(result['candidates'])]
            for label, rows in stages:
                records = [jsonable(r) for r in rows]
                if records:
                    with (view_path / (label+'_cells.csv')).open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                        writer.writeheader(); writer.writerows(records)
        return state

    def regional(self, patient, *, alignment_root=None, resume=False):
        from dataclasses import replace
        from copy import deepcopy
        from .optimization.regional_shape import fit_regional
        identity=self.identity(patient)
        folder=self.output/'patients'/patient.patient_id
        target=folder/'s2_state.pth'
        if target.exists():
            if not resume:raise FileExistsError(f'Existing S2 state: {target}')
            return TTOState.from_payload(load_state(target,identity=identity))
        if resume:raise FileNotFoundError(target)
        root=resolve(alignment_root) if alignment_root else self.output
        source=root/'patients'/patient.patient_id/'aligned.pth'
        initial=TTOState.from_payload(load_state(source))
        for key in ('patient_id','observations_sha256','checkpoint_sha256'):
            if initial.identity.get(key)!=identity.get(key):raise ValueError(f'S1/S2 {key} mismatch')
        prior=self.load_prior()
        with torch.no_grad():
            decoded=prior.decode_original(initial.z_shape,initial.z_motion,initial.conditions)
        if not torch.equal(decoded['vertices_4d'][0],initial.trajectory):
            raise ValueError('S1 source trajectory differs from original Decoder')
        pose_config=self.config['alignment']['pose']
        calibration=resolve(pose_config['initial_pose_file'])
        if sha256(calibration)!=pose_config['initial_pose_sha256']:raise ValueError('TRAIN latent calibration identity mismatch')
        stats=read_json(calibration)
        if stats['checkpoint_sha256']!=prior.checkpoint_sha256:raise ValueError('TRAIN latent scales checkpoint mismatch')
        bound=replace(initial,identity=identity,diagnostics=deepcopy(initial.diagnostics))
        bound.diagnostics['s2_source']=dict(path=str(source),sha256=sha256(source),identity=initial.identity,
            decoder_replay_max_abs_mm=0.,global32_frozen=True,motion32_frozen=True,network_frozen=True)
        return fit_regional(prior,patient,bound,options=self.config['alignment']['regional'],
            policy=self.config['supervision'],pose_config=pose_config,regional_std=stats['shape_std'][32:],
            output=folder,checkpoint=resolve(self.config['prior']['checkpoint']))

    def run(self, patient_ids=None, *, stage='preflight', resume=False, initialization_root=None, alignment_root=None, workers=1):
        if stage not in CAPABILITIES or stage == 'validate':
            raise ValueError(f'Unknown pipeline stage: {stage}')
        if stage not in ('preflight', 'initialize', 'global_alignment', 'chamber_candidates'):
            raise NotImplementedError(CAPABILITIES[stage])
        if resume and stage not in ('initialize', 'global_alignment', 'chamber_candidates'):
            raise ValueError('--resume applies only to saved initialization/alignment states')
        if type(workers) is not int or workers < 1 or (workers != 1 and stage not in ('global_alignment','chamber_candidates')):
            raise ValueError('Positive CPU workers only supported for S1 alignment')
        data_root = resolve(self.config['data']['root'])
        patients = list(patient_ids) if patient_ids else RealTTODataset(
            data_root, config=self.config['data']).patients
        if not patients or len(set(patients)) != len(patients):
            raise ValueError('At least one unique patient ID required')
        summary = dict(stage=stage, optimization_performed=False, completed=[], failures=[])
        if stage in ('global_alignment','chamber_candidates') and workers > 1:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs = {pool.submit(_align_worker, self.config, pid, initialization_root, resume, stage, alignment_root): pid for pid in patients}
                for job in as_completed(jobs):
                    pid = jobs[job]
                    try:
                        job.result(); summary['completed'].append(pid)
                    except Exception as error:
                        summary['failures'].append(dict(patient_id=pid, error=repr(error)))
            summary['completed'].sort()
            write_json(self.output / (stage+'_summary.json'), summary)
            return summary
        for patient_id in patients:
            try:
                patient = RealTTODataset(data_root, [patient_id], config=self.config['data'])[0]
                if stage == 'preflight':
                    report = self.preflight(patient)
                    write_json(self.output / 'patients' / patient_id / 'preflight.json', report)
                elif stage == 'global_alignment':
                    self.align(patient, initialization_root=initialization_root, resume=resume)
                elif stage == 'chamber_candidates':
                    self.regional(patient,alignment_root=alignment_root,resume=resume)
                else:
                    state = self.initialize(patient, resume=resume)
                    write_json(self.output / 'patients' / patient_id / 'initialization.json',
                        dict(patient_id=patient_id, stage=state.stage.value, identity=state.identity,
                             artifact='initialized.pth',
                             stored_fields=dict(shape_latent='z_shape [1,64]', motion_latent='z_motion [1,32]',
                                 full_prediction='trajectory [24,10631,3] canonical mm',
                                 reference='vertices_reference [10631,3] canonical mm',
                                 trained_decoder_conditions='conditions'),
                             diagnostics=cpu(state.diagnostics)))
                summary['completed'].append(patient_id)
            except Exception as error:
                summary['failures'].append(dict(patient_id=patient_id, error=repr(error)))
        write_json(self.output / (stage + '_summary.json'), summary)
        return summary


def _align_worker(config, patient_id, initialization_root, resume, stage='global_alignment', alignment_root=None):
    patient = RealTTODataset(resolve(config['data']['root']), [patient_id], config=config['data'])[0]
    pipeline=TTOPipeline(config)
    if stage=='chamber_candidates':pipeline.regional(patient,alignment_root=alignment_root,resume=resume)
    else:pipeline.align(patient, initialization_root=initialization_root, resume=resume)
    return patient_id
