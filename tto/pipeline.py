"""The only TTO stage controller. Only preflight and initialization run today."""
from pathlib import Path
import torch

from .config import resolve, validate_config
from .dataset import RealTTODataset, collate_patients
from .experiments import CAPABILITIES
from .io import cpu, digest_json, load_state, save_state, sha256, write_json
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
                    chamber_mode=self.config['alignment']['chamber_mode'])

    def initialize(self, patient, *, resume=False):
        self.preflight(patient)
        identity = self.identity(patient)
        path = self.output / 'patients' / patient.patient_id / 'initialized.pth'
        if path.exists():
            if not resume:
                raise FileExistsError(f'Existing state; use --resume or a different output: {path}')
            return TTOState.from_payload(load_state(path, identity=identity))
        if resume:
            raise FileNotFoundError(f'No initialization state to resume: {path}')
        prior = self.load_prior()
        initial = prior.initialize(collate_patients([patient])['observations'])
        state = TTOState(patient_id=patient.patient_id, identity=identity,
            z_shape=initial['z_shape_initial'], z_motion=initial['z_motion_initial'],
            conditions=initial['conditions'], vertices_reference=initial['vertices_reference_initial'][0],
            trajectory=initial['vertices_4d_initial'][0],
            global_transform=dict(R=torch.eye(3, device=prior.device), center_mm=torch.zeros(3, device=prior.device)),
            diagnostics=dict(forward_consistency=initial['forward_consistency'],
                             enabled_conditions=initial['enabled_conditions'],
                             alignment_status='UNREGISTERED', geometry_status='NOT_EVALUATED',
                             optimization_performed=False))
        prior.assert_frozen()
        save_state(path, state.to_payload())
        return state

    def run(self, patient_ids=None, *, stage='preflight', resume=False):
        if stage not in CAPABILITIES or stage == 'validate':
            raise ValueError(f'Unknown pipeline stage: {stage}')
        if stage not in ('preflight', 'initialize'):
            raise NotImplementedError(CAPABILITIES[stage])
        if resume and stage != 'initialize':
            raise ValueError('--resume applies only to initialized state')
        data_root = resolve(self.config['data']['root'])
        patients = list(patient_ids) if patient_ids else RealTTODataset(
            data_root, config=self.config['data']).patients
        if not patients or len(set(patients)) != len(patients):
            raise ValueError('At least one unique patient ID required')
        summary = dict(stage=stage, optimization_performed=False, completed=[], failures=[])
        for patient_id in patients:
            try:
                patient = RealTTODataset(data_root, [patient_id], config=self.config['data'])[0]
                if stage == 'preflight':
                    report = self.preflight(patient)
                    write_json(self.output / 'patients' / patient_id / 'preflight.json', report)
                else:
                    state = self.initialize(patient, resume=resume)
                    write_json(self.output / 'patients' / patient_id / 'initialization.json',
                        dict(patient_id=patient_id, stage=state.stage.value, identity=state.identity,
                             diagnostics=cpu(state.diagnostics)))
                summary['completed'].append(patient_id)
            except Exception as error:
                summary['failures'].append(dict(patient_id=patient_id, error=repr(error)))
        write_json(self.output / (stage + '_summary.json'), summary)
        return summary
