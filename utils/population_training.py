"""Online Base+Refiner training from random weights, using the original loop."""
import copy
import random

import numpy as np
import torch

import train_neural_ffd as base_training
from net.joint_ffd_refiner import JointFFDRefiner
from utils.joint_refiner_training import (DEFAULT_REGULARIZATION, load_refiner_checkpoint,
    save_refiner_checkpoint, tail_summary)

KIND = 'population-base-refiner-fromscratch-v1'


class PopulationFFDModel(JointFFDRefiner):
    """Explicit synthetic-training adapter; all physical features stay online.

    Shape/Motion stages train the randomly initialized Base. Joint trains Base
    and Refiner together. No predictions/features are cached across updates.
    Generic/real inference continues to use JointFFDRefiner with explicit Pose.
    """
    def __init__(self, base_model, refiner=None, **options):
        super().__init__(base_model, refiner, **options)
        self.active_stage = 'shape'
        self.evaluate_base_only = False

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Preserve the read interface used by the original loss, encoder
            # diagnostics, fixed-geometry evaluator and checkpoint metadata.
            if name == 'base_model' or name.startswith('_'):
                raise
            return getattr(super().__getattr__('base_model'), name)

    def forward(self, observations):
        if self.active_stage != 'joint' or self.evaluate_base_only:
            return self.base_model(observations)
        return super().forward(observations, pose=observations['pose_gt'])


class PopulationWorkflow:
    def build_model(self, dataset, config):
        model = PopulationFFDModel(base_training.build_model(dataset, config))
        model.active_stage = base_training.stage_for_epoch(0, config)
        model.initialization_metadata = dict(mode='random_base_and_zero_output_refiner',
            inherited_checkpoint=None, seed=config['training']['seed'])
        return model

    def set_stage(self, model, stage):
        if stage not in ('shape', 'motion', 'joint'):
            raise ValueError('Unknown population stage')
        model.active_stage = stage
        model.base_model.requires_grad_(True)
        model.refiner.requires_grad_(stage == 'joint')

    def configure_trainable(self, model, training):
        if training.get('freeze', 'none') != 'none':
            raise ValueError('From-scratch population training cannot freeze inherited Base weights')
        self.set_stage(model, model.active_stage)

    def make_optimizer(self, model, training):
        # Include dormant Refiner parameters now so its first Joint update is
        # optimized, and Adam group identity stays fixed across stage/resume.
        return torch.optim.Adam(model.parameters(), lr=training['learning_rate'])

    def compute_objective(self, model, batch, criterion, slicer, stage):
        output, losses, slices = base_training.compute_objective(model, batch, criterion, slicer, stage)
        if stage == 'joint':
            weights = DEFAULT_REGULARIZATION
            terms = output['control_regularization']
            penalty = sum((terms[k]*weights[k] for k in weights), losses['total']*0)
            losses['total'] = losses['total']+penalty
            losses['objective_weights'] = dict(losses['objective_weights'], L_R=1.)
            losses['terms'] = dict(losses['terms'], residual_penalty=penalty,
                **{'residual_'+k: v for k, v in terms.items()})
            losses['active_weights'] = dict(losses['active_weights'], residual_penalty=0.,
                **{'residual_'+k: v for k, v in weights.items()})
            losses['units'] = dict(losses['units'], residual_penalty='weighted residual regularity',
                residual_spatial_gradient_l2='dimensionless',
                residual_cyclic_velocity_l2='mm^2/cycle^2',
                residual_cyclic_acceleration_l2='mm^2/cycle^4')
        return output, losses, slices

    def save_checkpoint(self, path, model, optimizer, *, config, identity, epoch=0,
                        sample_offset=0, step=0, runtime=None):
        base = model.base_model
        numpy_rng = np.random.get_state()
        embedded = dict(version=base_training.CHECKPOINT_VERSION, architecture=base_training.ARCHITECTURE,
            specification=dict(encoder=base.encoder_config, decoder=base.decoder_config,
                template_sha256=base.template_sha256, asset_sha256=base.asset_sha256,
                norm_to_mm=base.norm_to_mm, geometry_sha256=base.geometry_sha256),
            model=base.state_dict(), config=config, data_identity=identity,
            epoch=epoch, sample_offset=sample_offset, step=step, runtime=runtime,
            initialization=model.initialization_metadata,
            rng=dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=(numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                cuda=torch.cuda.get_rng_state(base.template_vertices_mm.device)
                    if base.template_vertices_mm.is_cuda else None))
        experiment = dict(kind=KIND, active_stage=model.active_stage,
            regularization=dict(DEFAULT_REGULARIZATION), frozen_base=False,
            observations='online_current_base_every_update', pose_source='synthetic_GT',
            initialization=copy.deepcopy(model.initialization_metadata))
        save_refiner_checkpoint(path, model, optimizer, source_checkpoint=embedded,
            experiment=experiment, step=step)

    def load_checkpoint(self, path, device='cpu'):
        joint, raw = load_refiner_checkpoint(path, device)
        experiment = raw['experiment']
        if experiment.get('kind') != KIND or experiment.get('regularization') != DEFAULT_REGULARIZATION:
            raise ValueError('Resume requires a compatible from-scratch population checkpoint/objective')
        model = PopulationFFDModel(joint.base_model, joint.refiner, **joint.configuration)
        model.active_stage = experiment['active_stage']
        model.initialization_metadata = experiment['initialization']
        saved = dict(raw['base_checkpoint'], optimizer=raw['optimizer'])
        self.set_stage(model, model.active_stage)
        return model, saved

    @torch.no_grad()
    def evaluate(self, model, dataset, device, **options):
        report = base_training.evaluate(model, dataset, device, **options)
        def tails(value):
            rows = [dict(cell, case_id=case['case_id']) for case in value['cases']
                    for cell in case.get('exact', {}).get('cells', [])]
            return tail_summary(rows) if rows else None
        report['tails'] = tails(report)
        if model.active_stage == 'joint':
            model.evaluate_base_only = True
            try:
                baseline = base_training.evaluate(model, dataset, device, **options)
            finally:
                model.evaluate_base_only = False
            report['base_comparison'] = baseline
            report['base_summary'] = base_training.validation_summary(baseline)
            report['base_tails'] = tails(baseline)
            if 'performance' in report:
                report['performance']['final_prediction_seconds'] = report['performance']['seconds']
                report['performance']['base_comparison_seconds'] = baseline['performance']['seconds']
                report['performance']['seconds'] += baseline['performance']['seconds']
        return report

    def best_checkpoint_candidates(self, summary):
        candidates = base_training.best_checkpoint_candidates(summary)
        metrics = summary['validation']
        if summary['complete'] and summary['stage'] == 'joint' and metrics['minimum_exact_dice'] is not None:
            # Integer failure count dominates; minimum Dice breaks equal-count
            # ties. Keep best_joint and best_dice as independent alternatives.
            candidates['best_tail'] = dict(score=metrics['native_dice_fail_cells']+1-metrics['minimum_exact_dice'],
                criterion='Native Cell<0.98 count, then maximum minimum Dice; VAL only')
        return candidates
