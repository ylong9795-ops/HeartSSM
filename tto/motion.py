"""Explicit original-decoder and frozen-field 4D propagation semantics."""
import torch

from .motion_lock import FrozenMotionPropagator, motion_metrics, motion_guard


class MotionPropagation:
    def __init__(self, prior, initial, limits):
        self.prior, self.model, self.limits = prior, prior.model, dict(limits)
        self.z_motion = initial['z_motion_initial'].detach().clone()
        self.original = initial['vertices_4d_initial'].detach().clone()
        grids = initial['control_grids_initial']
        self.fields = grids['fine_motion'].detach().clone()
        self.regional_fields = grids.get('regional_motion')
        if self.regional_fields is not None:
            self.regional_fields = self.regional_fields.detach().clone()
        self.regional_fields_identity = self.regional_fields.clone() if self.regional_fields is not None else None
        self.conditions = initial['conditions']
        self.locked = FrozenMotionPropagator(self.model, self.fields)
        self.fields_identity = self.fields.clone()
        self.z_motion_identity = self.z_motion.clone()

    def decode(self, z_shape, mode='frozen_field'):
        self.assert_frozen()
        if mode == 'original_decoder':
            return self.prior.decode_original(z_shape, self.z_motion, self.conditions)['vertices_4d']
        if mode != 'frozen_field':
            raise ValueError(f'Unknown motion semantics: {mode}')
        reference = self.prior.decode_shape(z_shape, self.conditions)
        if self.regional_fields is None:
            return self.locked.propagate(reference)
        # Preserve the original order: fine FFD followed by regional motion FFD.
        trajectory = [reference]
        for phase in range(1, 24):
            updated = self.model._deform(trajectory[-1], self.fields[:, phase], 2)
            updated, _ = self.model.apply_motion_regions(updated, self.regional_fields[:, phase])
            trajectory.append(updated)
        return torch.stack(trajectory, 1)

    def assert_frozen(self):
        if not torch.equal(self.fields, self.fields_identity):
            raise AssertionError('Initial motion fields changed')
        if not torch.equal(self.z_motion, self.z_motion_identity):
            raise AssertionError('Motion latent changed')
        if self.regional_fields is not None and not torch.equal(self.regional_fields,self.regional_fields_identity):
            raise AssertionError('Frozen regional motion fields changed')

    def check(self, trajectory):
        return [dict(metrics=(m := motion_metrics(self.model, candidate, original)),
                     **motion_guard(m, self.limits))
                for candidate, original in zip(trajectory, self.original)]

    def preservation_loss(self, trajectory):
        # An explicitly named prior penalty; no real or hidden GT is consulted.
        u = trajectory-trajectory[:, :1]
        baseline = self.original-self.original[:, :1]
        return (u-baseline).square().sum(-1).mean((1, 2))
