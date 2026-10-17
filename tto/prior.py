"""Frozen adapter that retains every condition returned by the original encoder."""
import torch

from train_neural_ffd import load_checkpoint
from .io import device_tree, model_digest, sha256


class FrozenNeuralFFDPrior:
    def __init__(self, checkpoint, *, device='cpu', expected_sha256=None):
        self.checkpoint_sha256 = sha256(checkpoint)
        if expected_sha256 and self.checkpoint_sha256 != expected_sha256:
            raise ValueError('Synthetic checkpoint SHA256 mismatch')
        self.model, self.saved = load_checkpoint(checkpoint, device)
        self.model.eval().requires_grad_(False)
        if (self.model.shape_dim, self.model.motion_dim, self.model.frames) != (64, 32, 24):
            raise ValueError('This protocol requires Shape64 / Motion32 / 24 phases')
        if len(self.model.template_vertices_mm) != 10631:
            raise ValueError('Unexpected fixed topology vertex count')
        self.fingerprint = model_digest(self.model)

    @property
    def device(self):
        return self.model.template_vertices_mm.device

    @torch.no_grad()
    def encode(self, observations):
        # GT is never passed to the encoder, even during the synthetic oracle.
        forbidden = {'vertices_gt', 'pose_gt', 'phases_gt'} & observations.keys()
        if forbidden:
            raise ValueError(f'Optimizer observations contain hidden truth: {sorted(forbidden)}')
        return self.model.encode(device_tree(observations, self.device))

    @staticmethod
    def conditions(encoded):
        return {key: encoded[key] for key in ('reference_spatial', 'dynamic_spatial') if key in encoded}

    def decode_shape(self, z_shape, conditions):
        return self.model.decode_shape(z_shape, reference_spatial=conditions.get('reference_spatial'))

    def decode_original(self, z_shape, z_motion, conditions):
        return self.model.decode_4d(z_shape, z_motion, return_details=True, **conditions)

    @torch.no_grad()
    def initialize(self, observations):
        encoded = self.encode(observations)
        conditions = self.conditions(encoded)
        output = self.decode_original(encoded['z_shape'], encoded['z_motion'], conditions)
        reference = self.decode_shape(encoded['z_shape'], conditions)
        # Check the unmodified forward independently, including all conditions.
        direct = self.model(device_tree(observations, self.device))
        differences = {key: float((output[key]-direct[key]).abs().max())
                       for key in ('vertices_ref', 'vertices_4d')}
        differences.update({key: float((encoded[key]-direct[key]).abs().max())
                            for key in ('z_shape', 'z_motion')})
        if any(differences.values()) or not torch.equal(reference, output['vertices_ref']):
            raise AssertionError(f'TTO initialization differs from original forward: {differences}')
        if self.model.spatial_fusion is not None and 'reference_spatial' not in conditions:
            raise AssertionError('Trained reference spatial conditioning was lost')
        if self.model.dynamic_conditioner is not None and 'dynamic_spatial' not in conditions:
            raise AssertionError('Trained dynamic spatial conditioning was lost')
        return dict(z_shape_initial=encoded['z_shape'].detach().clone(),
                    z_motion_initial=encoded['z_motion'].detach().clone(),
                    vertices_reference_initial=output['vertices_ref'].detach().clone(),
                    vertices_4d_initial=output['vertices_4d'].detach().clone(),
                    control_grids_initial=device_tree(output['control_grids'], self.device),
                    conditions=conditions, forward_consistency=differences,
                    enabled_conditions=dict(reference_spatial=self.model.spatial_fusion is not None,
                                            dynamic_spatial=self.model.dynamic_conditioner is not None,
                                            regional_shape=self.model.regional_shape,
                                            regional_motion=self.model.regional_motion))

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise AssertionError('Pretrained network parameters must stay frozen and gradient-free')
        if model_digest(self.model) != self.fingerprint:
            raise AssertionError('Pretrained model parameters or fixed buffers changed')
