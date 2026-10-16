"""Strict adapter to the existing checkpoint loader and original decoder."""
import torch

from train_neural_ffd import load_checkpoint
from .io import sha256, model_digest, device_tree


class FrozenSyntheticPrior:
    def __init__(self, checkpoint, *, device='cpu', expected_sha256=None):
        self.checkpoint_sha256 = sha256(checkpoint)
        if expected_sha256 and self.checkpoint_sha256 != expected_sha256:
            raise ValueError('Synthetic checkpoint hash mismatch')
        self.model, self.saved = load_checkpoint(checkpoint, device)
        self.model.eval().requires_grad_(False)
        if (self.model.shape_dim, self.model.motion_dim, self.model.frames) != (64, 32, 24):
            raise ValueError('Original 64/32 latent, 24-frame NeuralFFDNet required')
        if self.model.regional_motion or self.model.dynamic_conditioner is not None or self.model.spatial_fusion is not None:
            raise ValueError('This frozen-field protocol targets the specified original checkpoint')
        self.fingerprint = model_digest(self.model)

    @property
    def device(self):
        return self.model.template_vertices_mm.device

    @torch.no_grad()
    def encode_real_observations(self, observations):
        return self.model.encode(device_tree(observations, self.device))

    def decode_initial_shape(self, z_shape):
        return self.model.decode_shape(z_shape)

    @torch.no_grad()
    def decode_initial_motion(self, z_shape, z_motion):
        output = self.model.decode_4d(z_shape, z_motion, return_details=True)
        output.update(z_shape=z_shape, z_motion=z_motion,
                      motion_fields=output['control_grids']['fine_motion'])
        return output

    @torch.no_grad()
    def export_motion_fields(self, z_shape, z_motion):
        # Fine0 is reference Shape; only indices1..23 are recurrent Motion.
        return self.decode_initial_motion(z_shape, z_motion)['control_grids']['fine_motion'].detach().clone()

    def reconstruct_4d(self, z_shape, z_motion, *, frozen_fields=None):
        if frozen_fields is None:
            output = self.model.decode_4d(z_shape, z_motion, return_details=True)
            output.update(z_shape=z_shape, z_motion=z_motion,
                          motion_fields=output['control_grids']['fine_motion'])
            return output
        from .motion_lock import FrozenMotionPropagator
        reference = self.model.decode_shape(z_shape)
        trajectory = FrozenMotionPropagator(self.model, frozen_fields).propagate(reference)
        return dict(z_shape=z_shape, z_motion=z_motion, vertices_ref=reference,
                    vertices_4d=trajectory, motion_fields=frozen_fields)

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise AssertionError('Pretrained network must remain eval/frozen with no parameter gradients')
        if model_digest(self.model) != self.fingerprint:
            raise AssertionError('Pretrained network buffers/weights changed')
