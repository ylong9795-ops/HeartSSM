"""Stage S: 24-phase observations -> [B,N,3] centered physical-mm V0.

The production fixed template has N=10631. Inherit G1 modules/state keys;
only orchestration differs. No Fine1..23 deformation is applied.
"""
import torch
from net.geometry_heartffdnet import GeometryHeartFFDNet, loss_output_adapter


class StagedHeartFFDNet(GeometryHeartFFDNet):
    def reference_output(self, observations):
        """Training/diagnostic output: vertices_4d has T=1; fine controls T=1."""
        for key, expected in (('asset_sha256', self.asset_sha256),
                              ('template_sha256', self.template_sha256)):
            values = observations.get(key, expected)
            values = [values] if isinstance(values, str) else values
            if any(value != expected for value in values):
                raise ValueError('Observation geometry identity differs: '+key)
        if 'norm_to_mm' in observations and not (torch.as_tensor(observations['norm_to_mm']) == self.norm_to_mm).all():
            raise ValueError('Observation normalization differs from template')
        spatial = self.encoder(observations)
        lifted = self.lifting(spatial, observations, self.query_points_mm, self.norm_to_mm)
        features = self.unet(lifted['features'].flatten(1, 2))
        controls = self.heads(features, self.control_spacings_mm)
        # Preserve the original joint head and its initialization; unused output
        # channels have zero gradient. No new encoder, head or spline evaluator.
        controls['fine'] = controls['fine'][:, :1]
        v = self.template_vertices_mm[None].expand(observations['masks'].shape[0], -1, -1)
        for level, grid in enumerate((controls['coarse'], controls['medium'], controls['fine'][:, 0])):
            v = self._deform(v, grid, level)
        return dict(vertices_ref=v, vertices_4d=v[:, None], control_grids=controls,
                    pose_mode=self.pose_mode,
                    observation_support_fraction=lifted['support'].float().mean((2,3,4)),
                    feature_shapes={k:list(value.shape) for k,value in features.items()})

    def predict_reference(self, observations):
        return self.reference_output(observations)['vertices_ref']

    def forward(self, observations):
        return self.predict_reference(observations)

    def freeze_reference(self):
        """Freeze ALL parameters and evaluation buffers affecting V0 for future M.

        Future Motion must own separate trainable Fine1..23 parameters and use
        inherited _deform on current vertices. Shared reference features remain
        frozen. Stage J requires an explicit, separately validated implementation.
        """
        self.requires_grad_(False)
        self._reference_frozen = True
        self.eval()
        return self

    def train(self, mode=True):
        return super().train(False if getattr(self, '_reference_frozen', False) else mode)

    def predict_motion(self, observations, reference_vertices):
        """Reserved M contract: frozen predicted V0 -> [B,23,N,3] future meshes.

        Motion controls must be [B,23,24,24,32,3], Fine1..23 only, recursively
        applied to current vertices. This release intentionally cannot train M/J.
        """
        raise NotImplementedError('Stage M/J are not implemented or authorized in Goal 04-S')


def reference_loss(criterion, output, vertices_gt, control_spacings_mm):
    """Reuse original shape objective with GT frame0 only; no loss reimplementation."""
    return criterion(loss_output_adapter(output), vertices_gt[:, :1],
                     control_spacings_mm=control_spacings_mm, stage='shape')
