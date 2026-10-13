"""Only the existing GroupNorm ResNet18 spatial path; no latent/readout heads."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from net.multiview_encoder import MultiViewTemporalEncoder, observation_validity


class GeometrySpatialEncoder(nn.Module):
    spatial_features = MultiViewTemporalEncoder.spatial_features

    def __init__(self, width=128, spatial_sides=(14, 7), sdf_scale_mm=15.,
                 cnn_chunk_size=6, checkpoint_cnn=True):
        super().__init__()
        source = MultiViewTemporalEncoder(width=width, spatial_sides=spatial_sides,
            sdf_scale_mm=sdf_scale_mm, cnn_chunk_size=cnn_chunk_size,
            checkpoint_cnn=checkpoint_cnn)
        for name in ('stem', 'layer1', 'layer2', 'layer3', 'projections'):
            setattr(self, name, getattr(source, name))
        self.width, self.spatial_sides = width, tuple(spatial_sides)
        self.sdf_scale_mm = sdf_scale_mm
        self.cnn_chunk_size, self.checkpoint_cnn = cnn_chunk_size, checkpoint_cnn

    def forward(self, observations):
        parameter = self.stem[0].weight
        masks, sdf = observations['masks'].to(parameter), observations['sdf'].to(parameter)
        if masks.ndim != 6 or masks.shape[1:4] != (24, 3, 4) or sdf.shape != masks.shape or min(masks.shape[-2:]) < 32:
            raise ValueError('Expected complete [B,24,3,4,H,W] Mask/SDF cycle')
        b, t, _, _, h, w = masks.shape
        valid = observation_validity(observations, b, t, masks.device)
        active = valid[..., None, None].expand_as(masks)
        if (not torch.isfinite(masks[active]).all() or not torch.isfinite(sdf[active]).all()
                or ((masks[active] != 0) & (masks[active] != 1)).any()):
            raise ValueError('Finite binary annotated masks and physical SDF required')
        phases = observations['phases'].to(parameter)
        expected = torch.arange(t, device=parameter.device, dtype=parameter.dtype)[None].expand(b, -1)/t
        if phases.shape != (b, t) or not torch.allclose(phases, expected):
            raise ValueError('Phases must be ordered 0,...,23/24; no duplicate endpoint')
        units = observations.get('sdf_units', 'mm')
        if units != 'mm' and not (isinstance(units, (list, tuple)) and all(v == 'mm' for v in units)):
            raise ValueError('Physical SDF in mm required')
        x = torch.cat((torch.where(active, masks, 0.),
                       torch.where(active, sdf, 0.)/self.sdf_scale_mm,
                       active.to(parameter)), dim=3).reshape(b*t*3, 12, h, w)
        pieces = [[], []]
        for chunk in x.split(self.cnn_chunk_size):
            values = (checkpoint(self.spatial_features, chunk, use_reentrant=False)
                      if self.training and self.checkpoint_cnn and torch.is_grad_enabled()
                      else self.spatial_features(chunk))
            for collection, value in zip(pieces, values):
                collection.append(value)
        maps = [torch.cat(values).reshape(b, t, 3, self.width, side, side)
                for values, side in zip(pieces, self.spatial_sides)]
        maps = [torch.where(valid.any(-1)[..., None, None, None], value, 0.) for value in maps]
        return dict(scale14=maps[0], scale7=maps[1], validity=valid)
