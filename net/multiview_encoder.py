"""Spatial multiview mask/SDF encoding to finite patient shape/motion codes.

Uses torchvision's BSD-licensed ResNet-18. Shared 2D multi-scale features are
motivated by DeepMesh; its SAX input, spatially aligned concatenation and
projection supervision are NOT assumed for A2C/A3C/A4C observations.
The optional direct_residual_v1 path pools each view/scale independently,
concatenates observations in canonical view order, and bypasses both attention
readouts. This is an adaptation, not DeepMesh's upsampling/3D conversion.
"""
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18


def observation_validity(observations, batch, frames, device):
    for key in ('view_present','annotated'):
        value = observations[key]
        if not ((value==0)|(value==1)).all():
            raise ValueError('Availability flags must be binary')
    present = observations['view_present'].to(device=device, dtype=torch.bool)
    annotated = observations['annotated'].to(device=device, dtype=torch.bool)
    if present.shape == (batch, 3):
        present = present[:, None].expand(-1, frames, -1)
    if annotated.shape == (batch, 3, 4):
        annotated = annotated[:, None].expand(-1, frames, -1, -1)
    if present.shape != (batch, frames, 3) or annotated.shape != (batch, frames, 3, 4):
        raise ValueError('Availability must be batched static or per-frame view/chamber flags')
    if (annotated & ~present[..., None]).any():
        raise ValueError('Annotated chamber in absent view')
    return present[..., None] & annotated


class MultiViewTemporalEncoder(nn.Module):
    def __init__(self, shape_dim=64, motion_dim=32, width=128, heads=4,
                 temporal_layers=2, spatial_sides=(14, 7), query_slots=8,
                 sdf_scale_mm=15., cnn_chunk_size=6, checkpoint_cnn=True,
                 feature_transfer='legacy', direct_pool_side=2,
                 direct_frame_gain=1., direct_shape_gain=1.):
        super().__init__()
        if (min(shape_dim, motion_dim, width, heads, temporal_layers, query_slots, cnn_chunk_size) < 1
                or width % heads or len(spatial_sides) != 2 or min(spatial_sides) < 2
                or not math.isfinite(sdf_scale_mm) or sdf_scale_mm <= 0):
            raise ValueError('Positive encoder dimensions, two spatial scales, compatible attention heads required')
        if feature_transfer not in ('legacy', 'direct_residual_v1'):
            raise ValueError('Unknown encoder feature transfer mode')
        if (type(direct_pool_side) is not int or not 1 <= direct_pool_side <= min(spatial_sides)
                or any(not math.isfinite(gain) or gain < 0 for gain in (direct_frame_gain, direct_shape_gain))):
            raise ValueError('Supported direct pooling size and finite nonnegative residual gains required')
        self.shape_dim, self.motion_dim = shape_dim, motion_dim
        self.width, self.spatial_sides = width, tuple(spatial_sides)
        self.sdf_scale_mm = sdf_scale_mm
        self.cnn_chunk_size, self.checkpoint_cnn = cnn_chunk_size, checkpoint_cnn
        self.feature_transfer, self.direct_pool_side = feature_transfer, direct_pool_side
        self.direct_frame_gain, self.direct_shape_gain = direct_frame_gain, direct_shape_gain
        # GroupNorm is independent of batch/chunk composition and supports B=1.
        backbone = resnet18(weights=None, norm_layer=partial(nn.GroupNorm, 8))
        backbone.conv1 = nn.Conv2d(12, 64, 7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.projections = nn.ModuleList([nn.Conv2d(128, width, 1), nn.Conv2d(256, width, 1)])
        self.position = nn.Linear(2, width, bias=False)
        self.view_embedding = nn.Parameter(torch.randn(3, width) * .02)
        self.scale_embedding = nn.Parameter(torch.randn(2, width) * .02)
        self.availability_embedding = nn.Linear(4, width, bias=False)
        self.spatial_queries = nn.Parameter(torch.randn(query_slots, width) * .02)
        self.view_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.frame_projection = nn.Sequential(nn.LayerNorm(query_slots*width), nn.Linear(query_slots*width, width))
        self.missing_frame = nn.Parameter(torch.zeros(width))
        self.time_projection = nn.Linear(5, width)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width*2,
                                           dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, temporal_layers, enable_nested_tensor=False)
        self.shape_query = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.motion_query = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.shape_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.motion_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.shape_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, shape_dim))
        self.motion_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, motion_dim))
        # Keep legacy parameters and initialization order exactly as before.
        # Missing mode in old checkpoints reconstructs the legacy state_dict.
        if self.feature_transfer == 'direct_residual_v1':
            self.direct_frame_projection = nn.Linear(3*2*width*direct_pool_side**2, width, bias=False)
            self.direct_shape_projection = nn.Linear(width, shape_dim, bias=False)
            nn.init.xavier_uniform_(self.direct_frame_projection.weight)
            nn.init.xavier_uniform_(self.direct_shape_projection.weight)

    def spatial_features(self, x):
        """Retain layer2/layer3 feature maps; no early global average pooling."""
        h = self.layer1(self.stem(x))
        middle = self.layer2(h)
        coarse = self.layer3(middle)
        return tuple(F.adaptive_avg_pool2d(proj(value), (side, side))
                     for proj, value, side in zip(self.projections, (middle, coarse), self.spatial_sides))

    def direct_descriptors(self, maps, view_valid):
        """[BT*3,C,H,W] maps -> [BT,3*2*C*K*K], ordered view then scale.

        Inputs are the same CNN maps used by attention, before any embeddings.
        A missing/unannotated view is zeroed AFTER the CNN, so convolution
        biases and normalization cannot create an observed direct feature.
        view_valid is [BT,3], derived from chamber annotation, not foreground.
        """
        if self.feature_transfer != 'direct_residual_v1':
            raise ValueError('Direct descriptors require direct_residual_v1')
        if len(maps) != 2 or view_valid.ndim != 2 or view_valid.shape[1] != 3:
            raise ValueError('Two CNN scales and [BT,3] view validity required')
        frames = view_valid.shape[0]
        pooled = []
        for value in maps:
            if value.ndim != 4 or value.shape[:2] != (frames*3, self.width):
                raise ValueError('CNN maps must retain flattened frame/view order')
            value = F.adaptive_avg_pool2d(value, self.direct_pool_side).reshape(
                frames, 3, self.width, self.direct_pool_side, self.direct_pool_side)
            value = torch.where(view_valid[...,None,None,None], value, 0.)
            pooled.append(value)
        # Concatenate scales inside each view, then flatten A2C/A3C/A4C.
        return torch.cat(pooled, dim=2).flatten(1)

    def forward(self, observations):
        # Only these observation fields are read. Mesh, SSM scores and GT pose
        # are never features, even if a Dataset item includes them.
        masks, sdf = observations['masks'], observations['sdf']
        if (masks.ndim != 6 or masks.shape[2:4] != (3, 4) or sdf.shape != masks.shape
                or min(masks.shape[:2])<1
                or min(masks.shape[-2:]) < 32):
            raise ValueError('Mask/SDF input must be [B,T,3,4,H,W], H,W>=32')
        parameter = self.view_embedding
        masks, sdf = masks.to(parameter), sdf.to(parameter)
        b, t, _, _, height, width = masks.shape
        valid = observation_validity(observations, b, t, masks.device)
        active = valid[..., None, None]
        if (not torch.isfinite(masks[active.expand_as(masks)]).all()
                or not torch.isfinite(sdf[active.expand_as(sdf)]).all()
                or ((masks[active.expand_as(masks)] != 0) & (masks[active.expand_as(masks)] != 1)).any()):
            raise ValueError('Finite binary masks and finite physical SDF required in annotated channels')
        phases = observations['phases'].to(parameter)
        if (phases.shape != (b, t) or not torch.isfinite(phases).all()
                or (phases < 0).any() or (phases >= 1).any()
                or not torch.allclose(phases, torch.arange(t, device=phases.device, dtype=phases.dtype)[None].expand(b,-1)/t)):
            raise ValueError('Synchronized independent phases must be 0,...,(T-1)/T; reference is source frame 0')
        if 'sdf_units' in observations:
            units = observations['sdf_units']
            if units != 'mm' and not (isinstance(units, (list, tuple)) and all(x == 'mm' for x in units)):
                raise ValueError('Encoder expects physical SDF in mm')
        masks = torch.where(active, masks, 0.)
        sdf = torch.where(active, sdf, 0.) / self.sdf_scale_mm
        flags = valid[..., None, None].expand(b, t, 3, 4, height, width).to(masks)
        x = torch.cat((masks, sdf, flags), dim=3).reshape(b*t*3, 12, height, width)
        chunks = [[], []]
        for piece in x.split(self.cnn_chunk_size):
            maps = (checkpoint(self.spatial_features, piece, use_reentrant=False)
                    if self.checkpoint_cnn and self.training and torch.is_grad_enabled()
                    else self.spatial_features(piece))
            for values, value in zip(chunks, maps):
                values.append(value)
        # Concatenate each CNN scale once and reuse it for both pathways.
        maps = tuple(torch.cat(pieces) for pieces in chunks)
        direct = None
        if self.feature_transfer == 'direct_residual_v1':
            descriptors = self.direct_descriptors(maps, valid.any(-1).reshape(b*t,3))
            direct = self.direct_frame_projection(descriptors).reshape(b,t,self.width)
        sequences, key_valid = [], []
        for scale, (value, side) in enumerate(zip(maps, self.spatial_sides)):
            feature = value.reshape(b*t, 3, self.width, side*side).permute(0,1,3,2)
            axis = torch.linspace(-1, 1, side, device=x.device, dtype=x.dtype)
            y, xx = torch.meshgrid(axis, axis, indexing='ij')
            position = self.position(torch.stack((xx, y), -1).reshape(side*side, 2))
            feature = (feature + position + self.view_embedding[None,:,None]
                       + self.scale_embedding[scale] + self.availability_embedding(valid.to(x).reshape(b*t,3,4))[:,:,None])
            sequences.append(feature.flatten(1,2))
            key_valid.append(valid.any(-1).reshape(b*t,3,1).expand(-1,-1,side*side).flatten(1))
        tokens, keys = torch.cat(sequences, 1), torch.cat(key_valid, 1)
        frame_present = keys.any(-1)
        if not frame_present.reshape(b,t).any(-1).all():
            raise ValueError('Every patient needs at least one annotated observed frame')
        queries = self.spatial_queries[None].expand(int(frame_present.sum()), -1, -1)
        routed = self.view_attention(queries, tokens[frame_present], tokens[frame_present],
                                     key_padding_mask=~keys[frame_present], need_weights=False)[0]
        h = self.missing_frame.expand(b*t, -1).clone()
        h[frame_present] = self.frame_projection(routed.flatten(1))
        h = h.reshape(b,t,self.width)
        if direct is not None:
            h = h + self.direct_frame_gain*direct
        time = torch.stack((phases, torch.sin(2*math.pi*phases), torch.cos(2*math.pi*phases),
                            torch.sin(4*math.pi*phases), torch.cos(4*math.pi*phases)), -1)
        missing = ~frame_present.reshape(b,t)
        context = self.temporal(h + self.time_projection(time), src_key_padding_mask=missing)
        shape = self.shape_attention(self.shape_query.expand(b,-1,-1) + h[:,:1], context, context,
                                     key_padding_mask=missing, need_weights=False)[0][:,0]
        motion = self.motion_attention(self.motion_query.expand(b,-1,-1), context, context,
                                       key_padding_mask=missing, need_weights=False)[0][:,0]
        z_shape = self.shape_head(shape)
        if direct is not None:
            # Add AFTER shape_head's normalization/projection: a true direct
            # CNN -> reference latent path, including all four regional codes.
            # Bias-free projections keep this residual zero if frame 0 is absent.
            z_shape = z_shape + self.direct_shape_gain*self.direct_shape_projection(direct[:,0])
        return {'z_shape': z_shape, 'z_motion': self.motion_head(motion)}
