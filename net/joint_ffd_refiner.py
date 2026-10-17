"""Shared observation-driven refinement on NeuralFFDNet's original Fine FFD.

Mechanisms reviewed (adapted, no external implementation/dependency copied):
* MedNeXt's independent depthwise 3D residual blocks:
  https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/network_architecture/mednextv1/blocks.py
* Circular temporal Conv1d in Time-Series-Library's TokenEmbedding:
  https://github.com/thuml/Time-Series-Library/blob/main/layers/Embed.py
* DiT's zero-initialised output projection and conditional modulation:
  https://github.com/facebookresearch/DiT/blob/main/models.py

Input channels remain LV,LA,RV,RA x (world dx,dy,dz mm, support). A grouped
stem preserves those blocks before learned fusion. Shape64/Motion32 condition
observation features; there is no latent-to-spatial-grid decoder, patient
embedding or per-patient optimisation. Output controls use mm and the exact
physical lattice/basis already used by NeuralFFDNet.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.ffd import bspline_basis
from utils.ffd_physical import bspline_ffd_physical, bspline_support_physical
from utils.ffd_observation_splat import build_missing_section_recovery, splat_observation_residuals
from utils.observation_residual import ObservationResidualExtractor


class _ChannelNorm(nn.Module):
    """Normalise channels at each node, without mixing patients or phases."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class _SpatialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm = _ChannelNorm(channels)
        self.expand = nn.Conv3d(channels, channels*2, 1, bias=False)
        self.project = nn.Conv3d(channels*2, channels, 1, bias=False)

    def forward(self, x):
        return x+self.project(F.silu(self.expand(self.norm(self.depthwise(x)))))


def smooth_control_limit(vectors_mm, max_norm_mm=3.):
    """Radial smooth cap, with derivative I at zero and norm <= max_norm_mm.

    Unlike tanh(norm)/norm.clamp_min(eps), this does not kill the gradient of
    an exactly zero-initialised head. This bounds vectors, not Jacobians.
    """
    if not math.isfinite(max_norm_mm) or max_norm_mm <= 0:
        raise ValueError('Positive finite physical control norm limit required')
    return vectors_mm/torch.sqrt(1+(vectors_mm/max_norm_mm).square().sum(-1, keepdim=True))


def _queries(phases, reference, batch, count):
    if phases is None:
        return (torch.arange(count, device=reference.device, dtype=reference.dtype)/count)[None].expand(batch, -1)
    if not isinstance(phases, torch.Tensor) or not phases.is_floating_point():
        raise ValueError('Floating query phases [Q] or [B,Q] required')
    phases = phases.to(reference)
    if phases.ndim == 1:
        phases = phases[None].expand(batch, -1)
    if phases.ndim != 2 or phases.shape[0] != batch or phases.shape[1] < 1 or not torch.isfinite(phases).all():
        raise ValueError('Finite nonempty query phases [Q] or [B,Q] required')
    return phases


def evaluate_residual_controls(static_raw_mm, dynamic_coefficients_mm, *, query_phases=None,
                                max_control_norm_mm=3., node_chunk_size=4096):
    """Evaluate a C2 periodic control field, then anchor and bound the TOTAL.

    Static raw [B,Gx,Gy,Gz,3], dynamic coefficients [B,T,Gx,Gy,Gz,3].
    Coefficients are UNIFORM PERIODIC cubic B-spline coefficients, not values
    to interpolate exactly. Queries are cycle phases, wrapped modulo one.
    The same cardinal cubic basis as the spatial FFD is used along time.

    C(t)=cap(S+D(t)-D(0)); C_static=cap(S); C_dynamic(t)=C(t)-C_static.
    Thus C_dynamic(0)=0, C=C_static+C_dynamic, and the total field is bounded
    for arbitrary query phases, including between the original 24 frames.
    The relative dynamic component by itself can have norm up to twice the
    total cap. First/last *sampled* frames need not be equal; phases 0 and 1 do.
    """
    s, d = static_raw_mm, dynamic_coefficients_mm
    if (s.ndim != 5 or s.shape[-1] != 3 or d.ndim != 6 or d.shape[0] != s.shape[0]
            or d.shape[2:] != s.shape[1:] or min(d.shape[:2]) < 1 or min(s.shape[1:4]) < 4
            or s.dtype not in (torch.float32, torch.float64) or d.dtype != s.dtype or d.device != s.device
            or not torch.isfinite(s).all() or not torch.isfinite(d).all()
            or type(node_chunk_size) is not int or node_chunk_size < 1):
        raise ValueError('Matching finite raw static/dynamic physical grids and a positive node chunk required')
    b, t = d.shape[:2]
    phases = _queries(query_phases, d, b, t)
    # Include phase zero in the same gather operation for exact anchoring.
    queries = torch.cat((phases, phases.new_zeros(b, 1)), 1)
    if t == 1:
        # One coefficient is a constant field. Avoid introducing phase-varying
        # roundoff by summing four copies with different floating weights.
        sampled = d.expand(-1, queries.shape[1], -1, -1, -1, -1)
    else:
        q = queries.remainder(1)*t
        cell = q.floor().long()
        basis = bspline_basis(q-cell.to(q))
        offsets = torch.arange(-1, 3, device=d.device)
        index = (cell[..., None]+offsets).remainder(t)
        flat = d.reshape(b, t, -1, 3)
        sampled_parts = []
        rows = torch.arange(b, device=d.device)[:, None, None]
        for start in range(0, flat.shape[2], node_chunk_size):
            values = flat[:, :, start:start+node_chunk_size][rows, index]
            sampled_parts.append((values*basis[..., None, None]).sum(2))
        sampled = torch.cat(sampled_parts, 2).reshape(b, queries.shape[1], *s.shape[1:])
    static = smooth_control_limit(s, max_control_norm_mm)
    total = smooth_control_limit(s[:, None]+(sampled[:, :-1]-sampled[:, -1:]), max_control_norm_mm)
    return dict(controls_res_mm=total, controls_static_mm=static,
                controls_dynamic_mm=total-static[:, None], query_phases=phases)


def apply_residual_ffd(vertices_base_mm, controls_res_mm, origin_mm, spacing_mm, *, frame_chunk_size=2):
    """Apply ONE residual FFD at current base vertices, preserving topology.

    [B,Q,N,3] + controls [B,Q,Gx,Gy,Gz,3] -> [B,Q,N,3]. These controls are
    absolute residuals on each base phase, not recurrent motion increments.
    Continuous-time callers must supply base vertices at the queried phases;
    this function does not redefine NeuralFFDNet's discrete base trajectory.
    """
    v, c = vertices_base_mm, controls_res_mm
    if (v.ndim != 4 or v.shape[-1] != 3 or c.ndim != 6 or c.shape[-1] != 3
            or v.shape[:2] != c.shape[:2] or min(v.shape[:3]) < 1 or min(c.shape[2:5]) < 4
            or v.dtype != c.dtype or v.device != c.device
            or type(frame_chunk_size) is not int or frame_chunk_size < 1):
        raise ValueError('Matching base trajectory and residual grids with a positive frame chunk required')
    b, q, n, _ = v.shape
    vflat, cflat = v.reshape(b*q, n, 3), c.reshape(b*q, *c.shape[2:])
    pieces = []
    for start in range(0, b*q, frame_chunk_size):
        base = vflat[start:start+frame_chunk_size]
        pieces.append(base+bspline_ffd_physical(base, cflat[start:start+frame_chunk_size], origin_mm, spacing_mm))
    return torch.cat(pieces).reshape_as(v)


def residual_control_regularization(controls_res_mm, spacing_mm):
    """Physical spatial gradients and cyclic temporal differences for training.

    Temporal fields must be sampled uniformly over one complete cycle. These
    differentiable penalties do not certify Jacobians, mesh quality or safety.
    The composed mesh still requires the existing complete geometry acceptance.
    """
    c = controls_res_mm
    if (c.ndim != 6 or c.shape[-1] != 3 or min(c.shape[:2]) < 1 or min(c.shape[2:5]) < 4
            or not c.is_floating_point() or not torch.isfinite(c).all()):
        raise ValueError('Finite physical control trajectory [B,T,Gx,Gy,Gz,3] required')
    spacing = torch.as_tensor(spacing_mm, device=c.device, dtype=c.dtype)
    if spacing.shape != (3,) or not torch.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Positive physical control spacing [3] required')
    spatial = torch.stack([(c.diff(dim=axis+2)/spacing[axis]).square().mean() for axis in range(3)]).mean()
    velocity = (c.roll(-1, 1)-c)*c.shape[1]
    acceleration = (c.roll(-1, 1)-2*c+c.roll(1, 1))*c.shape[1]**2
    return dict(spatial_gradient_l2=spatial, cyclic_velocity_l2=velocity.square().mean(),
                cyclic_acceleration_l2=acceleration.square().mean())


class ObservationFFDRefiner(nn.Module):
    """Small patient-shared spatial/temporal network on the ORIGINAL Fine grid.

    Default input [B,T,16,24,24,32], Shape64, Motion32 -> residual controls
    [B,T,24,24,32,3]. No spatial down/upsampling or free vertex prediction.
    All patients use the SAME parameters, irrespective of case IDs or batch
    size. A grouped four-chamber stem precedes learned channel coordination.

    Full uniform cycles anchored at phase zero are required for circular
    convolutions. Missing observations stay in their phase slots with zero
    support; do not delete frames. T=1 is supported for a static-only call.
    Query phases may be continuous; base mesh interpolation is a separate
    caller contract. All-zero support makes this module exactly neutral even
    after training. Spatial/cyclic coordination can fill locally absent input.
    """
    def __init__(self, input_channels=16, hidden_channels=24, shape_dim=64, motion_dim=32,
                 max_control_norm_mm=3., spatial_blocks=2, temporal_node_chunk_size=18432,
                 spatial_frame_chunk_size=24, displacement_scale_mm=15., support_scale_mm=1.):
        super().__init__()
        sizes = (input_channels, hidden_channels, shape_dim, motion_dim, spatial_blocks,
                 temporal_node_chunk_size, spatial_frame_chunk_size)
        if (any(type(n) is not int or n < 1 for n in sizes) or input_channels != 16 or hidden_channels % 4
                or any(not math.isfinite(x) or x <= 0 for x in
                       (max_control_norm_mm, displacement_scale_mm, support_scale_mm))):
            raise ValueError('16 chamber channels, hidden width divisible by four and positive finite sizes/scales required')
        self.shape_dim, self.motion_dim = shape_dim, motion_dim
        self.hidden_channels, self.max_control_norm_mm = hidden_channels, float(max_control_norm_mm)
        self.temporal_node_chunk_size, self.spatial_frame_chunk_size = temporal_node_chunk_size, spatial_frame_chunk_size
        self.displacement_scale_mm, self.support_scale_mm = float(displacement_scale_mm), float(support_scale_mm)
        self.configuration = dict(input_channels=input_channels, hidden_channels=hidden_channels,
            shape_dim=shape_dim, motion_dim=motion_dim, max_control_norm_mm=max_control_norm_mm,
            spatial_blocks=spatial_blocks, temporal_node_chunk_size=temporal_node_chunk_size,
            spatial_frame_chunk_size=spatial_frame_chunk_size, displacement_scale_mm=displacement_scale_mm,
            support_scale_mm=support_scale_mm)
        self.chamber_stem = nn.Conv3d(16, hidden_channels, 1, groups=4, bias=False)
        self.fusion = nn.Conv3d(hidden_channels, hidden_channels, 1, bias=False)
        self.spatial = nn.Sequential(*[_SpatialBlock(hidden_channels) for _ in range(spatial_blocks)])
        self.static_condition = nn.Linear(shape_dim, hidden_channels*2)
        self.dynamic_condition = nn.Linear(shape_dim+motion_dim, hidden_channels*2)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels,
                      padding_mode='circular', bias=False), nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1, bias=False))
        self.static_head = nn.Conv3d(hidden_channels, 3, 1)
        self.dynamic_head = nn.Conv1d(hidden_channels, 3, 1)
        for layer in (self.static_condition, self.dynamic_condition):
            nn.init.normal_(layer.weight, 0., .02)
            nn.init.zeros_(layer.bias)
        for head in (self.static_head, self.dynamic_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _condition(self, value, latent, layer):
        gamma, beta = layer(latent).chunk(2, -1)
        extra = (None,)*(value.ndim-2)
        return value*(1+gamma[(...,)+extra])+beta[(...,)+extra]

    def forward(self, observation_features, z_shape, z_motion, *, phases=None, query_phases=None):
        x = observation_features
        if (x.ndim != 6 or x.shape[2] != 16 or min(x.shape[:2]) < 1 or min(x.shape[3:]) < 4
                or x.dtype not in (torch.float32, torch.float64)):
            raise ValueError('Physical four-chamber features [B,T,16,Gx,Gy,Gz] required')
        b, t, _, gx, gy, gz = x.shape
        for latent, width in ((z_shape, self.shape_dim), (z_motion, self.motion_dim)):
            if (latent.shape != (b, width) or latent.dtype != x.dtype or latent.device != x.device
                    or not torch.isfinite(latent).all()):
                raise ValueError('Matching finite Shape/Motion codes required')
        if x.dtype != self.chamber_stem.weight.dtype or x.device != self.chamber_stem.weight.device:
            raise ValueError('Move/cast the shared Refiner to the feature device/dtype before calling')
        if phases is not None:
            supplied = _queries(phases, x, b, t)
            expected = _queries(None, x, b, t)
            if supplied.shape != expected.shape or not torch.allclose(supplied, expected, atol=1e-5, rtol=0):
                raise ValueError('Circular time modelling requires a complete uniform cycle anchored at phase zero')
        chamber = x.reshape(b, t, 4, 4, gx, gy, gz)
        support = chamber[:, :, :, 3:4]
        if not torch.isfinite(support).all() or (support < 0).any():
            raise ValueError('Finite nonnegative physical support channels required')
        observed = support > 0
        vectors = torch.where(observed, chamber[:, :, :, :3], 0.)
        if not torch.isfinite(vectors).all():
            raise ValueError('Supported physical displacement hints must be finite')
        normalised = torch.cat((vectors/self.displacement_scale_mm,
                                torch.log1p(support/self.support_scale_mm)), 3).reshape_as(x)
        flat = normalised.reshape(b*t, 16, gx, gy, gz)
        encoded = []
        for start in range(0, b*t, self.spatial_frame_chunk_size):
            h = F.silu(self.chamber_stem(flat[start:start+self.spatial_frame_chunk_size]))
            encoded.append(self.spatial(self.fusion(h)))
        h = torch.cat(encoded).reshape(b, t, self.hidden_channels, gx, gy, gz)
        static_features = self._condition(h.mean(1), z_shape, self.static_condition)
        static_raw = self.static_head(F.silu(static_features)).movedim(1, -1)
        node_features = h.flatten(3)
        dynamic_parts = []
        for start in range(0, gx*gy*gz, self.temporal_node_chunk_size):
            node = node_features[..., start:start+self.temporal_node_chunk_size]
            count = node.shape[-1]
            node = node.permute(0, 3, 2, 1).reshape(b*count, self.hidden_channels, t)
            latent = torch.cat((z_shape, z_motion), -1).repeat_interleave(count, 0)
            node = self._condition(node, latent, self.dynamic_condition)
            dynamic = self.dynamic_head(F.silu(node+self.temporal(node)))
            dynamic_parts.append(dynamic.reshape(b, count, 3, t).permute(0, 3, 1, 2))
        dynamic_coefficients = torch.cat(dynamic_parts, 2).reshape(b, t, gx, gy, gz, 3)
        has_evidence = observed.flatten(1).any(-1)
        static_raw = static_raw*has_evidence[:, None, None, None, None]
        dynamic_coefficients = dynamic_coefficients*has_evidence[:, None, None, None, None, None]
        result = evaluate_residual_controls(static_raw, dynamic_coefficients, query_phases=query_phases,
                    max_control_norm_mm=self.max_control_norm_mm, node_chunk_size=self.temporal_node_chunk_size)
        result.update(raw_static_mm=static_raw, dynamic_coefficients_mm=dynamic_coefficients,
                      has_observation_evidence=has_evidence)
        return result


class JointFFDRefiner(nn.Module):
    """Forward-only composition of base prediction, observations and shared H_phi.

    Wrap an existing NeuralFFDNet; its checkpoint/architecture is unchanged.
    Pose must be explicitly supplied (synthetic GT or an external estimate).
    The original Fine origin/spacing/shape are taken from the BASE MODEL,
    never fitted to individual patients. Recovery reads observation SDF and
    current predicted surfaces, not GT vertices. Base/latent/Refiner graphs
    remain connected; the caller decides which network weights to freeze.
    No automatic parameter fitting or optimiser step occurs in forward().
    Top-level control_grids remain the base model's recurrent control fields;
    absolute refinement controls and their regularizers are returned separately.
    """
    def __init__(self, base_model, refiner=None, *, max_observation_distance_mm=15.,
                 boundary_sample_step_mm=None, splat_point_chunk_size=65536, ffd_frame_chunk_size=8,
                 observation_cell_chunk_size=64):
        super().__init__()
        if any(not hasattr(base_model, key) for key in ('grid_shapes', 'control_origins_mm',
                'control_spacings_mm', 'chamber_ranges', 'faces', 'norm_to_mm', 'shape_dim', 'motion_dim')):
            raise ValueError('Existing physical NeuralFFDNet geometry/latent interface required')
        if any(type(n) is not int or n < 1 for n in (splat_point_chunk_size, ffd_frame_chunk_size, observation_cell_chunk_size)):
            raise ValueError('Positive splat/FFD chunk sizes required')
        self.base_model = base_model
        self.refiner = ObservationFFDRefiner(shape_dim=base_model.shape_dim, motion_dim=base_model.motion_dim) if refiner is None else refiner
        if self.refiner.shape_dim != base_model.shape_dim or self.refiner.motion_dim != base_model.motion_dim:
            raise ValueError('Refiner and base Shape/Motion dimensions must match')
        self.extractor = ObservationResidualExtractor(base_model.chamber_ranges, base_model.faces,
            base_model.norm_to_mm, max_distance_mm=max_observation_distance_mm, sample_step_mm=boundary_sample_step_mm,
            cell_chunk_size=observation_cell_chunk_size)
        self.max_observation_distance_mm = max_observation_distance_mm
        self.splat_point_chunk_size, self.ffd_frame_chunk_size = splat_point_chunk_size, ffd_frame_chunk_size
        self.configuration = dict(max_observation_distance_mm=max_observation_distance_mm,
            boundary_sample_step_mm=boundary_sample_step_mm, splat_point_chunk_size=splat_point_chunk_size,
            ffd_frame_chunk_size=ffd_frame_chunk_size, observation_cell_chunk_size=observation_cell_chunk_size)
        self.refiner.to(base_model.control_origins_mm)

    def forward(self, observations, *, pose):
        with torch.profiler.record_function('refiner/base_forward'):
            base = self.base_model(observations)
        splat = self.observation_features(base, observations, pose=pose)
        return self.refine_base(base, splat, phases=observations['phases'])

    def observation_features(self, base, observations, *, pose):
        """Explicit physical input stage; cache only while Base is frozen/eval."""
        vertices = base['vertices_4d']
        shape = self.base_model.grid_shapes[2]
        origin, spacing = self.base_model.control_origins_mm[2], self.base_model.control_spacings_mm[2]
        # Reject unsupported base geometry even if the observed subset happens
        # to lie in support. Zero initial controls are not an extrapolation rule.
        bspline_support_physical(vertices.reshape(-1, 3), origin, spacing, shape)
        with torch.profiler.record_function('refiner/physical_observation'):
            residuals = self.extractor(vertices, observations, pose=pose)
        with torch.profiler.record_function('refiner/missing_section_recovery'):
            recovery = build_missing_section_recovery(vertices, observations, residuals, pose=pose,
                chamber_ranges=self.base_model.chamber_ranges, faces=self.base_model.faces, norm_to_mm=self.base_model.norm_to_mm)
        with torch.profiler.record_function('refiner/splat'):
            splat = splat_observation_residuals(residuals, origin, spacing, shape,
                batch_size=vertices.shape[0], frames=vertices.shape[1], recovery=recovery,
                max_displacement_mm=self.max_observation_distance_mm, point_chunk_size=self.splat_point_chunk_size)
        return splat

    def refine_base(self, base, splat, *, phases):
        """Same shared prediction/application path for cached or online inputs."""
        vertices = base['vertices_4d']
        origin, spacing = self.base_model.control_origins_mm[2], self.base_model.control_spacings_mm[2]
        with torch.profiler.record_function('refiner/shared_network'):
            refinement = self.refiner(splat['features'], base['z_shape'], base['z_motion'], phases=phases)
        with torch.profiler.record_function('refiner/residual_ffd'):
            final = apply_residual_ffd(vertices, refinement['controls_res_mm'], origin, spacing,
                                       frame_chunk_size=self.ffd_frame_chunk_size)
        result = dict(base)
        result.update(vertices_4d=final, vertices_ref=final[:, 0], vertices_base_4d=vertices,
                      base_output=base, refinement=refinement, observation_splat=splat,
                      control_regularization=residual_control_regularization(refinement['controls_res_mm'], spacing))
        return result
