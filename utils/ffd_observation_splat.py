"""Chamber-separated observation features on the ORIGINAL physical FFD grid.

Mechanism reference reviewed: PyTorch3D points_to_volumes.py, weighted feature
and density scatter accumulation followed by normalisation:
https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/points_to_volumes.py
Its trilinear voxel coordinates/8-node kernel are not used or copied. Here
the 64-node kernel, flattening and closed boundary convention come directly
from utils.ffd_physical.bspline_support_physical, also used by FFD evaluation.

These are observation FEATURES, not solved control displacements. Applying
their averaged vectors as FFD controls need not reproduce the point hints.
No latent dimensions, learned parameters or per-patient optimisers are added.
"""
import math

import torch
from torch.nn import functional as F

from data.mask_transform import CHAMBERS
from net.multiview_encoder import observation_validity
from utils.ffd_physical import bspline_support_physical
from utils.slice_supervision import closest_surface_points


def _point_weights(points, displacement, confidence, sample_weights, valid):
    if (points.ndim != 2 or points.shape[1] != 3 or displacement.shape != points.shape
            or points.dtype not in (torch.float32, torch.float64)
            or displacement.dtype != points.dtype or displacement.device != points.device):
        raise ValueError('Matching float32/float64 physical points and displacements [K,3] required')
    k = len(points)
    confidence = points.new_ones(k) if confidence is None else torch.as_tensor(confidence).to(points)
    sample_weights = points.new_ones(k) if sample_weights is None else torch.as_tensor(sample_weights).to(points)
    valid = torch.ones(k, dtype=torch.bool, device=points.device) if valid is None else torch.as_tensor(valid, device=points.device)
    if confidence.shape != (k,) or sample_weights.shape != (k,) or valid.shape != (k,):
        raise ValueError('Confidence, sample weights and validity must be [K]')
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError('Sample validity must be binary')
    valid = valid.bool()
    if ((((confidence < 0) | (confidence > 1)) & torch.isfinite(confidence) & valid).any()
            or ((sample_weights < 0) & torch.isfinite(sample_weights) & valid).any()):
        raise ValueError('Confidence must be in [0,1] and sample weights nonnegative')
    # Explicitly remove invalid/nonfinite samples BEFORE constructing weights
    # and indices; multiplying NaN by zero would still contaminate the grid.
    eligible = (valid & torch.isfinite(points).all(-1) & torch.isfinite(displacement).all(-1)
                & torch.isfinite(confidence) & torch.isfinite(sample_weights)
                & (confidence > 0) & (sample_weights > 0))
    return confidence, sample_weights, eligible


def _splat_grouped(points, displacement, groups, group_count, origin, spacing, shape, *,
                   confidence, sample_weights, valid, max_displacement_mm, eps, point_chunk_size):
    if (not math.isfinite(max_displacement_mm) or max_displacement_mm <= 0
            or not math.isfinite(eps) or eps <= 0
            or type(point_chunk_size) is not int or point_chunk_size < 1):
        raise ValueError('Positive finite displacement cap/epsilon and positive integer point chunk required')
    confidence, sample_weights, eligible = _point_weights(points, displacement, confidence, sample_weights, valid)
    # Validate geometry even for empty observations. Shared support provides
    # strict integer dimensions and physical-origin/spacing validation.
    bspline_support_physical(points[:0], origin, spacing, shape)
    shape = tuple(int(s) for s in shape)
    node_count = math.prod(shape)
    sums = points.new_zeros(group_count*node_count, 4)
    used = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    # Chunk only the 64-node intermediates, never write a control node in a
    # Python point loop. Group offsets separate patients, frames AND chambers.
    for start in range(0, max(len(points), 1), point_chunk_size):
        end = start+point_chunk_size
        select = eligible[start:end]
        p, r = points[start:end][select], displacement[start:end][select]
        index, basis, supported = bspline_support_physical(p, origin, spacing, shape, strict=False)
        row_ids = torch.arange(start, min(end, len(points)), device=points.device)[select]
        used[row_ids] = supported
        magnitude = r.norm(dim=-1, keepdim=True)
        r = r*(max_displacement_mm/magnitude.clamp_min(eps)).clamp_max(1)
        weight = basis*(confidence[start:end][select]*sample_weights[start:end][select])[:, None]
        values = torch.cat((weight[..., None]*r[:, None], weight[..., None]), -1)
        index = index+groups[start:end][select, None]*node_count
        sums.index_add_(0, index.reshape(-1), values.reshape(-1, 4))
    support = sums[:, 3:]
    features = torch.cat((sums[:, :3]/(support+eps), support), -1)
    return features.reshape(group_count, *shape, 4).movedim(-1, 1).contiguous(), used


def splat_to_fine_controls(points_mm, displacement_mm, origin_mm, spacing_mm, grid_shape, *,
                           confidence=None, sample_weights=None, valid=None,
                           max_displacement_mm=15., eps=1e-8, point_chunk_size=8192):
    """One patient/frame/chamber -> [4,Gx,Gy,Gz] observation features.

    Channels are (world dx,dy,dz in mm, accumulated support). Each vote is
    B_j(p)*confidence*sample_weight. Three vector channels divide by total
    support+eps. Default sample weights are one; passing physical arc lengths
    makes support an integrated evidence measure in mm and avoids dependence
    on quadrature density. Support is not a calibrated probability.

    Displacement clipping limits the VECTOR NORM, preserving direction.
    Out-of-support, invalid, zero-weight and nonfinite points are excluded.
    Endpoints and floating roundoff follow the shared physical FFD kernel.
    Coordinates, displacements, confidence and weights retain autograd on
    their accepted piecewise-smooth branches. No inverse FFD is solved.
    """
    groups = torch.zeros(len(points_mm), dtype=torch.long, device=points_mm.device)
    features, _ = _splat_grouped(points_mm, displacement_mm, groups, 1, origin_mm, spacing_mm, grid_shape,
        confidence=confidence, sample_weights=sample_weights, valid=valid,
        max_displacement_mm=max_displacement_mm, eps=eps, point_chunk_size=point_chunk_size)
    return features[0]


def _check_indices(indices, batch_size, frames, device):
    if (not isinstance(indices, torch.Tensor) or indices.ndim != 2 or indices.shape[1] != 4
            or indices.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Canonical integer [K,4] patient/frame/view/chamber indices required')
    indices = indices.to(device=device, dtype=torch.long)
    limits = indices.new_tensor([batch_size, frames, 3, 4])
    if ((indices < 0) | (indices >= limits)).any():
        raise ValueError('Patient/frame/view/chamber index outside declared batch')
    return indices


def splat_observation_residuals(residuals, origin_mm, spacing_mm, grid_shape=(24, 24, 32), *,
                                batch_size, frames, recovery=None, max_displacement_mm=15.,
                                eps=1e-8, point_chunk_size=8192):
    """Pack ObservationResidualExtractor output into [B,T,16,Gx,Gy,Gz].

    Four contiguous channel blocks follow LV,LA,RV,RA; each has dx,dy,dz,
    support. Views of the SAME chamber combine; different chambers/patients/
    frames never combine. Uses residual confidence, FOV validity and physical
    weights_mm. Batch/time dimensions are explicit, even if all cells vanish.

    Missing sections with visible observed boundaries require an explicit
    ``recovery=build_missing_section_recovery(...)`` result. Recovery is
    accepted only for those cells, and provenance/availability flags remain
    in the result. An uninformative SDF or out-of-support anchors can leave
    recovery unavailable; zero support is accompanied by a false availability
    flag. Ordinary in-plane SDF hints and surface-capture hints are different
    evidence types, not exact 3D correspondences.
    """
    if type(batch_size) is not int or type(frames) is not int or min(batch_size, frames) < 1:
        raise ValueError('Explicit positive integer batch_size and frames required')
    points, delta = residuals['points_world_mm'], residuals['displacement_world_mm']
    device = points.device
    cells = _check_indices(residuals['cell_indices'], batch_size, frames, device)
    indices = _check_indices(residuals['point_indices'], batch_size, frames, device)
    has_section = residuals['cell_has_section'].to(device)
    has_boundary = residuals['cell_has_target_boundary'].to(device)
    if (has_section.shape != (len(cells),) or has_boundary.shape != (len(cells),)
            or has_section.dtype != torch.bool or has_boundary.dtype != torch.bool):
        raise ValueError('Boolean section and target-boundary flags must match cells')
    needed = ~has_section & has_boundary
    if needed.any() and recovery is None:
        raise ValueError('Missing observed sections require explicit build_missing_section_recovery output')
    confidence, weight, valid = (residuals[key].to(points) for key in ('confidence', 'weights_mm', 'valid'))
    main_count = len(points)
    if len(indices) != main_count:
        raise ValueError('Point indices must match residual points')
    cell_keys = ((cells[:, 0]*frames+cells[:, 1])*3+cells[:, 2])*4+cells[:, 3]
    recovery_map = torch.zeros(batch_size*frames*12, dtype=torch.bool, device=device)
    recovery_map[cell_keys] = needed
    if recovery is not None:
        ri = _check_indices(recovery['point_indices'], batch_size, frames, device)
        rp, rd = recovery['points_world_mm'].to(points), recovery['displacement_world_mm'].to(points)
        if len(ri) != len(rp) or not recovery_map[((ri[:, 0]*frames+ri[:, 1])*3+ri[:, 2])*4+ri[:, 3]].all():
            raise ValueError('Recovery points must belong to missing sections with observed boundaries')
        points, delta = torch.cat((points, rp)), torch.cat((delta, rd))
        indices = torch.cat((indices, ri))
        confidence = torch.cat((confidence, recovery['confidence'].to(confidence)))
        weight = torch.cat((weight, recovery['weights_mm'].to(weight)))
        valid = torch.cat((valid, recovery['valid'].to(valid)))
    group = (indices[:, 0]*frames+indices[:, 1])*4+indices[:, 3]
    features, used = _splat_grouped(points, delta, group, batch_size*frames*4, origin_mm, spacing_mm, grid_shape,
        confidence=confidence, sample_weights=weight, valid=valid, max_displacement_mm=max_displacement_mm,
        eps=eps, point_chunk_size=point_chunk_size)
    available = torch.zeros_like(recovery_map)
    ri = indices[main_count:][used[main_count:]]
    available[((ri[:, 0]*frames+ri[:, 1])*3+ri[:, 2])*4+ri[:, 3]] = True
    result = dict(features=features.reshape(batch_size, frames, 16, *features.shape[-3:]),
                  channel_names=tuple(c+'_'+f for c in CHAMBERS for f in ('dx_mm', 'dy_mm', 'dz_mm', 'support')),
                  point_used=used[:main_count], recovery_point_used=used[main_count:], cell_indices=cells,
                  cell_has_section=has_section, cell_has_target_boundary=has_boundary,
                  cell_needs_recovery=needed, cell_recovery_available=available[cell_keys])
    for key in ('cell_phases', 'cell_frame_indices', 'cell_case_ids'):
        if key in residuals:
            result[key] = residuals[key]
    return result


def _observed_zero_crossings(sdf, mask, spacing_yx):
    """Internal mask-transition edges sampled at the SDF zero, in local mm.

    No artificial contour is placed on the cropped FOV border. Edge weights
    dy (horizontal crossing) and dx (vertical) approximate physical boundary
    support with orientation bias; these are low-confidence recovery probes,
    not the true curve quadrature used by the normal residual extractor.
    """
    h, w = sdf.shape
    xy, weights = [], []
    for axis in (1, 0):
        first, last = (sdf[:, :-1], sdf[:, 1:]) if axis == 1 else (sdf[:-1], sdf[1:])
        m0, m1 = (mask[:, :-1], mask[:, 1:]) if axis == 1 else (mask[:-1], mask[1:])
        crosses = (m0 != m1) & ((first < 0) != (last < 0))
        y, x = crosses.nonzero(as_tuple=True)
        fraction = first[y, x]/(first[y, x]-last[y, x])
        pixel = torch.stack((x, y), -1).to(sdf)
        pixel[:, 0 if axis == 1 else 1] += fraction
        xy.append((pixel-sdf.new_tensor([w/2, h/2]))*spacing_yx.flip(0))
        weights.append(spacing_yx[0 if axis == 1 else 1].expand(len(pixel)))
    return torch.cat(xy), torch.cat(weights)


@torch.no_grad()
def build_missing_section_recovery(vertices_4d, observations, residuals, *, pose,
                                    chamber_ranges, faces, norm_to_mm, confidence=.25,
                                    query_chunk_size=64):
    """Observation-only nearest-surface capture for missing predicted sections.

    Lift observed MODEL-grid SDF zero crossings to world mm using the explicit
    pose. Anchor votes at the closest points on the CURRENT predicted chamber
    triangles, with delta=observed_point-anchor. Unlike ordinary SDF hints,
    this can contain a plane-normal component. Geometry matching is detached
    (ambiguous correspondences are only low-confidence input features); no GT
    vertices, hidden pose or patient optimisation is used. All splatted votes
    are still clipped/validated by the same splat routine.
    """
    if (vertices_4d.ndim != 4 or vertices_4d.shape[-1] != 3
            or vertices_4d.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(vertices_4d).all() or not math.isfinite(norm_to_mm) or norm_to_mm <= 0
            or not math.isfinite(confidence) or not 0 < confidence <= 1
            or type(query_chunk_size) is not int or query_chunk_size < 1):
        raise ValueError('Finite physical mesh/scale, confidence in (0,1] and positive query chunk required')
    b, t, n, _ = vertices_4d.shape
    units = observations.get('sdf_units')
    if not (units == 'mm' or (isinstance(units, (list, tuple)) and len(units) == b and all(u == 'mm' for u in units))):
        raise ValueError('Recovery SDF must declare physical sdf_units=mm')
    cells = _check_indices(residuals['cell_indices'], b, t, vertices_4d.device)
    availability = observation_validity(observations, b, t, vertices_4d.device)
    if not availability[tuple(cells.T)].all():
        raise ValueError('Recovery cells must be annotated in present views')
    ranges = torch.as_tensor(chamber_ranges)
    if (ranges.shape != (4, 2) or ranges.dtype not in (torch.int32, torch.int64)
            or ranges[0, 0] != 0 or ranges[-1, 1] != n or (ranges[:, 1] <= ranges[:, 0]).any()
            or not torch.equal(ranges[1:, 0], ranges[:-1, 1])):
        raise ValueError('Contiguous canonical four-chamber ranges must match vertices')
    bounds = ranges.cpu().tolist()
    sdf, masks, spacing = (observations[key].to(vertices_4d) for key in ('sdf', 'masks', 'pixel_spacing'))
    rotation, translation = pose['probe_R'].to(vertices_4d), pose['probe_t'].to(vertices_4d)*norm_to_mm
    if (sdf.ndim != 6 or sdf.shape[:4] != (b, t, 3, 4) or masks.shape != sdf.shape
            or min(sdf.shape[-2:]) < 2 or spacing.shape != (b, t, 3, 2)
            or rotation.shape != (b, t, 3, 3, 3) or translation.shape != (b, t, 3, 3)):
        raise ValueError('Recovery observations and explicit pose must match model grid and trajectory')
    has_section, has_boundary = (residuals[key].to(cells.device) for key in ('cell_has_section', 'cell_has_target_boundary'))
    if (has_section.shape != (len(cells),) or has_boundary.shape != (len(cells),)
            or has_section.dtype != torch.bool or has_boundary.dtype != torch.bool):
        raise ValueError('Boolean section and target-boundary flags must match cells')
    selected = ~has_section & has_boundary
    parts = {key: [] for key in ('points_world_mm', 'displacement_world_mm', 'target_points_world_mm',
                                'point_indices', 'weights_mm')}
    for cell in cells[selected].cpu().tolist():
        patient, frame, view, c = cell
        rr, tt, sp = rotation[patient, frame, view], translation[patient, frame, view], spacing[patient, frame, view]
        field, mask = sdf[patient, frame, view, c], masks[patient, frame, view, c]
        eye = torch.eye(3, device=rr.device, dtype=rr.dtype)
        if (not all(torch.isfinite(x).all() for x in (rr, tt, sp, field, mask)) or (sp <= 0).any()
                or not torch.allclose(rr.T@rr, eye, atol=1e-4, rtol=0) or torch.linalg.det(rr) <= 0
                or ((mask != 0) & (mask != 1)).any()):
            raise ValueError('Recovery requires a finite physical SDF, binary mask, positive spacing and proper pose')
        if ((mask.bool() & (field > 0)) | (~mask.bool() & (field < 0))).any():
            raise ValueError('Recovery SDF sign must be negative inside, positive outside the matching mask')
        xy, weight = _observed_zero_crossings(field, mask, sp)
        target = F.pad(xy, (0, 1))@rr.T+tt
        left, right = bounds[c]
        f = torch.as_tensor(faces[CHAMBERS[c]], device=vertices_4d.device)
        if (f.ndim != 2 or f.shape[1] != 3 or not f.numel() or f.dtype not in (torch.int32, torch.int64)
                or f.min() < 0 or f.max() >= right-left):
            raise ValueError('Recovery needs local triangle faces for the selected chamber')
        anchors = closest_surface_points(target, vertices_4d[patient, frame, left:right], f.long(), chunk_size=query_chunk_size)
        parts['points_world_mm'].append(anchors)
        parts['target_points_world_mm'].append(target)
        parts['displacement_world_mm'].append(target-anchors)
        parts['point_indices'].append(cells.new_tensor(cell).expand(len(target), -1))
        parts['weights_mm'].append(weight)
    result = {key: torch.cat(value) if value else vertices_4d.new_empty(
                  (0, 4) if key == 'point_indices' else ((0,) if key == 'weights_mm' else (0, 3)),
                  dtype=torch.long if key == 'point_indices' else vertices_4d.dtype)
              for key, value in parts.items()}
    k = len(result['points_world_mm'])
    result.update(confidence=vertices_4d.new_full((k,), confidence),
                  valid=torch.ones(k, dtype=torch.bool, device=vertices_4d.device),
                  source='observed_sdf_to_predicted_surface_v1')
    return result
