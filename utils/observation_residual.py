"""Observation-driven, physical mesh-section residuals (no learned parameters).

References reviewed: DeepMesh ``train_motion.py`` (world-to-image affine and
2D contour supervision) and ``utils.py::projection``:
https://github.com/ImperialCollegeLondon/DeepMesh/blob/main/train_motion.py
https://github.com/ImperialCollegeLondon/DeepMesh/blob/main/utils.py
DeepMesh's probabilistic vertex projection is NOT a triangle-plane section or
an SDF correction field. Neither its code nor its coordinate conventions are
copied here. We reuse HeartSSM's true sections and arc-length quadrature; the
SDF-normal correction implements the explicitly specified residual formula.

All distances/positions/displacements are mm, gradients are mm/mm. Raster EDT
has pixel bias even at GT; these features are hints, not exact correspondences
or a replacement for Native Exact / paired geometric acceptance criteria.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from data.mask_transform import CHAMBERS
from net.multiview_encoder import observation_validity
from utils.slice_supervision import batched_plane_segments, plane_segments, section_quadrature, packed_section_quadrature


def batched_sdf_residual(sdf, xy_mm, owners, spacing_yx, *, max_distance_mm=15., min_gradient_norm=1e-3):
    """Packed queries -> one physical gradient/grid_sample operation per batch.

    Caller has validated the active SDF/spacing/points. Only padded query
    storage is allocated; images are never replicated per query point.
    """
    cells, h, w = sdf.shape
    count = torch.bincount(owners, minlength=cells)
    size = max(1, int(count.max()))
    starts = count.cumsum(0)-count
    slots = torch.arange(len(owners), device=owners.device)-starts[owners]
    pixel = xy_mm/spacing_yx[owners].flip(-1)+xy_mm.new_tensor([w/2, h/2])
    valid = ((pixel >= 0) & (pixel <= pixel.new_tensor([w-1, h-1]))).all(-1)
    grid = sdf.new_zeros(cells, 1, size, 2)
    grid[owners, 0, slots] = 2*pixel/pixel.new_tensor([w-1, h-1])-1
    gy, gx = torch.gradient(sdf, dim=(1, 2))
    image = torch.stack((sdf, gx/spacing_yx[:, 1, None, None], gy/spacing_yx[:, 0, None, None]), 1)
    sampled = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    sampled = sampled[:, :, 0].transpose(1, 2)[owners, slots]
    distance, gradient = sampled[:, 0], sampled[:, 1:]
    norm = gradient.norm(dim=-1)
    confidence = valid & (norm > min_gradient_norm)
    displacement = -distance.clamp(-max_distance_mm, max_distance_mm)[:, None]*gradient/norm.clamp_min(1e-6)[:, None]
    return dict(displacement_xy_mm=torch.where(confidence[:, None], displacement, 0.),
                signed_distance_mm=distance, gradient_xy=gradient, gradient_norm=norm,
                valid=valid, confidence=confidence)


def sample_sdf_residual(sdf_hw, xy_mm, spacing_yx, *, max_distance_mm=15.,
                        min_gradient_norm=1e-3, gradient_eps=1e-6):
    """Sample one physical SDF and its gradient at [K,2] plane-local points.

    Pixel centres follow x_pixel=x_mm/dx+W/2 (likewise y), with bilinear
    ``grid_sample(..., align_corners=True)``. ``valid`` means within pixel
    centres [0,W-1] x [0,H-1]; border extrapolation is never confident. Return
    distance, raw gradient/norm, validity and confidence separately. Flat or
    clipped SDF plateaus produce zero displacement and false confidence.
    Out-of-FOV distance/gradient values are border diagnostics, not physical
    boundary measurements; downstream aggregation must use confidence.
    ``confidence`` is a boolean usability flag, not a calibrated probability.

    SDF must already be in mm on the grid described by spacing_yx. Gradients
    through sample coordinates and SDF values remain ordinary PyTorch graphs;
    clipping, validity and normalisation are piecewise differentiable.
    """
    if (sdf_hw.ndim != 2 or min(sdf_hw.shape) < 2 or xy_mm.ndim != 2 or xy_mm.shape[1] != 2
            or sdf_hw.dtype not in (torch.float32, torch.float64)
            or xy_mm.dtype != sdf_hw.dtype or xy_mm.device != sdf_hw.device):
        raise ValueError('Matching float32/float64 SDF [H,W>=2] and points [K,2] required')
    spacing = torch.as_tensor(spacing_yx, device=sdf_hw.device, dtype=sdf_hw.dtype)
    if (spacing.shape != (2,) or not torch.isfinite(spacing).all() or (spacing <= 0).any()
            or not torch.isfinite(sdf_hw).all() or not torch.isfinite(xy_mm).all()):
        raise ValueError('Finite SDF/points and positive finite y,x spacing in mm required')
    if (not math.isfinite(max_distance_mm) or max_distance_mm <= 0
            or not math.isfinite(min_gradient_norm) or min_gradient_norm < 0
            or not math.isfinite(gradient_eps) or gradient_eps <= 0):
        raise ValueError('Positive finite distance/epsilon and nonnegative gradient threshold required')
    h, w = sdf_hw.shape
    dy, dx = spacing.unbind()
    pixel = xy_mm / torch.stack((dx, dy)) + xy_mm.new_tensor([w/2, h/2])
    valid = ((pixel >= 0) & (pixel <= pixel.new_tensor([w-1, h-1]))).all(-1)
    grid = (2*pixel / pixel.new_tensor([w-1, h-1])-1).reshape(1, 1, -1, 2)
    # Scalar tensor spacing is not interpreted as a scalar by torch.gradient
    # on supported PyTorch versions. Divide index derivatives by physical mm.
    gy, gx = torch.gradient(sdf_hw, dim=(0, 1))
    image = torch.stack((sdf_hw, gx/dx, gy/dy))[None]
    if len(xy_mm):
        sampled = F.grid_sample(image, grid, mode='bilinear', padding_mode='border',
                                align_corners=True)[0, :, 0].T
    else:
        sampled = sdf_hw.new_empty((0, 3))
    distance, gradient = sampled[:, 0], sampled[:, 1:]
    norm = gradient.norm(dim=-1)
    confidence = valid & (norm > min_gradient_norm)
    displacement = -distance.clamp(-max_distance_mm, max_distance_mm)[:, None] * (
        gradient / norm.clamp_min(gradient_eps)[:, None])
    displacement = torch.where(confidence[:, None], displacement, torch.zeros_like(displacement))
    return dict(displacement_xy_mm=displacement, signed_distance_mm=distance,
                gradient_xy=gradient, gradient_norm=norm, valid=valid, confidence=confidence)


class ObservationResidualExtractor(nn.Module):
    """Pack true-section SDF hints with location, direction and cell identity.

    Input vertices are physical [B,T,N,3]. Observations use the Dataset's MODEL
    ``masks/sdf/pixel_spacing`` grid, canonical [B,T,3,4,H,W] channels and
    explicit ``sdf_units='mm'``. No native-grid spacing or GT vertices are used.
    Pose is required explicitly: synthetic callers pass
    ``extractor(vertices, observations, pose=observations['pose_gt'])``; real
    callers pass an estimated pose. In both cases probe_R columns are local
    x,y,normal, and probe_t is in source-normalised units (multiplied exactly
    once by norm_to_mm). No probe_s scaling applies to vectors in mm.

    Point fields are packed in lexicographic [patient,frame,view,chamber]
    order. ``cell_offsets[i:i+2]`` selects a cell's points; ``point_cell_ids``
    maps points to ``cell_indices`` and metadata. Arc-length weights (mm) are
    retained for later B-spline aggregation. Empty/full observed masks have
    no usable boundary direction; missing predicted sections have zero points
    and an explicit cell flag. This module does not invent recovery vectors.

    Discrete section topology and sample counts are locally constant; section
    coordinates, residuals and quadrature weights retain autograd. The module
    has geometry buffers only, with no patient parameters or optimisation.
    """
    def __init__(self, chamber_ranges, faces, norm_to_mm, *, max_distance_mm=15.,
                 sample_step_mm=None, min_gradient_norm=1e-3,
                 execution_backend='batched', cell_chunk_size=16):
        super().__init__()
        ranges = torch.as_tensor(chamber_ranges)
        if (ranges.shape != (4, 2) or ranges.dtype not in (torch.int32, torch.int64)
                or ranges[0, 0] != 0 or (ranges[:, 1] <= ranges[:, 0]).any()
                or not torch.equal(ranges[1:, 0], ranges[:-1, 1])):
            raise ValueError('Four contiguous nonempty canonical chamber ranges required')
        if (not math.isfinite(norm_to_mm) or norm_to_mm <= 0
                or not math.isfinite(max_distance_mm) or max_distance_mm <= 0
                or not math.isfinite(min_gradient_norm) or min_gradient_norm < 0
                or (sample_step_mm is not None and (not math.isfinite(sample_step_mm) or sample_step_mm <= 0))
                or execution_backend not in ('serial', 'batched')
                or type(cell_chunk_size) is not int or cell_chunk_size < 1):
            raise ValueError('Positive physical scales/step/chunk, valid gradient threshold and section backend required')
        self.register_buffer('chamber_ranges', ranges.long().clone())
        self._ranges = tuple(map(tuple, ranges.cpu().tolist()))
        for chamber, (left, right) in zip(CHAMBERS, self._ranges):
            f = torch.as_tensor(faces[chamber])
            if (f.ndim != 2 or f.shape[1] != 3 or not f.numel()
                    or f.dtype not in (torch.int32, torch.int64) or f.min() < 0 or f.max() >= right-left):
                raise ValueError('Local integer triangle faces required for each chamber')
            self.register_buffer('faces_'+chamber, f.long().clone())
        self.norm_to_mm, self.max_distance_mm = float(norm_to_mm), float(max_distance_mm)
        self.sample_step_mm, self.min_gradient_norm = sample_step_mm, min_gradient_norm
        self.execution_backend, self.cell_chunk_size = execution_backend, cell_chunk_size

    def forward(self, vertices_4d, observations, *, pose):
        if (vertices_4d.ndim != 4 or vertices_4d.shape[-1] != 3
                or vertices_4d.dtype not in (torch.float32, torch.float64)
                or vertices_4d.shape[2] != self._ranges[-1][1]
                or not torch.isfinite(vertices_4d).all()):
            raise ValueError('Finite physical float32/float64 vertices [B,T,N,3] matching topology required')
        b, t = vertices_4d.shape[:2]
        units = observations.get('sdf_units')
        if not (units == 'mm' or (isinstance(units, (list, tuple)) and len(units) == b
                                  and all(u == 'mm' for u in units))):
            raise ValueError('Explicit sdf_units=mm required; dimensionless SDF is not a physical distance')
        masks, sdf = (observations[key].to(vertices_4d) for key in ('masks', 'sdf'))
        spacing = observations['pixel_spacing'].to(vertices_4d)
        rotation = pose['probe_R'].to(vertices_4d)
        translation = pose['probe_t'].to(vertices_4d)*self.norm_to_mm
        phases = observations['phases'].to(vertices_4d)
        if (masks.ndim != 6 or masks.shape[:4] != (b, t, 3, 4) or min(masks.shape[-2:]) < 2
                or sdf.shape != masks.shape or spacing.shape != (b, t, 3, 2)
                or rotation.shape != (b, t, 3, 3, 3) or translation.shape != (b, t, 3, 3)
                or phases.shape != (b, t) or not torch.isfinite(phases).all()):
            raise ValueError('Observation model grid, pose, phase and trajectory shapes differ')
        valid_cells = observation_validity(observations, b, t, vertices_4d.device)
        active = valid_cells.any(-1)
        r, tr, sp = rotation[active], translation[active], spacing[active]
        eye = torch.eye(3, device=r.device, dtype=r.dtype)
        if (not all(torch.isfinite(x).all() for x in (r, tr, sp, masks[valid_cells], sdf[valid_cells]))
                or (sp <= 0).any() or ((r.transpose(-1, -2)@r-eye).abs() > 1e-4).any()
                or (torch.linalg.det(r) <= 0).any()
                or ((masks[valid_cells] != 0) & (masks[valid_cells] != 1)).any()):
            raise ValueError('Finite physical observations, binary masks, positive spacing and proper rotations required')
        cells = valid_cells.nonzero()
        entries = cells.cpu().tolist()
        curves = [None]*len(entries)
        segment_packs = []
        # Batch only cells sharing local chamber topology. Pack output back in
        # canonical order; padded segments never enter arc-length quadrature.
        for c, (left, right) in enumerate(self._ranges):
            selected = [i for i, cell in enumerate(entries) if cell[3] == c]
            faces = getattr(self, 'faces_'+CHAMBERS[c]).to(vertices_4d.device)
            chunk = 1 if self.execution_backend == 'serial' else self.cell_chunk_size
            for start in range(0, len(selected), chunk):
                ids = selected[start:start+chunk]
                patient, frame, view, _ = cells[ids].unbind(-1)
                v = vertices_4d[patient, frame, left:right]
                rr, tt = rotation[patient, frame, view], translation[patient, frame, view]
                if self.execution_backend == 'serial':
                    curves[ids[0]] = plane_segments(v[0], faces, rr[0], tt[0])
                else:
                    segments, present = batched_plane_segments(v, faces, rr, tt)
                    segment_packs.append((ids, segments, present))
        if self.execution_backend == 'batched' and entries:
            size = max(pack[1].shape[1] for pack in segment_packs)
            segments = vertices_4d.new_zeros(len(entries), size, 2, 2)
            present = valid_cells.new_zeros(len(entries), size)
            for ids, segment, mask in segment_packs:
                segments[ids, :segment.shape[1]] = segment
                present[ids, :mask.shape[1]] = mask
            return self._batched_features(segments, present, cells, sdf, masks, spacing, rotation, translation, observations)
        fields = {key: [] for key in ('points_xy_mm', 'points_world_mm', 'displacement_xy_mm',
                  'displacement_world_mm', 'signed_distance_mm', 'gradient_xy', 'gradient_norm',
                  'valid', 'confidence', 'weights_mm')}
        offsets, point_owners, has_section, has_boundary = [0], [], [], []
        for i, (patient, frame, view, c) in enumerate(entries):
            segments = curves[i]
            step = spacing[patient, frame, view].min().detach()
            if self.sample_step_mm is not None:
                step = torch.minimum(step, step.new_tensor(self.sample_step_mm))
            xy, weights = section_quadrature(segments, step)
            sampled = sample_sdf_residual(sdf[patient, frame, view, c], xy, spacing[patient, frame, view],
                                         max_distance_mm=self.max_distance_mm,
                                         min_gradient_norm=self.min_gradient_norm)
            mask = masks[patient, frame, view, c]
            boundary = mask.any() & ~mask.bool().all()
            sampled['confidence'] = sampled['confidence'] & boundary
            sampled['displacement_xy_mm'] = torch.where(sampled['confidence'][:, None],
                sampled['displacement_xy_mm'], torch.zeros_like(xy))
            rr, tt = rotation[patient, frame, view], translation[patient, frame, view]
            fields['points_xy_mm'].append(xy)
            fields['points_world_mm'].append(F.pad(xy, (0, 1))@rr.T+tt)
            fields['displacement_world_mm'].append(F.pad(sampled['displacement_xy_mm'], (0, 1))@rr.T)
            fields['weights_mm'].append(weights)
            for key, value in sampled.items():
                fields[key].append(value)
            offsets.append(offsets[-1]+len(xy))
            point_owners.extend([i]*len(xy))
            has_section.append(bool(len(segments)))
            has_boundary.append(boundary)
        result = {}
        for key, parts in fields.items():
            width = 3 if key in ('points_world_mm', 'displacement_world_mm') else (
                2 if key in ('points_xy_mm', 'displacement_xy_mm', 'gradient_xy') else None)
            shape = (0,) if width is None else (0, width)
            result[key] = torch.cat(parts) if parts else vertices_4d.new_empty(
                shape, dtype=torch.bool if key in ('valid', 'confidence') else vertices_4d.dtype)
        result.update(cell_indices=cells, cell_index_columns=('patient', 'frame', 'view', 'chamber'),
                      cell_offsets=cells.new_tensor(offsets), point_cell_ids=cells.new_tensor(point_owners),
                      cell_has_section=cells.new_tensor(has_section, dtype=torch.bool),
                      cell_has_target_boundary=torch.stack(has_boundary) if has_boundary else valid_cells.new_empty(0),
                      cell_phases=phases[cells[:, 0], cells[:, 1]])
        result['point_indices'] = cells[result['point_cell_ids']]
        result['point_phases'] = result['cell_phases'][result['point_cell_ids']]
        if 'frame_indices' in observations:
            frames = observations['frame_indices'].to(cells.device)
            if frames.shape != (b, t):
                raise ValueError('Source frame indices must match [B,T]')
            result['cell_frame_indices'] = frames[cells[:, 0], cells[:, 1]]
        if 'case_id' in observations:
            if len(observations['case_id']) != b:
                raise ValueError('One case ID per patient required')
            result['cell_case_ids'] = [observations['case_id'][cell[0]] for cell in entries]
        return result

    def _batched_features(self, segments, present, cells, sdf, masks, spacing, rotation, translation, observations):
        patient, frame, view, chamber = cells.unbind(-1)
        sp = spacing[patient, frame, view]
        steps = sp.amin(-1).detach()
        if self.sample_step_mm is not None:
            steps = steps.clamp_max(self.sample_step_mm)
        xy, weights, owners = packed_section_quadrature(segments, present, steps)
        result = batched_sdf_residual(sdf[patient, frame, view, chamber], xy, owners, sp,
            max_distance_mm=self.max_distance_mm, min_gradient_norm=self.min_gradient_norm)
        target = masks[patient, frame, view, chamber].bool().flatten(1)
        has_boundary = target.any(-1) & ~target.all(-1)
        result['confidence'] = result['confidence'] & has_boundary[owners]
        result['displacement_xy_mm'] = torch.where(result['confidence'][:, None], result['displacement_xy_mm'], 0.)
        rr = rotation[patient, frame, view][owners]
        tr = translation[patient, frame, view][owners]
        local = F.pad(xy, (0, 1))
        delta = F.pad(result['displacement_xy_mm'], (0, 1))
        result.update(points_xy_mm=xy, points_world_mm=torch.bmm(rr, local[..., None])[..., 0]+tr,
            displacement_world_mm=torch.bmm(rr, delta[..., None])[..., 0], weights_mm=weights,
            cell_indices=cells, cell_index_columns=('patient', 'frame', 'view', 'chamber'),
            cell_offsets=torch.cat((cells.new_zeros(1), torch.bincount(owners, minlength=len(cells)).cumsum(0))),
            point_cell_ids=owners, point_indices=cells[owners], cell_has_section=present.any(-1),
            cell_has_target_boundary=has_boundary, cell_phases=observations['phases'].to(xy)[patient, frame])
        result['point_phases'] = result['cell_phases'][owners]
        if 'frame_indices' in observations:
            frames = observations['frame_indices'].to(cells.device)
            if frames.shape != sdf.shape[:2]:
                raise ValueError('Source frame indices must match [B,T]')
            result['cell_frame_indices'] = frames[patient, frame]
        if 'case_id' in observations:
            if len(observations['case_id']) != sdf.shape[0]:
                raise ValueError('One case ID per patient required')
            result['cell_case_ids'] = [observations['case_id'][p] for p in patient.cpu().tolist()]
        return result
