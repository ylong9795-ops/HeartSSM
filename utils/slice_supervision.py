"""True mesh-plane sections with piecewise differentiable 2D distances.

GHDHeart/DeepMesh sparse-observation objectives were reviewed first. Their
occupancy quadrature/projection implementations are not copied. This minimal
operator intersects mesh edges with a physical plane, then uses a sigmoid of
the signed distance to the resulting section segments. It is NOT a silhouette
projection, winding-number gradient, or hard/soft straight-through estimator.
Discrete crossing/nearest-segment choices are locally constant; continuous
intersection coordinates/distances retain ordinary PyTorch gradients.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.mask_transform import CHAMBERS
from net.multiview_encoder import observation_validity


OBSERVATION_KINDS = ('paired_boundary', 'recovery', 'matched_empty',
                     'empty_target', 'no_visible_gt_boundary')


class MissingSectionError(ValueError):
    pass


def plane_segments(vertices_mm, faces, rotation, translation_mm):
    """Return [S,2,2] segments in plane-local mm; R columns are x,y,normal.

    Positive/nonpositive half-open classification handles on-plane edges.
    Entire coplanar triangles have no unique section curve and are exposed.
    """
    if (vertices_mm.ndim!=2 or vertices_mm.shape[-1]!=3 or faces.ndim!=2 or faces.shape[-1]!=3
            or rotation.shape!=(3,3) or translation_mm.shape!=(3,)
            or not torch.isfinite(vertices_mm).all() or not torch.isfinite(rotation).all()
            or not torch.isfinite(translation_mm).all()):
        raise ValueError('Finite mesh and physical plane coordinates required')
    eye = torch.eye(3,device=rotation.device,dtype=rotation.dtype)
    if not torch.allclose(rotation.T@rotation,eye,atol=1e-4,rtol=0) or torch.linalg.det(rotation)<=0:
        raise ValueError('Plane rotation must be right-handed orthonormal')
    local = (vertices_mm-translation_mm)@rotation
    tri = local[faces]
    d = tri[...,2]
    if (d.abs().amax(-1)<1e-9).any():
        raise ValueError('Coplanar triangle: section derivative is ambiguous')
    edge_pairs = torch.tensor([[0,1],[1,2],[2,0]],device=faces.device)
    side = d>0
    crossing = side[:,edge_pairs[:,0]] != side[:,edge_pairs[:,1]]
    selected = crossing.any(-1)
    if not selected.any():
        return local.new_empty((0,2,2))
    tri,d,crossing = tri[selected],d[selected],crossing[selected]
    if not (crossing.sum(-1)==2).all():
        raise ValueError('Invalid triangle-plane crossing topology')
    pairs = edge_pairs[crossing.nonzero()[:,1]].reshape(-1,2,2)
    owner = torch.arange(len(tri),device=tri.device)[:,None]
    a,b = tri[owner,pairs[...,0]],tri[owner,pairs[...,1]]
    fraction = a[...,2]/(a[...,2]-b[...,2])
    intersection = a[...,:2]+fraction[...,None]*(b[...,:2]-a[...,:2])
    nonzero = (intersection[:,1]-intersection[:,0]).square().sum(-1)>1e-16
    return intersection[nonzero]


def signed_segment_distance(points, segments, chunk_size=256):
    """In-plane signed Euclidean distance (mm), negative inside by odd parity.

    Nearest segment search is done without recording P*S intermediates; only
    the selected segment is evaluated with gradients. This is the exact
    derivative of min-distance almost everywhere, not a surrogate backward.
    """
    if points.ndim!=2 or points.shape[-1]!=2 or segments.ndim!=3 or segments.shape[1:]!=(2,2) or not len(segments):
        raise ValueError('2D points and nonempty plane segments required')
    if chunk_size<1:
        raise ValueError('Positive section query chunk required')
    distances = []
    for p in points.split(chunk_size):
        with torch.no_grad():
            a,b = segments[:,0],segments[:,1]
            ab = b-a
            u = ((p[:,None]-a)*ab).sum(-1)/ab.square().sum(-1).clamp_min(1e-16)
            closest = a+u.clamp(0,1)[...,None]*ab
            nearest = (p[:,None]-closest).square().sum(-1).argmin(-1)
            dy = ab[:,1]
            denominator = torch.where(dy.abs()>1e-16,dy,torch.ones_like(dy))
            x_cross = a[:,0]+(p[:,None,1]-a[:,1])*ab[:,0]/denominator
            crosses = ((a[:,1]>p[:,None,1])!=(b[:,1]>p[:,None,1])) & (p[:,None,0]<x_cross)
            sign = 1-2*(crosses.sum(-1)%2).to(p)
        chosen = segments[nearest]
        a,b = chosen[:,0],chosen[:,1]
        ab = b-a
        u = ((p-a)*ab).sum(-1)/ab.square().sum(-1).clamp_min(1e-16)
        closest = a+u.clamp(0,1)[...,None]*ab
        distances.append(sign*((p-closest).square().sum(-1)+1e-12).sqrt())
    return torch.cat(distances)


def batched_plane_segments(vertices_mm, faces, rotations, translations_mm):
    """Pack independent true sections into [C,S,2,2], with padding validity.

    Crossing edges retain triangle/edge order from plane_segments. Only the
    discrete section size is synchronized; intersection coordinates stay in
    the ordinary autograd graph. Callers validate physical poses once per batch.
    """
    local = torch.bmm(vertices_mm-translations_mm[:,None], rotations)
    tri = local[:,faces]
    d = tri[...,2]
    if (d.abs().amax(-1)<1e-9).any():
        raise ValueError('Coplanar triangle: section derivative is ambiguous')
    edge_pairs = torch.tensor([[0,1],[1,2],[2,0]],device=faces.device)
    side = d>0
    crossing = side[...,edge_pairs[:,0]] != side[...,edge_pairs[:,1]]
    selected = crossing.any(-1)
    # A binary triangle always has either zero or two crossing edges.
    owners, triangles = selected.nonzero(as_tuple=True)
    selected_tri, crosses = tri[owners,triangles], crossing[owners,triangles]
    if not len(owners):
        return local.new_zeros(len(local),1,2,2), selected.new_zeros(len(local),1)
    pairs = edge_pairs[crosses.nonzero()[:,1]].reshape(-1,2,2)
    rows = torch.arange(len(owners),device=tri.device)[:,None]
    a,b = selected_tri[rows,pairs[...,0]], selected_tri[rows,pairs[...,1]]
    fraction = a[...,2]/(a[...,2]-b[...,2])
    intersection = a[...,:2]+fraction[...,None]*(b[...,:2]-a[...,:2])
    nonzero = (intersection[:,1]-intersection[:,0]).square().sum(-1)>1e-16
    counts = selected.sum(-1)
    slots = selected.cumsum(-1)[owners,triangles]-1
    size = int(counts.max())
    segments = local.new_zeros(len(local),size,2,2)
    segments[owners,slots] = intersection
    present = selected.new_zeros(len(local),size)
    present[owners,slots] = nonzero
    return segments,present


def batched_signed_segment_distance(points, segments, present, chunk_size=256, *, signed=True):
    """Same nearest-segment distance and odd parity, batched over section cells.

    Invalid padded/zero-length segments participate in neither nearest search
    nor parity. As in the serial operator, the nearest choice is detached and
    distances through the chosen intersections remain differentiable.
    """
    distances = []
    for p in points.split(chunk_size,dim=1):
        with torch.no_grad():
            a,b = segments[:,:,0],segments[:,:,1]
            ab = b-a
            delta = p[:,:,None]-a[:,None]
            u = (delta*ab[:,None]).sum(-1)/ab.square().sum(-1)[:,None].clamp_min(1e-16)
            closest = a[:,None]+u.clamp(0,1)[...,None]*ab[:,None]
            squared = (p[:,:,None]-closest).square().sum(-1)
            nearest = squared.masked_fill(~present[:,None],float('inf')).argmin(-1)
            if signed:
                dy = ab[...,1]
                denominator = torch.where(dy.abs()>1e-16,dy,torch.ones_like(dy))
                x_cross = a[:,None,:,0]+(p[:,:,None,1]-a[:,None,:,1])*ab[:,None,:,0]/denominator[:,None]
                crosses = (((a[:,None,:,1]>p[:,:,None,1]) != (b[:,None,:,1]>p[:,:,None,1]))
                           & (p[:,:,None,0]<x_cross) & present[:,None])
                sign = 1-2*(crosses.sum(-1)%2).to(p)
            else:
                sign = 1.
        chosen = segments.gather(1,nearest[:,:,None,None].expand(-1,-1,2,2))
        a,b = chosen[:,:,0],chosen[:,:,1]
        ab = b-a
        u = ((p-a)*ab).sum(-1)/ab.square().sum(-1).clamp_min(1e-16)
        closest = a+u.clamp(0,1)[...,None]*ab
        distances.append(sign*((p-closest).square().sum(-1)+1e-12).sqrt())
    return torch.cat(distances,dim=1)



def section_quadrature(segments, step_mm):
    """Arc-length midpoint quadrature, at most one native pixel apart.

    Integer sample counts are locally constant. Coordinates and arc-length
    integration weights retain gradients, including for nonrigid curve changes.
    No maximum silently drops small or complicated section components.
    """
    length = (segments[:, 1]-segments[:, 0]).norm(dim=-1)
    counts = (length/step_mm).ceil().long().clamp_min(1)
    owners = torch.repeat_interleave(torch.arange(len(segments), device=segments.device), counts)
    offsets = counts.cumsum(0)-counts
    slot = torch.arange(len(owners), device=segments.device)-offsets[owners]
    fraction = (slot.to(segments)+.5)/counts[owners]
    points = segments[owners, 0]+fraction[:, None]*(segments[owners, 1]-segments[owners, 0])
    return points, length[owners]/counts[owners]


def clip_section_to_native(segments, spacing_yx, image_shape):
    """Clip curves to the physical support of native pixel centres/cells.

    Pixel (0,0) is (-W/2*dx,-H/2*dy); its edges extend half a pixel.
    Intersections with the FOV retain ordinary piecewise derivatives.
    """
    h, w = image_shape
    spacing_xy = spacing_yx.flip(0)
    lower = segments.new_tensor([-w/2-.5, -h/2-.5])*spacing_xy
    upper = segments.new_tensor([w/2-.5, h/2-.5])*spacing_xy
    a, delta = segments[:, 0], segments[:, 1]-segments[:, 0]
    parallel = delta.abs() < 1e-12
    safe = torch.where(parallel, torch.ones_like(delta), delta)
    first, last = (lower-a)/safe, (upper-a)/safe
    enter = torch.where(parallel, float('-inf'), torch.minimum(first, last)).amax(-1).clamp_min(0)
    leave = torch.where(parallel, float('inf'), torch.maximum(first, last)).amin(-1).clamp_max(1)
    keep = (leave > enter) & ~(parallel & ((a < lower) | (a > upper))).any(-1)
    return torch.stack((a+enter[:, None]*delta, a+leave[:, None]*delta), 1)[keep]


def _smooth_distance(distance, beta_mm):
    # signed_segment_distance adds 1e-12 under its square root. Remove that
    # numerical floor so identical curves have exactly zero loss.
    squared = (distance.square()-1e-12).clamp_min(0)
    return squared/((squared+beta_mm**2).sqrt()+beta_mm)


def batched_clip_section_to_native(segments, present, spacings_yx, image_shape):
    """Same physical clipping as clip_section_to_native; preserve padded cells."""
    h, w = image_shape
    spacing = spacings_yx.flip(-1)[:, None]
    lower = segments.new_tensor([-w/2-.5, -h/2-.5])*spacing
    upper = segments.new_tensor([w/2-.5, h/2-.5])*spacing
    a, delta = segments[:, :, 0], segments[:, :, 1]-segments[:, :, 0]
    parallel = delta.abs() < 1e-12
    safe = torch.where(parallel, torch.ones_like(delta), delta)
    first, last = (lower-a)/safe, (upper-a)/safe
    enter = torch.where(parallel, float('-inf'), torch.minimum(first, last)).amax(-1).clamp_min(0)
    leave = torch.where(parallel, float('inf'), torch.maximum(first, last)).amin(-1).clamp_max(1)
    keep = present & (leave > enter) & ~(parallel & ((a < lower) | (a > upper))).any(-1)
    clipped = torch.stack((a+enter[..., None]*delta, a+leave[..., None]*delta), 2)
    return torch.where(keep[..., None, None], clipped, 0.), keep


def packed_section_quadrature(segments, present, steps_mm):
    """Physical quadrature across cells; coordinates and weights retain gradients."""
    cell, slot = present.nonzero(as_tuple=True)
    curves = segments[cell, slot]
    length = (curves[:, 1]-curves[:, 0]).norm(dim=-1)
    counts = (length/steps_mm[cell]).ceil().long().clamp_min(1)
    owners = torch.repeat_interleave(torch.arange(len(curves), device=segments.device), counts)
    offsets = counts.cumsum(0)-counts
    sample = torch.arange(len(owners), device=segments.device)-offsets[owners]
    fraction = (sample.to(segments)+.5)/counts[owners]
    points = curves[owners, 0]+fraction[:, None]*(curves[owners, 1]-curves[owners, 0])
    return points, length[owners]/counts[owners], cell[owners]


def batched_paired_boundary_distance(predicted, pred_present, teacher, teacher_present,
                                     spacings_yx, image_shape, *, beta_mm=1., chunk_size=256,
                                     sample_step_mm=None):
    """Equivalent symmetric boundary objective batched over nonempty sections.

    The smooth distance uses squared unsigned distances, so ray parity is not
    needed. Missing/empty cells keep the supervisor's recovery/clearance paths.
    """
    teacher = teacher.detach()
    observed_pred, pred_visible = batched_clip_section_to_native(predicted, pred_present, spacings_yx, image_shape)
    observed_gt, gt_visible = batched_clip_section_to_native(teacher, teacher_present, spacings_yx, image_shape)
    step = spacings_yx.amin(-1).detach()
    if sample_step_mm is not None:
        step = torch.minimum(step, step.new_tensor(sample_step_mm))
    cell_count = len(predicted)
    parts, available = [], []
    for source, source_present, destination, destination_present in (
            (observed_gt, gt_visible, predicted, pred_present),
            (observed_pred, pred_visible, teacher, teacher_present)):
        points, weight, owner = packed_section_quadrature(source, source_present, step)
        counts = torch.bincount(owner, minlength=cell_count)
        size = max(1, int(counts.max()))
        starts = counts.cumsum(0)-counts
        slots = torch.arange(len(points), device=points.device)-starts[owner]
        queries = points.new_zeros(cell_count, size, 2)
        weights = weight.new_zeros(cell_count, size)
        queries[owner, slots], weights[owner, slots] = points, weight
        distance = batched_signed_segment_distance(queries, destination, destination_present, chunk_size, signed=False)
        denominator = weights.sum(-1)
        parts.append((_smooth_distance(distance, beta_mm)*weights).sum(-1)/denominator.clamp_min(1e-20))
        available.append(denominator > 0)
    return torch.stack(parts).sum(0)/torch.stack(available).sum(0).clamp_min(1)


def paired_boundary_distance(predicted, teacher, spacing_yx, image_shape, *, beta_mm=1., chunk_size=256,
                             sample_step_mm=None):
    """Symmetric physical curve distance (mm), with a smooth zero at equality.

    GT->prediction covers all observed GT curve pieces. Prediction->GT covers
    predicted pieces inside the native FOV. Distances use actual segments,
    not a raster boundary or a binary-mask EDT. Detached GT uses the same
    intersection operator, so rasterisation bias cannot push an exact GT away.
    """
    teacher = teacher.detach()
    observed_gt = clip_section_to_native(teacher, spacing_yx, image_shape)
    observed_pred = clip_section_to_native(predicted, spacing_yx, image_shape)
    parts = []
    step = spacing_yx.min().detach()
    if sample_step_mm is not None:
        step = torch.minimum(step, step.new_tensor(sample_step_mm))
    for source, destination in ((observed_gt, predicted), (observed_pred, teacher)):
        if len(source):
            points, weight = section_quadrature(source, step)
            distance = signed_segment_distance(points, destination, chunk_size)
            parts.append((_smooth_distance(distance, beta_mm)*weight).sum()/weight.sum())
    return torch.stack(parts).mean() if parts else predicted.sum()*0


def closest_surface_points(points_mm, vertices_mm, faces, *, chunk_size=64):
    """Closest predicted triangle points, sharing the surface-capture kernel.

    Face/barycentric choices are detached. Reconstructed points retain the
    selected barycentric vertex graph. This gives the correct distance-loss
    derivative by the envelope theorem, not a general derivative of the full
    closest-point map. Feature-only recovery callers can use no_grad().
    """
    if chunk_size < 1:
        raise ValueError('Positive closest-surface query chunk required')
    closest_points = []
    for points in points_mm.detach().split(chunk_size):
        with torch.no_grad():
            tri = vertices_mm.detach()[faces]
            a, u, v = tri[:, 0], tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
            uu, uv, vv = (u*u).sum(-1), (u*v).sum(-1), (v*v).sum(-1)
            determinant = uu*vv-uv.square()
            delta = points[:, None]-a
            du, dv = (delta*u).sum(-1), (delta*v).sum(-1)
            s = (du*vv-dv*uv)/determinant.clamp_min(1e-20)
            t = (dv*uu-du*uv)/determinant.clamp_min(1e-20)
            bary = torch.stack((1-s-t, s, t), -1)
            candidates = [(bary, (s >= 0) & (t >= 0) & (s+t <= 1) & (determinant > 1e-20))]
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = tri[:, j]-tri[:, i]
                fraction = ((points[:, None]-tri[:, i])*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-20)
                fraction = fraction.clamp(0, 1)
                bary_edge = points.new_zeros(len(points), len(tri), 3)
                bary_edge[..., i], bary_edge[..., j] = 1-fraction, fraction
                candidates.append((bary_edge, torch.ones_like(fraction, dtype=torch.bool)))
            best = points.new_full((len(points),), float('inf'))
            face_ids = torch.zeros(len(points), device=faces.device, dtype=torch.long)
            selected_bary = points.new_zeros(len(points), 3)
            rows = torch.arange(len(points), device=faces.device)
            for bary, allowed in candidates:
                closest = (bary[..., None]*tri[None]).sum(-2)
                squared = (points[:, None]-closest).square().sum(-1).masked_fill(~allowed, float('inf'))
                value, ids = squared.min(-1)
                improve = value < best
                best = torch.minimum(best, value)
                face_ids = torch.where(improve, ids, face_ids)
                selected_bary = torch.where(improve[:, None], bary[rows, ids], selected_bary)
        closest = (vertices_mm[faces[face_ids]]*selected_bary[..., None]).sum(-2)
        closest_points.append(closest)
    return torch.cat(closest_points) if closest_points else vertices_mm.new_empty((0, 3))


def surface_capture_distance(points_mm, vertices_mm, faces, *, beta_mm=1., chunk_size=64, weights=None):
    """Target boundary -> closest triangle distance for an absent section.

    By the envelope theorem, detached nearest-face/barycentric choices give
    the ordinary distance derivative almost everywhere. This supplies normal
    capture and in-plane localisation without fitting to GT vertices.
    """
    closest = closest_surface_points(points_mm, vertices_mm, faces, chunk_size=chunk_size)
    squared = (points_mm.detach()-closest).square().sum(-1)
    distances = squared/((squared+beta_mm**2).sqrt()+beta_mm)
    return distances.mean() if weights is None else (distances*weights).sum()/weights.sum()


def empty_section_clearance(vertices_mm, teacher_mm, rotation, translation_mm, spacing_yx, *, beta_mm=1.):
    """Suppress a spurious section toward the empty GT's physical half-space.

    The margin is min(half a native pixel, GT surface-to-plane clearance),
    fixed by detached GT and acquisition spacing. The smooth hinge is zero
    at GT and has a physical return direction even when no strided pixel
    samples the spurious curve. This is an absence constraint, not a curve
    ASSD, Dice threshold or correspondence/template attraction.
    """
    with torch.no_grad():
        gt_normal = ((teacher_mm-translation_mm)@rotation)[..., 2]
        if gt_normal.min() >= 0:
            side, clearance = 1., gt_normal.min()
        elif gt_normal.max() <= 0:
            side, clearance = -1., -gt_normal.max()
        else:
            raise ValueError('Empty GT section has no consistent physical half-space')
        margin = torch.minimum(clearance, spacing_yx.min()/2)
    normal = ((vertices_mm-translation_mm)@rotation)[..., 2]
    violation = (margin-side*normal).clamp_min(0)
    squared = violation.square()
    return (squared/((squared+beta_mm**2).sqrt()+beta_mm)).mean()


def section_soft_mask(vertices_mm, faces, rotation, translation_mm, pixel_spacing,
                      image_shape, *, sigma_mm=1., pixel_stride=1, chunk_size=256,
                      require_intersection=False):
    """Physical true section -> soft mask sampled at ORIGINAL pixel indices.

    Origin convention matches the Dataset: local x=(pixel_x-W/2)*spacing_x,
    local y=(pixel_y-H/2)*spacing_y. Striding samples those same coordinates;
    it does not resize masks or alter the acquisition/FOV.
    """
    if not np.isfinite(sigma_mm) or sigma_mm<=0 or pixel_stride<1:
        raise ValueError('Positive soft-boundary width (mm) and pixel stride required')
    spacing = torch.as_tensor(pixel_spacing,device=vertices_mm.device,dtype=vertices_mm.dtype)
    if spacing.shape!=(2,) or not torch.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('Positive physical y,x pixel spacing required')
    height,width = image_shape
    y = (torch.arange(0,height,pixel_stride,device=vertices_mm.device,dtype=vertices_mm.dtype)-height/2)*spacing[0]
    x = (torch.arange(0,width,pixel_stride,device=vertices_mm.device,dtype=vertices_mm.dtype)-width/2)*spacing[1]
    yy,xx = torch.meshgrid(y,x,indexing='ij')
    points = torch.stack((xx,yy),-1).reshape(-1,2)
    segments = plane_segments(vertices_mm,faces,rotation,translation_mm)
    if not len(segments):
        if require_intersection:
            raise MissingSectionError('Predicted chamber does not intersect an observed foreground plane; use GT shape/motion stages before 2D supervision')
        return dict(soft_mask=vertices_mm.new_zeros(len(y),len(x)),segments_mm=segments,
                    signed_distance_mm=None,logits=None)
    distance = signed_segment_distance(points,segments,chunk_size).reshape(len(y),len(x))
    logits = -distance/sigma_mm
    return dict(soft_mask=torch.sigmoid(logits),segments_mm=segments,
                signed_distance_mm=distance,logits=logits)


class MeshSliceSupervisor(nn.Module):
    """Paired physical supervision with an unreduced per-cell autograd API.

    ``cell_indices[C,4]`` identifies batch patient, phase slot, view and chamber
    (canonical A2C/A3C/A4C and LV/LA/RV/RA). ``cell_losses[C]`` and every vector
    in ``cell_terms`` retain gradients. Optional case IDs, actual cycle phases
    and source frame indices accompany them. ``total`` and scalar component
    keys remain available to existing callers as configured reductions.

    ``paired_boundary_distance_v1`` requires paired GT, uses arc-length curve
    distances in mm, and recovers missing predicted sections with distances
    from observed GT boundary points to the predicted triangle surface.
    ``legacy_probability_v1`` keeps historical target/loss behaviour.
    """
    def __init__(self, chamber_ranges, faces, norm_to_mm, *, sigma_mm=1., pixel_stride=4,
                 chunk_size=256, dice_weight=1., bce_weight=.1, boundary_weight=.1,
                 execution_backend='serial', cell_chunk_size=16,
                 target_mode='binary_native_v1', supervision_grid='model_grid', reduction='cell_mean',
                 boundary_objective='legacy_probability_v1', boundary_sample_step_mm=None,
                 boundary_fine_sample_step_mm=None, boundary_fine_start_fraction=.75,
                 boundary_only=False):
        super().__init__()
        self.register_buffer('chamber_ranges',torch.as_tensor(chamber_ranges).long().clone())
        self._ranges = tuple(map(tuple,self.chamber_ranges.cpu().tolist()))
        for c in CHAMBERS:
            self.register_buffer('faces_'+c,faces[c].long().clone())
        self.norm_to_mm = float(norm_to_mm)
        self.sigma_mm,self.pixel_stride,self.chunk_size = sigma_mm,pixel_stride,chunk_size
        if not np.isfinite(sigma_mm) or sigma_mm<=0 or pixel_stride<1 or chunk_size<1:
            raise ValueError('Positive soft-boundary width, pixel stride and query chunk required')
        if execution_backend not in ('serial','batched') or cell_chunk_size<1:
            raise ValueError('Slice backend must be serial/batched with positive cell chunk size')
        self.execution_backend,self.cell_chunk_size = execution_backend,cell_chunk_size
        if target_mode not in ('binary_native_v1','paired_soft_geometry_v1'):
            raise ValueError('Unknown slice supervision target mode')
        self.target_mode = target_mode
        if boundary_objective not in ('legacy_probability_v1', 'paired_boundary_distance_v1'):
            raise ValueError('Unknown boundary objective')
        if boundary_objective == 'paired_boundary_distance_v1' and target_mode != 'paired_soft_geometry_v1':
            raise ValueError('Paired boundary distances require paired soft GT supervision')
        if boundary_objective == 'paired_boundary_distance_v1' and boundary_weight <= 0:
            raise ValueError('Paired boundary supervision and section recovery require positive boundary_weight')
        self.boundary_objective = boundary_objective
        if type(boundary_only) is not bool or (boundary_only and boundary_objective != 'paired_boundary_distance_v1'):
            raise ValueError('Boundary-only execution requires paired geometric boundary supervision')
        self.boundary_only = boundary_only
        for step in (boundary_sample_step_mm, boundary_fine_sample_step_mm):
            if step is not None and (not np.isfinite(step) or step <= 0):
                raise ValueError('Boundary quadrature steps must be positive finite mm')
        if (not np.isfinite(boundary_fine_start_fraction) or not 0 <= boundary_fine_start_fraction <= 1
                or (boundary_fine_sample_step_mm is not None and boundary_sample_step_mm is None)
                or (boundary_fine_sample_step_mm is not None and boundary_fine_sample_step_mm > boundary_sample_step_mm)):
            raise ValueError('Declare a coarse step and a no-larger fine step with a start fraction in [0,1]')
        self.boundary_sample_step_mm = boundary_sample_step_mm
        self.boundary_fine_sample_step_mm = boundary_fine_sample_step_mm
        self.boundary_fine_start_fraction = boundary_fine_start_fraction
        self.joint_progress = 0.
        if supervision_grid not in ('model_grid','native_grid'):
            raise ValueError('Supervision grid must be model_grid or native_grid')
        if reduction not in ('cell_mean','chamber_mean'):
            raise ValueError('Slice reduction must be cell_mean or chamber_mean')
        self.supervision_grid,self.reduction=supervision_grid,reduction
        self.weights = dict(dice=dice_weight,bce=bce_weight,boundary_mm=boundary_weight)
        if (min(self.weights.values())<0 or sum(self.weights.values())<=0
                or not np.isfinite(norm_to_mm) or norm_to_mm<=0):
            raise ValueError('Nonnegative informative slice loss weights and physical scale required')

    def set_joint_progress(self, progress):
        """Deterministic epoch-derived schedule, restored without new forwards."""
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('Joint progress must be a finite fraction in [0,1]')
        self.joint_progress = float(progress)

    @property
    def boundary_step_cap_mm(self):
        if self.boundary_fine_sample_step_mm is not None and self.joint_progress >= self.boundary_fine_start_fraction:
            return self.boundary_fine_sample_step_mm
        return self.boundary_sample_step_mm

    def _terms(self,logits,truth,field,teacher_soft=None):
        """Same renderer, optional detached continuous GT target; no label edits.

        Squared-denominator soft Dice and Bernoulli KL have zero derivative
        when prediction equals continuous GT. Original fillPoly/binary Dice
        remains the independent, unchanged hard evaluation contract.
        """
        p = torch.sigmoid(logits)
        if teacher_soft is None:
            return dict(dice=1-(2*(p*truth).sum(-1)+1)/(p.sum(-1)+truth.sum(-1)+1),
                bce=F.binary_cross_entropy_with_logits(logits,truth,reduction='none').mean(-1),
                boundary_mm=((p-truth).abs()*field.abs()).mean(-1))
        q = teacher_soft.detach()
        return dict(dice=1-(2*(p*q).sum(-1)+1)/(p.square().sum(-1)+q.square().sum(-1)+1),
            bce=(F.binary_cross_entropy_with_logits(logits,q,reduction='none')-
                 (torch.special.entr(q)+torch.special.entr(1-q))).mean(-1),
            boundary_mm=((p-q).abs()*field.abs()).mean(-1))

    def _reduce(self,values,chambers):
        # Keep every annotated cell; no hard-cell mining or implicit masking.
        if self.reduction=='cell_mean':
            return {key:value.mean() for key,value in values.items()}
        present=chambers.unique()
        return {key:torch.stack([value[chambers==c].mean() for c in present]).mean()
                for key,value in values.items()}

    def _grid(self,observations,vertices):
        if self.supervision_grid=='native_grid':
            if self.boundary_objective == 'paired_boundary_distance_v1':
                if any(k not in observations for k in ('native_masks', 'native_pixel_spacing')):
                    raise ValueError('Native boundary supervision requires native_masks and native_pixel_spacing')
                masks = observations['native_masks'].to(vertices)
                # Paired GT geometry supplies the continuous target. A binary
                # EDT is neither required nor substituted for boundary distance.
                return masks, None, observations['native_pixel_spacing'].to(vertices)
            required=('native_masks','native_sdf','native_pixel_spacing','native_sdf_units')
            if any(k not in observations for k in required):
                raise ValueError('native_grid requires native_masks, native_sdf, native_pixel_spacing and native_sdf_units')
            units=observations['native_sdf_units']
            if units!='mm' and not (isinstance(units,(tuple,list)) and all(u=='mm' for u in units)):
                raise ValueError('Native SDF must be physical mm')
            return (observations['native_masks'].to(vertices),observations['native_sdf'].to(vertices),
                    observations['native_pixel_spacing'].to(vertices))
        return observations['masks'].to(vertices),observations['sdf'].to(vertices),observations['pixel_spacing'].to(vertices)

    def forward(self, vertices_4d, observations):
        if vertices_4d.ndim!=4 or vertices_4d.shape[-1]!=3:
            raise ValueError('Slice supervision requires [B,T,N,3] mesh trajectory')
        b,t,n,_ = vertices_4d.shape
        masks,sdf,spacings = self._grid(observations,vertices_4d)
        if (masks.ndim!=6 or masks.shape[:4]!=(b,t,3,4)
                or (sdf is not None and sdf.shape!=masks.shape) or n!=self.chamber_ranges[-1,1]):
            raise ValueError('Mesh, mask, SDF and topology shapes differ')
        if 'phases_gt' in observations and not torch.equal(observations['phases'],observations['phases_gt']):
            raise ValueError('GT mesh/mask phases differ from observation phases')
        valid = observation_validity(observations,b,t,vertices_4d.device)
        active=valid[...,None,None].expand_as(masks)
        if (not torch.isfinite(masks[active]).all()
                or (sdf is not None and not torch.isfinite(sdf[active]).all())
                or ((masks[active]!=0)&(masks[active]!=1)).any()):
            raise ValueError('Finite physical SDF and binary slice targets required')
        pose = observations['pose_gt']
        rotations,translations = pose['probe_R'].to(vertices_4d),pose['probe_t'].to(vertices_4d)*self.norm_to_mm
        if rotations.shape!=(b,t,3,3,3) or translations.shape!=(b,t,3,3) or spacings.shape!=(b,t,3,2):
            raise ValueError('Pose/time/pixel-grid contract mismatch')
        gt = None
        if self.target_mode == 'paired_soft_geometry_v1':
            if 'vertices_gt' not in observations:
                raise ValueError('Paired soft slice targets require complete synthetic GT supervision')
            gt = observations['vertices_gt'].to(vertices_4d).detach()
            if gt.shape != vertices_4d.shape or not torch.isfinite(gt).all():
                raise ValueError('Paired soft slice targets require finite matching GT topology/phases')
        if self.boundary_objective == 'paired_boundary_distance_v1':
            return self._boundary_forward(vertices_4d, masks, valid, rotations, translations, spacings, gt, observations)
        if self.execution_backend=='batched':
            return self._batched_forward(vertices_4d,masks,sdf,valid,rotations,translations,spacings,gt,observations)
        values = {key:[] for key in self.weights}
        matched_empty = 0
        cell_indices=valid.nonzero().tolist()
        for patient,frame,view,chamber in cell_indices:
            left,right = self.chamber_ranges[chamber].tolist()
            target = masks[patient,frame,view,chamber,::self.pixel_stride,::self.pixel_stride]
            field = sdf[patient,frame,view,chamber,::self.pixel_stride,::self.pixel_stride]
            if masks[patient,frame,view,chamber].any() and not target.any():
                raise ValueError('Slice pixel_stride misses an annotated foreground; reduce pixel_stride')
            try:
                rendered = section_soft_mask(vertices_4d[patient,frame,left:right],getattr(self,'faces_'+CHAMBERS[chamber]),
                    rotations[patient,frame,view],translations[patient,frame,view],spacings[patient,frame,view],masks.shape[-2:],
                    sigma_mm=self.sigma_mm,pixel_stride=self.pixel_stride,chunk_size=self.chunk_size,
                    require_intersection=bool(masks[patient,frame,view,chamber].any()))
            except (ValueError,MissingSectionError) as error:
                raise type(error)(f'Patient {patient}, frame {frame}, view {view}, chamber {CHAMBERS[chamber]}: {error}') from error
            if rendered['logits'] is None:
                # Mathematically empty/empty cells supply no local section
                # gradient. Report them explicitly; never invent leaf losses.
                matched_empty += 1
                for key in values:
                    values[key].append(vertices_4d[patient,frame,left:right].sum()*0)
                continue
            teacher_soft = None
            if gt is not None:
                with torch.no_grad():
                    teacher = section_soft_mask(gt[patient,frame,left:right],getattr(self,'faces_'+CHAMBERS[chamber]),
                        rotations[patient,frame,view],translations[patient,frame,view],spacings[patient,frame,view],masks.shape[-2:],
                        sigma_mm=self.sigma_mm,pixel_stride=self.pixel_stride,chunk_size=self.chunk_size,
                        require_intersection=bool(masks[patient,frame,view,chamber].any()))
                    teacher_soft = teacher['soft_mask']
            terms = self._terms(rendered['logits'].flatten()[None],target.flatten()[None],field.flatten()[None],
                None if teacher_soft is None else teacher_soft.flatten()[None])
            for key,value in terms.items():
                values[key].append(value[0])
        cells = int(valid.sum())
        if not cells or matched_empty==cells:
            raise ValueError('No informative plane sections in the supervised batch')
        packed = {key:torch.stack(value) for key,value in values.items()}
        result = self._reduce(packed,
                              torch.tensor([cell[3] for cell in cell_indices],device=vertices_4d.device))
        result['total'] = sum(self.weights[key]*result[key] for key in self.weights)
        result['diagnostics'] = dict(valid_cells=cells,matched_empty_sections=matched_empty,
                                     intersecting_cells=cells-matched_empty,pixel_stride=self.pixel_stride,
                                     soft_boundary_width_mm=self.sigma_mm)
        result['diagnostics'].update(target_mode=self.target_mode,supervision_grid=self.supervision_grid,reduction=self.reduction)
        if not torch.isfinite(result['total']):
            raise ValueError('Nonfinite slice supervision')
        return self._attach_cells(result, cell_indices, packed, observations)

    def _attach_cells(self, result, indices, terms, observations=None, missing=None, kinds=None):
        """Expose unreduced autograd tensors in canonical [B,T,V,C] order."""
        # Lexicographic sorting is independent of the serial/batched execution.
        order = sorted(range(len(indices)), key=lambda i: tuple(indices[i]))
        indices = torch.as_tensor(indices, device=self.chamber_ranges.device, dtype=torch.long)[order]
        terms = {key: value[order] for key, value in terms.items()}
        result['cell_indices'] = indices
        result['cell_index_columns'] = ('patient', 'phase', 'view', 'chamber')
        result['cell_terms'] = terms
        result['cell_losses'] = sum(self.weights[key]*terms[key] for key in self.weights)
        if missing is not None:
            result['cell_missing_sections'] = torch.as_tensor(missing, device=indices.device, dtype=torch.bool)[order]
        if kinds is not None:
            result['cell_observation_kind'] = torch.as_tensor(kinds, device=indices.device, dtype=torch.long)[order]
            result['cell_observation_kind_names'] = OBSERVATION_KINDS
        if observations is not None:
            owners, phases = indices[:, 0], indices[:, 1]
            if 'case_id' in observations:
                identities = observations['case_id']
                result['cell_case_ids'] = [identities[int(owner)] for owner in owners]
            for source, target in (('phases', 'cell_phases'), ('frame_indices', 'cell_frame_indices')):
                if source in observations:
                    result[target] = observations[source].to(indices.device)[owners, phases]
        return result

    def _boundary_forward(self, vertices, masks, valid, rotations, translations, spacings, gt, observations):
        """Versioned paired boundary objective; aggregate keys stay compatible.

        The unreduced boundary_mm vector is the primary geometric signal for
        a future constrained optimiser. total retains the caller's configured
        reduction/weights; this module does not implement a new Joint solver.
        """
        active = valid.any(-1)
        r, tr, spacing = rotations[active], translations[active], spacings[active]
        eye = torch.eye(3, device=vertices.device, dtype=vertices.dtype)
        if (not torch.isfinite(vertices).all() or not torch.isfinite(r).all()
                or not torch.isfinite(tr).all() or not torch.isfinite(spacing).all() or (spacing <= 0).any()
                or ((r.transpose(-1, -2)@r-eye).abs() > 1e-4).any() or (torch.linalg.det(r) <= 0).any()):
            raise ValueError('Finite mesh, right-handed orthonormal poses and positive physical spacing required')
        cells = valid.nonzero().cpu().tolist()
        if not cells:
            raise ValueError('No valid supervised cells')
        h, w = masks.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(0, h, self.pixel_stride, device=vertices.device, dtype=vertices.dtype)-h/2,
                                torch.arange(0, w, self.pixel_stride, device=vertices.device, dtype=vertices.dtype)-w/2,
                                indexing='ij')
        pixel_points = torch.stack((xx, yy), -1).reshape(-1, 2)
        values = {key: [] for key in (*self.weights, 'curve_mm', 'recovery_mm', 'empty_target_mm')}
        ordered_cells, missing_cells, cell_kinds = [], [], []
        empty_count = recovery_count = 0
        chunk = 1 if self.execution_backend == 'serial' else self.cell_chunk_size
        for chamber, (left, right) in enumerate(self._ranges):
            selected = [cell for cell in cells if cell[3] == chamber]
            faces = getattr(self, 'faces_'+CHAMBERS[chamber])
            for start in range(0, len(selected), chunk):
                entries = selected[start:start+chunk]
                indices = torch.tensor(entries, device=vertices.device)
                patient, frame, view, _ = indices.unbind(-1)
                v = vertices[patient, frame, left:right]
                r, tr, spacing = rotations[patient, frame, view], translations[patient, frame, view], spacings[patient, frame, view]
                segments, present = batched_plane_segments(v, faces, r, tr)
                intersects = present.any(-1)
                points = pixel_points[None]*spacing[:, None, [1, 0]]
                with torch.no_grad():
                    teacher_segments, teacher_present = batched_plane_segments(gt[patient, frame, left:right], faces, r, tr)
                    teacher_intersects = teacher_present.any(-1)
                    teacher_soft = points.new_zeros(points.shape[:2])
                    if not self.boundary_only and teacher_intersects.any():
                        teacher_distance = batched_signed_segment_distance(points[teacher_intersects],
                            teacher_segments[teacher_intersects], teacher_present[teacher_intersects], self.chunk_size)
                        teacher_soft[teacher_intersects] = torch.sigmoid(-teacher_distance/self.sigma_mm)
                # Never evaluate nearest distance against a dummy padded segment.
                logits = points.new_zeros(points.shape[:2])
                raster_cells = intersects & (~teacher_intersects if self.boundary_only else torch.ones_like(intersects))
                if raster_cells.any():
                    distance = batched_signed_segment_distance(points[raster_cells], segments[raster_cells], present[raster_cells], self.chunk_size)
                    logits = logits.index_copy(0, raster_cells.nonzero().flatten(), -distance/self.sigma_mm)
                paired = intersects & teacher_intersects
                fast_curves = None
                if self.boundary_only and paired.any():
                    fast_curves = v.sum((1, 2))*0
                    fast_curves = fast_curves.index_copy(0, paired.nonzero().flatten(),
                        batched_paired_boundary_distance(segments[paired], present[paired], teacher_segments[paired],
                            teacher_present[paired], spacing[paired], (h, w), beta_mm=self.sigma_mm,
                            chunk_size=self.chunk_size, sample_step_mm=self.boundary_step_cap_mm))
                # One small flag transfer per chunk, rather than repeated GPU
                # synchronisation for each chamber/phase branch decision.
                flags = torch.stack((intersects, teacher_intersects,
                    masks[patient, frame, view, chamber].bool().flatten(1).any(-1)), -1).cpu().tolist()
                if self.boundary_only and all(pred and truth for pred, truth, _ in flags):
                    # The common paired case needs no per-cell GPU indexing,
                    # scalar reads, rasterisation or Python nearest searches.
                    zeroes = v.sum((1, 2))*0
                    for key in self.weights:
                        values[key].extend((fast_curves if key == 'boundary_mm' else zeroes).unbind())
                    values['curve_mm'].extend(fast_curves.unbind())
                    values['recovery_mm'].extend(zeroes.unbind())
                    values['empty_target_mm'].extend(zeroes.unbind())
                    _, visible = batched_clip_section_to_native(teacher_segments, teacher_present, spacing, (h, w))
                    kinds = visible.any(-1).cpu().tolist()
                    ordered_cells.extend(entries)
                    missing_cells.extend([False]*len(entries))
                    cell_kinds.extend(OBSERVATION_KINDS.index('paired_boundary' if flag else 'no_visible_gt_boundary') for flag in kinds)
                    continue
                for row, cell in enumerate(entries):
                    p, t, view_id, c = cell
                    zero = v[row].sum()*0
                    curve_loss = recovery = empty_target = zero
                    has_pred, has_teacher, foreground = flags[row]
                    teacher_curve = teacher_segments[row, teacher_present[row]]
                    pred_curve = segments[row, present[row]]
                    if foreground and not has_teacher:
                        raise MissingSectionError(f'GT section absent for observed foreground cell {cell}')
                    if has_pred:
                        truth = masks[p, t, view_id, c, ::self.pixel_stride, ::self.pixel_stride].flatten()[None]
                        terms = dict(dice=zero, bce=zero, boundary_mm=zero) if self.boundary_only else {
                            key: value[0] for key, value in self._terms(logits[row:row+1], truth,
                            torch.zeros_like(truth), teacher_soft[row:row+1]).items()}
                        if len(teacher_curve):
                            curve_loss = fast_curves[row] if fast_curves is not None else paired_boundary_distance(pred_curve, teacher_curve,
                                spacing[row], (h, w), beta_mm=self.sigma_mm, chunk_size=self.chunk_size,
                                sample_step_mm=self.boundary_step_cap_mm)
                            visible_gt = clip_section_to_native(teacher_curve, spacing[row], (h, w))
                            kind = 'paired_boundary' if len(visible_gt) else 'no_visible_gt_boundary'
                        else:
                            visible_pred = clip_section_to_native(pred_curve, spacing[row], (h, w))
                            if len(visible_pred) or bool((logits[row] >= 0).any()):
                                empty_target = empty_section_clearance(v[row], gt[p, t, left:right],
                                    r[row], tr[row], spacing[row], beta_mm=self.sigma_mm)
                                kind = 'empty_target'
                            else:
                                kind = 'matched_empty'
                                empty_count += 1
                        terms['boundary_mm'] = curve_loss+empty_target
                    else:
                        observed_curve = clip_section_to_native(teacher_curve, spacing[row], (h, w))
                        if len(observed_curve) or foreground:
                            if len(observed_curve):
                                step = spacing[row].min()
                                if self.boundary_step_cap_mm is not None:
                                    step = torch.minimum(step, step.new_tensor(self.boundary_step_cap_mm))
                                capture_points, weight = section_quadrature(observed_curve, step)
                            else:
                                # A full foreground FOV can contain no visible
                                # GT boundary. Use its actual native pixels.
                                pixels = masks[p, t, view_id, c].nonzero().to(vertices)
                                capture_points = (pixels[:, [1, 0]]-vertices.new_tensor([w/2, h/2]))*spacing[row].flip(0)
                                weight = None
                            world = F.pad(capture_points, (0, 1))@r[row].T+tr[row]
                            recovery = surface_capture_distance(world, v[row], faces, beta_mm=self.sigma_mm, weights=weight)
                            terms = dict(dice=zero+1, bce=zero, boundary_mm=recovery)
                            kind = 'recovery'
                            recovery_count += 1
                        else:
                            terms = dict(dice=zero, bce=zero, boundary_mm=zero)
                            kind = 'matched_empty'
                            empty_count += 1
                    for key in self.weights:
                        values[key].append(terms[key])
                    values['recovery_mm'].append(recovery)
                    values['curve_mm'].append(curve_loss)
                    values['empty_target_mm'].append(empty_target)
                    ordered_cells.append(cell)
                    missing_cells.append(not has_pred)
                    cell_kinds.append(OBSERVATION_KINDS.index(kind))
        packed = {key: torch.stack(value) for key, value in values.items()}
        chambers = torch.tensor([cell[3] for cell in ordered_cells], device=vertices.device)
        result = self._reduce({key: packed[key] for key in self.weights}, chambers)
        result['total'] = sum(self.weights[key]*result[key] for key in self.weights)
        result['diagnostics'] = dict(valid_cells=len(cells), matched_empty_sections=empty_count,
            missing_predicted_sections=sum(missing_cells), recovery_cells=recovery_count,
            intersecting_cells=len(cells)-sum(missing_cells), pixel_stride=self.pixel_stride,
            soft_boundary_width_mm=self.sigma_mm, target_mode=self.target_mode,
            supervision_grid=self.supervision_grid, reduction=self.reduction, boundary_objective=self.boundary_objective,
            boundary_units='mm', boundary_quadrature='native_pixel_spacing',
            boundary_smoothing_mm=self.sigma_mm,
            boundary_sample_step_cap_mm=self.boundary_step_cap_mm, joint_progress=self.joint_progress,
            probability_terms_evaluated=not self.boundary_only,
            empty_target_policy='detached_gt_half_space; margin=min(native_half_pixel,gt_clearance)')
        if not all(torch.isfinite(value).all() for value in packed.values()):
            raise ValueError('Nonfinite per-cell slice supervision')
        return self._attach_cells(result, ordered_cells, packed, observations, missing_cells, cell_kinds)

    def _batched_forward(self, vertices, masks, sdf, valid, rotations, translations, spacings,gt=None,observations=None):
        # Validate once instead of synchronizing CPU/GPU for every cell.
        active_views = valid.any(-1)
        r,tr,spacing = rotations[active_views],translations[active_views],spacings[active_views]
        eye = torch.eye(3,device=vertices.device,dtype=vertices.dtype)
        checks = torch.stack((torch.isfinite(vertices).all(),torch.isfinite(r).all(),
            torch.isfinite(tr).all(),torch.isfinite(spacing).all(),(spacing>0).all(),
            ((r.transpose(-1,-2)@r-eye).abs()<=1e-4).all(),(torch.linalg.det(r)>0).all()))
        if not checks.all():
            raise ValueError('Finite mesh, right-handed orthonormal poses and positive physical spacing required')
        target = masks[...,::self.pixel_stride,::self.pixel_stride]
        field = sdf[...,::self.pixel_stride,::self.pixel_stride]
        foreground = masks.bool().flatten(-2).any(-1)
        if (valid & foreground & ~target.bool().flatten(-2).any(-1)).any():
            raise ValueError('Slice pixel_stride misses an annotated foreground; reduce pixel_stride')
        cells = valid.nonzero().cpu().tolist()
        if not cells:
            raise ValueError('No informative plane sections in the supervised batch')
        height,width = masks.shape[-2:]
        y = torch.arange(0,height,self.pixel_stride,device=vertices.device,dtype=vertices.dtype)-height/2
        x = torch.arange(0,width,self.pixel_stride,device=vertices.device,dtype=vertices.dtype)-width/2
        yy,xx = torch.meshgrid(y,x,indexing='ij')
        pixel_points = torch.stack((xx,yy),-1).reshape(-1,2)
        values = {key:[] for key in self.weights}
        matched_empty = vertices.new_zeros((),dtype=torch.long)
        chamber_indices=[]
        ordered_cells=[]
        for chamber,(left,right) in enumerate(self._ranges):
            selected = [cell[:3] for cell in cells if cell[3]==chamber]
            for start in range(0,len(selected),self.cell_chunk_size):
                indices = torch.tensor(selected[start:start+self.cell_chunk_size],device=vertices.device)
                patient,frame,view = indices.unbind(-1)
                segments,present = batched_plane_segments(vertices[patient,frame,left:right],
                    getattr(self,'faces_'+CHAMBERS[chamber]),rotations[patient,frame,view],translations[patient,frame,view])
                intersects = present.any(-1)
                if (foreground[patient,frame,view,chamber] & ~intersects).any():
                    raise MissingSectionError(f'Predicted {CHAMBERS[chamber]} does not intersect an observed foreground plane; '
                                              'use GT shape/motion stages before 2D supervision')
                matched_empty = matched_empty + (~intersects).sum()
                # Both coordinates stay at the same original strided pixel indices.
                points = pixel_points[None]*spacings[patient,frame,view][:,None,[1,0]]
                distance = batched_signed_segment_distance(points,segments,present,self.chunk_size)
                logits = -distance/self.sigma_mm
                p = torch.sigmoid(logits)
                truth = target[patient,frame,view,chamber].flatten(1)
                s = field[patient,frame,view,chamber].flatten(1)
                teacher_soft = None
                if gt is not None:
                    with torch.no_grad():
                        teacher_segments,teacher_present = batched_plane_segments(gt[patient,frame,left:right],
                            getattr(self,'faces_'+CHAMBERS[chamber]),rotations[patient,frame,view],translations[patient,frame,view])
                        if (foreground[patient,frame,view,chamber] & ~teacher_present.any(-1)).any():
                            raise MissingSectionError('GT soft section is absent in an observed foreground cell')
                        teacher_distance = batched_signed_segment_distance(points,teacher_segments,teacher_present,self.chunk_size)
                        teacher_soft = torch.where(teacher_present.any(-1)[:,None],
                            torch.sigmoid(-teacher_distance/self.sigma_mm),0.)
                terms = self._terms(logits,truth,s,teacher_soft)
                # Same cells/gradients in both reductions; absent sections retain zero.
                ordered_cells.extend([(*cell, chamber) for cell in selected[start:start+self.cell_chunk_size]])
                chamber_indices.append(torch.full_like(patient,chamber))
                for key,value in terms.items():
                    values[key].append(torch.where(intersects,value,0.))
        empty_count = int(matched_empty)
        if empty_count==len(cells):
            raise ValueError('No informative plane sections in the supervised batch')
        packed = {key:torch.cat(value) for key,value in values.items()}
        result = self._reduce(packed,torch.cat(chamber_indices))
        result['total'] = sum(self.weights[key]*result[key] for key in self.weights)
        result['diagnostics'] = dict(valid_cells=len(cells),matched_empty_sections=empty_count,
            intersecting_cells=len(cells)-empty_count,pixel_stride=self.pixel_stride,
            soft_boundary_width_mm=self.sigma_mm)
        result['diagnostics'].update(target_mode=self.target_mode,supervision_grid=self.supervision_grid,reduction=self.reduction)
        if not torch.isfinite(result['total']):
            raise ValueError('Nonfinite slice supervision')
        return self._attach_cells(result, ordered_cells, packed, observations)


def exact_section_mask(vertices_mm, faces, rotation, translation_mm, pixel_spacing, image_shape):
    """Independent non-differentiable trimesh+OpenCV true-section evaluator.

    Integer truncation/fillPoly follows saved synthetic pixel conventions.
    Uses trimesh path assembly, independently of the soft segment operator.
    No EchoGSF neural/rendering implementation is imported.
    """
    import cv2
    import trimesh
    vertices = np.asarray(vertices_mm,dtype=np.float64)
    rotation = np.asarray(rotation,dtype=np.float64)
    translation = np.asarray(translation_mm,dtype=np.float64)
    spacing = np.asarray(pixel_spacing,dtype=np.float64)
    height,width = image_shape
    mesh = trimesh.Trimesh(vertices=vertices,faces=np.asarray(faces),process=False)
    section = mesh.section(plane_normal=rotation[:,2],plane_origin=translation)
    image = np.zeros((height,width),np.uint8)
    if section is None:
        return image
    for loop in section.discrete:
        if len(loop)<4 or not np.allclose(loop[0],loop[-1],atol=1e-5,rtol=0):
            raise ValueError('Open exact chamber section')
        local = (loop-translation)@rotation
        pixels = np.stack((local[:,0]/spacing[1]+width/2,local[:,1]/spacing[0]+height/2),-1).astype(np.int32)
        pixels[:,0] = pixels[:,0].clip(0,width-1)
        pixels[:,1] = pixels[:,1].clip(0,height-1)
        cv2.fillPoly(image,[pixels],1)
    return image
