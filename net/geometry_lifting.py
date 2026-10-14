"""Known-pose physical feature lifting, using the original renderer pixel origin.

Row vectors: q=(X/norm_to_mm-t)@R; native x=(q_x/(s_x+1e-8)+1)*W/2.
model_to_native acts on homogeneous integer pixel indices (x,y,1).
Feature samples use exact ResNet receptive-center strides (8/16) followed by
adaptive-pool bin centers. align_corners=True maps these feature indices.
Unknown is encoded explicitly; image/native pixel coordinates stay unchanged.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def project_physical_points(points_mm, observations, norm_to_mm, model_shape):
    pose = observations.get('geometry', observations.get('pose_gt'))
    if pose is None:
        raise ValueError('Explicit known_pose geometry required; GT mesh cannot supply pose')
    r, t, s = (pose[k].to(points_mm) for k in ('probe_R', 'probe_t', 'probe_s'))
    b, frames = observations['masks'].shape[:2]
    if r.shape != (b, frames, 3, 3, 3) or t.shape != (b, frames, 3, 3) or s.shape != t.shape:
        raise ValueError('Per-case/per-phase [B,T,3] probe geometry required')
    affine = observations['model_to_native'].to(points_mm)
    native_spacing = observations['native_pixel_spacing'].to(points_mm)
    spacing = observations['pixel_spacing'].to(points_mm)
    if (affine.shape != r.shape or native_spacing.shape != (b, frames, 3, 2)
            or spacing.shape != native_spacing.shape):
        raise ValueError('Model/native pixel affine and calibrated spacing required')
    values = (r, t, s, affine, native_spacing, spacing)
    eye = torch.eye(3, device=r.device, dtype=r.dtype)
    if (not all(torch.isfinite(v).all() for v in values) or (s <= 0).any()
            or (spacing <= 0).any() or (native_spacing <= 0).any()
            or not torch.allclose(r.transpose(-1,-2)@r, eye.expand_as(r), atol=1e-4, rtol=0)
            or (torch.linalg.det(r) <= 0).any()
            or not torch.allclose(affine[...,2,:], eye[2].expand_as(affine[...,2,:]))):
        raise ValueError('Finite right-handed poses, affine pixel transform and positive spacing required')
    # Infer recorded native dimensions from calibrated probe scale/spacing,
    # without reading native_masks or any GT mesh/shape parameter.
    native_hw = 2*s[...,[1,0]]*norm_to_mm/native_spacing
    if not torch.allclose(native_hw, native_hw.round(), atol=.002, rtol=0):
        raise ValueError('Native calibration must imply integer image dimensions')
    if not torch.allclose(affine[...,0,0]*native_spacing[...,1], spacing[...,1], atol=1e-4, rtol=1e-5) or not torch.allclose(
            affine[...,1,1]*native_spacing[...,0], spacing[...,0], atol=1e-4, rtol=1e-5):
        raise ValueError('Pixel affine/physical spacing mismatch')
    q = torch.einsum('btvni,btvij->btvnj', points_mm[None,None,None]/norm_to_mm-t[...,None,:], r)
    native = (q[...,:2]/(s[...,None,:2]+1e-8)+1)*native_hw[...,[1,0]][...,None,:]/2
    homogeneous = torch.cat((native, torch.ones_like(native[...,:1])), -1)
    model = torch.einsum('btvij,btvnj->btvni', torch.linalg.inv(affine), homogeneous)[...,:2]
    hw = torch.tensor(model_shape, device=q.device, dtype=q.dtype)
    grid = 2*(model+.5)/hw.flip(0)-1
    inside = ((model >= -.5) & (model < hw.flip(0)-.5)).all(-1)
    return dict(grid=grid, inside=inside, distance_mm=q[...,2]*norm_to_mm,
                native_pixels=native, model_pixels=model)


def feature_sampling_grid(model_pixels, image_shape, map_shape, backbone_stride):
    """Invert pooled ResNet receptive-center positions, separately for x/y.

    Odd same-padding convolutions/pooling keep receptive center 0 at index 0.
    Native layer2/layer3 strides are 8/16. Adaptive pooling averages indices
    floor(j*N/M)..ceil((j+1)*N/M)-1, hence the explicit centroid below.
    For 224 -> 14/7, centers are 4+16*j and 8+32*j respectively.
    """
    coordinates = []
    for coordinate, size, side in zip(model_pixels.unbind(-1), reversed(image_shape), reversed(map_shape)):
        native_side = (size+backbone_stride-1)//backbone_stride
        if side > native_side or side < 2:
            raise ValueError('Lifting requires spatial maps no larger than the native CNN maps')
        # Pool bin endpoints are integer quantities. CUDA float division can
        # round an exact integer slightly upward before ceil(), shifting the
        # receptive center by half a backbone stride for selected bins.
        centers = model_pixels.new_tensor([
            (j*native_side//side + ((j+1)*native_side+side-1)//side-1)*backbone_stride/2
            for j in range(side)])
        index = torch.searchsorted(centers,coordinate.contiguous()).clamp(1,side-1)
        fraction = (coordinate-centers[index-1])/(centers[index]-centers[index-1])
        feature_index = index-1+fraction
        coordinates.append(2*feature_index/(side-1)-1)
    return torch.stack(coordinates,-1)


class GeometryFeatureLifting(nn.Module):
    def __init__(self, input_width=128, projected_width=4, sigma_mm=8., cutoff_sigma=3.):
        super().__init__()
        if projected_width < 1 or not math.isfinite(sigma_mm) or sigma_mm <= 0 or not math.isfinite(cutoff_sigma) or cutoff_sigma <= 0:
            raise ValueError('Positive lifting widths and finite plane-distance scales required')
        self.projections = nn.ModuleList([nn.Conv2d(input_width, projected_width, 1) for _ in range(2)])
        self.projected_width, self.sigma_mm, self.cutoff_sigma = projected_width, sigma_mm, cutoff_sigma
        self.output_width = 2*projected_width+6

    def forward(self, spatial, observations, query_mm, norm_to_mm):
        shape = query_mm.shape[:-1]
        projection = project_physical_points(query_mm.reshape(-1,3), observations, norm_to_mm,
                                              observations['masks'].shape[-2:])
        valid = spatial['validity']
        available = valid.any(-1)[...,None] & projection['inside']
        distance = projection['distance_mm'].abs()
        available = available & (distance <= self.cutoff_sigma*self.sigma_mm)
        weights = torch.where(available, torch.exp(-.5*(distance/self.sigma_mm).square()), 0.)
        denominator = weights.sum(2)
        fused = []
        b, t, v, n = weights.shape
        for key, project, stride in zip(('scale14', 'scale7'), self.projections, (8,16)):
            value = spatial[key]
            value = project(value.flatten(0,2))
            grid = feature_sampling_grid(projection['model_pixels'],observations['masks'].shape[-2:],value.shape[-2:],stride)
            sampled = F.grid_sample(value, grid.reshape(b*t*v,n,1,2),
                mode='bilinear', padding_mode='border', align_corners=True).reshape(b,t,v,self.projected_width,n)
            sampled = torch.where(available[...,None,:], sampled, 0.)
            fused.append((sampled*weights[...,None,:]).sum(2)/denominator[:, :, None].clamp_min(1e-8))
        # Support is not a background label. Channel support exposes which
        # chambers were annotated; far-plane extrapolation cannot become observed.
        support = denominator > 0
        chamber_support = (weights[...,None,:]*valid.to(weights)[...,None]).sum(2)/denominator[:,:,None].clamp_min(1e-8)
        nearest = torch.where(available, distance, torch.full_like(distance, self.cutoff_sigma*self.sigma_mm)).amin(2)
        features = torch.cat((*fused, support[:,:,None].to(weights),
                              nearest[:,:,None]/(self.cutoff_sigma*self.sigma_mm), chamber_support), 2)
        return dict(features=features.reshape(b,t,self.output_width,*shape),
                    support=support.reshape(b,t,*shape),
                    nearest_distance_mm=nearest.reshape(b,t,*shape))
