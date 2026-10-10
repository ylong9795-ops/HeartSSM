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
    def __init__(self, chamber_ranges, faces, norm_to_mm, *, sigma_mm=1., pixel_stride=4,
                 chunk_size=256, dice_weight=1., bce_weight=.1, boundary_weight=.1):
        super().__init__()
        self.register_buffer('chamber_ranges',torch.as_tensor(chamber_ranges).long().clone())
        for c in CHAMBERS:
            self.register_buffer('faces_'+c,faces[c].long().clone())
        self.norm_to_mm = float(norm_to_mm)
        self.sigma_mm,self.pixel_stride,self.chunk_size = sigma_mm,pixel_stride,chunk_size
        self.weights = dict(dice=dice_weight,bce=bce_weight,boundary_mm=boundary_weight)
        if (min(self.weights.values())<0 or sum(self.weights.values())<=0
                or not np.isfinite(norm_to_mm) or norm_to_mm<=0):
            raise ValueError('Nonnegative informative slice loss weights and physical scale required')

    def forward(self, vertices_4d, observations):
        if vertices_4d.ndim!=4 or vertices_4d.shape[-1]!=3:
            raise ValueError('Slice supervision requires [B,T,N,3] mesh trajectory')
        b,t,n,_ = vertices_4d.shape
        masks,sdf = observations['masks'].to(vertices_4d),observations['sdf'].to(vertices_4d)
        if masks.shape[:4]!=(b,t,3,4) or sdf.shape!=masks.shape or n!=self.chamber_ranges[-1,1]:
            raise ValueError('Mesh, mask, SDF and topology shapes differ')
        if 'phases_gt' in observations and not torch.equal(observations['phases'],observations['phases_gt']):
            raise ValueError('GT mesh/mask phases differ from observation phases')
        valid = observation_validity(observations,b,t,vertices_4d.device)
        pose = observations['pose_gt']
        rotations,translations = pose['probe_R'].to(vertices_4d),pose['probe_t'].to(vertices_4d)*self.norm_to_mm
        spacings = observations['pixel_spacing'].to(vertices_4d)
        if rotations.shape!=(b,t,3,3,3) or translations.shape!=(b,t,3,3) or spacings.shape!=(b,t,3,2):
            raise ValueError('Pose/time/pixel-grid contract mismatch')
        values = {key:[] for key in self.weights}
        matched_empty = 0
        for patient,frame,view,chamber in valid.nonzero().tolist():
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
            p = rendered['soft_mask']
            values['dice'].append(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1))
            values['bce'].append(F.binary_cross_entropy_with_logits(rendered['logits'],target))
            values['boundary_mm'].append(((p-target).abs()*field.abs()).mean())
        cells = int(valid.sum())
        if not cells or matched_empty==cells:
            raise ValueError('No informative plane sections in the supervised batch')
        result = {key:torch.stack(value).mean() for key,value in values.items()}
        result['total'] = sum(self.weights[key]*result[key] for key in self.weights)
        result['diagnostics'] = dict(valid_cells=cells,matched_empty_sections=matched_empty,
                                     intersecting_cells=cells-matched_empty,pixel_stride=self.pixel_stride,
                                     soft_boundary_width_mm=self.sigma_mm)
        if not torch.isfinite(result['total']):
            raise ValueError('Nonfinite slice supervision')
        return result


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
