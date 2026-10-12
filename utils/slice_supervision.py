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


def batched_signed_segment_distance(points, segments, present, chunk_size=256):
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
            dy = ab[...,1]
            denominator = torch.where(dy.abs()>1e-16,dy,torch.ones_like(dy))
            x_cross = a[:,None,:,0]+(p[:,:,None,1]-a[:,None,:,1])*ab[:,None,:,0]/denominator[:,None]
            crosses = (((a[:,None,:,1]>p[:,:,None,1]) != (b[:,None,:,1]>p[:,:,None,1]))
                       & (p[:,:,None,0]<x_cross) & present[:,None])
            sign = 1-2*(crosses.sum(-1)%2).to(p)
        chosen = segments.gather(1,nearest[:,:,None,None].expand(-1,-1,2,2))
        a,b = chosen[:,:,0],chosen[:,:,1]
        ab = b-a
        u = ((p-a)*ab).sum(-1)/ab.square().sum(-1).clamp_min(1e-16)
        closest = a+u.clamp(0,1)[...,None]*ab
        distances.append(sign*((p-closest).square().sum(-1)+1e-12).sqrt())
    return torch.cat(distances,dim=1)


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
                 chunk_size=256, dice_weight=1., bce_weight=.1, boundary_weight=.1,
                 execution_backend='serial', cell_chunk_size=16,
                 target_mode='binary_native_v1'):
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
        self.weights = dict(dice=dice_weight,bce=bce_weight,boundary_mm=boundary_weight)
        if (min(self.weights.values())<0 or sum(self.weights.values())<=0
                or not np.isfinite(norm_to_mm) or norm_to_mm<=0):
            raise ValueError('Nonnegative informative slice loss weights and physical scale required')

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
        gt = None
        if self.target_mode == 'paired_soft_geometry_v1':
            if 'vertices_gt' not in observations:
                raise ValueError('Paired soft slice targets require complete synthetic GT supervision')
            gt = observations['vertices_gt'].to(vertices_4d).detach()
            if gt.shape != vertices_4d.shape or not torch.isfinite(gt).all():
                raise ValueError('Paired soft slice targets require finite matching GT topology/phases')
        if self.execution_backend=='batched':
            return self._batched_forward(vertices_4d,masks,sdf,valid,rotations,translations,spacings,gt)
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
        result = {key:torch.stack(value).mean() for key,value in values.items()}
        result['total'] = sum(self.weights[key]*result[key] for key in self.weights)
        result['diagnostics'] = dict(valid_cells=cells,matched_empty_sections=matched_empty,
                                     intersecting_cells=cells-matched_empty,pixel_stride=self.pixel_stride,
                                     soft_boundary_width_mm=self.sigma_mm)
        result['diagnostics']['target_mode'] = self.target_mode
        if not torch.isfinite(result['total']):
            raise ValueError('Nonfinite slice supervision')
        return result

    def _batched_forward(self, vertices, masks, sdf, valid, rotations, translations, spacings,gt=None):
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
                # Empty/empty cells contribute graph-connected zero, just as serial.
                for key,value in terms.items():
                    values[key].append(torch.where(intersects,value,0.).sum())
        empty_count = int(matched_empty)
        if empty_count==len(cells):
            raise ValueError('No informative plane sections in the supervised batch')
        result = {key:torch.stack(value).sum()/len(cells) for key,value in values.items()}
        result['total'] = sum(self.weights[key]*result[key] for key in self.weights)
        result['diagnostics'] = dict(valid_cells=len(cells),matched_empty_sections=empty_count,
            intersecting_cells=len(cells)-empty_count,pixel_stride=self.pixel_stride,
            soft_boundary_width_mm=self.sigma_mm)
        result['diagnostics']['target_mode'] = self.target_mode
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
