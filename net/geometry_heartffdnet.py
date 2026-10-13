"""Geometry-aware, known-pose Ma-style joint 24-phase physical HeartFFDNet."""
import hashlib
import math
import torch
from torch import nn
from data.mask_transform import CHAMBERS
from net.geometry_spatial_encoder import GeometrySpatialEncoder
from net.geometry_lifting import GeometryFeatureLifting
from net.ffdnet import FFDNet
from torch.nn import functional as F
from utils.ffd_physical import control_geometry, bspline_ffd_physical

ARCHITECTURE = 'geometry-ma-joint-ffd-v1'


def geometry_digest(vertices_mm, ranges, faces):
    digest = hashlib.sha256()
    for value in (vertices_mm, ranges, *(faces[c] for c in CHAMBERS)):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GeometrySpatialUNet(nn.Module):
    """Reuse the existing FFDNet encoder/decoder layers and spatial skips.

    Lifted observations already start at fine control resolution, so omit the
    original dense-CMR conv0/conv1 and change conv2's stride to 1. The original
    FFDNet object is local: no baseline source code or global module is changed.
    Its forward's B=1 indexing/padding are replaced by batched mm adapters.
    """
    def __init__(self, input_channels, source, channels):
        super().__init__()
        self.conv0 = nn.Conv3d(input_channels, channels//2, 3, padding=1)
        for name in ('conv2','conv3','conv4','conv5','deconv5','deconv4','deconv3','deconv2'):
            setattr(self,name,getattr(source,name))
        self.conv2.conv.stride = (1,1,1)

    def forward(self, joint_input):
        fine = self.conv2(self.conv0(joint_input))
        medium = self.conv3(fine)
        coarse = self.conv4(medium)
        x = self.deconv5(self.conv5(coarse))
        x = self.deconv4(torch.cat((x,coarse),1))
        c = x
        x = F.interpolate(x,size=medium.shape[2:],mode='trilinear',align_corners=True)
        x = self.deconv3(torch.cat((x,medium),1))
        m = x
        x = F.interpolate(x,size=fine.shape[2:],mode='trilinear',align_corners=True)
        x = self.deconv2(torch.cat((x,fine),1))
        return dict(coarse=c,medium=m,fine=x)

def control_lipschitz_bound(grid, spacing):
    """Conservative operator bound: derivative columns are convex combinations
    of neighboring control differences / physical spacing for cardinal splines.
    Frobenius bound applies throughout the supported box, not just samples.
    Returns one bound per batch (and per phase for a time-batched field).
    """
    axes = grid.ndim-4
    columns = [(grid.diff(dim=axes+a)/spacing[a]).norm(dim=-1).flatten(axes).amax(-1)
               for a in range(3)]
    return torch.stack(columns, -1).square().sum(-1).sqrt()


class JointFFDHeads(nn.Module):
    def __init__(self, source, frames=24, reference_limit_mm=12., increment_limit_mm=2., max_lipschitz=.45):
        super().__init__()
        if frames != 24 or min(reference_limit_mm, increment_limit_mm) <= 0 or not 0 < max_lipschitz < 1:
            raise ValueError('24 phases and bounded positive physical deformation settings required')
        self.frames = frames
        self.reference_limit_mm, self.increment_limit_mm = reference_limit_mm, increment_limit_mm
        self.max_lipschitz = max_lipschitz
        # Reuse all three existing FFDNet output heads, including their small
        # random initialization; no independent head per chamber or phase.
        self.deform1, self.deform2, self.deform3 = source.deform1, source.deform2, source.deform3
        if self.deform3.conv.out_channels != 3*frames:
            raise ValueError('Source FFDNet head must jointly output 72 channels')

    def _bound(self, grid, limit, spacing):
        grid = limit*torch.tanh(grid/limit)
        bound = control_lipschitz_bound(grid, spacing)
        factor = (self.max_lipschitz/bound.clamp_min(1e-8)).clamp_max(1.)
        return grid*factor[(...,)+(None,)*4]

    def forward(self, features, spacings):
        coarse = self.deform1(features['coarse']).permute(0,2,3,4,1)
        medium = self.deform2(features['medium']).permute(0,2,3,4,1)
        raw = self.deform3(features['fine'])  # exactly one network call
        b, _, gx, gy, gz = raw.shape
        fine = raw.reshape(b,self.frames,3,gx,gy,gz).permute(0,1,3,4,5,2)
        limits = fine.new_full((1,self.frames,1,1,1,1), self.increment_limit_mm)
        limits[:,0] = self.reference_limit_mm
        return dict(coarse=self._bound(coarse,self.reference_limit_mm,spacings[0]),
                    medium=self._bound(medium,self.reference_limit_mm,spacings[1]),
                    fine=self._bound(fine,limits,spacings[2]))


class GeometryHeartFFDNet(nn.Module):
    def __init__(self, template_vertices_mm, chamber_ranges, faces, *, template_sha256,
                 asset_sha256, norm_to_mm, encoder_config=None, lifting_config=None,
                 decoder_config=None, pose_mode='known_pose'):
        super().__init__()
        if pose_mode != 'known_pose' or not math.isfinite(norm_to_mm) or norm_to_mm <= 0:
            raise ValueError('Version 1 requires explicit known_pose and positive mm normalization')
        self.pose_mode, self.norm_to_mm = pose_mode, float(norm_to_mm)
        self.template_sha256, self.asset_sha256 = template_sha256, asset_sha256
        self.encoder_config, self.lifting_config = dict(encoder_config or {}), dict(lifting_config or {})
        settings = dict(frames=24, channels=32, lower_mm=(-120.,-110.,-110.),
            upper_mm=(120.,110.,110.), reference_limit_mm=12., increment_limit_mm=2., max_lipschitz=.45)
        if set(decoder_config or {})-set(settings):
            raise ValueError('Unknown geometry decoder settings')
        settings.update(decoder_config or {})
        self.decoder_config = settings
        self.frames = settings['frames']
        self.grid_shapes = [(6,6,8), (12,12,16), (24,24,32)]
        template = torch.as_tensor(template_vertices_mm).detach().float().clone()
        ranges = torch.as_tensor(chamber_ranges).detach().long().clone()
        if (template.ndim != 2 or template.shape[-1] != 3 or not torch.isfinite(template).all()
                or ranges.shape != (4,2) or ranges[0,0] != 0 or ranges[-1,1] != len(template)
                or (ranges[:,1] <= ranges[:,0]).any() or not torch.equal(ranges[:-1,1],ranges[1:,0])):
            raise ValueError('Finite fixed four-chamber template/topology required')
        self.register_buffer('template_vertices_mm',template)
        self.register_buffer('chamber_ranges',ranges)
        for c,(a,b) in zip(CHAMBERS,ranges.tolist()):
            f = torch.as_tensor(faces[c]).detach().long().clone()
            if f.ndim != 2 or f.shape[-1] != 3 or not f.numel() or f.min() < 0 or f.max() >= b-a:
                raise ValueError('Invalid fixed chamber faces: '+c)
            self.register_buffer('faces_'+c,f)
        self.geometry_sha256 = geometry_digest(template,ranges,self.faces)
        self.register_buffer('lower_mm',torch.tensor(settings['lower_mm'],dtype=torch.float32))
        self.register_buffer('upper_mm',torch.tensor(settings['upper_mm'],dtype=torch.float32))
        geometries = [control_geometry(self.lower_mm,self.upper_mm,shape) for shape in self.grid_shapes]
        self.register_buffer('control_origins_mm',torch.stack([x[0] for x in geometries]))
        self.register_buffer('control_spacings_mm',torch.stack([x[1] for x in geometries]))
        if ((template < self.lower_mm) | (template > self.upper_mm)).any():
            raise ValueError('Template outside physical support')
        axes = [self.control_origins_mm[2,i]+torch.arange(n)*self.control_spacings_mm[2,i]
                for i,n in enumerate(self.grid_shapes[2])]
        self.register_buffer('query_points_mm',torch.stack(torch.meshgrid(*axes,indexing='ij'),-1))
        self.encoder = GeometrySpatialEncoder(**self.encoder_config)
        self.lifting = GeometryFeatureLifting(input_width=self.encoder.width, **self.lifting_config)
        if type(settings['channels']) is not int or settings['channels'] < 4 or settings['channels'] % 4:
            raise ValueError('FFDNet width must be a positive multiple of four')
        source = FFDNet(n_frame=self.frames,dim_hid=settings['channels'])
        self.unet = GeometrySpatialUNet(self.frames*self.lifting.output_width,source,settings['channels'])
        self.heads = JointFFDHeads(source,frames=self.frames,
            reference_limit_mm=settings['reference_limit_mm'],increment_limit_mm=settings['increment_limit_mm'],
            max_lipschitz=settings['max_lipschitz'])

    @property
    def faces(self):
        return {c:getattr(self,'faces_'+c) for c in CHAMBERS}

    @classmethod
    def from_dataset_geometry(cls, geometry, *, norm_to_mm, asset_sha256, **kwargs):
        return cls(geometry['vertices'].double()*norm_to_mm, geometry['chamber_ranges'],geometry['faces'],
            template_sha256=geometry['template_sha256'],asset_sha256=asset_sha256,norm_to_mm=norm_to_mm,**kwargs)

    def _deform(self, vertices, control, level):
        # The existing evaluator rejects out-of-support coordinates; no clipping.
        updated = vertices+bspline_ffd_physical(vertices,control,
            self.control_origins_mm[level],self.control_spacings_mm[level])
        if not torch.isfinite(updated).all() or ((updated < self.lower_mm) | (updated > self.upper_mm)).any():
            raise ValueError('Deformed mesh outside physical support or nonfinite')
        return updated

    def forward(self, observations):
        for key,expected in (('asset_sha256',self.asset_sha256),('template_sha256',self.template_sha256)):
            values = observations.get(key, expected)
            values = [values] if isinstance(values,str) else values
            if any(value != expected for value in values):
                raise ValueError('Observation geometry identity differs: '+key)
        if 'norm_to_mm' in observations and not (torch.as_tensor(observations['norm_to_mm']) == self.norm_to_mm).all():
            raise ValueError('Observation normalization differs from template')
        spatial = self.encoder(observations)
        lifted = self.lifting(spatial, observations,self.query_points_mm,self.norm_to_mm)
        features = self.unet(lifted['features'].flatten(1,2))
        controls = self.heads(features,self.control_spacings_mm)
        b = observations['masks'].shape[0]
        v = self.template_vertices_mm[None].expand(b,-1,-1)
        v = self._deform(v,controls['coarse'],0)
        v = self._deform(v,controls['medium'],1)
        v = self._deform(v,controls['fine'][:,0],2)
        trajectory = [v]
        for phase in range(1,self.frames):
            trajectory.append(self._deform(trajectory[-1],controls['fine'][:,phase],2))
        return dict(vertices_ref=trajectory[0], vertices_4d=torch.stack(trajectory,1),
                    control_grids=controls,pose_mode=self.pose_mode,
                    observation_support_fraction=lifted['support'].float().mean((2,3,4)),
                    feature_shapes={k:list(value.shape) for k,value in features.items()})


def loss_output_adapter(output):
    """Thin naming adapter; preserves every existing nine-term loss calculation."""
    grids = output['control_grids']
    return dict(output,control_grids=dict(coarse=grids['coarse'],medium=grids['medium'],
                                         fine_motion=grids['fine'],regions=None))
