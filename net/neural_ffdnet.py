"""HeartSSM-style coarse/medium/fine + incremental motion FFD.

Legacy net.ffdnet.FFDNet is intentionally untouched. No dense image, GT mesh,
SSM coefficients or acquisition pose is read by the decoder. Optional reference
spatial tokens condition the shared Medium grid without a latent bottleneck.
"""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from data.mask_transform import CHAMBERS
from net.multiview_encoder import MultiViewTemporalEncoder
from net.module import ConvBlock
from utils.ffd_physical import bspline_ffd_physical, control_geometry

ARCHITECTURE = 'neural-heartffd-incremental-v1'


def geometry_digest(vertices_mm, ranges, faces):
    digest = hashlib.sha256()
    for x in (vertices_mm.float(), ranges.long(), *[faces[c].long() for c in CHAMBERS]):
        digest.update(x.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _small_output(layer):
    nn.init.normal_(layer.weight, 0., 1e-3)
    nn.init.zeros_(layer.bias)


class ConditionedConv(nn.Module):
    """Original norm/activation/3D-conv idea, with FiLM AFTER normalization."""
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.norm = nn.InstanceNorm3d(channels, affine=False)
        self.condition = nn.Linear(latent_dim, channels*2)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
        nn.init.normal_(self.condition.weight, 0., .02)
        nn.init.zeros_(self.condition.bias)

    def forward(self, x, latent):
        gamma, beta = self.condition(latent).chunk(2, -1)
        value = self.norm(x)*(1+gamma[...,None,None,None])+beta[...,None,None,None]
        return self.conv(F.silu(value))


class FineMotionHead(nn.Module):
    """deform3 reference field and T-1 finite-phase incremental motion fields.

    Shares the spatial reference feature grid, but reference() accepts no
    motion code. This is a fixed-phase decoder, not a continuous-time model.
    """
    def __init__(self, channels, shape_dim, motion_dim, frames):
        super().__init__()
        self.reference = nn.Conv3d(channels, 3, 3, padding=1)
        self.norm = nn.InstanceNorm3d(channels, affine=False)
        self.phase_embedding = nn.Embedding(frames-1, channels)
        self.condition = nn.Sequential(nn.Linear(shape_dim+motion_dim+channels, channels*2), nn.SiLU(),
                                       nn.Linear(channels*2, channels*2))
        self.motion = nn.Conv3d(channels, 3, 3, padding=1)
        _small_output(self.reference)
        _small_output(self.motion)

    def motion_field(self, features, shape, motion, phase_index):
        phase = self.phase_embedding.weight[phase_index][None].expand(shape.shape[0],-1)
        gamma, beta = self.condition(torch.cat((shape,motion,phase), -1)).chunk(2,-1)
        h = self.norm(features)*(1+gamma[...,None,None,None])+beta[...,None,None,None]
        return self.motion(F.silu(h)).permute(0,2,3,4,1)


class ReferenceSpatialFusion(nn.Module):
    """One spatial 3D-query/2D-token attention residual at Medium resolution.

    Grid indices follow the existing FFD (x,y,z) tensor axes. Normalized
    positions identify queries, not probe pose or physical slice mappings.
    This is learned spatial conditioning, not geometric backprojection.
    """
    def __init__(self, channels, token_width, grid_shape, width=32, heads=4):
        super().__init__()
        axes = [torch.linspace(-1,1,n) for n in grid_shape]
        self.register_buffer('positions', torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3),
                             persistent=False)
        self.query = nn.Linear(channels,width)
        self.position = nn.Linear(3,width,bias=False)
        self.tokens = nn.Sequential(nn.LayerNorm(token_width),nn.Linear(token_width,width))
        self.attention = nn.MultiheadAttention(width,heads,batch_first=True)
        self.output = nn.Linear(width,channels,bias=False)
        # First update trains this output; subsequent updates reach attention
        # and the shared CNN. Inherited predictions are initially identical.
        nn.init.zeros_(self.output.weight)

    def forward(self, features, reference_spatial):
        tokens,valid = reference_spatial['tokens'],reference_spatial['valid']
        if (tokens.ndim!=3 or tokens.shape[0]!=features.shape[0]
                or tokens.shape[-1]!=self.tokens[0].normalized_shape[0]
                or valid.shape!=tokens.shape[:2] or valid.dtype!=torch.bool
                or tokens.device!=features.device or tokens.dtype!=features.dtype
                or valid.device!=features.device):
            raise ValueError('Reference tokens [B,N,C] and boolean validity [B,N] must match Medium features')
        present = valid.any(-1)
        delta = torch.zeros_like(features)
        # Avoid all-masked softmax, and make absent references exactly neutral
        # even after training. Unobserved tokens cannot inject NaNs or biases.
        if present.any():
            value = self.tokens(torch.where(valid[present,:,None],tokens[present],0.))
            query = self.query(features[present].flatten(2).transpose(1,2)) + self.position(self.positions)
            routed = self.attention(query,value,value,key_padding_mask=~valid[present],need_weights=False)[0]
            delta[present] = self.output(routed).transpose(1,2).reshape_as(features[present])
        return features + delta


class DynamicSpatialConditioner(nn.Module):
    """Shared adjacent-frame reads in a learned 3D query space.

    A broad, learnable view-specific coordinate bias distinguishes queries.
    Coordinates are normalized lattice/image indices, NOT source probe pose
    or a physical backprojection. Attention locality is an inductive bias,
    never evidence of anatomical correspondence by itself.
    """
    def __init__(self, channels, token_width, grid_shape, spatial_sides,
                 width=16, heads=4, frame_chunk=2, query_chunk=128, sigma=.6):
        super().__init__()
        if (min(width,heads,frame_chunk,query_chunk)<1 or width%heads
                or not math.isfinite(sigma) or not .2<=sigma<=1.5):
            raise ValueError('Compatible dynamic attention sizes and broad finite spatial sigma required')
        self.grid_shape=tuple(grid_shape)
        self.width,self.heads=width,heads
        self.frame_chunk,self.query_chunk=frame_chunk,query_chunk
        axes=[torch.linspace(-1,1,n) for n in grid_shape]
        self.register_buffer('positions',torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3),persistent=False)
        token_positions,token_views=[],[]
        for side in spatial_sides:
            y,x=torch.meshgrid(torch.linspace(-1,1,side),torch.linspace(-1,1,side),indexing='ij')
            token_positions.append(torch.stack((x,y),-1).reshape(-1,2).repeat(3,1))
            token_views.append(torch.arange(3).repeat_interleave(side*side))
        self.register_buffer('token_positions',torch.cat(token_positions),persistent=False)
        self.register_buffer('token_views',torch.cat(token_views),persistent=False)
        self.query=nn.Linear(channels,width)
        self.position=nn.Linear(3,width,bias=False)
        self.tokens=nn.Sequential(nn.LayerNorm(token_width),nn.Linear(token_width,width))
        self.attention=nn.MultiheadAttention(width,heads,batch_first=True)
        maps=torch.zeros(3,2,3)
        for view in range(3):
            angle=view*math.pi/3
            maps[view,0,0],maps[view,0,1]=math.cos(angle),math.sin(angle)
            maps[view,1,2]=1.
        self.query_to_uv=nn.Parameter(maps)
        self.query_uv_offset=nn.Parameter(torch.zeros(3,2))
        self.log_sigma=nn.Parameter(torch.full((3,),math.log(sigma)))
        self.local=ConvBlock(width,width,K=3)
        self.output=nn.Conv3d(width,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def spatial_bias(self):
        anchors=torch.tanh(torch.einsum('qd,vud->qvu',self.positions,self.query_to_uv)+self.query_uv_offset)
        delta=anchors[:,self.token_views]-self.token_positions[None]
        sigma=self.log_sigma.exp().clamp(.2,1.5)[self.token_views]
        return -delta.square().sum(-1)/(2*sigma.square()[None])

    def _attend(self,q,tokens,mask):
        return self.attention(q,tokens,tokens,attn_mask=mask,need_weights=False)[0]

    def _read(self,query,values,view_valid,bias):
        # [B,F,N,W]; F<=frame_chunk. Same queries and parameters for every frame.
        b,frames,n,_=values.shape
        key_valid=view_valid[:,None,:,].expand(-1,frames,-1)[...,self.token_views].reshape(b*frames,n)
        present=key_valid.any(-1)
        output=query.new_zeros(b*frames,query.shape[1],self.width)
        if present.any():
            q=query[:,None].expand(-1,frames,-1,-1).reshape(b*frames,-1,self.width)[present]
            value=values.reshape(b*frames,n,self.width)[present]
            valid=key_valid[present]
            pieces=[]
            for start in range(0,q.shape[1],self.query_chunk):
                part=q[:,start:start+self.query_chunk]
                mask=bias[None,start:start+self.query_chunk].expand(len(part),-1,-1)
                mask=mask.masked_fill(~valid[:,None],-torch.inf)
                mask=mask[:,None].expand(-1,self.heads,-1,-1).reshape(-1,part.shape[1],n)
                routed=(checkpoint(self._attend,part,value,mask,use_reentrant=False)
                    if self.training and torch.is_grad_enabled() else self._attend(part,value,mask))
                pieces.append(routed)
            output[present]=torch.cat(pieces,1)
        return output.reshape(b,frames,-1,self.width)

    def query_changes(self,features,dynamic_spatial):
        tokens,valid,channels=(dynamic_spatial[k] for k in ('tokens','valid','channel_valid'))
        b=features.shape[0]
        if (tokens.ndim!=4 or tokens.shape[0]!=b or tokens.shape[2]!=len(self.token_views)
                or tokens.shape[3]!=self.tokens[0].normalized_shape[0]
                or valid.shape!=tokens.shape[:3] or valid.dtype!=torch.bool
                or channels.shape!=(b,tokens.shape[1],3,4) or channels.dtype!=torch.bool
                or any(v.device!=features.device for v in (tokens,valid,channels))
                or tokens.dtype!=features.dtype):
            raise ValueError('Dynamic tokens [B,T,N,W], token validity and per-frame channel validity required')
        if not torch.equal(valid,channels.any(-1)[...,self.token_views]):
            raise ValueError('Dynamic token and channel observation validity differ')
        if not torch.isfinite(tokens[valid]).all():
            raise ValueError('Finite valid dynamic tokens required')
        # CNN maps mix channels. When annotation semantics change, discard the
        # entire view for this pair rather than pretending its missing channel
        # is empty or re-running the CNN with a different input.
        pairs=(channels[:,1:]==channels[:,:-1]).all(-1)&channels[:,1:].any(-1)&channels[:,:-1].any(-1)
        values=self.tokens(torch.where(valid[...,None],tokens,0.))
        medium=F.adaptive_avg_pool3d(features,self.grid_shape)
        query=self.query(medium.flatten(2).transpose(1,2))+self.position(self.positions)[None]
        bias=self.spatial_bias()
        if (pairs==pairs[:,:1]).all():
            # Normal full-cycle data: read each of the 24 frames exactly once.
            reads=torch.cat([self._read(query,values[:,start:start+self.frame_chunk],pairs[:,0],bias)
                for start in range(0,tokens.shape[1],self.frame_chunk)],1)
            difference=reads[:,1:]-reads[:,:-1]
        else:
            # Reuse any identical (frame, common-view-mask) query result.
            cache={}
            changes=[]
            for phase in range(tokens.shape[1]-1):
                mask=pairs[:,phase]
                signature=tuple(mask.reshape(-1).tolist())
                adjacent=[]
                for frame in (phase,phase+1):
                    key=(frame,signature)
                    if key not in cache:
                        cache[key]=self._read(query,values[:,frame:frame+1],mask,bias)[:,0]
                    adjacent.append(cache[key])
                changes.append(adjacent[1]-adjacent[0])
            difference=torch.stack(changes,1)
        return difference.permute(0,1,3,2).reshape(b,tokens.shape[1]-1,self.width,*self.grid_shape)

    def increment(self,features,change):
        # Preserve temporal-change magnitude across the local InstanceNorm.
        # A truly unchanged query read remains exactly neutral, even if the
        # convolution develops a bias. Norm has a defined zero gradient at 0.
        scale=torch.linalg.vector_norm(change.flatten(1),dim=-1)/math.sqrt(change[0].numel())
        raw=self.output(self.local(change))*scale[:,None,None,None,None]
        mean=raw.flatten(2).mean(-1)[...,None,None,None]
        medium=raw-mean
        fine=F.interpolate(medium,size=features.shape[-3:],mode='trilinear',align_corners=True)
        fine=fine-fine.flatten(2).mean(-1)[...,None,None,None]
        # Centering excludes broadcast modes by construction; it is not proof
        # of learned anatomical selectivity. Fine/mesh response is tested too.
        total=raw.square().flatten(1).mean(-1)
        centered=medium.square().flatten(1).mean(-1)
        stats=torch.stack((scale,centered.sqrt(),fine.square().flatten(1).mean(-1).sqrt(),
            centered/(total+1e-20)),dim=-1).detach()
        return features+fine,stats


class NeuralFFDNet(nn.Module):
    def __init__(self, template_vertices_mm, chamber_ranges, faces, *, template_sha256,
                 asset_sha256, norm_to_mm, encoder_config=None, decoder_config=None):
        super().__init__()
        self.encoder_config = dict(encoder_config or {})
        self.decoder_config = dict(decoder_config or {})
        self.encoder = MultiViewTemporalEncoder(**self.encoder_config)
        self.shape_dim, self.motion_dim = self.encoder.shape_dim, self.encoder.motion_dim
        settings = dict(frames=24, channels=32, seed_shape=(6,6,8),
                        lower_mm=(-120.,-110.,-110.), upper_mm=(120.,110.,110.),
                        regional_shape=True, global_shape_dim=32, regional_grid_shape=(4,4,4),
                        regional_motion=False,regional_motion_grid_shape=(5,5,5),
                        regional_motion_lower_mm=None,regional_motion_upper_mm=None,
                        regional_motion_hidden=32, spatial_fusion='none',dynamic_spatial='none',
                        dynamic_spatial_width=16,dynamic_spatial_heads=4,dynamic_spatial_frame_chunk=2,
                        dynamic_spatial_query_chunk=128,dynamic_spatial_sigma=.6)
        unknown = set(self.decoder_config)-set(settings)
        if unknown:
            raise ValueError('Unknown decoder settings: '+str(sorted(unknown)))
        settings.update(self.decoder_config)
        self.decoder_config = settings
        self.frames, channels = settings['frames'], settings['channels']
        if settings['spatial_fusion'] not in ('none','reference_cross_attention_v1'):
            raise ValueError('Unknown spatial fusion mode')
        if settings['dynamic_spatial'] not in ('none','query_difference_v1'):
            raise ValueError('Unknown dynamic spatial conditioning mode')
        if settings['dynamic_spatial']!='none' and (settings['spatial_fusion']!='none' or settings['regional_motion']):
            raise ValueError('Dynamic experiment requires the original reference and shared Motion head')
        self.seed_shape = tuple(settings['seed_shape'])
        if self.frames < 2 or channels < 1 or len(self.seed_shape)!=3 or min(self.seed_shape)<4:
            raise ValueError('At least two phases and a 3D seed >=4 nodes per axis required')
        self.regional_shape = settings['regional_shape']
        self.regional_motion = settings['regional_motion']
        if type(self.regional_motion) is not bool or (self.regional_motion and not self.regional_shape):
            raise ValueError('Regional motion requires the existing chamber Shape code partition')
        self.global_shape_dim = settings['global_shape_dim'] if self.regional_shape else self.shape_dim
        if (self.global_shape_dim < 1 or self.global_shape_dim > self.shape_dim
                or (self.regional_shape and (self.shape_dim-self.global_shape_dim < 4
                                             or (self.shape_dim-self.global_shape_dim)%4))):
            raise ValueError('Shape code must partition into global and four equal nonempty chamber codes')
        self.chamber_dim = (self.shape_dim-self.global_shape_dim)//4
        self.template_sha256, self.asset_sha256 = template_sha256, asset_sha256
        self.norm_to_mm = float(norm_to_mm)
        if not math.isfinite(self.norm_to_mm) or self.norm_to_mm <= 0:
            raise ValueError('Positive physical asset normalization required')
        template = torch.as_tensor(template_vertices_mm).detach().float().clone()
        ranges = torch.as_tensor(chamber_ranges).detach().long().clone()
        if (template.ndim!=2 or template.shape[-1]!=3 or not torch.isfinite(template).all()
                or ranges.shape!=(4,2) or ranges[0,0]!=0 or ranges[-1,1]!=len(template)
                or (ranges[:,1]<=ranges[:,0]).any() or not torch.equal(ranges[:-1,1], ranges[1:,0])):
            raise ValueError('Finite fixed-topology four-chamber template required')
        self.register_buffer('template_vertices_mm', template)
        self.register_buffer('chamber_ranges', ranges)
        for c,(left,right) in zip(CHAMBERS,ranges.tolist()):
            value = torch.as_tensor(faces[c]).detach().long().clone()
            if value.ndim!=2 or value.shape[-1]!=3 or value.numel()==0 or value.min()<0 or value.max()>=right-left:
                raise ValueError('Invalid local chamber faces: '+c)
            self.register_buffer('faces_'+c, value)
        self.geometry_sha256 = geometry_digest(template,ranges,self.faces)
        lower,upper = torch.tensor(settings['lower_mm'],dtype=torch.float32), torch.tensor(settings['upper_mm'],dtype=torch.float32)
        self.register_buffer('lower_mm', lower)
        self.register_buffer('upper_mm', upper)
        if ((template<lower)|(template>upper)).any():
            raise ValueError('Template outside the configured physical FFD box')
        self.grid_shapes = [tuple(n*scale for n in self.seed_shape) for scale in (1,2,4)]
        geometries = [control_geometry(lower,upper,shape) for shape in self.grid_shapes]
        self.register_buffer('control_origins_mm', torch.stack([g[0] for g in geometries]))
        self.register_buffer('control_spacings_mm', torch.stack([g[1] for g in geometries]))
        self.seed = nn.Linear(self.global_shape_dim, channels*math.prod(self.seed_shape))
        self.deconv5 = ConditionedConv(channels,self.global_shape_dim)
        self.deconv4 = ConditionedConv(channels,self.global_shape_dim)
        self.deconv3 = ConditionedConv(channels,self.global_shape_dim)
        self.deconv2 = ConditionedConv(channels,self.global_shape_dim)
        self.deform1 = nn.Conv3d(channels,3,3,padding=1)
        self.deform2 = nn.Conv3d(channels,3,3,padding=1)
        _small_output(self.deform1)
        _small_output(self.deform2)
        self.deform3 = FineMotionHead(channels,self.shape_dim,self.motion_dim,self.frames)
        if self.regional_shape:
            self.regional_grid_shape = tuple(settings['regional_grid_shape'])
            o,h = control_geometry(lower,upper,self.regional_grid_shape)
            self.register_buffer('regional_origin_mm',o)
            self.register_buffer('regional_spacing_mm',h)
            self.regional_heads = nn.ModuleList([nn.Linear(self.chamber_dim,math.prod(self.regional_grid_shape)*3)
                                                for _ in CHAMBERS])
            for head in self.regional_heads:
                _small_output(head)
        if self.regional_motion:
            self.regional_motion_grid_shape = tuple(settings['regional_motion_grid_shape'])
            hidden = settings['regional_motion_hidden']
            if (len(self.regional_motion_grid_shape)!=3 or any(type(n) is not int or n<4 for n in self.regional_motion_grid_shape)
                    or type(hidden) is not int or hidden<1
                    or settings['regional_motion_lower_mm'] is None or settings['regional_motion_upper_mm'] is None):
                raise ValueError('Regional motion requires explicit fixed physical boxes and small integer grids/hidden dimension')
            region_lower = torch.as_tensor(settings['regional_motion_lower_mm'],dtype=torch.float32)
            region_upper = torch.as_tensor(settings['regional_motion_upper_mm'],dtype=torch.float32)
            if region_lower.shape!=(4,3) or region_upper.shape!=(4,3):
                raise ValueError('Regional motion bounds must be [4,3] in LV/LA/RV/RA order, mm')
            geometries = [control_geometry(lo,hi,self.regional_motion_grid_shape) for lo,hi in zip(region_lower,region_upper)]
            for i,(left,right) in enumerate(ranges.tolist()):
                if ((template[left:right]<region_lower[i])|(template[left:right]>region_upper[i])).any():
                    raise ValueError('Template chamber outside its fixed regional motion support')
            self.register_buffer('regional_motion_lower_mm',region_lower)
            self.register_buffer('regional_motion_upper_mm',region_upper)
            self.register_buffer('regional_motion_origins_mm',torch.stack([g[0] for g in geometries]))
            self.register_buffer('regional_motion_spacings_mm',torch.stack([g[1] for g in geometries]))
            self.regional_motion_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(self.motion_dim+self.chamber_dim+channels,hidden),nn.SiLU(),
                              nn.Linear(hidden,math.prod(self.regional_motion_grid_shape)*3)) for _ in CHAMBERS])
            for head in self.regional_motion_heads:
                # Exactly preserve the inherited trajectory at initialization.
                # Output layers train immediately; inner/latent regional paths
                # gain gradients after that first ordinary optimizer update.
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

        # Append after all inherited modules to preserve their initialization.
        if settings['spatial_fusion'] == 'reference_cross_attention_v1':
            self.spatial_fusion = ReferenceSpatialFusion(channels,self.encoder.width,self.grid_shapes[1])
        else:
            self.spatial_fusion = None
        self.dynamic_conditioner=(DynamicSpatialConditioner(channels,self.encoder.width,self.grid_shapes[1],
            self.encoder.spatial_sides,width=settings['dynamic_spatial_width'],heads=settings['dynamic_spatial_heads'],
            frame_chunk=settings['dynamic_spatial_frame_chunk'],query_chunk=settings['dynamic_spatial_query_chunk'],
            sigma=settings['dynamic_spatial_sigma']) if settings['dynamic_spatial']!='none' else None)

    @property
    def faces(self):
        return {c:getattr(self,'faces_'+c) for c in CHAMBERS}

    @classmethod
    def from_dataset_geometry(cls, geometry, *, norm_to_mm, asset_sha256, **kwargs):
        return cls(geometry['vertices'].double()*norm_to_mm,geometry['chamber_ranges'],geometry['faces'],
                   norm_to_mm=norm_to_mm,asset_sha256=asset_sha256,
                   template_sha256=geometry['template_sha256'],**kwargs)

    def encode(self, observations):
        if observations['masks'].shape[1] != self.frames:
            raise ValueError('Observation cycle length differs from fixed decoder phases')
        for key,expected in (('asset_sha256',self.asset_sha256),('template_sha256',self.template_sha256)):
            if key in observations:
                values = observations[key]
                values = [values] if isinstance(values,str) else values
                if any(v != expected for v in values):
                    raise ValueError('Observation '+key+' differs from decoder geometry')
        if 'norm_to_mm' in observations:
            scales = torch.as_tensor(observations['norm_to_mm'])
            if not torch.isfinite(scales).all() or not (scales==self.norm_to_mm).all():
                raise ValueError('Observation physical normalization differs from decoder asset')
        return self.encoder(observations,return_reference_spatial=self.spatial_fusion is not None,
            return_dynamic_spatial=self.dynamic_conditioner is not None)

    def _check_latent(self, latent, dim):
        if (latent.ndim!=2 or latent.shape[1]!=dim or latent.dtype!=self.template_vertices_mm.dtype
                or latent.device!=self.template_vertices_mm.device or not torch.isfinite(latent).all()):
            raise ValueError('Finite latent with matching batch/dimension/device/dtype required')

    def _deform(self, vertices, controls, level):
        return vertices+bspline_ffd_physical(vertices,controls,self.control_origins_mm[level],self.control_spacings_mm[level])

    def _reference(self, z_shape, reference_spatial=None):
        self._check_latent(z_shape,self.shape_dim)
        global_code = z_shape[:,:self.global_shape_dim]
        h = self.seed(global_code).reshape(z_shape.shape[0],-1,*self.seed_shape)
        h = self.deconv4(self.deconv5(h,global_code),global_code)
        coarse = self.deform1(h).permute(0,2,3,4,1)
        v = self._deform(self.template_vertices_mm[None].expand(z_shape.shape[0],-1,-1),coarse,0)
        h = self.deconv3(F.interpolate(h,size=self.grid_shapes[1],mode='trilinear',align_corners=True),global_code)
        if reference_spatial is not None:
            if self.spatial_fusion is None:
                raise ValueError('Reference spatial tokens require an enabled fusion module')
            h = self.spatial_fusion(h,reference_spatial)
        medium = self.deform2(h).permute(0,2,3,4,1)
        v = self._deform(v,medium,1)
        h = self.deconv2(F.interpolate(h,size=self.grid_shapes[2],mode='trilinear',align_corners=True),global_code)
        fine = self.deform3.reference(h).permute(0,2,3,4,1)
        v = self._deform(v,fine,2)
        regions = None
        if self.regional_shape:
            codes = z_shape[:,self.global_shape_dim:].reshape(z_shape.shape[0],4,self.chamber_dim)
            regions = torch.stack([head(codes[:,i]).reshape(z_shape.shape[0],*self.regional_grid_shape,3)
                                   for i,head in enumerate(self.regional_heads)],1)
            v = self.apply_regions(v,regions)
        return v,h,coarse,medium,fine,regions

    def apply_regions(self, vertices, controls):
        """Fixed one-hot chamber supports; each displacement is a smooth FFD.

        The four chambers are separate surface components. Constant support
        within each component never creates view-specific or free-vertex meshes.
        """
        if not self.regional_shape or controls.shape != (vertices.shape[0],4,*self.regional_grid_shape,3):
            raise ValueError('Four fixed chamber-region control fields required')
        parts = []
        for i,(left,right) in enumerate(self.chamber_ranges.tolist()):
            v = vertices[:,left:right]
            parts.append(v+bspline_ffd_physical(v,controls[:,i],self.regional_origin_mm,self.regional_spacing_mm))
        return torch.cat(parts,1)

    def decode_shape(self, z_shape, reference_spatial=None):
        """Omitting reference_spatial explicitly evaluates the latent-only path."""
        return self._reference(z_shape,reference_spatial)[0]

    def apply_motion_regions(self,vertices,controls):
        """Smooth mm increments with fixed chamber supports and fixed domains."""
        if not self.regional_motion or controls.shape != (vertices.shape[0],4,*self.regional_motion_grid_shape,3):
            raise ValueError('Four fixed regional motion control fields required')
        parts,displacement_rms = [],[]
        for i,(left,right) in enumerate(self.chamber_ranges.tolist()):
            v = vertices[:,left:right]
            delta = bspline_ffd_physical(v,controls[:,i],self.regional_motion_origins_mm[i],self.regional_motion_spacings_mm[i])
            parts.append(v+delta)
            displacement_rms.append(delta.square().sum(-1).mean(-1).sqrt())
        return torch.cat(parts,1),torch.stack(displacement_rms,-1)

    def decode_4d(self, z_shape, z_motion, *, reference_spatial=None, dynamic_spatial=None, return_details=False):
        self._check_latent(z_motion,self.motion_dim)
        if z_shape.shape[0]!=z_motion.shape[0]:
            raise ValueError('Shape and motion batches differ')
        reference,h,coarse,medium,fine,regions = self._reference(z_shape,reference_spatial)
        changes=None
        dynamic_stats=[]
        if dynamic_spatial is not None:
            if self.dynamic_conditioner is None or dynamic_spatial['tokens'].shape[1]!=self.frames:
                raise ValueError('Enabled dynamic module and matching complete cycle required')
            changes=self.dynamic_conditioner.query_changes(h,dynamic_spatial)
        trajectory,fields = [reference],[fine]
        regional_fields,regional_rms = [],[]
        if self.regional_motion:
            codes = z_shape[:,self.global_shape_dim:].reshape(z_shape.shape[0],4,self.chamber_dim)
            regional_fields.append(reference.new_zeros(z_shape.shape[0],4,*self.regional_motion_grid_shape,3))
            regional_rms.append(reference.new_zeros(z_shape.shape[0],4))
        for phase in range(self.frames-1):
            features=h
            if changes is not None:
                features,stats=self.dynamic_conditioner.increment(h,changes[:,phase])
                dynamic_stats.append(stats)
            field = self.deform3.motion_field(features,z_shape,z_motion,phase)
            updated = self._deform(trajectory[-1],field,2)
            if self.regional_motion:
                time = self.deform3.phase_embedding.weight[phase][None].expand(z_shape.shape[0],-1)
                controls = torch.stack([head(torch.cat((z_motion,codes[:,i],time),-1)).reshape(
                    z_shape.shape[0],*self.regional_motion_grid_shape,3) for i,head in enumerate(self.regional_motion_heads)],1)
                updated,rms = self.apply_motion_regions(updated,controls)
                regional_fields.append(controls)
                regional_rms.append(rms)
            trajectory.append(updated)
            fields.append(field)
        vertices = torch.stack(trajectory,1)
        if not torch.isfinite(vertices).all():
            raise ValueError('Nonfinite decoded 4D vertices')
        if not return_details:
            return vertices
        return dict(vertices_ref=reference,vertices_4d=vertices,
                    dynamic_statistics=torch.stack(dynamic_stats,1) if dynamic_stats else None,
                    regional_motion_rms_mm=torch.stack(regional_rms,1) if self.regional_motion else None,
                    control_grids=dict(coarse=coarse,medium=medium,fine_motion=torch.stack(fields,1),regions=regions,
                        regional_motion=torch.stack(regional_fields,1) if self.regional_motion else None))

    def forward(self, observations):
        latents = self.encode(observations)
        output = self.decode_4d(**latents,return_details=True)
        output.update(latents)
        return output
