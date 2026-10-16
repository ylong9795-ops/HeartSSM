"""Shared learned Shape/Motion latents -> original bounded physical FFD.

The GT encoders are training-only posterior encoders. Conditional inference
accepts only frozen observation features and uses separate observation latent
heads. No patient-specific control optimization or GT projection is involved.
This is a deterministic learned latent manifold, not yet a probabilistic VAE.
"""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from net.geometry_heartffdnet import JointFFDHeads
from net.geometry_heartffdnet_v2 import bound_control_field
from utils.ffd_physical import bspline_ffd_physical

ARCHITECTURE='learned-shape-periodic-motion-latent-ffd-v1'


class LatentHeartFFDPrior(nn.Module):
    def __init__(self,geometry,*,shape_dim=64,motion_dim=64,hidden=128,motion_width=32):
        super().__init__()
        if min(shape_dim,motion_dim,hidden,motion_width)<1:raise ValueError('Positive learned latent dimensions required')
        if (geometry.heads.max_lipschitz,geometry.heads.reference_limit_mm,geometry.heads.increment_limit_mm)!=(.45,12.,2.):
            raise ValueError('Original .45/12/2mm geometry contract required')
        self.shape_dim,self.motion_dim=int(shape_dim),int(motion_dim)
        self.hidden,self.motion_width=int(hidden),int(motion_width)
        self.observation_width=geometry.decoder_config['channels']
        self.norm_to_mm=geometry.norm_to_mm;self.pose_mode='known_pose';self.frames=24
        self.template_sha256,self.asset_sha256=geometry.template_sha256,geometry.asset_sha256
        self.max_lipschitz=.45
        self.heads=SimpleNamespace(max_lipschitz=.45,reference_limit_mm=12.,increment_limit_mm=2.)
        self.grid_shapes=list(geometry.grid_shapes)
        for name in ['template_vertices_mm','chamber_ranges','lower_mm','upper_mm','control_origins_mm','control_spacings_mm']:
            self.register_buffer(name,getattr(geometry,name).detach().clone())
        for name,value in geometry.faces.items():self.register_buffer('faces_'+name,value.detach().clone())
        n=len(self.template_vertices_mm);pool_width=self.observation_width*3*3*4
        self.shape_posterior=nn.Sequential(nn.Linear(n*3,hidden),nn.SiLU(),nn.Linear(hidden,shape_dim))
        self.motion_spatial_posterior=nn.Linear(n*3,motion_width)
        self.motion_temporal_posterior=nn.Sequential(nn.Conv1d(motion_width,motion_width,3,padding=1,padding_mode='circular'),nn.SiLU())
        self.motion_posterior=nn.Linear(24*motion_width,motion_dim)
        self.observation_shape=nn.Sequential(nn.Linear(pool_width,hidden),nn.SiLU(),nn.Linear(hidden,shape_dim))
        self.observation_motion=nn.Sequential(nn.Linear(pool_width,hidden),nn.SiLU(),nn.Linear(hidden,motion_dim))
        self.shape_trunk=nn.Sequential(nn.Linear(shape_dim,hidden),nn.SiLU())
        self.shape_readouts=nn.ModuleList([nn.Linear(hidden,int(torch.tensor(s).prod())*3) for s in self.grid_shapes])
        # One shared nonlinear phase-conditioned decoder, not23 copied models.
        self.motion_trunk=nn.Sequential(nn.Linear(shape_dim+motion_dim+8,motion_width),nn.SiLU())
        self.motion_readout=nn.Linear(motion_width,int(torch.tensor(self.grid_shapes[2]).prod())*3,bias=False)
        phase=torch.arange(24,dtype=torch.float32)/24
        angle=2*torch.pi*phase[:,None]*torch.arange(1,5,dtype=torch.float32)[None]
        self.register_buffer('phase_embedding',torch.cat((angle.sin(),angle.cos()),1))
        for head in [*self.shape_readouts,self.motion_readout]:
            nn.init.normal_(head.weight,std=1e-4)
            if head.bias is not None:nn.init.zeros_(head.bias)
        self._shape_frozen=False

    @property
    def faces(self):return {c:getattr(self,'faces_'+c) for c in ['LV','LA','RV','RA']}

    def _bound(self,raw,limit,spacing):
        return JointFFDHeads._bound(self,raw,limit,spacing)

    def _deform(self,vertices,field,level):
        return vertices+bspline_ffd_physical(vertices,field,self.control_origins_mm[level],self.control_spacings_mm[level])

    def encode_shape_gt(self,vertices):
        if vertices.ndim!=3 or vertices.shape[1:]!=self.template_vertices_mm.shape or not torch.isfinite(vertices).all():
            raise ValueError('Training posterior requires finite[B,N,3] complete reference GT')
        return self.shape_posterior(((vertices-self.template_vertices_mm)/self.norm_to_mm).flatten(1))

    def encode_motion_gt(self,vertices):
        if vertices.ndim!=4 or vertices.shape[1:]!=(24,*self.template_vertices_mm.shape) or not torch.isfinite(vertices).all():
            raise ValueError('Training posterior requires finite[B,24,N,3] complete cycle GT')
        relative=(vertices-vertices[:,:1])/self.norm_to_mm
        spatial=self.motion_spatial_posterior(relative.flatten(2))
        temporal=self.motion_temporal_posterior(spatial.transpose(1,2)).transpose(1,2)
        return self.motion_posterior(temporal.flatten(1))

    def observation_latents(self,features):
        if features.ndim!=5 or features.shape[1]!=self.observation_width or not torch.isfinite(features).all():
            raise ValueError('Finite frozen observation feature tensor[B,C,X,Y,Z] required; no GT input')
        pooled=F.adaptive_avg_pool3d(features,(3,3,4)).flatten(1)
        return dict(shape=self.observation_shape(pooled),motion=self.observation_motion(pooled))

    def reference_controls(self,shape_latent):
        if shape_latent.ndim!=2 or shape_latent.shape[1]!=self.shape_dim:raise ValueError('Declared shape latent required')
        hidden=self.shape_trunk(shape_latent);controls=[]
        for i,(head,shape) in enumerate(zip(self.shape_readouts,self.grid_shapes)):
            raw=head(hidden).reshape(-1,*shape,3)
            controls.append(bound_control_field(self,raw,12.,self.control_spacings_mm[i]))
        return controls

    def motion_controls(self,shape_latent,motion_latent):
        if motion_latent.shape!=(len(shape_latent),self.motion_dim):raise ValueError('Declared motion latent required')
        b=len(shape_latent);phase=self.phase_embedding[None].expand(b,-1,-1)
        combined=torch.cat((shape_latent[:,None].expand(-1,24,-1),motion_latent[:,None].expand(-1,24,-1),phase),-1)
        potential=self.motion_readout(self.motion_trunk(combined))
        # Periodic phase potential is an inductive bias. Nonlinear bounding
        # and recursive evaluation do NOT imply exact vertex-cycle closure.
        increments=(potential[:,1:]-potential[:,:-1]).reshape(b,23,*self.grid_shapes[2],3)
        return bound_control_field(self,increments,2.,self.control_spacings_mm[2])

    def forward(self,shape_latent,motion_latent=None):
        c,m,f=self.reference_controls(shape_latent)
        v=self.template_vertices_mm[None].expand(len(shape_latent),-1,-1)
        for i,field in enumerate([c,m,f]):v=self._deform(v,field,i)
        reference=v;frames=[v];fine=f[:,None];incremental=None
        if motion_latent is not None:
            incremental=self.motion_controls(shape_latent,motion_latent)
            for field in incremental.unbind(1):
                v=self._deform(v,field,2);frames.append(v)
            fine=torch.cat((fine,incremental),1)
        return dict(vertices_ref=reference,vertices_4d=torch.stack(frames,1),
            control_grids=dict(coarse=c,medium=m,fine=fine),pose_mode=self.pose_mode,
            motion_control_grids=incremental,shape_latent=shape_latent,motion_latent=motion_latent)

    def conditional(self,observation_features,*,with_motion=True):
        z=self.observation_latents(observation_features)
        return self(z['shape'],z['motion'] if with_motion else None)

    def freeze_shape(self):
        for module in [self.shape_posterior,self.observation_shape,self.shape_trunk,self.shape_readouts]:
            module.requires_grad_(False);module.eval()
        self._shape_frozen=True
        return self

    def shape_modules(self):return [self.shape_posterior,self.observation_shape,self.shape_trunk,self.shape_readouts]

    def motion_modules(self):return [self.motion_spatial_posterior,self.motion_temporal_posterior,self.motion_posterior,self.observation_motion,self.motion_trunk,self.motion_readout]
