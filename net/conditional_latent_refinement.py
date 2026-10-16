"""Observation-conditioned global anatomical residuals around the effective S1.

The latent is a shared control representation, not a replacement for S1 detail.
No GT posterior/latent optimization occurs in this model. The learned residual
changes the SAME three bounded reference fields; Motion has its own latent path.
"""
import torch
from torch import nn
from torch.nn import functional as F
from net.staged_heartffdnet import StagedHeartFFDNet
from net.geometry_heartffdnet_v2 import bound_control_field

ARCHITECTURE='conditional-anatomical-latent-residual-ffd-v1'
MODEL_KEYS={'masks','sdf','pixel_spacing','native_pixel_spacing','model_to_native','pose_gt','geometry',
    'view_present','annotated','phases','sdf_units','norm_to_mm','asset_sha256','template_sha256'}
NEW_PREFIXES=('shape_latent_head.','shape_latent_trunk.','shape_latent_fields.','motion_latent_head.','motion_decoder.')


class PeriodicLatentMotion(nn.Module):
    def __init__(self,shape_dim,motion_dim,width,grid_shape):
        super().__init__();self.grid_shape=tuple(grid_shape)
        self.trunk=nn.Sequential(nn.Linear(shape_dim+motion_dim+8,width),nn.SiLU())
        self.readout=nn.Linear(width,int(torch.tensor(grid_shape).prod())*3,bias=False)
        nn.init.normal_(self.readout.weight,std=1e-4)
        phase=torch.arange(24).float()[:,None]/24;angle=2*torch.pi*phase*torch.arange(1,5).float()[None]
        self.register_buffer('phase_embedding',torch.cat((angle.sin(),angle.cos()),1))

    def forward(self,shape,motion):
        b=len(shape);phase=self.phase_embedding[None].expand(b,-1,-1)
        combined=torch.cat((shape[:,None].expand(-1,24,-1),motion[:,None].expand(-1,24,-1),phase),-1)
        potential=self.readout(self.trunk(combined))
        return (potential[:,1:]-potential[:,:-1]).reshape(b,23,*self.grid_shape,3)


class ConditionalLatentHeartFFD(StagedHeartFFDNet):
    def install_latent_residual(self,*,shape_dim=64,motion_dim=64,hidden=128,motion_width=32):
        if hasattr(self,'shape_latent_head') or hasattr(self,'geometry_certificate'):raise ValueError('Verified original S1 required')
        if (self.heads.max_lipschitz,self.heads.reference_limit_mm,self.heads.increment_limit_mm)!=(.45,12.,2.):raise ValueError('Original .45/12/2mm required')
        self.latent_specification=dict(shape_dim=shape_dim,motion_dim=motion_dim,hidden=hidden,motion_width=motion_width)
        pooled=self.decoder_config['channels']*3*3*4
        def encoder(dim):return nn.Sequential(nn.Linear(pooled,hidden),nn.SiLU(),nn.Linear(hidden,dim))
        self.shape_latent_head=encoder(shape_dim);self.motion_latent_head=encoder(motion_dim)
        self.shape_latent_trunk=nn.Sequential(nn.Linear(shape_dim,hidden),nn.SiLU())
        self.shape_latent_fields=nn.ModuleList([nn.Linear(hidden,int(torch.tensor(s).prod())*3) for s in self.grid_shapes])
        for head in self.shape_latent_fields:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.motion_decoder=PeriodicLatentMotion(shape_dim,motion_dim,motion_width,self.grid_shapes[2])
        self.to(self.template_vertices_mm.device);self.set_latent_stage('shape')

    def set_latent_stage(self,stage):
        if stage not in ['shape','motion']:raise ValueError('Explicit Shape/Motion latent stage required')
        self.requires_grad_(False);self.latent_stage=stage
        modules=[self.shape_latent_head,self.shape_latent_trunk,self.shape_latent_fields] if stage=='shape' else [self.motion_latent_head,self.motion_decoder]
        for m in modules:m.requires_grad_(True)
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        for name in ['encoder','lifting','unet','heads']:
            getattr(self,name).eval()
        if hasattr(self,'latent_stage'):
            for name in ['shape_latent_head','shape_latent_trunk','shape_latent_fields']:
                getattr(self,name).train(mode and self.latent_stage=='shape')
            self.motion_latent_head.train(mode and self.latent_stage=='motion');self.motion_decoder.train(mode and self.latent_stage=='motion')
        return self

    @staticmethod
    def migrate_strict_stage1(model,**specification):
        if type(model) is not StagedHeartFFDNet:raise ValueError('Verified original Stage S required')
        before={n:v.detach().clone() for n,v in model.state_dict().items()}
        model.__class__=ConditionalLatentHeartFFD;model.install_latent_residual(**specification)
        for n,v in before.items():
            if not torch.equal(v,model.state_dict()[n]):raise ValueError('Inherited S1 tensor changed: '+n)
        if any(not n.startswith(NEW_PREFIXES) for n in set(model.state_dict())-set(before)):raise ValueError('Undeclared latent state')
        return model

    def observation_context(self,observations):
        observations={k:v for k,v in observations.items() if k in MODEL_KEYS}
        for key,expected in [('asset_sha256',self.asset_sha256),('template_sha256',self.template_sha256)]:
            values=observations.get(key,expected);values=[values] if isinstance(values,str) else values
            if any(value!=expected for value in values):raise ValueError('Observation geometry identity differs: '+key)
        if 'norm_to_mm' in observations and not (torch.as_tensor(observations['norm_to_mm'])==self.norm_to_mm).all():
            raise ValueError('Observation normalization differs from template')
        with torch.no_grad():
            spatial=self.encoder(observations);lifted=self.lifting(spatial,observations,self.query_points_mm,self.norm_to_mm)
            features=self.unet(lifted['features'].flatten(1,2));base=self.heads(features,self.control_spacings_mm);base['fine']=base['fine'][:,:1]
        return dict(features=features['fine'],base_controls=base)

    def reference_from_context(self,context):
        feature=context['features'];base=context['base_controls'];pooled=F.adaptive_avg_pool3d(feature,(3,3,4)).flatten(1)
        if feature.requires_grad or any(v.requires_grad for v in base.values()):raise ValueError('Detached frozen S1 context required')
        shape=self.shape_latent_head(pooled);hidden=self.shape_latent_trunk(shape);controls={}
        for i,key in enumerate(['coarse','medium','fine']):
            anchor=base[key][:,0] if key=='fine' else base[key]
            residual=self.shape_latent_fields[i](hidden).reshape_as(anchor)
            # Inverse tanh preserves the strong predicted base to the declared
            # original1e-4mm numeric contract, not an identity/template reset.
            ratio=(anchor/12).clamp(-1+torch.finfo(anchor.dtype).eps,1-torch.finfo(anchor.dtype).eps)
            raw=12*torch.atanh(ratio)+residual
            bounded=bound_control_field(self.heads,raw,12.,self.control_spacings_mm[i])
            controls[key]=bounded[:,None] if key=='fine' else bounded
        v=self.template_vertices_mm[None].expand(len(shape),-1,-1)
        for i,key in enumerate(['coarse','medium','fine']):v=self._deform(v,controls[key][:,0] if key=='fine' else controls[key],i)
        return dict(vertices_ref=v,vertices_4d=v[:,None],control_grids=controls,pose_mode=self.pose_mode,shape_latent=shape)

    def reference_output(self,observations):return self.reference_from_context(self.observation_context(observations))

    def cycle_from_context(self,context):
        reference=self.reference_from_context(context);shape=reference['shape_latent'];feature=context['features']
        pooled=F.adaptive_avg_pool3d(feature,(3,3,4)).flatten(1);motion=self.motion_latent_head(pooled)
        raw=self.motion_decoder(shape,motion);fields=bound_control_field(self.heads,raw,2.,self.control_spacings_mm[2])
        v=reference['vertices_ref'];frames=[v]
        for field in fields.unbind(1):v=self._deform(v,field,2);frames.append(v)
        controls=dict(reference['control_grids'],fine=torch.cat((reference['control_grids']['fine'],fields),1))
        return dict(reference,vertices_4d=torch.stack(frames,1),control_grids=controls,motion_latent=motion,motion_control_grids=fields)

    def cycle_output(self,observations):return self.cycle_from_context(self.observation_context(observations))
