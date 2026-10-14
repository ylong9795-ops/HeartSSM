"""V2: phase-specific local conditions + shared context -> ONE joint FFD head.

Frozen v1 supplies the CNN, spatial U-Net, reference heads, mm geometry and FFD
operator. Additional modules preserve view/time identity, sampled 224px slice
SDF, direct/propagated/unknown support, and explicit Eulerian temporal changes.
"""
import torch
from torch import nn
from torch.nn import functional as F
from net.module import ConvBlock
from net.geometry_heartffdnet import GeometryHeartFFDNet,control_lipschitz_bound
from net.geometry_observation_lifting import GeometryObservationLifting
from net.geometry_view_fusion import SupportAwareViewFusion
from net.local_temporal_motion import (LocalTemporalMotionConditioner,FineLocalMotionSkip,
                                       resample_physical_features)

ARCHITECTURE = 'geometry-local-temporal-joint-ffd-v2'


def bound_control_field(head,raw,limit,spacing):
    """Reuse frozen active bound; exact tanh derivative when it is inactive.

    The certificate's sqrt(0) has an undefined autograd derivative for uniform
    translations. In the inactive region the complete mapping is simply tanh;
    using that expression preserves the output and its proper derivative.
    """
    flat=raw.reshape(-1,*raw.shape[-4:])
    limits=torch.as_tensor(limit,device=raw.device,dtype=raw.dtype).expand_as(raw).reshape_as(flat)
    result=limits*torch.tanh(flat/limits)
    with torch.no_grad():active=control_lipschitz_bound(result.detach(),spacing)>head.max_lipschitz
    if active.any():result[active]=head._bound(flat[active],limits[active],spacing)
    return result.reshape_as(raw)


class TimeConditionedJointFFDGenerator(nn.Module):
    def __init__(self, context_width, temporal_width=8, fine_width=8, phase_width=8):
        super().__init__()
        self.phase_width = phase_width
        self.phase_project = nn.Conv3d(context_width+temporal_width+fine_width,phase_width,1)
        self.readout = ConvBlock(24*phase_width,72)
        # One grouped 72-channel call preserves the explicit phase identity.
        # Shared per-phase extraction and circular temporal mixing precede it.
        self.readout.conv = nn.Conv3d(24*phase_width,72,3,padding=1,groups=24)
        nn.init.normal_(self.readout.conv.weight,std=1e-5)
        nn.init.zeros_(self.readout.conv.bias)

    def forward(self, context, motion, fine_skip):
        b,t = motion.shape[:2]
        if t != 24 or fine_skip.shape[:2] != (b,t):
            raise ValueError('Complete 24-phase local conditions required')
        shared = context[:,None].expand(-1,t,-1,-1,-1,-1)
        phase_input = torch.cat((shared,motion,fine_skip),2)
        phase_features = F.silu(self.phase_project(phase_input.flatten(0,1))).reshape(b,t,self.phase_width,*context.shape[-3:])
        # Single network call yields 72 channels; no per-phase decode loop.
        return self.readout(phase_features.flatten(1,2))


class GeometryHeartFFDNetV2(GeometryHeartFFDNet):
    def __init__(self,*args,motion_config=None,**kwargs):
        super().__init__(*args,**kwargs)
        settings = dict(variant='A4',temporal_width=8,fine_width=8,phase_width=8,
                        direct_half_thickness_mm=.75)
        if set(motion_config or {})-set(settings):
            raise ValueError('Unknown local temporal motion configuration')
        settings.update(motion_config or {})
        if settings['variant'] not in ('A1','A2','A3','A4') or any(type(settings[x]) is not int or settings[x]<1 for x in ('temporal_width','fine_width','phase_width')):
            raise ValueError('Explicit supported ablation and positive integer widths required')
        self.motion_config = settings
        self.variant = settings['variant']
        self.use_sdf = self.variant in ('A2','A3','A4')
        self.use_fusion = self.variant in ('A3','A4')
        self.use_fine_skip = self.variant == 'A4'
        self.observation_lifting = GeometryObservationLifting(sigma_mm=self.lifting.sigma_mm,
            cutoff_sigma=self.lifting.cutoff_sigma,direct_half_thickness_mm=settings['direct_half_thickness_mm'],
            sdf_scale_mm=self.encoder.sdf_scale_mm)
        view_width = 2*self.lifting.projected_width+4
        self.view_fusion = SupportAwareViewFusion(view_width)
        # Per-view evidence + conflict + support metadata has a stable width
        # across ablations, so corresponding initialization remains comparable.
        local_width = self.lifting.output_width+4*view_width+8
        self.local_temporal = LocalTemporalMotionConditioner(local_width,settings['temporal_width'])
        self.fine_motion_skip = FineLocalMotionSkip(view_width*3,settings['fine_width'])
        self.heads.deform3 = TimeConditionedJointFFDGenerator(self.decoder_config['channels'],
            settings['temporal_width'],settings['fine_width'],settings['phase_width'])
        axes = [self.control_origins_mm[1,i]+torch.arange(n)*self.control_spacings_mm[1,i]
                for i,n in enumerate(self.grid_shapes[1])]
        self.register_buffer('medium_query_points_mm',torch.stack(torch.meshgrid(*axes,indexing='ij'),-1))

    def _local_observation(self,spatial,observations,query,legacy):
        lifted = self.observation_lifting(spatial,observations,query,self.norm_to_mm,self.lifting.projections)
        if not self.use_sdf:
            lifted = dict(lifted,view_features=lifted['view_features'].clone())
            lifted['view_features'][:,:,:, -4:] = 0.
        fused = self.view_fusion(lifted,learned=self.use_fusion)
        extra = fused['features']
        validity = fused['feature_validity']
        if not self.use_fusion:
            # A1 adds temporal legacy conditions; A2 adds only per-view SDF.
            extra = extra.clone();extra[:,:,:4*(2*self.lifting.projected_width+4)] = 0.
            if self.use_sdf:
                width = 2*self.lifting.projected_width+4
                for view in range(3):
                    sl = slice(view*width+width-4,view*width+width)
                    extra[:,:,sl] = lifted['view_features'][:,:,view,-4:]
        available = legacy['support'][:,:,None]
        legacy_valid = available.expand_as(legacy['features'])
        features = torch.cat((legacy['features'],extra),2)
        valid = torch.cat((legacy_valid,validity),2)
        difference_masks = {}
        fine_difference_masks = {}
        signature = lifted['chamber_support'].flatten(2,3)
        annotation = lifted['channel_annotated']
        view_width = lifted['view_features'].shape[3]
        for name in ('reference','previous'):
            other_signature = signature[:,:1].expand_as(signature) if name == 'reference' else signature.roll(1,1)
            other_annotation = annotation[:,:1].expand_as(annotation) if name == 'reference' else annotation.roll(1,1)
            same_sources = (signature == other_signature).all(2)
            allowed = torch.ones_like(valid)
            allowed[:,:,:self.lifting.output_width] = same_sources[:,:,None]
            # The conflict statistic also depends on which views are present.
            start = self.lifting.output_width+3*view_width
            allowed[:,:,start:start+view_width] = same_sources[:,:,None]
            fine_allowed = torch.ones_like(lifted['feature_validity'].flatten(2,3))
            same_annotations = (annotation == other_annotation).all(-1)
            for view in range(3):
                cnn_slice = slice(view*view_width,view*view_width+view_width-4)
                same = same_annotations[:,:,view,None,None,None,None]
                fine_allowed[:,:,cnn_slice] = same
                allowed[:,:,slice(self.lifting.output_width+cnn_slice.start,self.lifting.output_width+cnn_slice.stop)] = same
            difference_masks[name] = allowed
            fine_difference_masks[name] = fine_allowed
        return dict(features=features,validity=valid,lifted=lifted,fused=fused,
                    difference_masks=difference_masks,fine_difference_masks=fine_difference_masks)

    def encode_conditions(self,observations):
        # Reuse frozen v1 observation identity checks without executing its
        # forward (that would duplicate the CNN and produce unwanted controls).
        for key,expected in (('asset_sha256',self.asset_sha256),('template_sha256',self.template_sha256)):
            values = observations.get(key,expected)
            values = [values] if isinstance(values,str) else values
            if any(x != expected for x in values):raise ValueError('Observation geometry identity differs: '+key)
        if 'norm_to_mm' in observations and not (torch.as_tensor(observations['norm_to_mm'])==self.norm_to_mm).all():
            raise ValueError('Physical normalization differs')
        spatial = self.encoder(observations)
        spatial_fp32 = dict(spatial,scale14=spatial['scale14'].float(),scale7=spatial['scale7'].float())
        with torch.autocast(device_type=self.template_vertices_mm.device.type,enabled=False):
            legacy_fine = self.lifting(spatial_fp32,observations,self.query_points_mm,self.norm_to_mm)
        context = self.unet(legacy_fine['features'].flatten(1,2))
        with torch.autocast(device_type=self.template_vertices_mm.device.type,enabled=False):
            legacy_medium = self.lifting(spatial_fp32,observations,self.medium_query_points_mm,self.norm_to_mm)
            medium = self._local_observation(spatial_fp32,observations,self.medium_query_points_mm,legacy_medium)
        temporal = self.local_temporal(medium['features'],medium['validity'],observations['phases'],medium['difference_masks'])
        motion = resample_physical_features(temporal['features'],self.query_points_mm,
                                            self.control_origins_mm[1],self.control_spacings_mm[1])
        with torch.autocast(device_type=self.template_vertices_mm.device.type,enabled=False):
            fine = self._local_observation(spatial_fp32,observations,self.query_points_mm,legacy_fine)
        fine_views = fine['lifted']['view_features'].flatten(2,3)
        fine_valid = fine['lifted']['feature_validity'].flatten(2,3)
        skip = (self.fine_motion_skip(fine_views,fine_valid,observations['phases'],fine['fine_difference_masks']) if self.use_fine_skip
                else motion.new_zeros(*motion.shape[:2],self.motion_config['fine_width'],*motion.shape[-3:]))
        return dict(spatial=spatial,context=context,motion=motion,fine_skip=skip,
                    medium=medium,fine=fine,temporal=temporal,legacy_fine=legacy_fine)

    def generate_controls(self,encoded,*,motion_override=None,fine_skip_override=None):
        features = encoded['context']
        coarse = self.heads.deform1(features['coarse']).permute(0,2,3,4,1).float()
        medium = self.heads.deform2(features['medium']).permute(0,2,3,4,1).float()
        raw = self.heads.deform3(features['fine'],encoded['motion'] if motion_override is None else motion_override,
                               encoded['fine_skip'] if fine_skip_override is None else fine_skip_override).float()
        b,_,gx,gy,gz = raw.shape
        fine = raw.reshape(b,24,3,gx,gy,gz).permute(0,1,3,4,5,2)
        limits = fine.new_full((1,24,1,1,1,1),self.heads.increment_limit_mm);limits[:,0] = self.heads.reference_limit_mm
        if not all(torch.isfinite(x).all() for x in (coarse,medium,fine)):
            raise ValueError('Nonfinite raw neural control fields')
        controls = dict(coarse=bound_control_field(self.heads,coarse,self.heads.reference_limit_mm,self.control_spacings_mm[0]),
            medium=bound_control_field(self.heads,medium,self.heads.reference_limit_mm,self.control_spacings_mm[1]),
            fine=bound_control_field(self.heads,fine,limits,self.control_spacings_mm[2]))
        tanh_field = limits*torch.tanh(fine/limits)
        lip = control_lipschitz_bound(tanh_field,self.control_spacings_mm[2])
        stats = dict(tanh_saturation_fraction=(fine.abs()/limits>1.8318).float().mean((2,3,4,5)),
            lipschitz_triggered=lip>self.heads.max_lipschitz,
            lipschitz_scale=(self.heads.max_lipschitz/lip.clamp_min(1e-8)).clamp_max(1.),
            raw_control_rms_mm=fine.square().mean((2,3,4,5)).sqrt(),
            final_control_rms_mm=controls['fine'].square().mean((2,3,4,5)).sqrt())
        return controls,stats

    def apply_controls(self,controls):
        b = controls['coarse'].shape[0]
        v = self._deform(self.template_vertices_mm[None].expand(b,-1,-1),controls['coarse'],0)
        v = self._deform(v,controls['medium'],1)
        trajectory = [self._deform(v,controls['fine'][:,0],2)]
        for phase in range(1,24):trajectory.append(self._deform(trajectory[-1],controls['fine'][:,phase],2))
        return torch.stack(trajectory,1)

    def forward(self,observations,*,return_features=False):
        encoded = self.encode_conditions(observations)
        # Physical control/FFD arithmetic remains float32 under CUDA AMP.
        controls,stats = self.generate_controls(encoded)
        with torch.autocast(device_type=self.template_vertices_mm.device.type,enabled=False):
            vertices = self.apply_controls(controls)
        result = dict(vertices_ref=vertices[:,0],vertices_4d=vertices,control_grids=controls,
            pose_mode=self.pose_mode,constraint_statistics=stats,
            observation_support_fraction=encoded['legacy_fine']['support'].float().mean((2,3,4)),
            view_fusion_weights=encoded['fine']['fused']['weights'],
            support_regions={k:encoded['fine']['fused'][k] for k in ('direct','propagated','unknown')},
            feature_shapes=dict(motion=list(encoded['motion'].shape),fine_skip=list(encoded['fine_skip'].shape)))
        if return_features:result['features'] = encoded
        return result
