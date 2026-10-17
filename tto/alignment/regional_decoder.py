"""Exact original Regional Shape decoder with the Global32 branch cached."""
import numpy as np
import torch
from utils.ffd_physical import bspline_ffd_physical


class FixedGlobalRegionalDecoder:
    def __init__(self, prior, initial_shape, conditions, regional_std):
        self.prior, self.model=prior,prior.model
        m=self.model
        if not m.regional_shape or (m.shape_dim,m.global_shape_dim,m.chamber_dim)!=(64,32,8):
            raise ValueError('S2 requires checkpoint-verified Global32 + four Regional8 codes')
        self.initial=initial_shape.detach().to(prior.device).clone()
        self.conditions=conditions
        self.std=torch.as_tensor(regional_std,device=prior.device,dtype=self.initial.dtype).reshape(4,8)
        if not torch.isfinite(self.std).all() or (self.std<=0).any(): raise ValueError('Positive TRAIN regional scale required')
        with torch.no_grad():
            ref,_,coarse,medium,fine,_=m._reference(self.initial,reference_spatial=conditions.get('reference_spatial'))
            v=m.template_vertices_mm[None]
            for level,controls in enumerate((coarse,medium,fine)): v=m._deform(v,controls,level)
            self.pre_regional=v.detach()
            self.reference=ref.detach()
            bases=[]
            for ci,(a,b) in enumerate(m.chamber_ranges.tolist()):
                controls=m.regional_heads[ci].weight.T.reshape(8,*m.regional_grid_shape,3)
                points=v[:,a:b].expand(8,-1,-1)
                field=bspline_ffd_physical(points,controls,m.regional_origin_mm,m.regional_spacing_mm)
                bases.append(field.permute(1,2,0).contiguous())
            self.bases=bases
            # Gauge in NORMALISED latent coordinates; rank generally three.
            common=torch.cat([basis.mean(0)*self.std[ci][None]/4 for ci,basis in enumerate(bases)],dim=1)
            c=common.double()
            projection=torch.eye(32,device=prior.device,dtype=torch.float64)-c.T@torch.linalg.pinv(c@c.T)@c
            self.projection=projection.to(self.initial)
            self.common_constraint=common
        if not torch.equal(self.decode(torch.zeros(4,8,device=prior.device))[0], self.reference):
            raise AssertionError('Cached Regional decoder differs from original reference')

    def shape_code(self, raw_normalized_delta):
        change=(self.projection@raw_normalized_delta.reshape(32))*self.std.reshape(32)
        return torch.cat((self.initial[:,:32],self.initial[:,32:]+change[None]),dim=1)

    def decode(self, raw_normalized_delta):
        z=self.shape_code(raw_normalized_delta)
        codes=z[:,32:].reshape(1,4,8)
        controls=torch.stack([head(codes[:,ci]).reshape(1,*self.model.regional_grid_shape,3)
                              for ci,head in enumerate(self.model.regional_heads)],dim=1)
        return self.model.apply_regions(self.pre_regional,controls),z

    @torch.no_grad()
    def fit_hint(self, displacement, ridge):
        raw=[]
        for ci,basis in enumerate(self.bases):
            a=(basis*self.std[ci][None,None]).reshape(-1,8).double()
            target=torch.as_tensor(displacement[ci],device=a.device,dtype=a.dtype).expand(len(basis),3).reshape(-1)
            scale=max(1,len(basis))
            q=torch.linalg.solve(a.T@a/scale+ridge*torch.eye(8,device=a.device),a.T@target/scale)
            raw.append(q.to(self.initial))
        return torch.stack(raw)

    @torch.no_grad()
    def assert_original(self, raw_normalized_delta):
        v,z=self.decode(raw_normalized_delta)
        direct=self.prior.decode_shape(z,self.conditions)
        if not torch.equal(v,direct):
            raise AssertionError(f'Cached regional path changed original Decoder: {(v-direct).abs().max().item()}')
        return z
