"""Reference-frame native 2D objective and explicit anatomy soft regularisers.

These soft terms are not geometry certificates or real 3D supervision. Every
candidate is separately checked with the original hard GeometryGuard.
"""
import numpy as np
import torch
from torch.nn import functional as F

from data.mask_transform import CHAMBERS
from utils.geometry_pair_policy import SourceAVCapPolicy, SETTINGS
from utils.slice_supervision import section_soft_mask
from ..alignment.centres import centroid
from .observations import observation_cells, native_cell


class RegionalShapeObjective:
    def __init__(self, model, patient, state, config, policy):
        self.model,self.config=model,config
        device=model.template_vertices_mm.device
        self.pose={k:v.detach().to(device=device,dtype=torch.float32) for k,v in state.observation_pose.items()}
        self.rows=[]
        for row in observation_cells([patient],policy,phases=[0]):
            if not row['used']:continue
            mask,sdf,spacing=native_cell(patient,row)
            if not mask.any():continue
            h,w=mask.shape;stride=config['pixel_stride']
            yy,xx=torch.meshgrid((torch.arange(0,h,stride,device=device)-h/2)*float(spacing[0]),
                (torch.arange(0,w,stride,device=device)-w/2)*float(spacing[1]),indexing='ij')
            truth=torch.as_tensor(mask[::stride,::stride],device=device,dtype=torch.float32)
            if not truth.any():raise ValueError('S2 stride misses annotated chamber foreground')
            mu=(centroid(mask)-[w/2,h/2])*spacing[::-1]
            self.rows.append(dict(evidence=row,truth=truth,field=torch.as_tensor(sdf[::stride,::stride],device=device),
                centroid=torch.as_tensor(mu,device=device,dtype=torch.float32),
                xy=torch.stack((xx,yy),-1),spacing=spacing,shape=(h,w)))
        if not self.rows:raise ValueError('S2 reference phase has no annotated foreground')
        self.initial=state.vertices_reference.detach().to(device)
        self.local=[]
        for chamber,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist()):
            faces=model.faces[chamber]
            edges=torch.unique(torch.cat((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])).sort(-1).values,dim=0)
            v=self.initial[a:b];tri=v[faces]
            normal=F.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
            self.local.append(dict(a=a,b=b,faces=faces,edges=edges,
                edge=(v[edges[:,0]]-v[edges[:,1]]).norm(dim=-1).clamp_min(1e-6),normal=normal))
        policy_source=SourceAVCapPolicy(model,SETTINGS)
        self.av=[]
        for ac,bc in (('LV','LA'),('RV','RA')):
            ai,bi=CHAMBERS.index(ac),CHAMBERS.index(bc)
            a0,a1=model.chamber_ranges[ai].tolist();b0,b1=model.chamber_ranges[bi].tolist()
            aa=np.unique(policy_source.faces[ac][policy_source.caps[ac]])
            bb=np.unique(policy_source.faces[bc][policy_source.caps[bc]])
            # The immutable source cap support selects the neighbourhood; no
            # outcome-dependent expansion of allowed contact is performed.
            aa=aa[np.linspace(0,len(aa)-1,min(64,len(aa))).round().astype(int)]
            ia=torch.as_tensor(aa,device=device)+a0;ib=torch.as_tensor(bb,device=device)+b0
            nearest=torch.cdist(self.initial[ia],self.initial[ib]).argmin(-1)
            ib=ib[nearest]
            self.av.append((ia,ib,self.initial[ia]-self.initial[ib]))
        self.cap_policy_identity=policy_source.identity

    def __call__(self, vertices, raw_delta):
        v=vertices[0]
        chamber_terms={}
        missing=0
        for record in self.rows:
            row=record['evidence'];vi,ci=row['view'],row['chamber'];a,b=self.model.chamber_ranges[ci].tolist()
            output=section_soft_mask(v[a:b],self.model.faces[CHAMBERS[ci]],self.pose['R'][vi],self.pose['center_mm'][vi],
                record['spacing'],record['shape'],sigma_mm=self.config['sigma_mm'],
                pixel_stride=self.config['pixel_stride'],chunk_size=self.config['chunk_size'])
            p,truth=output['soft_mask'],record['truth']
            mass=p.sum()
            if output['logits'] is None:
                missing+=1
                centre=v.sum()*0+self.config['missing_center_penalty']
            else:
                mu=(p[...,None]*record['xy']).sum((0,1))/mass.clamp_min(1e-8)
                centre=F.smooth_l1_loss(mu/self.config['center_scale_mm'],
                    record['centroid']/self.config['center_scale_mm'],beta=.3)
            terms=dict(center=centre,dice=1-(2*(p*truth).sum()+1)/(mass+truth.sum()+1),
                boundary_mm=((p-truth).abs()*record['field'].abs()).mean())
            chamber_terms.setdefault(ci,[]).append((terms,row['confidence']))
        components={}
        for name in ('center','dice','boundary_mm'):
            components[name]=torch.stack([sum(t[name]*w for t,w in rows)/sum(w for _,w in rows)
                                         for rows in chamber_terms.values()]).mean()
        edges=[];flips=[]
        for r in self.local:
            points=v[r['a']:r['b']];e=r['edges'];tri=points[r['faces']]
            edges.append(((points[e[:,0]]-points[e[:,1]]).norm(dim=-1)/r['edge']-1).square().mean())
            normal=F.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
            flips.append(F.relu(.1-(normal*r['normal']).sum(-1)).square().mean())
        av=torch.stack([F.smooth_l1_loss(v[a]-v[b],reference,beta=1.) for a,b,reference in self.av]).mean()
        anchor=F.smooth_l1_loss(v,self.initial,beta=1.)
        latent=raw_delta.square().mean()
        anatomy=(self.config['edge_prior_weight']*torch.stack(edges).mean()+
            self.config['normal_prior_weight']*torch.stack(flips).mean()+self.config['av_prior_weight']*av+
            self.config['surface_prior_weight']*anchor+self.config['latent_prior_weight']*latent)
        components.update(anatomy=anatomy,av_neighbourhood=av,surface_anchor=anchor,latent_prior=latent)
        total=sum(self.config['weights'][k]*components[k] for k in ('center','boundary_mm','dice','anatomy'))
        return dict(total=total,components=components,missing_sections=missing,
            evaluated_chambers=sorted(chamber_terms),reference_cells=len(self.rows))
