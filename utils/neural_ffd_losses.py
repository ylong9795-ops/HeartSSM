"""Paired four-chamber GT supervision in physical mm and mL.

Kong HeartFFDNet src/loss.py compares predicted local edge/Laplacian geometry
with GT. We retain that idea, using fixed correspondence rather than Chamfer,
and do not force each patient to retain the average-template geometry.
"""
import torch
from torch import nn
from torch.nn import functional as F

from data.mask_transform import CHAMBERS


def chamber_mean(values, ranges):
    return torch.stack([values[...,left:right].mean() for left,right in ranges.tolist()]).mean()


def mesh_volumes_ml(vertices, ranges, faces):
    volumes = []
    for c,(left,right) in zip(CHAMBERS,ranges.tolist()):
        v = vertices[...,left:right,:]
        v = v-v.mean(-2,keepdim=True)
        tri = v[...,faces[c],:]
        cross = torch.cross(tri[...,1,:]-tri[...,0,:],tri[...,2,:]-tri[...,0,:],dim=-1)
        volumes.append((tri[...,0,:]*cross).sum((-1,-2)).abs()/6000.)
    return torch.stack(volumes,-1)


class NeuralFFDLoss(nn.Module):
    def __init__(self, chamber_ranges, faces, *, huber_beta_mm=1., weights=None):
        super().__init__()
        self.register_buffer('chamber_ranges',chamber_ranges.long().clone())
        self.beta = huber_beta_mm
        self.weights = dict(shape=1.,motion=1.,velocity=.2,volume=.01,
                            edge=.05,laplacian=.05,normal=.01,control=.001,slice=.5)
        if weights is not None:
            if set(weights)-set(self.weights):
                raise ValueError('Unknown supervised loss weights')
            self.weights.update(weights)
        if min(self.weights.values())<0 or sum(self.weights.values())<=0 or self.beta<=0:
            raise ValueError('Nonnegative informative loss weights and positive Huber scale required')
        for c,(left,right) in zip(CHAMBERS,self.chamber_ranges.tolist()):
            f = faces[c].long().clone()
            edges = torch.cat((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
            edges = torch.unique(edges.sort(-1).values,dim=0)
            directed = torch.cat((edges,edges.flip(-1)))
            degree = torch.bincount(directed[:,0],minlength=right-left)
            if (degree==0).any():
                raise ValueError('Isolated fixed-topology vertices')
            self.register_buffer('faces_'+c,f)
            self.register_buffer('edges_'+c,edges)
            self.register_buffer('directed_'+c,directed)
            self.register_buffer('degree_'+c,degree.float())

    @property
    def faces(self):
        return {c:getattr(self,'faces_'+c) for c in CHAMBERS}

    def _paired_huber(self,p,g):
        error = F.smooth_l1_loss(p,g,beta=self.beta,reduction='none').mean(-1)
        return chamber_mean(error,self.chamber_ranges)

    def _laplacian(self,vertices,chamber):
        edges = getattr(self,'directed_'+chamber)
        neighbors = torch.zeros_like(vertices).index_add(-2,edges[:,0],vertices[...,edges[:,1],:])
        return vertices-neighbors/getattr(self,'degree_'+chamber)[:,None]

    def forward(self, output, vertices_gt, *, slice_result=None, control_spacings_mm=None,
                regional_spacing_mm=None, stage='joint'):
        p,g = output['vertices_4d'],vertices_gt.to(output['vertices_4d']).detach()
        if (p.shape!=g.shape or p.ndim!=4 or p.shape[-1]!=3 or p.shape[-2]!=self.chamber_ranges[-1,1]
                or not torch.isfinite(p).all() or not torch.isfinite(g).all()):
            raise ValueError('Finite paired [B,T,N,3] prediction/GT required')
        if stage not in ('shape','motion','joint'):
            raise ValueError('Training stage must be shape, motion or joint')
        u,ug = p-p[:,:1],g-g[:,:1]
        values = dict(shape=self._paired_huber(p[:,0],g[:,0]),motion=self._paired_huber(u,ug),
                      velocity=self._paired_huber(p.roll(-1,1)-p,g.roll(-1,1)-g),
                      volume=F.smooth_l1_loss(mesh_volumes_ml(p,self.chamber_ranges,self.faces),
                                            mesh_volumes_ml(g,self.chamber_ranges,self.faces),beta=1.))
        edge_losses,lap_losses,normal_losses = [],[],[]
        for c,(left,right) in zip(CHAMBERS,self.chamber_ranges.tolist()):
            pv,gv = p[...,left:right,:],g[...,left:right,:]
            if stage=='shape':
                pv,gv = pv[:,:1],gv[:,:1]
            edges,f = getattr(self,'edges_'+c),getattr(self,'faces_'+c)
            pe,ge = pv[...,edges[:,1],:]-pv[...,edges[:,0],:],gv[...,edges[:,1],:]-gv[...,edges[:,0],:]
            edge_losses.append((pe-ge).square().mean())
            lap_losses.append((self._laplacian(pv,c)-self._laplacian(gv,c)).square().mean())
            pt,gt = pv[...,f,:],gv[...,f,:]
            pn = F.normalize(torch.cross(pt[...,1,:]-pt[...,0,:],pt[...,2,:]-pt[...,0,:],dim=-1),dim=-1,eps=1e-8)
            gn = F.normalize(torch.cross(gt[...,1,:]-gt[...,0,:],gt[...,2,:]-gt[...,0,:],dim=-1),dim=-1,eps=1e-8)
            normal_losses.append((1-(pn*gn).sum(-1)).mean())
        values.update(edge=torch.stack(edge_losses).mean(),laplacian=torch.stack(lap_losses).mean(),
                      normal=torch.stack(normal_losses).mean())
        controls = output['control_grids']
        penalty = []
        if control_spacings_mm is not None:
            for level,key in enumerate(('coarse','medium','fine_motion')):
                grid = controls[key]
                if key=='fine_motion' and stage=='shape':
                    grid = grid[:,:1]
                for axis in range(3):
                    penalty.append((grid.diff(dim=grid.ndim-4+axis)/control_spacings_mm[level,axis]).square().mean())
            if controls['regions'] is not None:
                if regional_spacing_mm is None:
                    raise ValueError('Physical spacing required for regional control regularization')
                for axis in range(3):
                    penalty.append((controls['regions'].diff(dim=2+axis)/regional_spacing_mm[axis]).square().mean())
        elif self.weights['control']:
            raise ValueError('Physical control spacings required for active control regularization')
        if penalty:
            values['control'] = torch.stack(penalty).mean()
        if slice_result is not None:
            values['slice'] = slice_result['total']
        if stage=='joint' and self.weights['slice'] and slice_result is None:
            raise ValueError('Joint training requires active differentiable slice supervision')
        active = dict(self.weights)
        if stage=='shape':
            active.update(motion=0.,velocity=0.,volume=0.,slice=0.)
        elif stage=='motion':
            active['slice'] = 0.
        total = sum(active[key]*value for key,value in values.items() if active[key])
        if not torch.isfinite(total):
            raise ValueError('Nonfinite supervised objective')
        return dict(total=total,terms=values,active_weights=active,
                    units=dict(shape='mm',motion='mm',velocity='mm per phase step',volume='mL',
                               edge='mm^2',laplacian='mm^2',normal='dimensionless',control='dimensionless',
                               slice='weighted dimensionless Dice/BCE + boundary mm'))


@torch.no_grad()
def reconstruction_metrics(prediction, truth, ranges, faces):
    """Four-chamber trajectory/volume and lightweight topology diagnostics."""
    p,g = prediction,truth.to(prediction)
    if p.shape!=g.shape:
        raise ValueError('Metrics require paired fixed-topology trajectories')
    vp,vg = mesh_volumes_ml(p,ranges,faces),mesh_volumes_ml(g,ranges,faces)
    result = {}
    for i,(c,(left,right)) in enumerate(zip(CHAMBERS,ranges.tolist())):
        pv,gv = p[...,left:right,:],g[...,left:right,:]
        error = (pv-gv).norm(dim=-1)
        u,ug = pv-pv[:,:1],gv-gv[:,:1]
        moving = ug.norm(dim=-1)>1e-3
        cosine = F.cosine_similarity(u,ug,dim=-1,eps=1e-8)
        tri = pv[...,faces[c],:]
        area_twice = torch.cross(tri[...,1,:]-tri[...,0,:],tri[...,2,:]-tri[...,0,:],dim=-1).norm(dim=-1)
        peak = (vp[...,i].argmax(1)-vg[...,i].argmax(1)).abs()
        trough = (vp[...,i].argmin(1)-vg[...,i].argmin(1)).abs()
        period = p.shape[1]
        amplitude_gt = ug.square().sum(-1).mean().sqrt()
        volume_amplitude_gt = (vg[...,i].amax(1)-vg[...,i].amin(1)).mean()
        result[c] = dict(reference_vertex_mae_mm=float(error[:,0].mean()),
            reference_vertex_rmse_mm=float(error[:,0].square().mean().sqrt()),
            trajectory_vertex_mae_mm=float(error.mean()),trajectory_vertex_rmse_mm=float(error.square().mean().sqrt()),
            relative_motion_rmse_mm=float((u-ug).square().sum(-1).mean().sqrt()),
            motion_amplitude_ratio=float(u.square().sum(-1).mean().sqrt()/amplitude_gt) if amplitude_gt>1e-8 else None,
            mean_motion_direction_cosine=float(cosine[moving].mean()) if moving.any() else None,
            volume_mae_ml=float((vp[...,i]-vg[...,i]).abs().mean()),
            volume_amplitude_ratio=float((vp[...,i].amax(1)-vp[...,i].amin(1)).mean()/volume_amplitude_gt) if volume_amplitude_gt>1e-8 else None,
            volume_peak_phase_error=float(torch.minimum(peak,period-peak).float().mean()/period),
            volume_trough_phase_error=float(torch.minimum(trough,period-trough).float().mean()/period),
            closing_step_motion_error_mm=float(((pv[:,0]-pv[:,-1])-(gv[:,0]-gv[:,-1])).square().sum(-1).mean().sqrt()),
            predicted_closing_gap_mm=float((pv[:,0]-pv[:,-1]).norm(dim=-1).mean()),
            gt_closing_gap_mm=float((gv[:,0]-gv[:,-1]).norm(dim=-1).mean()),
            degenerate_triangles=int((area_twice<1e-8).sum()),nonfinite_vertices=int((~torch.isfinite(pv).all(-1)).sum()),
            predicted_volume_curve_ml=vp[...,i].cpu().tolist(),gt_volume_curve_ml=vg[...,i].cpu().tolist())
    return result
