"""Paired four-chamber GT supervision in physical mm and mL.

Kong HeartFFDNet src/loss.py compares predicted local edge/Laplacian geometry
with GT. We retain that idea, using fixed correspondence rather than Chamfer,
and do not force each patient to retain the average-template geometry.
"""
import math

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
    """Stage-aware GT objectives, combined by one ordinary backward pass.

    ``L_S`` contains reference/trajectory GT positions, local GT structure and
    reference control regularity. ``L_M`` contains relative GT motion, cyclic
    velocity, volume and motion control regularity. ``L_O`` is an unweighted
    observation objective; ``objective_weights`` specifies its outer weight.
    Internal Shape/Motion component weights remain in ``active_weights``.

    grouped_gt_boundary_v1 adds all-phase position supervision in Joint and
    balances physical boundary mean + upper-quartile tail across observed
    chambers. legacy_v1 preserves saved configurations and pretraining stages.
    """
    def __init__(self, chamber_ranges, faces, *, huber_beta_mm=1., weights=None,
                 objective_mode='legacy_v1', observation_tail_fraction=.25,
                 observation_tail_weight=1., observation_boundary_threshold_mm=None):
        super().__init__()
        self.register_buffer('chamber_ranges',chamber_ranges.long().clone())
        self.beta = huber_beta_mm
        if objective_mode not in ('legacy_v1', 'grouped_gt_boundary_v1'):
            raise ValueError('Unknown supervised objective mode')
        if (not math.isfinite(observation_tail_fraction) or not 0 < observation_tail_fraction <= 1
                or not math.isfinite(observation_tail_weight) or observation_tail_weight < 0):
            raise ValueError('Observation tail fraction must be in (0,1] and tail weight finite/nonnegative')
        self.objective_mode = objective_mode
        self.observation_tail_fraction = observation_tail_fraction
        self.observation_tail_weight = observation_tail_weight
        if observation_boundary_threshold_mm is not None:
            raise ValueError('Uncalibrated boundary_mm thresholds are disabled; require TRAIN-only view/chamber calibration and validated proxy ranking')
        self.observation_boundary_threshold_mm = None
        self.weights = dict(shape=1.,motion=1.,velocity=.2,volume=.01,
                            edge=.05,laplacian=.05,normal=.01,control=.001,slice=.5,
                            trajectory=1. if objective_mode == 'grouped_gt_boundary_v1' else 0.)
        if weights is not None:
            if set(weights)-set(self.weights):
                raise ValueError('Unknown supervised loss weights')
            self.weights.update(weights)
        if (not all(math.isfinite(value) for value in self.weights.values())
                or min(self.weights.values())<0 or sum(self.weights.values())<=0
                or not math.isfinite(self.beta) or self.beta<=0):
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

    def _observation_objective(self, slices, prediction):
        """Unreduced physical boundary residuals; no Dice ranking surrogate."""
        policy = slices.get('diagnostics', {})
        if (policy.get('boundary_objective') != 'paired_boundary_distance_v1'
                or policy.get('target_mode') != 'paired_soft_geometry_v1'):
            raise ValueError('Grouped observations require paired continuous GT boundary distances')
        indices = slices.get('cell_indices')
        boundary = slices.get('cell_terms', {}).get('boundary_mm')
        if (not isinstance(indices, torch.Tensor) or indices.ndim != 2 or indices.shape[1] != 4
                or indices.dtype != torch.long or not len(indices)
                or not isinstance(boundary, torch.Tensor) or boundary.shape != (len(indices),)
                or boundary.device != prediction.device or indices.device != prediction.device
                or not torch.isfinite(boundary).all() or (boundary < 0).any()):
            raise ValueError('Finite nonnegative per-cell boundary_mm and [C,4] integer indices required')
        bounds = indices.new_tensor([prediction.shape[0], prediction.shape[1], 3, 4])
        if ((indices < 0) | (indices >= bounds)).any() or len(indices.unique(dim=0)) != len(indices):
            raise ValueError('Observation cells must be unique valid patient/phase/view/chamber indices')
        means, tails, counts, tail_counts = [], [], [], []
        for chamber in range(4):
            values = boundary[indices[:, 3] == chamber]
            count = len(values)
            k = max(1, math.ceil(count*self.observation_tail_fraction)) if count else 0
            means.append(values.mean() if count else prediction.sum()*0)
            tails.append(values.topk(k).values.mean() if count else prediction.sum()*0)
            counts.append(count)
            tail_counts.append(k)
        means, tails = torch.stack(means), torch.stack(tails)
        counts = indices.new_tensor(counts)
        present = counts > 0
        objective = (means[present]+self.observation_tail_weight*tails[present]).mean()
        # Explicitly report absent chambers, rather than diluting the objective
        # with invented zero observations. All four are equal when available.
        summary = dict(chambers=list(CHAMBERS), mean_boundary_mm=means.detach(),
            tail_boundary_mm=tails.detach(), cell_counts=counts.detach(),
            tail_counts=indices.new_tensor(tail_counts), present=present.detach(),
            tail_fraction=self.observation_tail_fraction, tail_weight=self.observation_tail_weight,
            tail_definition='largest ceil(fraction*valid_chamber_cells); not Exact Dice failures')
        summary.update(self._boundary_status(slices, boundary.detach(), indices))
        return objective, summary

    def _boundary_status(self, slices, boundary, indices):
        """Report residual types without inventing proxy pass/fail thresholds.

        Curve, recovery, absence and unobservable states are counted
        separately. None of them receives a proxy PASS/FAIL before a future
        TRAIN-only calibration and ranking-validation protocol is implemented.
        """
        status = dict(boundary_threshold_mm=None,
            boundary_threshold_state='DISABLED_UNCALIBRATED',
            boundary_threshold_metric='smoothed physical boundary proxy; not Native Exact Dice or raw ASSD',
            proxy_calibration_requirement='TRAIN only; Native Dice>=0.98 and ASSD<=0.5mm; validate view/chamber ranking first',
            proxy_boundary_pass_counts=None, proxy_boundary_exceedance_counts=None,
            proxy_failure_counts=None)
        kinds = slices.get('cell_observation_kind')
        names = slices.get('cell_observation_kind_names')
        required = ('paired_boundary', 'recovery', 'matched_empty', 'empty_target', 'no_visible_gt_boundary')
        status.update({name+'_counts': None for name in required})
        status['cell_kind_counts_available'] = kinds is not None
        if kinds is None:
            return status
        if (not isinstance(kinds, torch.Tensor) or kinds.shape != (len(indices),)
                or kinds.dtype != torch.long or kinds.device != indices.device
                or not isinstance(names, (tuple, list)) or tuple(names) != required
                or ((kinds < 0) | (kinds >= len(required))).any()):
            raise ValueError('Boundary classification requires explicit valid per-cell observation kinds')
        def counts(mask):
            return torch.stack([(mask & (indices[:, 3] == chamber)).sum() for chamber in range(4)])
        status.update({name+'_counts': counts(kinds == i) for i, name in enumerate(required)})
        return status

    def _control_objectives(self, controls, prediction, spacings, regional_spacing,
                            regional_motion_spacings, stage):
        """Partition the existing regularizer without changing its total scale.

        Global/Shape regional source-axis terms share the original denominator.
        Fine reference versus motion terms carry 1/T and (T-1)/T factors.
        The existing additional regional Motion regularizer belongs only to M.
        """
        zero = prediction.sum()*0
        shape, motion = [], []
        sources = 0
        if spacings is not None:
            for level, key in enumerate(('coarse', 'medium', 'fine_motion')):
                grid = controls[key]
                for axis in range(3):
                    dimension = grid.ndim-4+axis
                    if key == 'fine_motion':
                        frames = 1 if stage == 'shape' else grid.shape[1]
                        shape.append((grid[:, :1].diff(dim=dimension)/spacings[level, axis]).square().mean()/frames)
                        if frames > 1:
                            motion.append((grid[:, 1:].diff(dim=dimension)/spacings[level, axis]).square().mean()*(frames-1)/frames)
                    else:
                        shape.append((grid.diff(dim=dimension)/spacings[level, axis]).square().mean())
                    sources += 1
            if controls.get('regions') is not None:
                if regional_spacing is None:
                    raise ValueError('Physical spacing required for regional control regularization')
                for axis in range(3):
                    shape.append((controls['regions'].diff(dim=2+axis)/regional_spacing[axis]).square().mean())
                    sources += 1
        elif self.weights['control']:
            raise ValueError('Physical control spacings required for active control regularization')
        shape_total = sum(shape, zero)/sources if sources else zero
        motion_total = sum(motion, zero)/sources if sources else zero
        if controls.get('regional_motion') is not None and stage != 'shape':
            if regional_motion_spacings is None or regional_motion_spacings.shape != (4, 3):
                raise ValueError('Four physical regional motion spacings required for control regularization')
            grid = controls['regional_motion'][:, 1:]
            if grid.shape[1]:
                parts = [(grid.diff(dim=3+axis)/regional_motion_spacings[:, axis][None, None, :, None, None, None, None]).square().mean()
                         for axis in range(3)]
                motion_total = motion_total+torch.stack(parts).mean()
        return shape_total, motion_total

    @torch.no_grad()
    def _motion_diagnostics(self, motion, truth):
        """GT amplitude/direction diagnostics from the already decoded cycle."""
        predicted_rms, gt_rms, ratios, cosines, amplitude_valid, direction_valid = [], [], [], [], [], []
        for left, right in self.chamber_ranges.tolist():
            p, g = motion[:, 1:, left:right], truth[:, 1:, left:right]
            pr = p.square().sum(-1).mean().sqrt() if p.numel() else motion.new_zeros(())
            gr = g.square().sum(-1).mean().sqrt() if g.numel() else motion.new_zeros(())
            moving = g.norm(dim=-1) > 1e-3
            cosine = F.cosine_similarity(p, g, dim=-1, eps=1e-8)
            valid = moving.any()
            predicted_rms.append(pr)
            gt_rms.append(gr)
            amplitude_valid.append(gr > 1e-8)
            ratios.append(torch.where(gr > 1e-8, pr/gr.clamp_min(1e-8), gr*0))
            direction_valid.append(valid)
            cosines.append(torch.where(valid, (cosine*moving).sum()/moving.sum().clamp_min(1), gr*0))
        return dict(chambers=list(CHAMBERS), predicted_displacement_rms_mm=torch.stack(predicted_rms),
            gt_displacement_rms_mm=torch.stack(gt_rms), amplitude_ratio=torch.stack(ratios),
            amplitude_defined=torch.stack(amplitude_valid), direction_cosine=torch.stack(cosines),
            direction_defined=torch.stack(direction_valid))

    def forward(self, output, vertices_gt, *, slice_result=None, control_spacings_mm=None,
                regional_spacing_mm=None, regional_motion_spacings_mm=None, stage='joint'):
        p,g = output['vertices_4d'],vertices_gt.to(output['vertices_4d']).detach()
        if (p.shape!=g.shape or p.ndim!=4 or p.shape[-1]!=3 or p.shape[-2]!=self.chamber_ranges[-1,1]
                or not torch.isfinite(p).all() or not torch.isfinite(g).all()):
            raise ValueError('Finite paired [B,T,N,3] prediction/GT required')
        if stage not in ('shape','motion','joint'):
            raise ValueError('Training stage must be shape, motion or joint')
        u,ug = p-p[:,:1],g-g[:,:1]
        values = dict(shape=self._paired_huber(p[:,0],g[:,0]),trajectory=self._paired_huber(p,g),
                      motion=self._paired_huber(u,ug),
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
        control_shape, control_motion = self._control_objectives(output['control_grids'], p,
            control_spacings_mm, regional_spacing_mm, regional_motion_spacings_mm, stage)
        values.update(control_shape=control_shape, control_motion=control_motion,
                      control=control_shape+control_motion)
        observation_summary = None
        if slice_result is not None:
            if self.objective_mode == 'grouped_gt_boundary_v1':
                values['slice'], observation_summary = self._observation_objective(slice_result, p)
            else:
                values['slice'] = slice_result['total']
        elif stage == 'joint' and self.weights['slice']:
            raise ValueError('Joint training requires active differentiable slice supervision')
        else:
            values['slice'] = p.sum()*0
        active = dict(self.weights)
        # Full-cycle position supervision is a versioned Joint change. Keep
        # historical reference-only Shape and pre-Joint Motion schedules.
        if self.objective_mode == 'legacy_v1' or stage != 'joint':
            active['trajectory'] = 0.
        if stage == 'shape':
            active.update(motion=0., velocity=0., volume=0., slice=0.)
        elif stage == 'motion':
            active['slice'] = 0.
        active['control_shape'] = active['control']
        active['control_motion'] = active['control'] if stage != 'shape' else 0.
        active['control'] = 0.  # Diagnostic aggregate only; never count twice.
        shape_terms = ('shape', 'trajectory', 'edge', 'laplacian', 'normal', 'control_shape')
        motion_terms = ('motion', 'velocity', 'volume', 'control_motion')
        L_S = sum((active[key]*values[key] for key in shape_terms), p.sum()*0)
        L_M = sum((active[key]*values[key] for key in motion_terms), p.sum()*0)
        L_O = values['slice']
        objective_weights = dict(L_S=1., L_M=1., L_O=active['slice'])
        total = L_S+L_M+objective_weights['L_O']*L_O
        if not torch.isfinite(total):
            raise ValueError('Nonfinite supervised objective')
        return dict(total=total, L_S=L_S, L_M=L_M, L_O=L_O,
                    objective_mode=self.objective_mode, objective_weights=objective_weights,
                    objective_terms=dict(L_S=shape_terms, L_M=motion_terms, L_O=('slice',)),
                    observation_summary=observation_summary,
                    motion_diagnostics=self._motion_diagnostics(u.detach(), ug.detach()),
                    terms=values, active_weights=active,
                    units=dict(shape='mm', trajectory='mm', motion='mm', velocity='mm per phase step', volume='mL',
                               edge='mm^2', laplacian='mm^2', normal='dimensionless', control='dimensionless',
                               control_shape='dimensionless', control_motion='dimensionless',
                               slice=('chamber-balanced boundary mean + tail (mm)' if self.objective_mode == 'grouped_gt_boundary_v1'
                                      else 'weighted dimensionless Dice/BCE + boundary mm')))



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
            per_frame_trajectory_rmse_mm=error.square().mean((0,2)).sqrt().cpu().tolist(),
            per_frame_relative_motion_rmse_mm=(u-ug).square().sum(-1).mean((0,2)).sqrt().cpu().tolist(),
            degenerate_triangles=int((area_twice<1e-8).sum()),nonfinite_vertices=int((~torch.isfinite(pv).all(-1)).sum()),
            predicted_volume_curve_ml=vp[...,i].cpu().tolist(),gt_volume_curve_ml=vg[...,i].cpu().tolist())
    return result
