"""Immutable initial Motion controls applied through original recursive FFD."""
import numpy as np
import torch

from data.mask_transform import CHAMBERS
from utils.neural_ffd_losses import mesh_volumes_ml


class FrozenMotionPropagator:
    def __init__(self, model, fields, prior=None, limits=None):
        self.model = model
        if fields.ndim != 6 or fields.shape[1] != 24 or fields.requires_grad:
            raise ValueError('Detached [B,24,X,Y,Z,3] original fine fields required')
        self.fields = fields.detach().clone()
        self._original = fields.detach().clone()
        self.prior = prior.detach().clone() if prior is not None else None
        self.limits = dict(limits) if limits is not None else None

    def propagate(self, reference):
        if not torch.equal(self.fields, self._original):
            raise AssertionError('Frozen Motion control fields were mutated')
        if reference.shape[0] != self.fields.shape[0]:
            raise ValueError('Motion and reference patient batches differ')
        values = [reference]
        for phase in range(1, 24):
            values.append(self.model._deform(values[-1], self.fields[:, phase], 2))
        return torch.stack(values, 1)

    def check(self, trajectory):
        if self.prior is None or self.limits is None:
            raise ValueError('Motion check requires the immutable initial trajectory and fixed limits')
        if trajectory.shape != self.prior.shape:
            raise ValueError('Candidate and original Motion trajectory must match')
        reports = []
        for candidate, prior in zip(trajectory, self.prior):
            metrics = motion_metrics(self.model, candidate, prior)
            reports.append(dict(metrics=metrics, **motion_guard(metrics, self.limits)))
        return reports


# Metrics preserved from the audited frozen-field protocol; no FFD kernel copied.
def motion_metrics(model,candidate,prior):
    """Relative-to-frozen-prior diagnostics. 'prior' is NOT real/synthetic GT."""
    candidate=torch.as_tensor(candidate,device=model.template_vertices_mm.device,dtype=torch.float32)
    prior=torch.as_tensor(prior,device=candidate.device,dtype=candidate.dtype)
    if candidate.ndim==3:candidate=candidate[None]
    if prior.ndim==3:prior=prior[None]
    with torch.no_grad():
        vp=mesh_volumes_ml(candidate,model.chamber_ranges,model.faces)[0]
        vq=mesh_volumes_ml(prior,model.chamber_ranges,model.faces)[0]
        records={}
        for ci,(c,(a,b)) in enumerate(zip(CHAMBERS,model.chamber_ranges.tolist())):
            p=candidate[0,:,a:b];q=prior[0,:,a:b];u=p-p[:1];uq=q-q[:1]
            rms=lambda x:float(x.square().sum(-1).mean().sqrt())
            amp=rms(u);ampq=rms(uq);moving=uq.norm(dim=-1)>1e-3
            cos=torch.nn.functional.cosine_similarity(u,uq,dim=-1)
            vel=(p.roll(-1,0)-p)*24;velq=(q.roll(-1,0)-q)*24
            acc=(vel.roll(-1,0)-vel)*24;accq=(velq.roll(-1,0)-velq)*24
            jerk=(acc.roll(-1,0)-acc)*24;jerkq=(accq.roll(-1,0)-accq)*24
            normjerk=rms(jerk)/max(amp,1e-6);normjerkq=rms(jerkq)/max(ampq,1e-6)
            curve=vp[:,ci];curveq=vq[:,ci];swing=float(curve.max()-curve.min());swingq=float(curveq.max()-curveq.min())
            peak=int(curve.argmax());peakq=int(curveq.argmax());trough=int(curve.argmin());troughq=int(curveq.argmin())
            records[c]=dict(relative_motion_rmse_mm=rms(u-uq),per_phase_relative_motion_rmse_mm=(u-uq).square().sum(-1).mean(-1).sqrt().tolist(),
                per_phase_motion_amplitude_mm=u.square().sum(-1).mean(-1).sqrt().tolist(),amplitude_rms_mm=amp,prior_amplitude_rms_mm=ampq,
                amplitude_ratio=amp/max(ampq,1e-6),direction_mean_cosine=float(cos[moving].mean()) if moving.any() else None,
                velocity_rms_mm_per_cycle=rms(vel),velocity_error_rms_mm_per_cycle=rms(vel-velq),
                acceleration_rms_mm_per_cycle2=rms(acc),acceleration_error_rms_mm_per_cycle2=rms(acc-accq),
                normalized_jerk_per_cycle3=normjerk,prior_normalized_jerk_per_cycle3=normjerkq,
                normalized_jerk_ratio=normjerk/max(normjerkq,1e-6),closing_step_rms_mm=rms(p[0]-p[-1]),
                prior_closing_step_rms_mm=rms(q[0]-q[-1]),closing_step_error_mm=rms((p[0]-p[-1])-(q[0]-q[-1])),
                volume_curve_ml=curve.tolist(),prior_volume_curve_ml=curveq.tolist(),volume_swing_ratio=swing/max(swingq,1e-6),
                volume_max_ml=float(curve.max()),volume_min_ml=float(curve.min()),volume_peak_phase=peak/24,volume_trough_phase=trough/24,
                volume_peak_shift_cycles=min(abs(peak-peakq),24-abs(peak-peakq))/24,
                volume_trough_shift_cycles=min(abs(trough-troughq),24-abs(trough-troughq))/24,
                volume_function_error_rms_ml=float(((curve-curve[0])-(curveq-curveq[0])).square().mean().sqrt()))
        return dict(chambers=records,reference='frozen predicted prior, not ground truth',time_units='cardiac cycle; cyclic differences include23->0')

def motion_guard(metrics,limits):
    failures=[]
    required = {'relative_motion_rmse_mm', 'amplitude_log_ratio', 'direction_one_minus_cosine',
        'velocity_error_rms_mm_per_cycle', 'acceleration_error_rms_mm_per_cycle2',
        'normalized_jerk_log_ratio', 'closing_step_error_mm', 'volume_swing_log_ratio',
        'volume_peak_shift_cycles', 'volume_trough_shift_cycles'}
    if set(limits) != required or any(not np.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Complete finite predeclared Motion trust limits required')
    for c,m in metrics['chambers'].items():
        values=dict(relative_motion_rmse_mm=m['relative_motion_rmse_mm'],amplitude_log_ratio=abs(np.log(max(m['amplitude_ratio'],1e-8))),
            direction_one_minus_cosine=1-m['direction_mean_cosine'] if m['direction_mean_cosine'] is not None else 0.,
            velocity_error_rms_mm_per_cycle=m['velocity_error_rms_mm_per_cycle'],acceleration_error_rms_mm_per_cycle2=m['acceleration_error_rms_mm_per_cycle2'],
            normalized_jerk_log_ratio=abs(np.log(max(m['normalized_jerk_ratio'],1e-8))),closing_step_error_mm=m['closing_step_error_mm'],
            volume_swing_log_ratio=abs(np.log(max(m['volume_swing_ratio'],1e-8))),volume_peak_shift_cycles=m['volume_peak_shift_cycles'],
            volume_trough_shift_cycles=m['volume_trough_shift_cycles'])
        for k,v in values.items():
            if not np.isfinite(v):
                failures.append(dict(chamber=c,metric=k,value=None,limit=limits[k],reason='nonfinite; unverified'))
            elif v>limits[k]+1e-6:failures.append(dict(chamber=c,metric=k,value=float(v),limit=limits[k]))
    return dict(passed=not failures,failures=failures)
