"""Pure 4D diagnostics; no field replay, acceptance gate or optimization."""
import torch
from data.mask_transform import CHAMBERS
from utils.neural_ffd_losses import mesh_volumes_ml


def motion_metrics(model,candidate,prior):
    """Independent relative-motion diagnostics. 'prior' is NOT real/synthetic GT."""
    candidate=torch.as_tensor(candidate,device=model.template_vertices_mm.device,dtype=torch.float32)
    prior=torch.as_tensor(prior,device=candidate.device,dtype=candidate.dtype)
    if candidate.ndim==3:candidate=candidate[None]
    if prior.ndim==3:prior=prior[None]
    expected = (1, 24, len(model.template_vertices_mm), 3)
    if candidate.shape != expected or prior.shape != expected:
        raise ValueError('Compare one matching pair of 24-phase fixed-topology trajectories')
    if not torch.isfinite(candidate).all() or not torch.isfinite(prior).all():
        raise ValueError('Finite trajectories required for motion diagnostics')
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
        return dict(chambers=records,reference='comparison predicted trajectory, not ground truth',time_units='cardiac cycle; cyclic differences include23->0')
