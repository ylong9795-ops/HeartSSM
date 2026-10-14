"""Detailed four-chamber diagnostics; frozen losses/geometry stay unchanged.

Volumes reuse the existing mm-to-mL operator. Vertex observation regions use
the original physical projection at GT locations, solely for evaluation;
GT-derived regions never enter the network. Time units are cardiac cycles,
since no patient-specific cycle duration is provided.
"""
import torch
from data.mask_transform import CHAMBERS
from net.geometry_lifting import project_physical_points
from net.multiview_encoder import observation_validity
from utils.neural_ffd_losses import mesh_volumes_ml


@torch.no_grad()
def observed_vertex_regions(truth,observations,ranges,norm_to_mm,*,direct_half_thickness_mm=.75,
                            propagated_distance_mm=24.):
    b,t,n,_=truth.shape
    if t!=24 or not 0<direct_half_thickness_mm<propagated_distance_mm:
        raise ValueError('24 phases and ordered positive observation distances required')
    valid=observation_validity(observations,b,t,truth.device)
    direct=torch.zeros(b,t,n,dtype=torch.bool,device=truth.device)
    supported=torch.zeros_like(direct)
    pose=observations.get('geometry',observations.get('pose_gt'))
    if pose is None:raise ValueError('Explicit observation pose required')
    for patient in range(b):
        for phase in range(t):
            # The frozen projector accepts a shared query list. Explicitly
            # inspect EVERY patient/phase with its own points and pose.
            single=dict(masks=observations['masks'][patient:patient+1,phase:phase+1],
                geometry={key:value[patient:patient+1,phase:phase+1] for key,value in pose.items()},
                **{key:observations[key][patient:patient+1,phase:phase+1] for key in
                   ('model_to_native','pixel_spacing','native_pixel_spacing')})
            projection=project_physical_points(truth[patient,phase],single,norm_to_mm,
                                                observations['masks'].shape[-2:])
            inside=projection['inside'].squeeze(0).squeeze(0)
            distance=projection['distance_mm'].abs().squeeze(0).squeeze(0)
            for chamber,(left,right) in enumerate(ranges.tolist()):
                available=inside[:,left:right]&valid[patient,phase,:,chamber,None]
                direct[patient,phase,left:right]=(available&(distance[:,left:right]<=direct_half_thickness_mm)).any(0)
                supported[patient,phase,left:right]=(available&(distance[:,left:right]<=propagated_distance_mm)).any(0)
    return dict(direct=direct,propagated=supported&~direct,unknown=~supported,
                not_direct=~direct)


def rms(vector):
    return float(vector.square().sum(-1).mean().sqrt())


def extrema(curve,phases,ventricle):
    maximum,minimum=curve.max(),curve.min()
    flat=bool(maximum-minimum<1e-8)
    maximum_phase=None if flat else float(phases[curve.argmax()])
    minimum_phase=None if flat else float(phases[curve.argmin()])
    return dict(curve_ml=curve.tolist(),maximum_ml=float(maximum),minimum_ml=float(minimum),
        peak_to_trough_ml=float(maximum-minimum),maximum_phase=maximum_phase,minimum_phase=minimum_phase,
        flat_curve=flat,maximum_ties=int((curve==maximum).sum()),minimum_ties=int((curve==minimum).sum()),
        edv_extremum_proxy_ml=float(maximum) if ventricle else None,
        esv_extremum_proxy_ml=float(minimum) if ventricle else None,
        ef_extremum_proxy=float((maximum-minimum)/maximum) if ventricle and maximum>1e-8 else None)


@torch.no_grad()
def detailed_motion_metrics(prediction,truth,ranges,faces,phases,*,regions=None):
    p,g=prediction,truth.to(prediction)
    if (p.shape!=g.shape or p.ndim!=4 or p.shape[1]!=24 or p.shape[-1]!=3
            or p.shape[-2]!=int(ranges[-1,1]) or not torch.isfinite(p).all() or not torch.isfinite(g).all()):
        raise ValueError('Finite paired full-cycle fixed-topology trajectories required')
    phases=phases.to(prediction)
    expected=torch.arange(24,device=p.device,dtype=p.dtype)[None].expand(len(p),-1)/24
    if phases.shape!=expected.shape or not torch.allclose(phases,expected,atol=1e-6,rtol=0):
        raise ValueError('Independent uniformly sampled 24-phase cycle required')
    if regions is not None:
        if set(regions)!= {'direct','propagated','unknown','not_direct'}:
            raise ValueError('Explicit direct/propagated/unknown region partition required')
        if any(value.shape!=p.shape[:-1] or value.dtype!=torch.bool or value.device!=p.device for value in regions.values()):
            raise ValueError('Region masks must be bool [B,24,N]')
        if not torch.all((regions['direct'].int()+regions['propagated'].int()+regions['unknown'].int())==1):
            raise ValueError('Region classes must partition all vertices')
        if not torch.equal(regions['not_direct'],~regions['direct']):raise ValueError('Invalid not-direct region')
    predicted_volume=mesh_volumes_ml(p,ranges,faces)
    truth_volume=mesh_volumes_ml(g,ranges,faces)
    cases=[]
    for patient in range(len(p)):
        chambers={}
        for chamber,(name,(left,right)) in enumerate(zip(CHAMBERS,ranges.tolist())):
            pv,gv=p[patient,:,left:right],g[patient,:,left:right]
            u,ug=pv-pv[:1],gv-gv[:1]
            error=(pv-gv).norm(dim=-1)
            delta,gt_delta=pv.roll(-1,0)-pv,gv.roll(-1,0)-gv
            velocity,gt_velocity=delta*24,gt_delta*24
            acceleration=(velocity.roll(-1,0)-velocity)*24
            gt_acceleration=(gt_velocity.roll(-1,0)-gt_velocity)*24
            jerk=(acceleration.roll(-1,0)-acceleration)*24
            gt_jerk=(gt_acceleration.roll(-1,0)-gt_acceleration)*24
            amplitude,gt_amplitude=rms(u),rms(ug)
            moving=ug.norm(dim=-1)>1e-3
            cosine=torch.nn.functional.cosine_similarity(u,ug,dim=-1,eps=1e-8)
            record=dict(reference_rmse_mm=rms(pv[0]-gv[0]),full_4d_rmse_mm=rms(pv-gv),
                per_phase_shape_rmse_mm=error.square().mean(-1).sqrt().tolist(),
                per_phase_shape_p95_mm=torch.quantile(error,.95,dim=-1).tolist(),
                per_phase_motion_rmse_mm=(u-ug).square().sum(-1).mean(-1).sqrt().tolist(),
                relative_motion_rmse_mm=rms(u-ug),motion_amplitude_rms_mm=amplitude,
                gt_motion_amplitude_rms_mm=gt_amplitude,
                motion_amplitude_ratio=amplitude/gt_amplitude if gt_amplitude>1e-8 else None,
                mean_motion_direction_cosine=float(cosine[moving].mean()) if moving.any() else None,
                closing_23_to_0_step_rms_mm=rms(delta[-1]),
                gt_closing_23_to_0_step_rms_mm=rms(gt_delta[-1]),
                closing_step_error_rms_mm=rms(delta[-1]-gt_delta[-1]),
                velocity_rms_mm_per_cycle=rms(velocity),velocity_error_rms_mm_per_cycle=rms(velocity-gt_velocity),
                acceleration_rms_mm_per_cycle2=rms(acceleration),
                acceleration_error_rms_mm_per_cycle2=rms(acceleration-gt_acceleration),
                jerk_rms_mm_per_cycle3=rms(jerk),gt_jerk_rms_mm_per_cycle3=rms(gt_jerk),
                normalized_jerk_per_cycle3=rms(jerk)/amplitude if amplitude>1e-8 else None,
                gt_normalized_jerk_per_cycle3=rms(gt_jerk)/gt_amplitude if gt_amplitude>1e-8 else None,
                volume=extrema(predicted_volume[patient,:,chamber],phases[patient],name in ('LV','RV')),
                gt_volume=extrema(truth_volume[patient,:,chamber],phases[patient],name in ('LV','RV')))
            if regions is not None:
                record['regions']={}
                for region,mask in regions.items():
                    selected=mask[patient,:,left:right];count=int(selected.sum())
                    record['regions'][region]=dict(vertex_phase_count=count,
                        full_4d_rmse_mm=rms((pv-gv)[selected]) if count else None,
                        relative_motion_rmse_mm=rms((u-ug)[selected]) if count else None,
                        motion_amplitude_rms_mm=rms(u[selected]) if count else None,
                        gt_motion_amplitude_rms_mm=rms(ug[selected]) if count else None)
            chambers[name]=record
        coordination={}
        for label,atrium,ventricle in (('left','LA','LV'),('right','RA','RV')):
            offsets={}
            for field in ('volume','gt_volume'):
                a,v=chambers[atrium][field]['maximum_phase'],chambers[ventricle][field]['maximum_phase']
                offsets[field]=((a-v+.5)%1-.5) if a is not None and v is not None else None
            coordination[label]=offsets
        cases.append(dict(patient=patient,chambers=chambers,
                          atrial_minus_ventricular_volume_peak_offset_cycles=coordination))
    return dict(cases=cases,time_units='cycle; forward periodic differences include 23->0',
                extrema_policy='per-chamber extrema proxies; frame0 is not assumed ED',
                region_policy='GT physical locations; evaluation only; no encoder input')
