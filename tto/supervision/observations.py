"""Prediction-independent native evidence selection and physical time mapping."""
import numpy as np
from ..dataset import sdf_mm


def observation_cells(patients, policy, phases=range(24)):
    """Fixed evidence-based membership; predictions/Dice never enter this policy."""
    phases = tuple(phases)
    rows = []
    for owner, patient in enumerate(patients):
        contract = patient.metadata['contract']
        synthetic = contract.get('acquisition_status') == 'synthetic source phase mapping verified'
        timing_factor = 1. if synthetic else policy['unverified_phase_confidence']
        if not synthetic and all('times' in view for view in patient.views):
            start,duration,anchor=(float(contract[key]) for key in ('cycle_start_s','cycle_duration_s','anchor_time_s'))
            for view_index,view in enumerate(patient.views):
                for native_index,time_s in enumerate(view['times']):
                    if not start<=time_s<start+duration:continue
                    position=float(((time_s-anchor)/duration)%1)*24
                    lower=int(np.floor(position))%24;alpha=position-np.floor(position)
                    if alpha<1e-7:alpha=0.
                    if alpha>1-1e-7:lower=(lower+1)%24;alpha=0.
                    if len(phases)==1 and list(phases)[0]==0:
                        if native_index!=patient.mapping[0][view_index]['nearest_index']:continue
                    elif lower not in phases:continue
                    for chamber in range(4):
                        if not view['annotated'][native_index,chamber]:continue
                        source_confidence=float(view['weights'][native_index,chamber])
                        confidence=source_confidence*timing_factor
                        rows.append(dict(owner=owner,phase=lower,mesh_upper=(lower+1)%24,mesh_alpha=float(alpha),
                            time_error_ms=0.,
                            view=view_index,chamber=chamber,native_index=native_index,
                            native_frame=view['frame_ids'][native_index],native_time_s=float(time_s),
                            source=view['sources'][native_index][chamber],confidence=confidence,
                            source_confidence=source_confidence,phase_confidence=timing_factor,used=confidence>0,
                            reason='Unique original frame inside selected RR; mesh interpolated at observed time; source phase alignment estimated'))
            continue
        for phase in phases:
            native = patient.native(phase)
            for view in range(3):
                mapping = patient.mapping[phase][view]
                timing_ok = abs(mapping['nearest_time_error_ms']) <= policy['max_nearest_time_error_ms']
                for chamber in range(4):
                    if not native['annotated'][view, chamber]:
                        continue
                    source = native['sources'][view][chamber]
                    source_confidence = float(native['weights'][view, chamber])
                    confidence = source_confidence*timing_factor if timing_ok else 0.
                    rows.append(dict(owner=owner, phase=phase, view=view, chamber=chamber,
                        time_error_ms=mapping['nearest_time_error_ms'],
                        mesh_upper=(phase+1)%24,mesh_alpha=0.,native_index=mapping['nearest_index'],
                        native_frame=mapping['nearest_native_frame'],native_time_s=mapping.get('target_time_s',phase/24),
                        source=source, confidence=confidence, source_confidence=source_confidence,
                        phase_confidence=timing_factor, used=confidence > 0,
                        reason=('nearest native time outside declared tolerance' if not timing_ok else
                                'synthetic phase verified' if synthetic else
                                'ED anchoring is an estimate; source frame0 equivalence unverified')))
    return rows


def native_cell(patient,row):
    """One original observation, never a resampled binary mask."""
    view=patient.views[row['view']];index=row['native_index'];chamber=row['chamber']
    mask=view['masks'][index,chamber]
    cache_key=('observed_sdf',row['view'],index,chamber)
    if cache_key not in patient._native_cache:
        patient._native_cache[cache_key]=sdf_mm(mask,view['spacing'])
    return mask,patient._native_cache[cache_key],view['spacing']


def interpolate_cell(trajectory,row):
    alpha=row['mesh_alpha']
    return (1-alpha)*trajectory[row['phase']]+alpha*trajectory[row['mesh_upper']]

