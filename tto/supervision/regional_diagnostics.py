"""Exact native all-cell S2 metrics, including areas/boundaries and relations."""
import numpy as np
from data.mask_transform import CHAMBERS, VIEWS
from train_neural_ffd import boundary_metrics
from ..alignment.centres import measure
from ..alignment.global_alignment import NativePlaneSearch
from ..evaluator import native_exact


def evaluate_candidate(model,patient,state,trajectory,policy,pose_config):
    values=np.asarray(trajectory)
    engine=NativePlaneSearch(model,patient,values,policy,pose_config)
    pose={k:v.detach().cpu().numpy() for k,v in state.observation_pose.items()}
    rows=[]
    for vi in range(3):
        for cell in (r for r in engine.cells if r['view']==vi):
            ci,t=cell['chamber'],cell['phase'];view=patient.views[vi]
            target=view['masks'][cell['native_index'],ci];a,b=model.chamber_ranges[ci].tolist()
            error=None
            try:
                prediction=native_exact(values[t,a:b],engine.faces[CHAMBERS[ci]],pose['R'][vi],pose['center_mm'][vi],view['spacing'],target.shape)
            except Exception as exception:
                error=repr(exception);prediction=np.zeros_like(target)
            rows.append(dict(cell,view_name=VIEWS[vi],chamber_name=CHAMBERS[ci],renderer_error=error,**measure(prediction,target,view['spacing']),
                **boundary_metrics(prediction,target,view['spacing'])))
    result={}
    for ci,c in enumerate(CHAMBERS):
        subset=[r for r in rows if r['chamber']==ci]
        trusted=[r for r in subset if r['fit_weight']>0]
        def avg(key,selected):
            pairs=[(r[key],r['fit_weight']) for r in selected if r[key] is not None]
            return float(np.average([p[0] for p in pairs],weights=[p[1] for p in pairs])) if pairs else None
        def mean(key):
            x=[r[key] for r in subset if r[key] is not None]
            return float(np.mean(x)) if x else None
        result[c]=dict(cells=len(subset),trusted_dice=avg('exact_hard_dice',trusted),
            native_dice=mean('exact_hard_dice'),center_error_mm=mean('centroid_error_norm_mm'),
            boundary_assd_mm=mean('boundary_assd_mm'),boundary_hd95_mm=mean('boundary_hd95_mm'),
            area_ratio_median=float(np.median([r['area_ratio'] for r in subset if r['area_ratio'] is not None])) if subset else None,
            prediction_empty_cells=sum(r['prediction_area_mm2']==0 for r in subset))
    centers=np.stack([values[:,a:b].mean(1) for a,b in model.chamber_ranges.tolist()],1)
    pairs={}
    for a in range(4):
        for b in range(a+1,4):
            vector=centers[:,b]-centers[:,a]
            pairs[CHAMBERS[a]+'-'+CHAMBERS[b]]=dict(vector_mm=vector.tolist(),distance_mm=np.linalg.norm(vector,axis=1).tolist())
    trusted=[r['trusted_dice'] for r in result.values() if r['trusted_dice'] is not None]
    return dict(chambers=result,macro_trusted_dice=float(np.mean(trusted)) if trusted else None,
        macro_native_dice=float(np.mean([r['native_dice'] for r in result.values() if r['native_dice'] is not None])),
        chamber_centers_mm=centers.tolist(),relative_chamber_positions=pairs,
        renderer_errors=sum(bool(r['renderer_error']) for r in rows)),rows
