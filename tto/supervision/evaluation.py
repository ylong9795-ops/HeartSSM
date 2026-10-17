"""Native evaluation with explicit all-label / trusted / optimization coverage."""
from ..evaluator import ExactNativeEvaluator, dice_stats
from .observations import observation_cells,native_cell,interpolate_cell
from data.mask_transform import CHAMBERS, VIEWS


def proportions(rows):
    result = dice_stats(rows)
    count = result['cells']
    result.update(fraction_ge095=result['ge095']/count if count else None,
                  fraction_ge098=result['ge098']/count if count else None,
                  all_cells_ge098=bool(count and result['ge098'] == count))
    return result


class NativeEvaluator(ExactNativeEvaluator):
    def __init__(self, model, policy):
        super().__init__(model)
        self.policy = policy

    def annotated_cells(self, trajectory, pose, patient, stage):
        import numpy as np
        from ..evaluator import native_exact
        rows=[]
        for evidence in observation_cells([patient],self.policy):
            vi,ci=evidence['view'],evidence['chamber'];chamber=CHAMBERS[ci]
            target,_,spacing=native_cell(patient,evidence);target=target.astype(bool)
            a,b=self.model.chamber_ranges[ci].tolist();vertices=interpolate_cell(trajectory,evidence)[a:b]
            error=None
            try:
                prediction=native_exact(vertices,self.faces[chamber],pose['R'][vi],pose['center_mm'][vi],spacing,target.shape).astype(bool)
                denominator=int(prediction.sum()+target.sum())
                dice=2*int((prediction&target).sum())/denominator if denominator else 1.
            except Exception as exception:
                error=repr(exception);prediction=np.zeros_like(target);dice=0.
            rows.append(dict(patient=patient.patient_id,stage=stage,phase=evidence['phase'],
                phase_upper=evidence['mesh_upper'],phase_alpha=evidence['mesh_alpha'],
                view=VIEWS[vi],chamber=chamber,exact_hard_dice=float(dice),
                native_frame=evidence['native_frame'],native_time_s=evidence['native_time_s'],time_error_ms=evidence['time_error_ms'],
                label_source=evidence['source'],source_weight=evidence['source_confidence'],
                prediction_area_mm2=float(prediction.sum()*np.prod(spacing)),target_area_mm2=float(target.sum()*np.prod(spacing)),
                renderer_error=error,confidence=evidence['confidence'],phase_confidence=evidence['phase_confidence'],
                used_for_optimization=evidence['used'],supervision_reason=evidence['reason']))
        return rows

    def cells(self, trajectory, pose, patient, stage, phases=None):
        rows = self.annotated_cells(trajectory, pose, patient, stage)
        return rows if phases is None else [r for r in rows if r['phase'] in phases]

    def evaluate_state(self, state, patient):
        state.validate()
        if state.patient_id != patient.patient_id or state.identity['observations_sha256'] != patient.identity:
            raise ValueError('Evaluation state and observations differ')
        if state.observation_pose is None:
            raise ValueError('Native evaluation requires registered observation-plane poses')
        trajectory = state.patient_trajectory().detach().cpu().numpy()
        pose = {key: value.detach().cpu().numpy() for key, value in state.observation_pose.items()}
        rows = self.annotated_cells(trajectory, pose, patient, state.stage.value)
        return self.summarize(rows), rows

    def summarize(self, rows):
        used = [r for r in rows if r['used_for_optimization']]
        trusted = [r for r in rows if r['confidence'] >= self.policy['trusted_confidence_threshold']]
        return dict(all_annotated=proportions(rows),trusted=proportions(trusted),
            optimization_observations=proportions(used),
            supervision_coverage=len(used)/len(rows) if rows else 0.,
            trusted_coverage=len(trusted)/len(rows) if rows else 0.,
            by_source={s:proportions([r for r in rows if r['label_source']==s]) for s in sorted({r['label_source'] for r in rows})},
            by_view={s:proportions([r for r in rows if r['view']==s]) for s in VIEWS},
            by_chamber={s:proportions([r for r in rows if r['chamber']==s]) for s in CHAMBERS},
            failed_cells=[r for r in rows if r['exact_hard_dice']<.98],
            interpretation='Native 2D agreement at estimated Pose; no real 3D/Pose GT; all failed cells retained')
