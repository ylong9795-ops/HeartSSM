"""Hidden truth boundary and post-save evaluation for the new state schema.

Case selection and solver recovery gates are intentionally not implemented.
Truth is never accepted by the TTO pipeline or optimizer contracts.
"""
import numpy as np
import torch
from data.mask_transform import CHAMBERS
from .dataset import PatientObservations
from .io import load_state
from .state import TTOState


def hidden_patient(sample, name):
    """Expose ONLY observed inputs. Truth is returned separately to the exporter."""
    masks=sample['native_masks'].numpy().copy()
    spacing=sample['native_pixel_spacing'].numpy()
    if not np.allclose(spacing,spacing[:1],atol=1e-7,rtol=0):
        raise ValueError('Synthetic acquisition changes pixel spacing; shared-Pose contract unsupported')
    annotated=sample['annotated'].numpy()
    if annotated.ndim==2:annotated=np.broadcast_to(annotated,(24,3,4)).copy()
    present=sample['view_present'].numpy()
    if present.ndim==1:present=np.broadcast_to(present,(24,3)).copy()
    annotated=annotated & present[...,None]
    views=[]
    for view in range(3):
        views.append(dict(masks=masks[:,view],spacing=spacing[0,view],
            sources=[['synthetic_observed']*4]*24,weights=annotated[:,view].astype(np.float32),
            annotated=annotated[:,view],frame_ids=list(range(24))))
    mapping=[[dict(nearest_index=phase,nearest_native_frame=phase,nearest_time_error_ms=0.,
                   model_phase=phase/24,target_time_s=phase/24) for _ in range(3)] for phase in range(24)]
    encoder={key:sample[key].clone() for key in ('masks','sdf','phases')}
    encoder.update(annotated=torch.from_numpy(annotated),view_present=torch.from_numpy(present),sdf_units='mm')
    metadata=dict(contract=dict(version='synthetic-hidden-observations-v1',patient_id=name,
        acquisition_status='synthetic source phase mapping verified',case_id=sample['case_id'],
        sample_sha256=sample['sample_sha256'],source_manifest_sha256=sample['source_manifest_sha256']),
        source_hashes=dict(sample=sample['sample_sha256']),mapping=mapping,
        hidden_truth_policy='Mesh and R/t removed; encoder and optimizer receive observed masks/SDF/spacing/validity/time only')
    patient=PatientObservations(name,views,mapping,encoder,metadata)
    truth=dict(vertices_gt=sample['vertices_gt'].float(),
               R=sample['pose_gt']['probe_R'].float(),
               center_mm=sample['pose_gt']['probe_t'].float()*sample['norm_to_mm'],
               case_id=sample['case_id'],sample_sha256=sample['sample_sha256'])
    return patient,truth


def surface_distances(vertices, truth, faces, *, return_samples=False):
    """Symmetric vertex-sampled point-to-triangle distances, explicitly named."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    def distances(source,target):
        points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.ascontiguousarray(target,dtype=np.float64),deep=True))
        cells=vtk.vtkCellArray()
        packed=np.column_stack([np.full(len(faces),3,dtype=np.int64),faces]).reshape(-1)
        cells.SetCells(len(faces),numpy_to_vtkIdTypeArray(packed,deep=True))
        poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
        implicit=vtk.vtkImplicitPolyDataDistance();implicit.SetInput(poly)
        return np.array([abs(implicit.EvaluateFunction(point)) for point in source])
    distance=np.concatenate([distances(vertices,truth),distances(truth,vertices)])
    metrics=dict(ASSD_mm=float(distance.mean()),HD95_mm=float(np.quantile(distance,.95)),
                definition='symmetric vertex-sampled point-to-opposite-triangle-surface distance; no GT used for optimization')
    return (metrics,distance) if return_samples else metrics


def evaluate_hidden(model, endpoint_path, truth):
    """Only read an already committed endpoint; never return truth to a solver."""
    state = TTOState.from_payload(load_state(endpoint_path))
    case = truth['case_id']
    expected_case = state.identity.get('case_id')
    if expected_case != case or state.identity.get('sample_sha256') != truth['sample_sha256']:
        raise ValueError('Hidden truth belongs to a different synthetic observation')
    vertices = state.trajectory.detach().cpu().numpy()
    target = torch.as_tensor(truth['vertices_gt']).cpu().numpy()
    if vertices.shape != target.shape:
        raise ValueError('Hidden truth topology/phases differ from committed endpoint')
    rmse = lambda value: float(np.sqrt(np.mean(np.sum(value**2, axis=-1))))
    records = {}
    for chamber, (a, b) in zip(CHAMBERS, model.chamber_ranges.tolist()):
        records[chamber] = dict(reference_vertex_rmse_mm=rmse(vertices[0, a:b]-target[0, a:b]),
            **surface_distances(vertices[0, a:b], target[0, a:b], model.faces[chamber].cpu().numpy()))
    pose_metrics = dict(status='NOT_EVALUATED', reason='No registered observation pose')
    if state.observation_pose is not None:
        # Plane poses are in patient space; compare after canonical-to-patient mapping.
        global_r = state.global_transform['R'].cpu().numpy()
        global_t = state.global_transform['center_mm'].cpu().numpy()
        r = global_r.T @ state.observation_pose['R'].cpu().numpy()
        t = (state.observation_pose['center_mm'].cpu().numpy()-global_t) @ global_r
        gt_r = torch.as_tensor(truth['R']).cpu().numpy()
        gt_t = torch.as_tensor(truth['center_mm']).cpu().numpy()
        stable = np.allclose(gt_r, gt_r[:1]) and np.allclose(gt_t, gt_t[:1])
        if stable:
            relative = np.swapaxes(r, -1, -2) @ gt_r[0]
            angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))
            pose_metrics = dict(status='EVALUATED', rotation_degrees=angle.tolist(),
                                translation_mm=np.linalg.norm(t-gt_t[0], axis=-1).tolist())
        else:
            pose_metrics = dict(status='NOT_EVALUATED', reason='Synthetic plane poses vary over the cycle')
    return dict(case_id=case, stage=state.stage.value,
        reference_vertex_rmse_mm=rmse(vertices[0]-target[0]),
        trajectory_vertex_rmse_mm=rmse(vertices-target),
        relative_motion_rmse_mm=rmse((vertices-vertices[:1])-(target-target[:1])),
        by_chamber=records, pose=pose_metrics,
        hidden_gt_access='Independent evaluation of a committed endpoint only',
        recovery_gate='NOT_IMPLEMENTED', coordinates='Canonical physical mm without GT alignment')
