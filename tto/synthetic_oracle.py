"""Hide TRAIN Mesh/Pose truth, run the real pipeline, then evaluate independently."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from data.mask_transform import CHAMBERS
from train_neural_ffd import dataset_from_config
from .config import resolve
from .dataset import PatientObservations
from .io import cpu, load_state, read_json, save_state, write_json
from .losses import observation_cells


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


def select_oracle_cases(config, calibration, count=None):
    """Membership is fixed by TRAIN manifest order, before loading any target."""
    data_config=yaml.safe_load(resolve(config['synthetic']['data_config']).read_text())
    dataset=dataset_from_config(data_config,'train')
    excluded=set(calibration['sample_ids']) if config['synthetic']['exclude_calibration_cases'] else set()
    selected=[]
    for index,source in enumerate(dataset.sources):
        case_id=source.get('case_id',f"{dataset.asset_sha256}:{source['candidate_id']}:{source['coordinate_sha256']}")
        if case_id not in excluded:selected.append(index)
        if len(selected)>=(count or config['synthetic']['cases']):break
    return dataset,selected,excluded


def validate_source(dataset, prior):
    if (dataset.asset_sha256!=prior.model.asset_sha256 or dataset.norm_to_mm!=prior.model.norm_to_mm or
        dataset.template_sha256!=prior.model.template_sha256):
        raise ValueError('Synthetic oracle source asset/physical scale/fixed topology differs from pretrained model')
    declared=prior.saved.get('data_identity',{})
    if 'train_manifest' in declared and declared['train_manifest']!=dataset.manifest_sha256:
        raise ValueError('Oracle TRAIN manifest differs from frozen pretraining membership')


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


def evaluate_hidden(model, state, truth):
    """Called only AFTER the optimization endpoint has been committed to disk."""
    initial=state['initial'];result=state['result'];records={}
    rotation_gt=truth['R'].numpy();translation_gt=truth['center_mm'].numpy()
    stable_pose=bool(np.allclose(rotation_gt,rotation_gt[:1],atol=1e-6,rtol=0) and
                     np.allclose(translation_gt,translation_gt[:1],atol=1e-5,rtol=0))
    for name,trajectory,pose in [('before',initial['vertices_4d_initial'][0],state['pose_initial']),
                                 ('after',result['trajectory'],result['pose'])]:
        vertices=trajectory.detach().cpu().numpy();gt=truth['vertices_gt'].numpy()
        r=pose['R'].detach().cpu().numpy();t=pose['center_mm'].detach().cpu().numpy()
        relative=np.swapaxes(r,-1,-2) @ rotation_gt[0]
        angle=np.degrees(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))
        shape={};surface_samples=[]
        for chamber,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist()):
            surface,samples=surface_distances(vertices[0,a:b],gt[0,a:b],model.faces[chamber].cpu().numpy(),return_samples=True)
            surface_samples.append(samples)
            shape[chamber]=dict(reference_vertex_rmse_mm=float(np.sqrt(np.mean(np.sum((vertices[0,a:b]-gt[0,a:b])**2,axis=-1)))),
                **surface)
        relative_motion=vertices-vertices[:1];gt_motion=gt-gt[:1]
        records[name]=dict(pose_rotation_degrees=angle.tolist(),pose_translation_mm=np.linalg.norm(t-translation_gt[0],axis=-1).tolist(),
            reference_vertex_rmse_mm=float(np.sqrt(np.mean(np.sum((vertices[0]-gt[0])**2,axis=-1)))),
            trajectory_vertex_rmse_mm=float(np.sqrt(np.mean(np.sum((vertices-gt)**2,axis=-1)))),
            relative_motion_rmse_mm=float(np.sqrt(np.mean(np.sum((relative_motion-gt_motion)**2,axis=-1)))),by_chamber=shape)
        pooled=np.concatenate(surface_samples)
        records[name].update(reference_ASSD_mm=float(pooled.mean()),reference_HD95_mm=float(np.quantile(pooled,.95)))
        from utils.neural_ffd_losses import reconstruction_metrics
        device=model.template_vertices_mm.device
        records[name]['gt_trajectory_motion_volume_by_chamber']=reconstruction_metrics(
            torch.as_tensor(vertices,device=device)[None],torch.as_tensor(gt,device=device)[None],model.chamber_ranges,model.faces)
    return dict(case_id=truth['case_id'],sample_sha256=truth['sample_sha256'],source_pose_shared_across_cycle=stable_pose,
        metrics=records,coordinates='raw canonical physical mm; no GT alignment or candidate selection',
        hidden_gt_access='independent evaluator after endpoint saved',status=result['status'],
        geometry=result['geometry'],motion=result['motion_report'])


def run_oracle(pipeline, *, resume=False):
    config=pipeline.config;folder=pipeline.output/'synthetic_hidden_gt';folder.mkdir(exist_ok=True)
    dataset,indices,excluded=select_oracle_cases(config,pipeline.calibration)
    validate_source(dataset,pipeline.prior)
    patients=[];hidden_paths=[];selection=[]
    for index in indices:
        sample=dataset[index]
        if sample['case_id'] in excluded:raise AssertionError('Manifest/sample oracle identity differs')
        name=f'SYN_{len(patients):03d}'
        patient,truth=hidden_patient(sample,name)
        path=folder/'hidden_truth'/(name+'.pth')
        if path.exists():
            previous=load_state(path)
            if previous['sample_sha256']!=truth['sample_sha256']:raise ValueError('Oracle selection changed')
        else:save_state(path,truth)
        patients.append(patient);hidden_paths.append(path)
        selection.append(dict(patient=name,case_id=sample['case_id'],sample_sha256=sample['sample_sha256'],
                              manifest_index=index,optimizer_fields=sorted(patient.encoder)))
    if len(patients)!=config['synthetic']['cases']:raise ValueError('Insufficient independent TRAIN oracle cases')
    write_json(folder/'selection.json',dict(split='TRAIN',manifest_sha256=dataset.manifest_sha256,
        calibration_excluded=sorted(excluded),cases=selection,selection_rule=config['synthetic']['selection']))
    # The exact same process() executes both datasets. A separate directory keeps
    # synthetic endpoints and all checkpoint identities away from real patients.
    original_output=pipeline.output;pipeline.output=folder
    failures=[];records=[];size=config['runtime']['batch_size']
    try:
        for start in range(0,len(patients),size):
            batch=patients[start:start+size]
            try:
                states=pipeline.process(batch,stage='all',resume=resume)
                for offset,(patient,state) in enumerate(zip(batch,states)):
                    record=evaluate_hidden(pipeline.model,state,load_state(hidden_paths[start+offset]))
                    write_json(folder/'patients'/patient.patient_id/'hidden_gt_evaluation.json',record);records.append(record)
            except Exception as error:
                failure=dict(patients=[p.patient_id for p in batch],error=repr(error));failures.append(failure)
                write_json(folder/f'failure_{start:03d}.json',failure)
            write_json(folder/'progress.json',dict(completed=len(records),required=len(patients),failures=failures))
    finally:pipeline.output=original_output
    criteria=config['synthetic']['gate']
    admitted=[r for r in records if r['status']=='COMPLETE']
    if admitted:
        before_rotation=np.median([np.mean(r['metrics']['before']['pose_rotation_degrees']) for r in admitted])
        after_rotation=np.median([np.mean(r['metrics']['after']['pose_rotation_degrees']) for r in admitted])
        before_translation=np.median([np.mean(r['metrics']['before']['pose_translation_mm']) for r in admitted])
        after_translation=np.median([np.mean(r['metrics']['after']['pose_translation_mm']) for r in admitted])
        before_shape=np.median([r['metrics']['before']['reference_vertex_rmse_mm'] for r in admitted])
        after_shape=np.median([r['metrics']['after']['reference_vertex_rmse_mm'] for r in admitted])
    else:before_rotation=after_rotation=before_translation=after_translation=before_shape=after_shape=None
    checks=dict(all_cases_evaluated=len(records)==len(patients) and len(patients)>=criteria['minimum_cases'],
        admitted_fraction=len(admitted)/len(patients)>=criteria['minimum_admitted_fraction'],
        shared_pose_supported=bool(records) and all(r['source_pose_shared_across_cycle'] for r in records),
        pose_rotation_recovery=after_rotation is not None and after_rotation<=criteria['median_rotation_degrees'] and after_rotation<=before_rotation,
        pose_translation_recovery=after_translation is not None and after_translation<=criteria['median_translation_mm'] and after_translation<=before_translation,
        shape_preserved=after_shape is not None and after_shape<=criteria['median_shape_rmse_mm'] and after_shape<=before_shape+criteria['shape_rmse_growth_mm'],
        geometry_and_motion=bool(admitted) and all(r['geometry']['passed'] and r['motion']['passed'] for r in admitted))
    gate=dict(passed=all(checks.values()),checks=checks,criteria=criteria,failures=failures,
        total=len(patients),evaluated=len(records),admitted=len(admitted),
        medians=dict(rotation_before=before_rotation,rotation_after=after_rotation,translation_before=before_translation,
                     translation_after=after_translation,shape_before=before_shape,shape_after=after_shape),
        protocol_sha256=pipeline.protocol['config_sha256'],source_sha256=pipeline.protocol['source_sha256'])
    gate['validation_sha256']=pipeline.protocol['validation_sha256']
    gate['seed']=pipeline.config['runtime']['seed']
    write_json(folder/'gate.json',gate)
    return dict(completed=[r['case_id'] for r in records],failures=failures,synthetic_gate=gate)
