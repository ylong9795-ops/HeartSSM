"""Independent artifact audit; never changes live TTO source or checkpoints.

python scripts/audit_real_tto_joint.py --output outputs/real_tto_joint_v1_20261016
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from tto.config import load_config,resolve
from tto.dataset import RealTTODataset
from tto.evaluator import dice_stats
from tto.io import digest_json,load_state,read_json,sha256,write_json
from tto.losses import observation_cells
from tto.pose import estimated_pose


def same_stats(expected,actual):
    errors={}
    for key,value in expected.items():
        reported=actual.get(key)
        if value is None or reported is None:
            if value!=reported:errors[key]=dict(expected=value,reported=reported)
        elif isinstance(value,(float,int)) and not math.isclose(value,reported,rel_tol=1e-10,abs_tol=1e-12):
            errors[key]=dict(expected=value,reported=reported)
    return errors


def load_cells(path):
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key in ('exact_hard_dice','confidence','phase_confidence','prediction_area_mm2','target_area_mm2','phase_alpha'):
            row[key]=float(row[key])
        for key in ('phase','phase_upper'):row[key]=int(row[key])
        row['used_for_optimization']=row['used_for_optimization']=='True'
    return rows


def observation_key(row):
    return (str(row['native_frame']),row['view'],row['chamber'])


def failure_reason(row):
    if row.get('renderer_error'):return 'renderer_error'
    if row['prediction_area_mm2']==0 and row['target_area_mm2']>0:return 'missing_predicted_native_section'
    if row['target_area_mm2']==0 and row['prediction_area_mm2']>0:return 'predicted_section_against_empty_annotation'
    if row['exact_hard_dice']==0:return 'disjoint_native_sections'
    return 'native_contour_disagreement_at_estimated_pose; no 3D/Pose GT to assign a unique cause'


def pose_changes(first,last):
    # Saved float32 frames can be slightly non-orthogonal. acos(trace(R.T@R))
    # then invents a nonzero angle even for an identical frame. Project only
    # reporting copies to SO(3), use float64 atan2, and preserve exact equality.
    def reporting_rotation(value):
        value=value.double()
        orthogonality=(value.transpose(-1,-2)@value-torch.eye(3,dtype=value.dtype,device=value.device)).abs().amax()
        if not torch.isfinite(value).all() or float(orthogonality)>1e-3 or not (torch.linalg.det(value)>0).all():
            raise ValueError('Pose angle reporting requires finite near-SO(3) frames')
        u,_,vh=torch.linalg.svd(value)
        projected=u@vh
        correction=torch.ones_like(value[...,0,:])
        correction[...,-1]=torch.linalg.det(projected)
        return (u*correction.unsqueeze(-2))@vh
    r_first=reporting_rotation(first['R']);r_last=reporting_rotation(last['R'])
    rotation=r_first.transpose(-1,-2)@r_last
    skew=torch.stack((rotation[...,2,1]-rotation[...,1,2],
                      rotation[...,0,2]-rotation[...,2,0],
                      rotation[...,1,0]-rotation[...,0,1]),dim=-1)
    sine=skew.norm(dim=-1)/2
    cosine=((rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    angles=torch.rad2deg(torch.atan2(sine,cosine))
    identical=(first['R']==last['R']).all(dim=-1).all(dim=-1)
    angles=torch.where(identical,torch.zeros_like(angles),angles)
    translation=(last['center_mm'].double()-first['center_mm'].double()).norm(dim=-1)
    normals_first=r_first[...,:,2];normals_last=r_last[...,:,2]
    pairs=[]
    for a,b in ((0,1),(0,2),(1,2)):
        angle=lambda normals:float(torch.rad2deg(torch.atan2(torch.linalg.cross(normals[a],normals[b]).norm(),(normals[a]*normals[b]).sum())))
        pairs.append(dict(views=[a,b],normal_angle_before_degrees=angle(normals_first),
                          normal_angle_after_degrees=angle(normals_last),
                          center_distance_before_mm=float((first['center_mm'][a].double()-first['center_mm'][b].double()).norm()),
                          center_distance_after_mm=float((last['center_mm'][a].double()-last['center_mm'][b].double()).norm())))
    return dict(rotation_degrees=angles.tolist(),translation_mm=translation.tolist(),relative_view_geometry=pairs,
                rotation_metric='Float64 nearest-SO(3) reporting copies and atan2; identical saved matrices give exact zero. Saved poses are unchanged.')


def audit_candidate_selection(history,initial_z,baseline_rows,progress,arm):
    """Rebuild the legal candidate ranking without calling the optimizer ranker."""
    errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    def rank(stats):
        return tuple(stats[key] for key in ('ge098','minimum','P01','P05','mean'))
    baseline=dice_stats(baseline_rows)
    check(bool(baseline_rows),'Selection baseline has no eligible native cells')
    selected=dict(update=0,score=rank(baseline),z_shape=initial_z,
                  delta=torch.zeros_like(progress['best']['delta']))
    previous_z=initial_z;previous_delta=selected['delta'];legal=1
    for row in history:
        update=row['update']
        expected_groups=(['pose'] if arm=='pose_only' or (arm=='alternate' and update%2==1)
                         else ['shape'] if arm=='shape_only' or (arm=='alternate' and update%2==0)
                         else ['shape','pose'])
        check(row.get('parameter_groups_updated')==expected_groups,
              f'Declared parameter groups differ at update {update}')
        if 'shape' not in expected_groups:
            check(torch.equal(row['proposed']['z_shape'],previous_z),
                  f'Frozen Shape changed at update {update}')
        if 'pose' not in expected_groups:
            check(torch.equal(row['proposed']['delta'],previous_delta),
                  f'Frozen Pose changed at update {update}')
        if row['accepted']:
            stats=row.get('native_stats')
            check(stats is not None,f'Accepted candidate lacks native metrics at update {update}')
            if stats is not None:
                legal+=1;candidate_score=rank(stats)
                if candidate_score>selected['score']:
                    selected=dict(update=update,score=candidate_score,**row['proposed'])
        previous_z=row['retained']['z_shape'];previous_delta=row['retained']['delta']
    best=progress['best']
    check(tuple(best['score'])==selected['score'],'Saved best is not the maximum legal native score')
    check(torch.equal(best['z_shape'],selected['z_shape']),'Saved best Shape differs from ranked candidate')
    check(torch.equal(best['delta'],selected['delta']),'Saved best Pose differs from ranked candidate')
    check(torch.equal(progress['z_shape'],previous_z),'Current Shape differs from last retained state')
    check(torch.equal(progress['delta'],previous_delta),'Current Pose differs from last retained state')
    return dict(passed=not errors,errors=errors,selected_update=selected['update'],
        selected_score=list(selected['score']),baseline_cells=len(baseline_rows),legal_candidates=legal,
        ranking=['ge098_count','minimum','P01','P05','mean'],
        scope='Saved history consistency, parameter-group isolation and legal-candidate selection; not independent candidate rerendering or intermediate Adam-state verification')


def export_parameter_trajectories(output,state,baseline_rows,calibration,selection,state_sha256):
    """Plot recorded proposals/retained states; candidate Dice is never retained Dice."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    metadata_path=output/'joint_parameter_trajectory.json'
    code_sha=sha256(Path(__file__))
    if metadata_path.exists():
        previous=read_json(metadata_path)
        if (previous.get('state_sha256')==state_sha256 and previous.get('auditor_sha256')==code_sha
            and all((output/name).exists() for name in ('joint_parameter_trajectory.csv','joint_parameter_trajectory.png','joint_parameter_trajectory.pdf'))):
            return previous
    initial=state['initial']['z_shape_initial'][0].double()
    std=torch.as_tensor(calibration['shape_std'],dtype=torch.float64)
    baseline=dice_stats(baseline_rows)
    def record(update,accepted,proposed,retained,stats):
        row=dict(update=update,accepted=accepted)
        for prefix,parameters in (('proposed',proposed),('retained',retained)):
            delta=parameters['delta'].double()
            row[prefix+'_shape_rms_train_std']=float(((parameters['z_shape'].double()-initial)/std).square().mean().sqrt())
            for index,view in enumerate(('A2C','A3C','A4C')):
                row[f'{prefix}_{view}_translation_mm']=float(delta[index,:3].norm())
                row[f'{prefix}_{view}_rotation_degrees']=float(delta[index,3:].norm())
        for key in ('cells','mean','minimum','P01','P05','ge095','ge098'):
            row['candidate_native_'+key]=stats[key] if stats is not None else None
        return row
    base=dict(z_shape=initial,delta=torch.zeros(3,6))
    rows=[record(0,None,base,base,baseline)]
    rows.extend(record(h['update'],h['accepted'],h['proposed'],h['retained'],h['native_stats'])
                for h in state['result']['history'])
    with (output/'joint_parameter_trajectory.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    steps=[row['update'] for row in rows]
    selected=selection.get('selected_update',0)
    figure,axes=plt.subplots(2,2,figsize=(12,8))
    for prefix,style in (('proposed','--'),('retained','-')):
        axes[0,0].plot(steps,[row[prefix+'_shape_rms_train_std'] for row in rows],style,label=prefix)
    axes[0,0].set(ylabel='RMS displacement / measured TRAIN std',title='Shape latent relative to Joint initialization')
    for index,view in enumerate(('A2C','A3C','A4C')):
        for axis,suffix,title in ((axes[0,1],'translation_mm','Pose translation increments (mm)'),
                                  (axes[1,0],'rotation_degrees','Pose rotation increments (degrees)')):
            axis.plot(steps,[row[f'retained_{view}_{suffix}'] for row in rows],color=f'C{index}',label=view+' retained')
            axis.plot(steps,[row[f'proposed_{view}_{suffix}'] for row in rows],'--',color=f'C{index}',alpha=.55,label=view+' proposed')
            axis.set_title(title)
    for key in ('mean','minimum','P05'):
        axes[1,1].plot(steps,[np.nan if row['candidate_native_'+key] is None else row['candidate_native_'+key] for row in rows],
                       marker='o',markersize=3,label=key)
    axes[1,1].set(title='Proposed candidate Native Dice (Joint subset)',ylabel='Dice',ylim=(0,1))
    for axis in axes.flat:
        axis.axvline(selected,color='black',linestyle=':',label=f'selected update {selected}')
        axis.set_xlim(0,max(1,max(steps)))
        axis.set_xlabel('Joint update (0 = initialization)');axis.legend(fontsize=7);axis.grid(alpha=.2)
    figure.suptitle('Recorded optimization states; dashed proposals can be rejected; selected endpoint can precede last update')
    figure.tight_layout(rect=(0,0,1,.95))
    for extension in ('png','pdf'):figure.savefig(output/f'joint_parameter_trajectory.{extension}',dpi=150)
    plt.close(figure)
    metadata=dict(state_sha256=state_sha256,auditor_sha256=code_sha,updates=len(rows)-1,
        selected_update=selected,rows=len(rows),
        native_metric_updates=[row['update'] for row in rows if row['candidate_native_mean'] is not None],
        missing_native_metric_updates=[row['update'] for row in rows if row['candidate_native_mean'] is None],
        scope='Joint stage only, relative to its shared Pose initializer; Shape displacement uses measured TRAIN std. Native metrics describe proposed candidates, not retained or final states. Missing candidate metrics stay missing. No GT accuracy or geometry inference from these curves.')
    write_json(metadata_path,metadata);return metadata


def audit_visualizations(folder,patient):
    """Check saved visuals, without treating an image as a geometry certificate."""
    from PIL import Image
    expected={name:None for name in ('reference_mesh.png','optimization.png','motion_volumes.png','mesh_24phases.gif')}
    for phase in (0,6,12,18):
        native=patient.native(phase)
        for view,name in enumerate(('A2C','A3C','A4C')):
            h,w=native['masks'][view].shape[-2:]
            expected[f'{name}_phase_{phase:02d}_native.png']=(int(w),int(h))
    files={};errors=[]
    for name,size in expected.items():
        path=folder/name
        try:
            with Image.open(path) as picture:
                record=dict(sha256=sha256(path),size=list(picture.size),format=picture.format,
                    frames=picture.n_frames if hasattr(picture,'n_frames') else 1)
                if size is not None and picture.size!=size:errors.append(f'Native visualization dimensions differ: {name}')
                if min(picture.size)<=0:errors.append(f'Empty visualization dimensions: {name}')
                if name.endswith('.gif'):
                    durations=[]
                    for frame in range(record['frames']):
                        picture.seek(frame);picture.load();durations.append(picture.info.get('duration',0))
                    record['frame_durations_ms']=durations
                    if record['frames']!=24:errors.append(f'Animation lacks 24 model phases: {name}')
                    if any(duration<=0 for duration in durations):errors.append(f'Animation has invalid frame durations: {name}')
                else:picture.load()
                files[name]=record
        except Exception as error:
            errors.append(f'Missing or unreadable visualization {name}: {error!r}')
    return dict(folder=str(folder),passed=not errors,files=files,errors=errors,
        scope='File decodability, native preview dimensions and 24 animation frames only; images do not certify geometry or real 3D accuracy',
        native_preview_timing='Fixed model slots with nearest native labels; exact-native-time clinical overlays are separate audit outputs')


class ArtifactAuditor:
    def __init__(self,root,config,verify_renderer=False,render_overlays=False):
        self.root=Path(root);self.config=config;self.verify_renderer=verify_renderer
        self.patient_cache={};self.synthetic_dataset=None;self.synthetic_selection=None
        self.failures=[]
        self.render_overlays=render_overlays;self.cohort_cells=[]

    def patient(self,name,synthetic=False):
        key=(name,synthetic)
        if key not in self.patient_cache:
            if synthetic:
                from tto.synthetic_oracle import select_oracle_cases,hidden_patient
                if self.synthetic_dataset is None:
                    calibration=read_json(resolve(self.config['prior']['calibration']))
                    self.synthetic_dataset,self.synthetic_selection,_=select_oracle_cases(self.config,calibration)
                sample_index=self.synthetic_selection[int(name.split('_')[-1])]
                patient,_=hidden_patient(self.synthetic_dataset[sample_index],name)
            else:patient=RealTTODataset(resolve(self.config['data']['root']),[name],config=self.config['data'])[0]
            self.patient_cache[key]=patient
        return self.patient_cache[key]

    def audit_patient(self,path,protocol,arm,synthetic=False):
        name=path.name;state_path=path/'states'/'export.pth';errors=[]
        def check(condition,message):
            if not condition:errors.append(message)
        if not state_path.exists():
            failure=path/'failure.json'
            if failure.exists():return dict(patient=name,arm=arm,status='FAILED',failure=read_json(failure),integrity_passed=False)
            stages=sorted(p.stem for p in (path/'states').glob('*.pth'))
            return dict(patient=name,arm=arm,status='IN_PROGRESS' if stages else 'PENDING',stages=stages,integrity_passed=None)
        saved=load_state(state_path);state=saved['state'];result=state['result'];initial=state['initial']
        completion_path=path/'export'/'completed.json'
        check(completion_path.exists(),'Export completion marker missing')
        if completion_path.exists():
            completion=read_json(completion_path)
            check(completion['identity']==saved['identity'],'Export completion identity differs')
            check(completion['status']==result['status'],'Export completion status differs')
        check(saved['identity']['protocol_sha256']==digest_json(protocol),'Checkpoint protocol identity differs')
        patient=self.patient(name,synthetic)
        check(saved['identity']['input_sha256']==patient.identity,'Current input manifest differs from committed checkpoint')
        check(result['arm']==arm,'Result arm differs')
        check(tuple(result['trajectory'].shape)==(24,10631,3),'Full24 fixed-topology mesh contract differs')
        check(tuple(initial['z_shape_initial'].shape)==(1,64),'Shape64 initialization contract differs')
        check(tuple(initial['z_motion_initial'].shape)==(1,32),'Motion32 initialization contract differs')
        check(all(v==0 for v in state['forward_consistency'].values()),'Initial forward was not exactly original forward')
        check(torch.isfinite(result['trajectory']).all().item(),'Exported trajectory is nonfinite')
        calibration=read_json(resolve(self.config['prior']['calibration']))
        std=torch.tensor(calibration['shape_std']);drift=(result['z_shape']-initial['z_shape_initial'][0])/std
        check(float(drift.square().mean().sqrt())<=self.config['joint']['rms_limit']+1e-6,'Shape RMS trust domain exceeded')
        check(float(drift.abs().max())<=self.config['joint']['coordinate_limit']+1e-6,'Shape coordinate trust domain exceeded')
        history=result['history'];expected_updates=0 if arm=='frozen_prior' else self.config['joint']['updates']
        check(result['motion_mode']==('original_decoder' if arm=='original_motion' else 'frozen_field'),
              'Motion forward differs from declared experiment arm')
        if arm in ('frozen_prior','pose_only'):
            check(torch.equal(result['z_shape'],initial['z_shape_initial'][0]),'Frozen Shape arm changed Shape')
        if arm in ('frozen_prior','shape_only'):
            check(all(torch.allclose(state['pose'][k],result['pose'][k],atol=2e-5,rtol=0) for k in state['pose']),
                  'Frozen Pose arm changed Pose')
        if result['status']=='COMPLETE':
            check(len(history)==expected_updates,'Declared optimizer budget not executed')
            check(result['geometry']['passed'] and result['motion_report']['passed'],'Complete result lacks final geometry/Motion admission')
        previous_z=initial['z_shape_initial'][0].clone();previous_delta=torch.zeros(3,6)
        rollback_count=0
        for row in history:
            retained=row['retained']
            if not row['accepted']:
                rollback_count+=1
                check(torch.equal(retained['z_shape'],previous_z),f"Shape rollback differs at update {row['update']}")
                check(torch.equal(retained['delta'],previous_delta),f"Pose rollback differs at update {row['update']}")
            else:
                check(bool(row['geometry'] and row['geometry']['passed']),f"Accepted geometry not proven at {row['update']}")
                check(row['motion']['passed'],f"Accepted motion not proven at {row['update']}")
            previous_z=retained['z_shape'];previous_delta=retained['delta']
        optimization_budget=dict(attempted_updates=len(history),
            accepted_updates=sum(row['accepted'] for row in history),rejected_updates=rollback_count,
            rejection_reasons=dict(Counter(reason for row in history if not row['accepted']
                                           for reason in row['rejected_by'])),
            attempted_group_steps=dict(Counter(group for row in history for group in row.get('parameter_groups_updated',[]))),
            accepted_group_steps=dict(Counter(group for row in history if row['accepted']
                                               for group in row.get('parameter_groups_updated',[]))),
            final_lr_factor=None,
            scope='Proposals and admitted parameter-group steps, not necessarily nonzero numerical changes. A rejection can have multiple reasons; reason counts need not sum to rejected updates.')
        unknown=result['unobserved_vertices'];distance=(result['trajectory'][0]-initial['vertices_4d_initial'][0,0]).norm(dim=-1)
        unknown_stats=dict(vertices=int(unknown.sum()),rms_mm=float(distance[unknown].square().mean().sqrt()) if unknown.any() else 0.,
                           maximum_mm=float(distance[unknown].max()) if unknown.any() else 0.,all_vertex_maximum_mm=float(distance.max()))
        check(unknown_stats['maximum_mm']<=self.config['joint']['unknown_max_mm']+1e-5,'Unobserved structure hard bound exceeded')
        check(unknown_stats['all_vertex_maximum_mm']<=self.config['joint']['all_vertex_max_mm']+1e-5,'Reference vertex hard bound exceeded')
        joint_pose=pose_changes(state['pose'],result['pose'])
        check(max(joint_pose['translation_mm'])<=self.config['joint']['translation_limit_mm']+1e-5,'Joint Pose translation bound exceeded')
        check(max(joint_pose['rotation_degrees'])<=self.config['joint']['rotation_limit_degrees']+1e-3,'Joint Pose rotation bound exceeded')
        container=path.parents[1];progress_path=container/'checkpoints'/name/'joint.pth'
        if history:
            check(progress_path.exists(),'Restorable final optimization checkpoint missing')
            if progress_path.exists():
                progress=load_state(progress_path)['progress']
                optimization_budget['final_lr_factor']=progress['lr_factor']
                check(torch.equal(progress['frozen_fields'],initial['control_grids_initial']['fine_motion'][0]),'Frozen fields mutated')
                check(torch.equal(progress['z_motion_initial'],initial['z_motion_initial'][0]),'Motion latent mutated')
                check(torch.equal(result['z_shape'],progress['best']['z_shape']),'Final Shape differs from selected legal checkpoint')
                best_pose=estimated_pose(progress['best']['delta'],state['pose'])
                check(all(torch.allclose(best_pose[k],result['pose'][k],atol=2e-5,rtol=0) for k in best_pose),'Final Pose differs from selected legal checkpoint')
                check(all(key in progress for key in ('optimizer','best','rng','lr_factor','frozen_fields')),'Incomplete resumable state')
        exported=np.load(path/'export'/'reconstruction.npz',allow_pickle=False)
        for key,tensor in [('z_shape_final',result['z_shape']),('z_motion_initial',initial['z_motion_initial'][0]),
            ('frozen_fields',initial['control_grids_initial']['fine_motion'][0]),('vertices_4d_initial',initial['vertices_4d_initial'][0]),
            ('vertices_4d_frozen_field',result['frozen_field_trajectory']),('vertices_4d_original_decoder',result['original_decoder_trajectory'])]:
            check(np.array_equal(exported[key],tensor.numpy()),f'NPZ/state disagreement: {key}')
        report=read_json(path/'evaluation'/'metrics.json');all_failure_cells=[];sampling={}
        eligibility=observation_cells([patient],self.config['supervision'])
        expected_ids={(str(r['native_frame']),('A2C','A3C','A4C')[r['view']],('LV','LA','RV','RA')[r['chamber']]) for r in eligibility}
        joint_ids={(str(r['native_frame']),('A2C','A3C','A4C')[r['view']],('LV','LA','RV','RA')[r['chamber']])
                   for r in result['supervision_manifest'] if r['used'] and history}
        selection_audit=dict(status='NOT_APPLICABLE',reason='No Joint history')
        baseline_rows=[row for row in load_cells(path/'evaluation'/'pose_only_initialization_cells.csv')
                       if row['used_for_optimization'] and (not history or observation_key(row) in joint_ids)]
        if history and progress_path.exists():
            selection_audit=audit_candidate_selection(history,initial['z_shape_initial'][0],baseline_rows,progress,arm)
            errors.extend('Candidate selection: '+error for error in selection_audit['errors'])
        for stage in ('initial','pose_only_initialization','final','original_decoder_at_final_shape'):
            rows=load_cells(path/'evaluation'/(stage+'_cells.csv'));keys=[observation_key(r) for r in rows]
            check(len(keys)==len(set(keys)),f'Duplicate native observation in {stage}')
            check(set(keys)==expected_ids,f'Native observation omissions/additions in {stage}')
            summary=report['comparisons'][stage]
            check(not same_stats(dice_stats(rows),summary['all_annotated']),f'Reported all-label metrics differ in {stage}')
            used=[r for r in rows if r['used_for_optimization']]
            trusted=[r for r in rows if r['confidence']>=self.config['supervision']['trusted_confidence_threshold']]
            check(not same_stats(dice_stats(used),summary['optimization_observations']),f'Reported eligible-label metrics differ in {stage}')
            check(not same_stats(dice_stats(trusted),summary['trusted']),f'Reported trusted-label metrics differ in {stage}')
            if stage=='final':
                sampling=dict(annotated_cells=len(rows),eligible_cells=len(used),trusted_cells=len(trusted),
                    joint_loss_cells=len(joint_ids),joint_loss_coverage=len(joint_ids)/len(expected_ids) if expected_ids else 0.,
                    scope='Joint-stage usage; common encoder/Pose initialization can observe the full cycle')
                for row in rows:
                    if not synthetic:self.cohort_cells.append(dict(**row,arm=arm,joint_loss_used=observation_key(row) in joint_ids))
                    if row['exact_hard_dice']<.98:
                        all_failure_cells.append(dict(**row,joint_loss_used=observation_key(row) in joint_ids,failure_reason=failure_reason(row)))
        if self.verify_renderer:
            from train_neural_ffd import load_checkpoint
            from tto.evaluation import RealTTOEvaluator
            if not hasattr(self,'model'):self.model,_=load_checkpoint(resolve(self.config['prior']['checkpoint']),'cpu')
            recomputed=RealTTOEvaluator(self.model,self.config['supervision']).annotated_cells(result['trajectory'].numpy(),
                {k:v.numpy() for k,v in result['pose'].items()},patient,'audit')
            original={observation_key(r):r['exact_hard_dice'] for r in load_cells(path/'evaluation'/'final_cells.csv')}
            check(all(abs(r['exact_hard_dice']-original[observation_key(r)])<=1e-12 for r in recomputed),'Independent exact re-render differs')
        output=self.root/'independent_audit'/(('synthetic/'+name) if synthetic else (arm+'/'+name))
        write_json(output/'failed_cells.json',all_failure_cells)
        parameter_trajectories=export_parameter_trajectories(output,state,baseline_rows,calibration,
                                                           selection_audit,sha256(state_path))
        if self.render_overlays and not synthetic:self.native_overlays(patient,state,output/'native_time_overlays',exported)
        visuals=audit_visualizations(container/'visualizations'/name,patient)
        errors.extend(visuals['errors'])
        audit=dict(patient=name,arm=arm,status=result['status'],integrity_passed=not errors,errors=errors,
            state_sha256=sha256(state_path),sampling=sampling,unobserved_structure=unknown_stats,visualizations=visuals,
            pose_initial_to_final=pose_changes(state['pose_initial'],result['pose']),pose_joint_change=joint_pose,
            rollbacks_verified=rollback_count,candidate_selection=selection_audit,
            optimization_budget=optimization_budget,
            parameter_trajectories=parameter_trajectories,
            final_geometry_admitted=result['geometry']['passed'],
            final_motion_preserved=result['motion_report']['passed'],
            original_decoder_geometry=report['original_decoder_comparison']['geometry']['passed'],
            unverifiable_spatial_ffd_certificate=result['geometry'].get('spatial_ffd_certificate'),
            physiology_scope='Relative preservation does not establish correct real motion or real 3D accuracy',
            rollback_scope='Parameter rollback checked for every rejection; intermediate Adam states are covered by executed unit/resume tests, not individually archived')
        write_json(output/'audit.json',audit);return audit

    def native_overlays(self,patient,state,output,exported):
        """Render at the ORIGINAL observation timestamp, not its nearest model slot."""
        import cv2
        import pydicom
        from tto.evaluator import native_exact
        from tto.losses import native_cell,interpolate_cell
        from data.mask_transform import VIEWS,CHAMBERS
        output.mkdir(parents=True,exist_ok=True)
        contract=patient.metadata['contract'];pixels=None
        if 'dicom' in contract:
            path=Path(contract['dicom'])
            if not path.is_absolute():path=resolve(self.config['data']['root'])/patient.patient_id/path
            pixels=pydicom.dcmread(path).pixel_array
        rows=observation_cells([patient],self.config['supervision'])
        trajectory=state['result']['trajectory'].numpy();pose={k:v.numpy() for k,v in state['result']['pose'].items()}
        metadata=[]
        for vi,view in enumerate(VIEWS):
            native_indices=sorted({r['native_index'] for r in rows if r['view']==vi})
            if not native_indices:continue
            selected=[native_indices[k] for k in np.linspace(0,len(native_indices)-1,min(4,len(native_indices)),dtype=int)]
            for native_index in selected:
                cells=[r for r in rows if r['view']==vi and r['native_index']==native_index]
                row=cells[0];mask,_,spacing=native_cell(patient,row);h,w=mask.shape
                image=None
                if pixels is not None:
                    region=contract['views'][vi]['region']
                    image=pixels[native_index,region['y0']:region['y0']+h,region['x0']:region['x0']+w]
                else:
                    for slot,mapping in enumerate(patient.mapping):
                        if mapping[vi]['nearest_index']==native_index and patient.images:
                            image=patient.images[vi][slot];break
                if image is None:continue
                if image.ndim==2:image=np.repeat(image[...,None],3,-1)
                canvas=image[...,:3].astype(np.uint8).copy();vertices=interpolate_cell(trajectory,row)
                for cell in cells:
                    chamber=CHAMBERS[cell['chamber']];a,b=map(int,exported['chamber_ranges'][cell['chamber']])
                    faces=exported[f'faces_{chamber}']
                    target,_,spacing=native_cell(patient,cell)
                    prediction=native_exact(vertices[a:b],faces,pose['R'][vi],pose['center_mm'][vi],spacing,(h,w))
                    for source,color in ((target,(0,255,0)),(prediction,(255,0,255))):
                        contours,_=cv2.findContours(source.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(canvas,contours,-1,color,1)
                filename=f'{view}_native_{native_index:03d}.png'
                cv2.imwrite(str(output/filename),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
                metadata.append(dict(file=filename,native_frame=row['native_frame'],native_time_s=row['native_time_s'],
                    model_lower=row['phase'],model_upper=row['mesh_upper'],upper_weight=row['mesh_alpha'],
                    truth='original binary annotation, green',prediction='mesh interpolated at actual native time, magenta'))
        write_json(output/'index.json',metadata)

    def run(self,verify_legacy=False):
        previous_path=self.root/'independent_audit'/'audit_summary.json'
        previous=read_json(previous_path) if previous_path.exists() else None
        if previous is not None:
            write_json(self.root/'independent_audit'/'history'/f'audit_{previous["updated"]:.6f}.json',previous)
        protocol=read_json(self.root/'protocol.json')
        source_errors=[]
        for name,digest in protocol['sources'].items():
            for kind,path in [('current',ROOT/name),('snapshot',self.root/'source_snapshot'/name)]:
                if not path.exists() or sha256(path)!=digest:source_errors.append(dict(source=name,location=kind))
        prior_hash=sha256(resolve(self.config['prior']['checkpoint']))
        if prior_hash!=self.config['prior']['sha256']:source_errors.append(dict(source='frozen checkpoint'))
        records=[]
        for arm in self.config['experiment']['arms']:
            folder=self.root if arm=='joint' else self.root/'ablations'/arm
            local=read_json(folder/'protocol.json') if (folder/'protocol.json').exists() else None
            for patient in self.config['experiment']['cohort']:
                if local is None:records.append(dict(patient=patient,arm=arm,status='PENDING',integrity_passed=None));continue
                try:records.append(self.audit_patient(folder/'patients'/patient,local,arm))
                except Exception as error:records.append(dict(patient=patient,arm=arm,status='AUDIT_ERROR',error=repr(error),integrity_passed=False))
        synthetic=[]
        for index in range(self.config['synthetic']['cases']):
            path=self.root/'synthetic_hidden_gt'/'patients'/f'SYN_{index:03d}'
            try:synthetic.append(self.audit_patient(path,protocol,'joint',True))
            except Exception as error:synthetic.append(dict(patient=path.name,status='AUDIT_ERROR',error=repr(error),integrity_passed=False))
        legacy=previous.get('legacy') if previous else None
        if legacy is not None and 'verified_at' not in legacy:legacy=dict(legacy,verified_at=previous['updated'])
        if verify_legacy:
            inventory=read_json(self.root/'legacy_inventory.json');bad=[]
            for item in inventory['states']:
                path=resolve(item['path'])
                if not path.exists() or sha256(path)!=item['sha256']:bad.append(item['path'])
            legacy=dict(checked=len(inventory['states']),unchanged=not bad,changed_or_missing=bad,verified_at=time.time())
        summaries={}
        for arm in self.config['experiment']['arms']:
            cells=[r for r in self.cohort_cells if r['arm']==arm]
            summaries[arm]=dict(all_annotated=dice_stats(cells),
                eligible=dice_stats([r for r in cells if r['used_for_optimization']]),
                trusted=dice_stats([r for r in cells if r['confidence']>=self.config['supervision']['trusted_confidence_threshold']]),
                actual_joint_loss_cells=sum(r['joint_loss_used'] for r in cells),
                actual_joint_loss_coverage=sum(r['joint_loss_used'] for r in cells)/len(cells) if cells else None,
                by_source={s:dice_stats([r for r in cells if r['label_source']==s]) for s in sorted({r['label_source'] for r in cells})},
                cohort_denominator=len(self.config['experiment']['cohort']))
        output=dict(updated=time.time(),auditor_sha256=sha256(Path(__file__)),source_errors=source_errors,
            checkpoint_sha256=prior_hash,legacy=legacy,real=records,synthetic=synthetic,
            aggregate_native_metrics=summaries,
            all_required_results_exist=all(r['status'] not in ('PENDING','IN_PROGRESS','AUDIT_ERROR') for r in records+synthetic),
            complete_integrity_audit=not source_errors and all(r.get('integrity_passed') is True for r in records+synthetic),
            scope='Artifact integrity and declared coverage; independent of scientific recovery/geometry gates; pending is never pass')
        write_json(self.root/'independent_audit'/'audit_summary.json',output)
        return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/real_tto_joint_v1.yaml')
    parser.add_argument('--output',default='outputs/real_tto_joint_v1_20261016')
    parser.add_argument('--verify-renderer',action='store_true')
    parser.add_argument('--verify-legacy',action='store_true')
    parser.add_argument('--render-overlays',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    result=ArtifactAuditor(resolve(args.output),load_config(resolve(args.config)),args.verify_renderer,args.render_overlays).run(args.verify_legacy)
    print(json.dumps(dict(source_errors=result['source_errors'],legacy=result['legacy'],
        real_completed=sum(r.get('integrity_passed') is True for r in result['real']),
        synthetic_completed=sum(r.get('integrity_passed') is True for r in result['synthetic']),
        all_required_results_exist=result['all_required_results_exist'],complete_integrity_audit=result['complete_integrity_audit'])))
    return 0 if not result['source_errors'] and not any(r.get('integrity_passed') is False for r in result['real']+result['synthetic']) else 1


if __name__=='__main__':raise SystemExit(main())
