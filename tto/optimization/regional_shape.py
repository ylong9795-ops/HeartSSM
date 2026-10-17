"""S2: original Regional8 candidates, fixed Global32/Pose/Motion32.

Temporary translations initialise latent candidates only. All saved candidate
meshes come from a fresh original full24 Decoder, with motion changes reported.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from data.mask_transform import CHAMBERS
from ..alignment.chamber_alignment import generate_candidates
from ..alignment.regional_decoder import FixedGlobalRegionalDecoder
from ..geometry.guard import GeometryGuard
from ..geometry.motion_metrics import motion_metrics
from ..io import cpu,read_json,save_state,write_json
from ..state import Stage
from ..supervision.regional_shape import RegionalShapeObjective
from ..supervision.regional_diagnostics import evaluate_candidate


def observation_acceptance(baseline,candidate,options):
    reasons=[]
    if candidate['renderer_errors']:reasons.append('native renderer errors')
    gain=candidate['macro_trusted_dice']-baseline['macro_trusted_dice']
    if gain<options['minimum_dice_gain']:reasons.append('no predeclared all-chamber native Dice gain')
    for c in CHAMBERS:
        b,a=baseline['chambers'][c]['trusted_dice'],candidate['chambers'][c]['trusted_dice']
        if b is not None and (a is None or a<b-options['native_dice_regression_tolerance']):
            reasons.append(f'{c} trusted native Dice regression')
    return not reasons,reasons


def _write_rows(path,rows):
    import csv
    from ..io import jsonable
    if not rows:return
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows(jsonable(rows))


def strict_full_audit(candidate_path,checkpoint,output,timeout):
    command=[sys.executable,'-m','tto.geometry.audit_worker','--candidate',str(candidate_path),
             '--checkpoint',str(checkpoint),'--output',str(output)]
    timed_out=False
    with Path(output).with_suffix('.log').open('w') as log:
        try:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1.,timeout),check=False)
            error=None if result.returncode==0 else f'geometry worker exit {result.returncode}'
        except subprocess.TimeoutExpired:
            timed_out=True;error='strict geometry audit wall budget exhausted'
    report=read_json(output) if Path(output).exists() else dict(passed=False,status='UNVERIFIED')
    if error:
        report.update(passed=False,status='UNVERIFIED',error=error,timed_out=timed_out)
        write_json(output,report)
    return report


def fit_regional(prior,patient,initial,*,options,policy,pose_config,regional_std,output,checkpoint):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+options['max_seconds_per_case']
    initial.validate()
    if initial.observation_pose is None or initial.diagnostics.get('pose_gauge')!='canonical_identity_global':
        raise ValueError('S2 requires S1 shared canonical poses and identity global gauge')
    baseline,rows=evaluate_candidate(prior.model,patient,initial,initial.trajectory.numpy(),policy,pose_config)
    hints=generate_candidates(prior,patient,initial,rows=rows,config=options)
    _write_rows(output/'baseline_cells.csv',rows)
    write_json(output/'relative_translation_hints.json',hints)
    decoder=FixedGlobalRegionalDecoder(prior,initial.z_shape,initial.conditions,regional_std)
    objective=RegionalShapeObjective(prior.model,patient,initial,options,policy)
    fit=decoder.fit_hint(hints['displacements_mm'],options['latent_fit_ridge'])
    with torch.no_grad():
        fit_v,fit_z=decoder.decode(fit)
    fit_error=[]
    for ci,(a,b) in enumerate(prior.model.chamber_ranges.tolist()):
        actual=(fit_v[0,a:b]-initial.vertices_reference[a:b]).mean(0).cpu().numpy()
        fit_error.append(dict(chamber=CHAMBERS[ci],desired_initial_hint_mm=hints['displacements_mm'][ci],
                             decoder_realised_mean_shift_mm=actual,
                             hint_mismatch_mm=float(np.linalg.norm(actual-hints['displacements_mm'][ci]))))
    write_json(output/'hint_to_latent_initialization.json',dict(chambers=fit_error,
        raw_normalized_regional_delta=fit,teacher_is_3d_gt=False,temporary_offsets_used_in_final_mesh=False))
    guard=GeometryGuard(prior.model)
    baseline_geometry=guard.discrete(initial.vertices_reference.numpy())
    write_json(output/'baseline_reference_geometry.json',baseline_geometry)
    temporary=initial.trajectory.clone()
    for ci,(a,b) in enumerate(prior.model.chamber_ranges.tolist()):
        temporary[:,a:b]+=torch.as_tensor(hints['displacements_mm'][ci],dtype=temporary.dtype)
    relative_error=((temporary-temporary[:1])-(initial.trajectory-initial.trajectory[:1])).abs().max().item()
    write_json(output/'temporary_hint_audit.json',dict(
        relative_motion_max_abs_error_mm=relative_error,expected_motion_contract='constant whole-cycle translation preserves relative motion up to roundoff',
        reference_geometry=guard.discrete(temporary[0].numpy()),
        geometry_is_guaranteed=False,temporary_mesh_saved_or_used_as_final=False))
    priority=max(range(4),key=lambda ci: baseline['chambers'][CHAMBERS[ci]]['center_error_mm'] or 0.)
    candidates=[];states=[];budget_hit=False
    for branch,scale in enumerate(options['seed_scales']):
        if time.monotonic()>=deadline:budget_hit=True;break
        raw=torch.nn.Parameter((fit*scale).detach().clone())
        optimizer=torch.optim.Adam([raw],lr=options['learning_rate'])
        history=[];best=None;best_value=float('inf')
        for step in range(options['steps']+1):
            if time.monotonic()>=deadline:budget_hit=True;break
            optimizer.zero_grad(set_to_none=True)
            vertices,z=decoder.decode(raw)
            result=objective(vertices,raw)
            if not torch.isfinite(result['total']):raise ValueError('Nonfinite S2 reference objective')
            value=float(result['total'].detach())
            history.append(dict(step=step,total=value,components={k:float(v.detach()) for k,v in result['components'].items()},
                all_observed_chambers_evaluated=result['evaluated_chambers'],missing_sections=result['missing_sections']))
            if value<best_value:
                best_value=value;best=raw.detach().clone()
            if step==options['steps']:break
            result['total'].backward()
            if raw.grad is None or not torch.isfinite(raw.grad).all():raise ValueError('Invalid Regional8 gradients')
            if step<options['priority_steps']:
                mask=torch.zeros_like(raw);mask[priority]=1;raw.grad.mul_(mask)
            history[-1]['gradient_norm']=float(raw.grad.norm())
            optimizer.step()
        if best is None:break
        z=decoder.assert_original(best)
        with torch.no_grad():
            generated=prior.decode_original(z,initial.z_motion.to(prior.device),initial.conditions)
            vertices=generated['vertices_4d'][0].detach().cpu()
        if not torch.equal(z[:,:32].cpu(),initial.z_shape[:,:32]):raise AssertionError('Global32 changed')
        diagnostics=deepcopy(initial.diagnostics)
        diagnostics.update(network_updates=0,latent_updates=len(history)-1,
            global_shape_updates=0,motion_latent_updates=0,pose_updates=0,
            geometry_status='CANDIDATE_NOT_ACCEPTED',optimization_performed=True,
            s2_candidate=dict(branch=branch,initialization_scale=scale,priority_chamber=CHAMBERS[priority],
                partition_lock_version='regional-only-v1',
                frozen_parameters=dict(global32=initial.z_shape[:,:32].clone(),motion32=initial.z_motion.clone(),
                    pose={k:v.clone() for k,v in initial.observation_pose.items()}),
                checkpoint_partition=dict(global_dim=32,regional_dims=[8]*4,motion_dim=32),
                original_decoder_verified=True,reference_decoder_cached_exactly=True,
                temporary_offsets_used_in_final_mesh=False,
                relative_motion_changes=motion_metrics(prior.model,vertices,initial.trajectory),
                motion_contract='Motion32 frozen; full original Decoder regenerated; relative motion may change and is NOT locked',
                objective_scope='reference phase0 native 2D cells; full24 hard evaluation for every endpoint',
                phase_semantics='S0/S1 UNVERIFIED unchanged'))
        state=replace(initial,z_shape=z.detach().cpu(),z_motion=initial.z_motion.clone(),
            vertices_reference=vertices[0].clone(),trajectory=vertices,diagnostics=diagnostics,
            stage=Stage.CHAMBER_INITIALIZED)
        state.validate()
        metrics,cells=evaluate_candidate(prior.model,patient,state,vertices.numpy(),policy,pose_config)
        eligible,reasons=observation_acceptance(baseline,metrics,options)
        quick=guard.discrete(vertices[0].numpy())
        if not quick['passed']:reasons.append('reference geometry FAIL or UNVERIFIED');eligible=False
        common=np.stack([(vertices[0,a:b]-initial.vertices_reference[a:b]).mean(0).numpy()
                         for a,b in prior.model.chamber_ranges.tolist()]).mean(0)
        if np.linalg.norm(common)>1e-4:raise AssertionError('S2 reintroduced a shared chamber translation')
        record=dict(branch=branch,initialization_scale=scale,reference_loss=best_value,metrics=metrics,
            eligible_for_full_geometry=eligible,rejection_reasons=reasons,accepted=False,
            geometry_reference=quick,mean_common_reference_shift_mm=common,
            relative_motion_changes=diagnostics['s2_candidate']['relative_motion_changes'],
            completed_steps=len(history)-1)
        folder=output/f'candidate_{branch}';folder.mkdir(exist_ok=True)
        save_state(folder/'state.pth',state.to_payload())
        save_state(folder/'optimizer.pth',dict(raw_normalized_regional_delta=raw.detach(),best_raw_delta=best,
                                             optimizer=optimizer.state_dict(),identity=state.identity))
        write_json(folder/'history.json',history);_write_rows(folder/'cells.csv',cells)
        write_json(folder/'report.json',record)
        candidates.append(record);states.append(state)
        print(f"S2 {patient.patient_id}/branch{branch}: Dice {baseline['macro_trusted_dice']:.4f}->{metrics['macro_trusted_dice']:.4f}; geometry-eligible={eligible}",flush=True)
    # The fixed protocol fully audits the highest-scoring eligible candidate;
    # other hypotheses stay provisional. Never choose one on LV Dice alone.
    eligible=[i for i,c in enumerate(candidates) if c['eligible_for_full_geometry']]
    selected=None
    if eligible:
        index=max(eligible,key=lambda i:candidates[i]['metrics']['macro_trusted_dice'])
        folder=output/f'candidate_{index}'
        remaining=max(1.,deadline-time.monotonic())
        geometry=strict_full_audit(folder/'state.pth',checkpoint,folder/'geometry.json',
            min(options['geometry_timeout_seconds'],remaining))
        candidates[index]['geometry_full']=geometry
        if geometry['passed']:
            selected=index;candidates[index]['accepted']=True
            candidates[index]['rejection_reasons']=[]
        else:candidates[index]['rejection_reasons'].append('full24/AV/continuous geometry FAIL or UNVERIFIED')
        write_json(folder/'report.json',candidates[index])
    final=states[selected] if selected is not None else replace(initial,diagnostics=deepcopy(initial.diagnostics))
    final.diagnostics['s2']=dict(status='ACCEPTED_REGIONAL_CANDIDATE' if selected is not None else 'NO_ACCEPTED_CANDIDATE',
        selected_branch=selected,baseline_metrics=baseline,candidates=candidates,
        parameter_contract=dict(network_weights='FROZEN',global32='FROZEN',regional8x4='ADJUSTED_WITH_COMMON_SHIFT_GAUGE',
            motion32='FROZEN',shared_view_poses='FROZEN',final_mesh='ORIGINAL_DECODER_ONLY'),
        policy=options,geometry_acceptance='strict original guard including full24 and23->0; UNVERIFIED rejects',
        motion_acceptance='finite full Decoder output only; no motion-identity tolerance',
        phase_semantics='UNVERIFIED from S0/S1, not repaired by S2',
        accepted=selected is not None,budget_exhausted=budget_hit,elapsed_seconds=time.monotonic()-started)
    if selected is not None:final.diagnostics['geometry_status']='PASS_ORIGINAL_GUARD'
    prior.assert_frozen()
    save_state(output/'s2_state.pth',final.to_payload())
    write_json(output/'s2_report.json',dict(patient_id=patient.patient_id,diagnostics=final.diagnostics['s2']))
    return final
