"""Fixed TRAIN/VAL diagnostics, independent of formal supervised training.

Oracle starts at GT frame0 solely to test the motion representation. Local
response uses geometrically consistent mask warps with regenerated mm SDF;
finite-difference Jacobians describe a sampled subspace, never a full rank.
"""
import argparse
import copy
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from scipy.ndimage import binary_erosion, map_coordinates
WORKSPACE_ROOT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('HEARTFFD_AUDIT_SOURCE_ROOT',str(WORKSPACE_ROOT))).resolve()
sys.path.insert(0,str(ROOT))
from data.mask_transform import mask_to_sdf,CHAMBERS,VIEWS
from train_geometry_heartffd_v2 import (load_config,dataset_from_config,build_model,
    load_checkpoint,to_device,write_json,code_identity,data_identity)
from utils.geometry_pair_policy import validate_source_geometry,validate_output_geometry
from utils.geometry_v2_checks import GeometrySafetyError,validate_observed_sections
from utils.geometry_motion_metrics import detailed_motion_metrics,observed_vertex_regions
from net.geometry_lifting import project_physical_points
from net.geometry_heartffdnet_v2 import bound_control_field


def relative(vertices):return vertices-vertices[:,:1]


def equal_chamber_mse(delta,ranges):
    return torch.stack([delta[:,:,a:b].square().sum(-1).mean() for a,b in ranges.tolist()]).mean()


def motion_oracle(model,batch,steps=48):
    """All 23 increment fields optimized together under the frozen bounds.

    Candidate safety checked after every step. Unsafe steps are rolled back;
    no target labels, initial reference geometry, or limit is modified.
    """
    gt=batch['vertices_gt'].to(model.template_vertices_mm);reference=gt[:,0].detach()
    model._source_av_cap_policy.audit_frames(reference.cpu().numpy())
    raw=torch.nn.Parameter(reference.new_zeros(1,23,*model.grid_shapes[2],3))
    optimizer=torch.optim.Adam([raw],lr=.2)
    def decode():
        bounded=bound_control_field(model.heads,raw,model.heads.increment_limit_mm,model.control_spacings_mm[2])
        fine=torch.cat((torch.zeros_like(bounded[:,:1]),bounded),1)
        trajectory=[reference]
        for t in range(1,24):trajectory.append(model._deform(trajectory[-1],fine[:,t],2))
        vertices=torch.stack(trajectory,1)
        controls=dict(fine=fine,coarse=reference.new_zeros(1,*model.grid_shapes[0],3),
                      medium=reference.new_zeros(1,*model.grid_shapes[1],3))
        return dict(vertices_4d=vertices,vertices_ref=vertices[:,0],control_grids=controls)
    initial=decode();initial_rmse=float(equal_chamber_mse(relative(initial['vertices_4d'])-relative(gt),model.chamber_ranges).sqrt())
    validate_output_geometry(model,initial)
    history=[];started=time.perf_counter()
    for step in range(steps):
        previous=raw.detach().clone();previous_optimizer=copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True);out=decode()
        loss=equal_chamber_mse(relative(out['vertices_4d'])-relative(gt),model.chamber_ranges)
        loss.backward()
        if not torch.isfinite(raw.grad).all():raise ValueError('Nonfinite oracle gradient')
        optimizer.step()
        accepted=False;failure=None
        for attempt in range(6):
            with torch.no_grad():candidate=decode()
            try:
                safety=validate_output_geometry(model,candidate);accepted=True;break
            except (GeometrySafetyError,ValueError) as error:
                failure=str(error)
                with torch.no_grad():raw.copy_(previous+.5*(raw-previous))
        if not accepted:
            with torch.no_grad():raw.copy_(previous)
            optimizer.load_state_dict(previous_optimizer)
        with torch.no_grad():
            out=decode();rmse=float(equal_chamber_mse(relative(out['vertices_4d'])-relative(gt),model.chamber_ranges).sqrt())
        history.append(dict(step=step+1,rmse_mm=rmse,accepted=accepted,backtracks=attempt,reason=failure))
        if step%8==0:print('oracle',step+1,rmse,flush=True)
    out=decode();safety=validate_output_geometry(model,out)
    bounded=out['control_grids']['fine'][:,1:]
    from net.geometry_heartffdnet import control_lipschitz_bound
    tanh=model.heads.increment_limit_mm*torch.tanh(raw.detach()/model.heads.increment_limit_mm)
    lip=control_lipschitz_bound(tanh,model.control_spacings_mm[2])
    regions=observed_vertex_regions(gt,batch,model.chamber_ranges,model.norm_to_mm)
    metrics=detailed_motion_metrics(out['vertices_4d'].detach(),gt,model.chamber_ranges,model.faces,batch['phases'],regions=regions)
    return dict(scope='restricted motion representation at exact GT frame0; no encoder or shape oracle',
        initial_rmse_mm=initial_rmse,final_rmse_mm=history[-1]['rmse_mm'],history=history,
        actual_motion_steps=steps,seconds=time.perf_counter()-started,safety=safety,metrics=metrics,
        tanh_saturation_fraction=float((raw.detach().abs()/2>1.8318).float().mean()),
        lipschitz_trigger_fraction=float((lip>.45).float().mean()),
        minimum_lipschitz_scale=float((.45/lip.clamp_min(1e-8)).clamp_max(1).min()),
        final_control_rms_mm=float(bounded.square().mean().sqrt())),out


def perturb_observation(batch,phase,view,chamber,*,model=None,displacement_mm=1.,radius_mm=10.):
    masks=batch['masks'].detach().cpu().numpy().copy()
    original=masks[0,phase,view,chamber]
    boundary=np.argwhere(original.astype(bool)&~binary_erosion(original))
    if not len(boundary):raise ValueError('Prespecified cell has no contour')
    # Rightmost quarter of the contour, median row; fixed rule, no result selection.
    outer=boundary[boundary[:,1]>=np.quantile(boundary[:,1],.75)]
    anchor=outer[np.argmin(abs(outer[:,0]-np.median(outer[:,0])))].astype(float)
    selector='rightmost contour quartile / median row'
    if CHAMBERS[chamber]=='RV' and model is not None:
        import trimesh
        from scipy.spatial import cKDTree
        a,b=model.chamber_ranges[chamber].tolist();la,lb=model.chamber_ranges[0].tolist()
        gt=batch['vertices_gt'][0,phase].cpu().numpy()
        rotation=batch['pose_gt']['probe_R'][0,phase,view].cpu().numpy()
        translation=batch['pose_gt']['probe_t'][0,phase,view].cpu().numpy()*model.norm_to_mm
        cut=trimesh.Trimesh(vertices=gt[a:b],faces=model.faces['RV'].cpu().numpy(),process=False).section(
            plane_origin=translation,plane_normal=rotation[:,2])
        if cut is None:raise ValueError('RV free-wall selector has no GT section')
        distance=cKDTree(gt[la:lb]).query(cut.vertices)[0]
        point=batch['sdf'].new_tensor(cut.vertices[np.argmax(distance)][None])
        projected=project_physical_points(point,batch,model.norm_to_mm,original.shape)
        rc=projected['model_pixels'][0,phase,view,0].flip(0).cpu().numpy()
        anchor=boundary[np.argmin(((boundary-rc)**2).sum(1))].astype(float)
        selector='GT RV section point farthest from LV surface; nearest model-mask boundary'
    spacing=batch['pixel_spacing'][0,phase,view].cpu().numpy()
    yy,xx=np.mgrid[:original.shape[0],:original.shape[1]]
    radius2=((yy-anchor[0])*spacing[0])**2+((xx-anchor[1])*spacing[1])**2
    shift=(displacement_mm/spacing[1])*np.exp(-.5*radius2/radius_mm**2)
    masks[0,phase,view,chamber]=map_coordinates(original,[yy,xx-shift],order=0,mode='constant',cval=0)
    sdf=mask_to_sdf(masks[0],batch['pixel_spacing'][0].cpu().numpy(),
        view_present=batch['view_present'][0].cpu().numpy(),annotated=batch['annotated'][0].cpu().numpy())
    modified=dict(batch,masks=torch.as_tensor(masks,device=batch['masks'].device),
                  sdf=torch.as_tensor(sdf[None],device=batch['sdf'].device))
    return modified,dict(anchor_row_col=anchor.tolist(),displacement_mm=displacement_mm,
        radius_mm=radius_mm,selector=selector,changed_mask_pixels=int(np.count_nonzero(original!=masks[0,phase,view,chamber])))


@torch.no_grad()
def local_response(model,batch,*,jacobian_directions=32):
    model.eval();baseline=model(batch,return_features=True);encoded=baseline['features']
    baseline_geometry={};geometry_accepted=True
    for name,check in (('physical',validate_output_geometry),('sections',lambda m,o:validate_observed_sections(m,o,batch))):
        try:baseline_geometry[name]=check(model,baseline)
        except (GeometrySafetyError,ValueError) as error:
            geometry_accepted=False;baseline_geometry[name]=dict(status='rejected',reason=str(error),detail=getattr(error,'report',None))
    p,gt=baseline['vertices_4d'],batch['vertices_gt'];residual=relative(gt)-relative(p)
    regions=observed_vertex_regions(gt,batch,model.chamber_ranges,model.norm_to_mm)
    targets=[('A4C','RV'),('A4C','RA'),('A2C','LV'),('A3C','LV')]
    records=[]
    for view_name,chamber_name in targets:
        view,chamber=VIEWS.index(view_name),CHAMBERS.index(chamber_name)
        left,right=model.chamber_ranges[chamber].tolist()
        controls_response=[];mesh_response=[];local=[];feature_changes=[];direction_cosines=[]
        for phase in range(24):
            modified,meta=perturb_observation(batch,phase,view,chamber,model=model)
            changed=model(modified,return_features=True)
            dctrl=changed['control_grids']['fine']-baseline['control_grids']['fine']
            dmesh=relative(changed['vertices_4d'])-relative(p)
            controls_response.append(dctrl.square().mean((0,2,3,4,5)).sqrt().cpu().tolist())
            mesh_response.append(dmesh.square().sum(-1).mean((0,2)).sqrt().cpu().tolist())
            # Physical locality: corresponding GT vertices close to the warped contour anchor.
            projection=project_physical_points(gt[0,phase,left:right],batch,model.norm_to_mm,batch['masks'].shape[-2:])
            pixel=projection['model_pixels'][0,phase,view]
            anchor=pixel.new_tensor(meta['anchor_row_col'])[ [1,0] ]
            spacing=batch['pixel_spacing'][0,phase,view].flip(0)
            radius=((pixel-anchor)*spacing).norm(dim=-1)
            plane=projection['distance_mm'][0,phase,view].abs()
            target=(radius<=15)&(plane<=8)
            if not target.any():raise ValueError('Prespecified perturbation lacks a corresponding 3D target region')
            # Restoration of the known correct contour is the directional
            # intervention. The arbitrary outward corruption is not expected
            # to improve GT; comparing it that way would invert the hypothesis.
            r=(residual-dmesh)[0,phase,left:right][target]
            update=-dmesh[0,phase,left:right][target]
            cosine=float(torch.nn.functional.cosine_similarity(update.flatten(),r.flatten(),dim=0,eps=1e-12))
            before=float(r.square().sum(-1).mean().sqrt())
            after=float((r-update).square().sum(-1).mean().sqrt())
            direction_cosines.append(cosine)
            unknown=regions['unknown']
            local.append(dict(phase=phase,target_vertices=int(target.sum()),**meta,
                target_response_rms_mm=float(update.square().sum(-1).mean().sqrt()),
                unknown_response_rms_mm=float(dmesh[unknown].square().sum(-1).mean().sqrt()) if unknown.any() else None,
                gt_residual_cosine=cosine,target_motion_rmse_before_mm=before,target_motion_rmse_after_mm=after,
                direction_intervention='corrupted contour -> original GT-compatible clean contour'))
            feature_changes.append({key:float((changed['features'][key]-encoded[key]).square().mean().sqrt())
                for key in ('motion','fine_skip')})
            feature_changes[-1].update(cnn_scale14_rms=float((changed['features']['spatial']['scale14']-encoded['spatial']['scale14']).square().mean().sqrt()),
                lifted_medium_rms=float((changed['features']['medium']['features']-encoded['medium']['features']).square().mean().sqrt()))
            del changed
        records.append(dict(view=view_name,chamber=chamber_name,
            direct_control_response_output_by_input=np.asarray(controls_response).T.tolist(),
            accumulated_mesh_response_output_by_input=np.asarray(mesh_response).T.tolist(),
            local_records=local,feature_changes=feature_changes,
            positive_residual_cosine_fraction=float(np.mean(np.asarray(direction_cosines)>0))))
    # Decoder-only subspace: random, spatially localized phase-motion probes.
    generator=torch.Generator(device=p.device).manual_seed(2026)
    indices=torch.cat([torch.linspace(a,b-1,64,device=p.device).long() for a,b in model.chamber_ranges.tolist()])
    epsilon=.1;columns=[]
    for direction in range(jacobian_directions):
        noise=torch.randn(encoded['motion'].shape,device=p.device,generator=generator)
        noise=noise/noise.square().mean().sqrt()
        field_plus,_=model.generate_controls(encoded,motion_override=encoded['motion']+epsilon*noise)
        field_minus,_=model.generate_controls(encoded,motion_override=encoded['motion']-epsilon*noise)
        plus=model.apply_controls(field_plus);minus=model.apply_controls(field_minus)
        columns.append(((relative(plus)-relative(minus))/(2*epsilon))[:,:,indices].reshape(-1).cpu())
    jac=torch.stack(columns,1).double();r=residual[:,:,indices].reshape(-1).cpu().double()
    u,s,v=torch.linalg.svd(jac,full_matrices=False);rank=int((s>s.max()*1e-3).sum())
    projected=u[:,:rank]@(u[:,:rank].T@r)
    explained=float(projected.square().sum()/r.square().sum().clamp_min(1e-20))
    disabled_controls,_=model.generate_controls(encoded,motion_override=torch.zeros_like(encoded['motion']),
                                               fine_skip_override=torch.zeros_like(encoded['fine_skip']))
    disabled=model.apply_controls(disabled_controls)
    variances={key:float(value.float().var(dim=(-3,-2,-1),unbiased=False).mean())
               for key,value in dict(motion=encoded['motion'],fine_skip=encoded['fine_skip'],
                   context=encoded['context']['fine'],lifted=encoded['medium']['features']).items()}
    return dict(scope='fixed geometry-compatible boundary perturbations; not motion ground truth',
        baseline_geometry=baseline_geometry,baseline_geometry_accepted=geometry_accepted,
        counterfactual_actual_triangle_and_section_safety_all_checked=False,
        targets=records,jacobian=dict(method='central finite differences of local motion features; 32-direction random sketch',
            epsilon=epsilon,directions=jacobian_directions,singular_values=s.tolist(),
            sampled_effective_rank=rank,sampled_gt_residual_explained_fraction=explained,
            rank_threshold_relative=1e-3,full_decoder_rank_claim=False,
            precision='float32 mesh finite differences; initial very small heads can be below coordinate resolution'),spatial_feature_variances=variances,
        local_branch_disabled_mesh_change_rms_mm=float((disabled-p).square().sum(-1).mean().sqrt()),
        motion_rmse_mm=float(equal_chamber_mse(relative(p)-relative(gt),model.chamber_ranges).sqrt()),
        local_branch_disabled_motion_rmse_mm=float(equal_chamber_mse(relative(disabled)-relative(gt),model.chamber_ranges).sqrt()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('oracle','response'),required=True)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--steps',type=int,default=48)
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(2026)
    config,data,protocol=load_config(ROOT/'configs/geometry_heartffd_v2.yaml')
    dataset=dataset_from_config(data)
    if args.checkpoint:model,saved=load_checkpoint(args.checkpoint,args.device,identity=data_identity(data))
    else:model=build_model(dataset,config).to(args.device)
    validate_source_geometry(model)
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=dict(mode=args.mode,code_identity=code_identity(),data_identity=data_identity(data),
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_source_root=str(ROOT),
        checkpoint=str(args.checkpoint) if args.checkpoint else None,initialization='checkpoint' if args.checkpoint else 'random',
        selection='first fixed TRAIN and first fixed VAL; no TEST or result selection')
    write_json(args.output/'experiment.json',metadata)
    for split in ('train','val'):
        sample=dataset_from_config(data,split)[0];batch=to_device(default_collate([sample]),args.device)
        try:
            if args.mode=='oracle':
                report,out=motion_oracle(model,batch,args.steps)
                torch.save(dict(control_grids={k:v.detach().cpu() for k,v in out['control_grids'].items()},
                    vertices_ref=out['vertices_ref'].detach().cpu(),scope='GT reference motion oracle',**metadata),args.output/(split+'_oracle.pth'))
            else:report=local_response(model,batch)
            report.update(case_id=sample['case_id'],candidate_id=sample['candidate_id'])
            write_json(args.output/(split+'.json'),report)
        except (GeometrySafetyError,ValueError) as error:
            write_json(args.output/(split+'.json'),dict(status='rejected',reason=str(error),
                detail=getattr(error,'report',None),case_id=sample['case_id']))
            raise
        print(args.mode,split,'complete',flush=True)


if __name__=='__main__':main()
