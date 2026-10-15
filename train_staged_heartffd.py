"""Goal04-S only: fixed TRAIN4/VAL8 experiment, no formal 1K or Motion mode."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import default_collate
import yaml
from data.mask_transform import CHAMBERS
from data.synthetic_4d import _sha256
from net.staged_heartffdnet import StagedHeartFFDNet, reference_loss
from train_geometry_heartffdnet import (load_config as load_base, data_identity, geometry_identity,
    save_checkpoint as save_g1, load_checkpoint as load_g1, training_collate, finite_gradient_report, write_json)
from train_neural_ffd import dataset_from_config, to_device, exact_metrics, restore_rng
from utils.neural_ffd_losses import NeuralFFDLoss, reconstruction_metrics
from utils.geometry_joint_checks import validate_template, validate_prediction, triangle_self_intersections
from utils.geometry_motion_metrics import observed_vertex_regions
from utils.geometry_pair_policy import SourceAVCapPolicy, SETTINGS
ROOT = Path(__file__).resolve().parent


def code_identity():
    files = sorted(p for directory in ('net','utils','data') for p in (ROOT/directory).glob('*.py'))
    files += sorted(ROOT.glob('train*.py'))
    files += [ROOT/'configs/staged_heartffd.yaml',ROOT/'configs/geometry_heartffd_1k.yaml',
              ROOT/'configs/synthetic_data_1k.yaml',ROOT/'tests/test_staged_heartffdnet.py']
    return {str(p.relative_to(ROOT)):_sha256(p) for p in files}


def phase0(batch):
    """Slice evaluation observations only; model still consumes all 24 phases."""
    def take(value):
        if isinstance(value,dict):return {k:take(v) for k,v in value.items()}
        if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==24:return value[:,:1]
        return value
    return {k:take(v) for k,v in batch.items()}


def load_model(path, device, identity=None, config=None):
    model,saved = load_g1(path,device,identity=identity,config=config)
    if saved['runtime'].get('stage_contract')!='goal04s-reference-only-v1':
        raise ValueError('Stage S checkpoint required')
    if saved['runtime']['code_identity']!=code_identity():raise ValueError('Source identity differs')
    model.__class__ = StagedHeartFFDNet
    return model,saved


@torch.no_grad()
def evaluate(model,samples,device,full=False,surface_samples=8192):
    model.eval();rows=[]
    for sample in samples:
        batch=to_device(default_collate([sample]),device)
        out=model.reference_output(batch);truth=batch['vertices_gt'][:,:1]
        metrics=reconstruction_metrics(out['vertices_4d'],truth,model.chamber_ranges,model.faces)
        safety=validate_prediction(model,out)
        row=dict(case_id=sample['case_id'],metrics=metrics,geometry=safety,
                 control_rms_mm={k:float(v.square().mean().sqrt()) for k,v in out['control_grids'].items()})
        if full:
            import trimesh
            from net.geometry_heartffdnet import control_lipschitz_bound
            regions=observed_vertex_regions(batch['vertices_gt'],batch,model.chamber_ranges,model.norm_to_mm)
            for c,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist()):
                p=out['vertices_ref'][0,a:b].cpu().numpy();g=truth[0,0,a:b].cpu().numpy()
                f=model.faces[c].cpu().numpy();delta=p-g;error=np.linalg.norm(delta,axis=-1)
                pm=trimesh.Trimesh(p,f,process=False);gm=trimesh.Trimesh(g,f,process=False)
                ps=trimesh.sample.sample_surface(pm,surface_samples,seed=2026)[0]
                gs=trimesh.sample.sample_surface(gm,surface_samples,seed=2026)[0]
                dp=trimesh.proximity.closest_point(gm,ps)[1]
                dg=trimesh.proximity.closest_point(pm,gs)[1]
                normal=(delta*gm.vertex_normals).sum(-1)
                tangent=np.sqrt(np.maximum(0,error**2-normal**2))
                m=metrics[c]
                m.update(reference_vertex_p95_mm=float(np.quantile(error,.95)),
                    surface_assd_mm=float((dp.mean()+dg.mean())/2),
                    surface_hd95_mm=float(max(np.quantile(dp,.95),np.quantile(dg,.95))),
                    normal_rmse_mm=float(np.sqrt(np.mean(normal**2))),
                    tangential_rmse_mm=float(np.sqrt(np.mean(tangent**2))),
                    normal_mae_mm=float(np.abs(normal).mean()),
                    tangential_mae_mm=float(tangent.mean()),
                    self_intersections=triangle_self_intersections(p,f))
                vp=m['predicted_volume_curve_ml'][0][0];vg=m['gt_volume_curve_ml'][0][0]
                m['volume_relative_error_percent']=100*(vp-vg)/vg
                m['regions']={}
                for region in ('direct','propagated','unknown'):
                    selected=regions[region][0,0,a:b].cpu().numpy();e=error[selected]
                    m['regions'][region]=dict(count=int(selected.sum()),
                        rmse_mm=float(np.sqrt(np.mean(e**2))) if len(e) else None,
                        mae_mm=float(e.mean()) if len(e) else None,
                        p95_mm=float(np.quantile(e,.95)) if len(e) else None)
            row['exact']=exact_metrics(model,out['vertices_4d'],phase0(batch))
            # Audit adapter only: 23 identity maps permit reuse of frozen G1
            # Jacobian code whose loop is hard-coded to 24. No network recurrence.
            grids=out['control_grids'];fine=grids['fine']
            audit=dict(out,control_grids=dict(grids,fine=torch.cat((fine,torch.zeros_like(fine).expand(-1,23,-1,-1,-1,-1)),1)))
            row['geometry']=validate_prediction(model,audit,sample_jacobian=True)
            row['geometry']['actual_mesh_self_intersecting_pairs']=sum(metrics[c]['self_intersections'] for c in CHAMBERS)
            row['geometry']['inter_chamber']=SourceAVCapPolicy(model,SETTINGS).audit_frames(out['vertices_4d'][0].cpu().numpy())
            row['control_lipschitz']={k:control_lipschitz_bound(v,model.control_spacings_mm[i]).cpu().tolist()
                                      for i,(k,v) in enumerate(grids.items())}
        rows.append(row)
    ms=[r['metrics'][c] for r in rows for c in CHAMBERS]
    summary=dict(reference_vertex_rmse_mm=float(np.sqrt(np.mean([m['reference_vertex_rmse_mm']**2 for m in ms]))),
                 reference_vertex_mae_mm=float(np.mean([m['reference_vertex_mae_mm'] for m in ms])))
    if full:
        for key in ('reference_vertex_p95_mm','surface_assd_mm','surface_hd95_mm','normal_rmse_mm','tangential_rmse_mm','volume_mae_ml'):
            summary[key]=float(np.mean([m[key] for m in ms]))
        summary['mean_absolute_volume_relative_error_percent']=float(np.mean([abs(m['volume_relative_error_percent']) for m in ms]))
        cells=[dict(cell,case_id=r['case_id']) for r in rows for cell in r['exact']['cells']]
        summary.update(cells=len(cells),cells_dice_ge_098=sum(c['exact_dice']>=.98 for c in cells),
                       minimum_dice=min(c['exact_dice'] for c in cells),mean_dice=float(np.mean([c['exact_dice'] for c in cells])),
                       worst_cell=min(cells,key=lambda c:c['exact_dice']))
    return dict(cases=rows,summary=summary)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/staged_heartffd.yaml')
    ap.add_argument('--schedule',choices=('constant','cosine'),required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:2')
    ap.add_argument('--resume',type=Path);ap.add_argument('--evaluate',type=Path)
    args=ap.parse_args()
    protocol=yaml.safe_load(args.config.read_text());config,data=load_base(ROOT/protocol['base_config'])
    config=copy.deepcopy(config);config['stage_s']=protocol
    torch.set_num_threads(4);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    train=dataset_from_config(data);val=dataset_from_config(data,'val')
    samples=[train[i] for i in protocol['train_indices']];validation=[val[i] for i in protocol['val_indices']]
    identity=data_identity(data);outdir=args.output
    runtime=dict(stage_contract='goal04s-reference-only-v1',schedule=args.schedule,code_identity=code_identity(),
                 selection=protocol['selection'],train_sources=[train.sources[i] for i in protocol['train_indices']],
                 val_sources=[val.sources[i] for i in protocol['val_indices']],
                 torch_version=str(torch.__version__),device=args.device)
    outdir.mkdir(parents=True,exist_ok=True)
    if args.evaluate:
        model,saved=load_model(args.evaluate,args.device,identity,config);validate_template(model)
        for split,subset in (('train',samples),('val',validation)):
            report=evaluate(model,subset,args.device,True,protocol['surface_samples'])
            report.update(checkpoint_sha256=_sha256(args.evaluate),step=saved['step'],runtime=saved['runtime'])
            write_json(outdir/(split+'_evaluation.json'),report)
            print(split,report['summary'],flush=True)
        return
    if not args.resume and (outdir/'protocol.json').exists():raise ValueError('Fresh run directory required')
    if args.resume:
        model,saved=load_model(args.resume,args.device,identity,config)
        if saved['runtime']!=runtime:raise ValueError('Runtime identity differs')
        step=saved['step']
    else:
        model=StagedHeartFFDNet.from_dataset_geometry(train.shared_geometry,norm_to_mm=train.norm_to_mm,
            asset_sha256=train.asset_sha256,encoder_config=config['encoder'],lifting_config=config['lifting'],
            decoder_config=config['decoder'],pose_mode=config['pose_mode']).to(args.device);step=0
    if len(model.template_vertices_mm)!=10631:raise ValueError('Production reference must have 10631 vertices')
    validate_template(model)
    criterion=NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    if args.resume:
        optimizer.load_state_dict(saved['optimizer_state']);restore_rng({'rng':saved['rng_state']},args.device)
    else:
        write_json(outdir/'protocol.json',dict(config=config,data_identity=identity,runtime=runtime,geometry=geometry_identity(model)))
        for name in runtime['code_identity']:
            dest=outdir/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    best=json.loads((outdir/'best.json').read_text()) if args.resume and (outdir/'best.json').exists() else dict(score=float('inf'))
    def save(name):
        save_g1(outdir/name,model,optimizer,config=config,identity=identity,step=step,training_stage='shape',runtime=runtime)
    if not args.resume:
        save('initial.pth')
        write_json(outdir/'initial_metrics.json',dict(train=evaluate(model,samples,args.device),val=evaluate(model,validation,args.device)))
    with (outdir/'history.jsonl').open('a') as stream:
        while step<protocol['updates']:
            epoch,offset=divmod(step,len(samples))
            order=torch.randperm(len(samples),generator=torch.Generator().manual_seed(protocol['seed']+epoch)).tolist()
            sample=samples[order[offset]];batch=to_device(training_collate([sample]),args.device)
            fraction=step/(protocol['updates']-1)
            lr=protocol['learning_rate']
            if args.schedule=='cosine':lr=protocol['minimum_learning_rate']+(lr-protocol['minimum_learning_rate'])*(1+math.cos(math.pi*fraction))/2
            for group in optimizer.param_groups:group['lr']=lr
            model.train();optimizer.zero_grad(set_to_none=True);start=time.perf_counter()
            output=model.reference_output(batch)
            losses=reference_loss(criterion,output,batch['vertices_gt'],model.control_spacings_mm)
            losses['total'].backward();grads=finite_gradient_report(model);optimizer.step();step+=1
            record=dict(step=step,case_id=sample['case_id'],lr=lr,loss=float(losses['total'].detach()),
                        terms={k:float(v.detach()) for k,v in losses['terms'].items() if losses['active_weights'][k]},
                        gradients=grads,seconds=time.perf_counter()-start)
            if step%protocol['evaluate_every']==0 or step==protocol['updates']:
                record['train']=evaluate(model,samples,args.device)['summary']
                record['val']=evaluate(model,validation,args.device)['summary']
                score=record['val']['reference_vertex_rmse_mm']
                if score<best['score']:
                    best=dict(step=step,score=score,criterion=protocol['selection']);save('best.pth');write_json(outdir/'best.json',best)
                print(json.dumps(record),flush=True)
            elif step==1 or step%16==0:print(f"step {step}/{protocol['updates']} loss={record['loss']:.6f} lr={lr:.8f} seconds={record['seconds']:.2f}",flush=True)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            if step%protocol['checkpoint_every']==0 or step==protocol['updates']:save('latest.pth')
        save('endpoint.pth')
    write_json(outdir/'completed.json',dict(step=step,best=best,endpoint_sha256=_sha256(outdir/'endpoint.pth')))


if __name__=='__main__':main()
