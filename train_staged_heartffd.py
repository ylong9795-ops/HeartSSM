"""Stage S only: configured TRAIN/VAL subsets, lazy case loading, no Motion mode."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import default_collate, DataLoader
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
from utils.staged_data import (CachedStageSamples, EpochOrder, make_loader,
                               configure_runtime_temp, workspace_path)
ROOT = Path(__file__).resolve().parent
STOP_REQUESTED=False

def request_stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


FULL_TRAINER_SHA256 = 'f68a7e79e15aa3dd76ccb97f55c52a0e51f1a42c883b5342ba202f1ca4cbb776'

CACHED_TRAINER_SHA256 = 'b4b9747cc7da88c4f3379e3837253d36ed405f2b48bfbf4b933ba919e57702a6'

LEGACY_LOADER_TRAINER_SHA256 = '14cab496bdfe9faba7f9aa128b8a599e6f64c9df28ced2214b8e0fa80c817d12'


def check_resume_code(saved, upgrade=False):
    previous=saved['runtime']['code_identity'];current=code_identity()
    if previous==current:return
    allowed={'train_staged_heartffd.py','tests/test_staged_heartffdnet.py'}
    added=set(current)-set(previous)
    compatible=(not set(previous)-set(current)
                and ((previous.get('train_staged_heartffd.py')==LEGACY_LOADER_TRAINER_SHA256 and added=={'utils/staged_data.py'})
                     or (previous.get('train_staged_heartffd.py') in (CACHED_TRAINER_SHA256,FULL_TRAINER_SHA256) and not added))
                and all(current[k]==v for k,v in previous.items() if k not in allowed))
    if not upgrade or not compatible:
        raise ValueError('Source identity differs; only pinned compatible trainers can use --upgrade-resume')



def code_identity():
    files = sorted(p for directory in ('net','utils','data') for p in (ROOT/directory).glob('*.py'))
    files += sorted(ROOT.glob('train*.py'))
    files += [ROOT/'configs/staged_heartffd.yaml',ROOT/'configs/geometry_heartffd_1k.yaml',
              ROOT/'configs/synthetic_data_1k.yaml',ROOT/'tests/test_staged_heartffdnet.py',
              ROOT/'configs/staged_heartffd_train800.yaml',ROOT/'configs/staged_heartffd_refine.yaml']
    return {str(p.relative_to(ROOT)):_sha256(p) for p in files}


def phase0(batch):
    """Slice evaluation observations only; model still consumes all 24 phases."""
    def take(value):
        if isinstance(value,dict):return {k:take(v) for k,v in value.items()}
        if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==24:return value[:,:1]
        return value
    return {k:take(v) for k,v in batch.items()}


def configure_execution(model,mode='saved',saved=None):
    if mode=='saved':
        enabled=(saved or {}).get('runtime',{}).get('execution',{}).get(
            'cnn_checkpoint',model.encoder_config.get('checkpoint_cnn',True))
    elif mode in ('on','off'):enabled=mode=='on'
    else:raise ValueError('Unknown CNN checkpoint execution mode')
    model.encoder.checkpoint_cnn=bool(enabled)
    return dict(cnn_checkpoint=bool(enabled),cnn_chunk_size=model.encoder.cnn_chunk_size)


def load_model(path, device, identity=None, config=None, upgrade=False):
    model,saved = load_g1(path,device,identity=identity,config=config)
    if saved['runtime'].get('stage_contract')!='goal04s-reference-only-v1':
        raise ValueError('Stage S checkpoint required')
    check_resume_code(saved,upgrade)
    model.__class__ = StagedHeartFFDNet
    configure_execution(model,saved=saved)
    return model,saved


@torch.no_grad()
def evaluate(model,samples,device,full=False,surface_samples=8192,deadline=None,prediction_dir=None,
             native_exact=False,reference_cache=None):
    """Optional run-scoped case cache stores only immutable GT renderer references.

    exact_metrics revalidates content/pose/topology/renderer signatures on every
    use. Predictions and their metrics are always recomputed. S2 shares this
    cache across its fixed TRAIN32/VAL100 cohorts and final candidate evaluation.
    """
    model.eval();rows=[];started=time.perf_counter()
    batches=samples if isinstance(samples,DataLoader) else (default_collate([sample]) for sample in samples)
    for index,cpu_batch in enumerate(batches):
        if STOP_REQUESTED or (deadline is not None and time.monotonic()>=deadline):break
        batch=to_device(cpu_batch,device,non_blocking=True)
        out=model.reference_output(batch);truth=batch['vertices_gt'][:,:1]
        metrics=reconstruction_metrics(out['vertices_4d'],truth,model.chamber_ranges,model.faces)
        for c,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist()):
            error=(out['vertices_ref'][:,a:b]-truth[:,0,a:b]).norm(dim=-1)
            metrics[c]['reference_vertex_p95_mm']=float(torch.quantile(error,.95))
            vp=metrics[c]['predicted_volume_curve_ml'][0][0];vg=metrics[c]['gt_volume_curve_ml'][0][0]
            metrics[c]['volume_relative_error_percent']=100*(vp-vg)/vg
        safety=validate_prediction(model,out)
        case_reference = None
        if reference_cache is not None and (full or native_exact):
            case_reference = reference_cache.setdefault(cpu_batch['case_id'][0], {})
        row=dict(case_id=cpu_batch['case_id'][0],metrics=metrics,geometry=safety,
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
            row['exact']=exact_metrics(model,out['vertices_4d'],phase0(batch),reference_cache=case_reference)
            grids=out['control_grids']
            row['geometry']=validate_prediction(model,out,sample_jacobian=True,reference_only=True)
            row['geometry']['actual_mesh_self_intersecting_pairs']=sum(metrics[c]['self_intersections'] for c in CHAMBERS)
            row['geometry']['inter_chamber']=SourceAVCapPolicy(model,SETTINGS).audit_frames(out['vertices_4d'][0].cpu().numpy())
            row['control_lipschitz']={k:control_lipschitz_bound(v,model.control_spacings_mm[i]).cpu().tolist()
                                      for i,(k,v) in enumerate(grids.items())}
        if prediction_dir is not None:
            import hashlib
            prediction_dir=workspace_path(prediction_dir);prediction_dir.mkdir(parents=True,exist_ok=True)
            artifact=prediction_dir/(hashlib.sha256(row['case_id'].encode()).hexdigest()+'.npz')
            np.savez_compressed(artifact,vertices_ref_mm=out['vertices_ref'][0].cpu().numpy(),
                vertices_gt_frame0_mm=truth[0,0].cpu().numpy(),case_id=np.asarray(row['case_id']),
                chamber_ranges=model.chamber_ranges.cpu().numpy(),
                **{'faces_'+c:model.faces[c].cpu().numpy() for c in CHAMBERS},
                **{'controls_'+k:v[0].cpu().numpy() for k,v in out['control_grids'].items()})
            row['reference_artifact']=dict(path=str(artifact),sha256=_sha256(artifact))
        if native_exact and not full:row['exact']=exact_metrics(model,out['vertices_4d'],phase0(batch),reference_cache=case_reference)
        rows.append(row)
        if index==0 or (index+1)%50==0 or index+1==len(samples):
            print(f'evaluation {index+1}/{len(samples)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    if not rows:return dict(cases=[],summary={},complete=False)
    ms=[r['metrics'][c] for r in rows for c in CHAMBERS]
    summary=dict(reference_vertex_rmse_mm=float(np.sqrt(np.mean([m['reference_vertex_rmse_mm']**2 for m in ms]))),
                 reference_vertex_mae_mm=float(np.mean([m['reference_vertex_mae_mm'] for m in ms])))
    summary['reference_vertex_p95_mm']=float(np.mean([m['reference_vertex_p95_mm'] for m in ms]))
    summary['volume_mae_ml']=float(np.mean([m['volume_mae_ml'] for m in ms]))
    summary['per_chamber']={c:dict(reference_vertex_rmse_mm=float(np.sqrt(np.mean([row['metrics'][c]['reference_vertex_rmse_mm']**2 for row in rows]))),
        reference_vertex_p95_mm=float(np.mean([row['metrics'][c]['reference_vertex_p95_mm'] for row in rows])),
        volume_mae_ml=float(np.mean([row['metrics'][c]['volume_mae_ml'] for row in rows]))) for c in CHAMBERS}
    if full:
        for key in ('reference_vertex_p95_mm','surface_assd_mm','surface_hd95_mm','normal_rmse_mm','tangential_rmse_mm','volume_mae_ml'):
            summary[key]=float(np.mean([m[key] for m in ms]))
        summary['mean_absolute_volume_relative_error_percent']=float(np.mean([abs(m['volume_relative_error_percent']) for m in ms]))
    if full or native_exact:
        cells=[dict(cell,case_id=r['case_id']) for r in rows for cell in r['exact']['cells']]
        summary.update(cells=len(cells),cells_dice_ge_098=sum(c['exact_dice']>=.98 for c in cells),
                       minimum_dice=min(c['exact_dice'] for c in cells),mean_dice=float(np.mean([c['exact_dice'] for c in cells])),
                       worst_cell=min(cells,key=lambda c:c['exact_dice']),
                       p01_dice=float(np.quantile([c['exact_dice'] for c in cells],.01)),
                       p05_dice=float(np.quantile([c['exact_dice'] for c in cells],.05)),
                       all_cells_ge098_cases=sum(all(c['exact_dice']>=.98 for c in row['exact']['cells']) for row in rows))
    return dict(cases=rows,summary=summary,complete=len(rows)==len(samples))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/staged_heartffd.yaml')
    ap.add_argument('--schedule',choices=('constant','cosine'),required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:2')
    ap.add_argument('--resume',type=Path);ap.add_argument('--evaluate',type=Path)
    ap.add_argument('--cache-dir',type=Path,default=ROOT/'outputs/goal04s/preprocessed_cache')
    ap.add_argument('--refine-arm',choices=('A','B','C'))
    ap.add_argument('--refine-config',type=Path,default=ROOT/'configs/staged_heartffd_refine.yaml')
    ap.add_argument('--calibrate-refinement',action='store_true')
    ap.add_argument('--refinement-calibration',type=Path)
    ap.add_argument('--evaluate-refinement',type=Path)
    ap.add_argument('--cnn-checkpoint',choices=('saved','on','off'),default='saved',
                    help='Activation recomputation only; off trades memory for speed, saved preserves checkpoint execution settings')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--prefetch',type=int,default=2)
    ap.add_argument('--prepare-cache',action='store_true',help='Precompute TRAIN/VAL only, without GPU training')
    ap.add_argument('--upgrade-resume',action='store_true',help='Migrate pinned old loader checkpoint into a NEW output directory')
    ap.add_argument('--max-wall-seconds',type=float,help='Save and stop on elapsed wall-time budget; does not claim full update completion')
    ap.add_argument('--train-evaluate-every-epochs',type=int,default=1,
                    help='Complete TRAIN evaluation interval; VAL always follows configured epoch cadence')
    ap.add_argument('--train-probe-cases',type=int,default=32,help='Fixed TRAIN subset between full TRAIN evaluations')
    ap.add_argument('--evaluation-case-limit',type=int,help='Explicit prefix for diagnostic evaluation only')
    ap.add_argument('--evaluation-splits',choices=('both','train','val'),default='both')
    args=ap.parse_args()
    if args.train_evaluate_every_epochs<1 or args.train_probe_cases<1:ap.error('Positive TRAIN evaluation interval/probe required')
    if args.evaluation_case_limit is not None and (not args.evaluate or args.evaluation_case_limit<1):ap.error('Case limit requires evaluation and positive count')
    if args.max_wall_seconds is not None and args.max_wall_seconds<=0:ap.error('Wall-time budget must be positive')
    deadline=None if args.max_wall_seconds is None else time.monotonic()+args.max_wall_seconds
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    args.output=workspace_path(args.output);args.cache_dir=workspace_path(args.cache_dir)
    configure_runtime_temp(args.output/'runtime_tmp')
    if args.upgrade_resume and not args.resume:ap.error('--upgrade-resume requires --resume')
    if args.prepare_cache and (args.resume or args.evaluate):ap.error('Cache preparation is independent of resume/evaluation')
    if args.resume and args.evaluate:ap.error('Resume and evaluation are mutually exclusive')
    if args.refine_arm or args.calibrate_refinement or args.evaluate_refinement:
        if args.evaluate or args.upgrade_resume:ap.error('Use the dedicated S2 initialization/resume contract')
        from utils.reference_refinement import run_refinement
        run_refinement(args);return
    protocol=yaml.safe_load(args.config.read_text());config,data=load_base(ROOT/protocol['base_config'])
    config=copy.deepcopy(config);config['stage_s']=protocol
    torch.set_num_threads(4);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    train=dataset_from_config(data);val=dataset_from_config(data,'val')
    samples=CachedStageSamples(train,protocol['train_indices'],args.cache_dir)
    validation=CachedStageSamples(val,protocol['val_indices'],args.cache_dir)
    loader_options=dict(workers=args.workers,prefetch=args.prefetch,seed=protocol['seed'],
                        pin_memory=torch.device(args.device).type=='cuda')
    def evaluation_loader(dataset):
        return make_loader(dataset,collate_fn=default_collate if args.evaluate else training_collate,**loader_options)
    train_eval=evaluation_loader(samples);val_eval=evaluation_loader(validation)
    train_probe=evaluation_loader(CachedStageSamples(train,protocol['train_indices'][:args.train_probe_cases],args.cache_dir))
    if args.prepare_cache:
        started=time.perf_counter()
        for label,loader in (('TRAIN',train_eval),('VAL',val_eval)):
            for i,_ in enumerate(loader):
                if i==0 or (i+1)%50==0 or i+1==len(loader):
                    print(f'cache {label} {i+1}/{len(loader)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        write_json(args.output/'cache_prepared.json',dict(train=samples.identity,val=validation.identity,
                   train_cases=len(samples),val_cases=len(validation),seconds=time.perf_counter()-started))
        return
    if protocol['batch_size'] != 1:
        raise ValueError('Stage S update/sample ordering currently requires batch_size=1')
    if 'epochs' in protocol and protocol['updates'] != protocol['epochs']*len(samples):
        raise ValueError('Epoch/update budget differs from selected TRAIN size')
    identity=data_identity(data);outdir=args.output
    runtime=dict(stage_contract='goal04s-reference-only-v1',schedule=args.schedule,code_identity=code_identity(),
                 selection=protocol['selection'],train_sources=[train.sources[i] for i in protocol['train_indices']],
                 val_sources=[val.sources[i] for i in protocol['val_indices']],
                 torch_version=str(torch.__version__),device=args.device,
                 input_pipeline=dict(cache_dir=str(args.cache_dir),train_cache=samples.identity,
                     val_cache=validation.identity,workers=args.workers,prefetch=args.prefetch),
                 evaluation_policy=dict(train_full_every_epochs=args.train_evaluate_every_epochs,
                                        train_probe_cases=args.train_probe_cases,val='full VAL every configured interval'))
    outdir.mkdir(parents=True,exist_ok=True)
    if args.evaluate:
        model,saved=load_model(args.evaluate,args.device,identity,config);validate_template(model)
        for split,subset in (('train',train_eval),('val',val_eval)):
            if args.evaluation_splits not in ('both',split):continue
            if args.evaluation_case_limit is not None:
                dataset=samples if split=='train' else validation
                count=min(args.evaluation_case_limit,len(dataset))
                subset=evaluation_loader(CachedStageSamples(dataset.dataset,dataset.indices[:count],args.cache_dir))
            report=evaluate(model,subset,args.device,True,protocol['surface_samples'],deadline=deadline,prediction_dir=outdir/(split+'_predictions'))
            report.update(checkpoint_sha256=_sha256(args.evaluate),step=saved['step'],runtime=saved['runtime'])
            write_json(outdir/(split+'_evaluation.json'),report)
            print(split,report['summary'],flush=True)
        return
    if not args.resume and (outdir/'protocol.json').exists():raise ValueError('Fresh run directory required')
    if args.resume:
        model,saved=load_model(args.resume,args.device,identity,config,upgrade=args.upgrade_resume)
        runtime['execution']=configure_execution(model,args.cnn_checkpoint,saved)
        if args.upgrade_resume:
            if (outdir/'protocol.json').exists() or outdir.resolve()==args.resume.resolve().parent:
                raise ValueError('Loader migration requires a fresh, separate output directory')
            unchanged=('stage_contract','schedule','selection','train_sources','val_sources','torch_version')
            if any(saved['runtime'][key]!=runtime[key] for key in unchanged):
                raise ValueError('Loader migration must preserve training/data/device semantics')
            runtime['loader_upgrade']=dict(checkpoint_sha256=_sha256(args.resume),step=saved['step'],
                                          previous_code_identity=saved['runtime']['code_identity'],previous_device=saved['runtime']['device'])
        else:
            if 'loader_upgrade' in saved['runtime']:runtime['loader_upgrade']=saved['runtime']['loader_upgrade']
            if saved['runtime']!=runtime:raise ValueError('Runtime identity differs')
        step=saved['step']
    else:
        model=StagedHeartFFDNet.from_dataset_geometry(train.shared_geometry,norm_to_mm=train.norm_to_mm,
            asset_sha256=train.asset_sha256,encoder_config=config['encoder'],lifting_config=config['lifting'],
            decoder_config=config['decoder'],pose_mode=config['pose_mode']).to(args.device);step=0
        runtime['execution']=configure_execution(model,args.cnn_checkpoint)
    if len(model.template_vertices_mm)!=10631:raise ValueError('Production reference must have 10631 vertices')
    validate_template(model)
    criterion=NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    if args.resume:
        if saved['optimizer_state'] is not None:optimizer.load_state_dict(saved['optimizer_state'])
        elif step!=0:raise ValueError('Noninitial checkpoint is missing optimizer state')
        restore_rng({'rng':saved['rng_state']},args.device)
    if not args.resume or args.upgrade_resume:
        write_json(outdir/'protocol.json',dict(config=config,data_identity=identity,runtime=runtime,geometry=geometry_identity(model)))
        for name in runtime['code_identity']:
            dest=outdir/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    best=json.loads((outdir/'best.json').read_text()) if args.resume and (outdir/'best.json').exists() else dict(score=float('inf'))
    if args.upgrade_resume:
        old_best=args.resume.parent/'best.json'
        if old_best.exists():
            candidate=json.loads(old_best.read_text())
            if candidate['step']<=step:
                _,best_saved=load_model(args.resume.parent/'best.pth','cpu',identity,config,upgrade=True)
                if best_saved['step']!=candidate['step']:raise ValueError('Best checkpoint metadata differs')
                best_saved['runtime']=runtime
                torch.save(best_saved,outdir/'best.pth');best=candidate;write_json(outdir/'best.json',best)
            else:raise ValueError('Best selection is newer than resume checkpoint')
        shutil.copy2(args.resume,outdir/'pre_upgrade_checkpoint.pth')
        restore_rng({'rng':saved['rng_state']},args.device)
    def save(name):
        save_g1(outdir/name,model,optimizer,config=config,identity=identity,step=step,training_stage='shape',runtime=runtime)
    if not args.resume:
        save('initial.pth')
        write_json(outdir/'initial_metrics.json',dict(train=evaluate(model,train_eval,args.device,deadline=deadline),val=evaluate(model,val_eval,args.device,deadline=deadline)))
    if args.upgrade_resume:save('latest.pth')
    sampler=EpochOrder(len(samples),protocol['seed'])
    train_loader=make_loader(samples,sampler=sampler,collate_fn=training_collate,**loader_options)
    iterator=None
    with (outdir/'history.jsonl').open('a') as stream:
        while step<protocol['updates']:
            if STOP_REQUESTED or (deadline is not None and time.monotonic()>=deadline):break
            wall_start=time.perf_counter()
            epoch,offset=divmod(step,len(samples))
            if iterator is None or offset==0:
                sampler.set_position(epoch,offset);iterator=iter(train_loader)
            wait_start=time.perf_counter();cpu_batch=next(iterator)
            data_wait=time.perf_counter()-wait_start
            transfer_start=time.perf_counter();batch=to_device(cpu_batch,args.device,non_blocking=True)
            if model.template_vertices_mm.is_cuda:torch.cuda.synchronize(args.device)
            transfer_seconds=time.perf_counter()-transfer_start
            fraction=step/(protocol['updates']-1)
            lr=protocol['learning_rate']
            if args.schedule=='cosine':lr=protocol['minimum_learning_rate']+(lr-protocol['minimum_learning_rate'])*(1+math.cos(math.pi*fraction))/2
            for group in optimizer.param_groups:group['lr']=lr
            model.train();optimizer.zero_grad(set_to_none=True);start=time.perf_counter()
            output=model.reference_output(batch)
            losses=reference_loss(criterion,output,batch['vertices_gt'],model.control_spacings_mm)
            losses['total'].backward();grads=finite_gradient_report(model);optimizer.step();step+=1
            record=dict(step=step,case_id=cpu_batch['case_id'][0],lr=lr,loss=float(losses['total'].detach()),
                        terms={k:float(v.detach()) for k,v in losses['terms'].items() if losses['active_weights'][k]},
                        gradients=grads,seconds=time.perf_counter()-start,
                        data_wait_seconds=data_wait,transfer_seconds=transfer_seconds,
                        update_wall_seconds=time.perf_counter()-wall_start)
            if step%protocol['evaluate_every']==0 or step==protocol['updates']:
                evaluation_start=time.perf_counter()
                epoch_completed=step//len(samples)
                full_train=(epoch_completed%args.train_evaluate_every_epochs==0 or step==protocol['updates'])
                train_report=evaluate(model,train_eval if full_train else train_probe,args.device,deadline=deadline)
                record['train_scope']='TRAIN800' if full_train else f'TRAIN_prefix{min(args.train_probe_cases,len(samples))}'
                val_report=evaluate(model,val_eval,args.device,deadline=deadline)
                record['train']=train_report['summary'];record['val']=val_report['summary']
                record['evaluation_complete']=train_report['complete'] and val_report['complete']
                record['evaluation_seconds']=time.perf_counter()-evaluation_start
                evaluation_dir=outdir/'evaluations';evaluation_dir.mkdir(exist_ok=True)
                write_json(evaluation_dir/f'step{step:05d}.json',dict(step=step,train_scope=record['train_scope'],train=train_report,val=val_report))
                score=record['val'].get('reference_vertex_rmse_mm',float('inf'))
                if val_report['complete'] and score<best['score']:
                    best=dict(step=step,score=score,criterion=protocol['selection']);save('best.pth');write_json(outdir/'best.json',best)
                print(json.dumps(record),flush=True)
            elif step==1 or step%16==0:print(f"step {step}/{protocol['updates']} loss={record['loss']:.6f} lr={lr:.8f} seconds={record['seconds']:.2f}",flush=True)
            record['iteration_wall_seconds']=time.perf_counter()-wall_start
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            if step%protocol['checkpoint_every']==0 or step==protocol['updates']:save('latest.pth')
        save('latest.pth')
        final_name='endpoint.pth' if step==protocol['updates'] else 'time_limit.pth'
        save(final_name)
    status='completed' if step==protocol['updates'] else 'stopped_early'
    write_json(outdir/(status+'.json'),dict(step=step,target_updates=protocol['updates'],
        full_budget_completed=step==protocol['updates'],best=None if not math.isfinite(best['score']) else best,
        reason='full_update_budget' if status=='completed' else 'signal' if STOP_REQUESTED else 'wall_time_budget',
        checkpoint=final_name,checkpoint_sha256=_sha256(outdir/final_name)))
    print(f'{status}: step={step}/{protocol["updates"]} checkpoint={final_name}',flush=True)


if __name__=='__main__':
    import sys
    sys.modules.setdefault('train_staged_heartffd',sys.modules[__name__])
    main()
