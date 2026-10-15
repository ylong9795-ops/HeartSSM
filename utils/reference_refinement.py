"""Stage S2 helpers: cached native frame0 targets, original3D loss + one slice loss.

No encoder, pose transform, intersection, spline or renderer is implemented here.
The A/B/C experiment deliberately disables any additional prior/hard-cell loss.
"""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
import yaml
from data.mask_transform import mask_to_sdf, CHAMBERS
from data.synthetic_4d import _sha256
from utils.staged_data import CachedStageSamples, EpochOrder, make_loader, workspace_path, digest
from net.staged_heartffdnet import StagedHeartFFDNet, reference_loss
from utils.slice_supervision import MeshSliceSupervisor
ROOT=Path(__file__).resolve().parents[1]
S1_TRAINER_SHA256='d16d7dce024d5631c968a0c44e2772a8c2741781fe0d1a1a701ceeaf033772ca'


class NativeReferenceSamples(Dataset):
    """Append lossless physical native_sdf [1,3,4,Hn,Wn] to an existing sample.

    Existing24-phase model-grid caches remain byte-for-byte untouched. Native
    SDF uses phase0 only, same mask_to_sdf and native spacing, clip15mm, no warp.
    """
    def __init__(self,samples,cache_dir,clip_mm=15.):
        self.samples=samples;self.root=workspace_path(cache_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.clip_mm=float(clip_mm)
        if not math.isfinite(self.clip_mm) or self.clip_mm<=0:raise ValueError('Positive native SDF clip required')
        self.verified={}
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        sample=self.samples[index]
        native=sample['native_masks'][:1].cpu().numpy()
        spacing=sample['native_pixel_spacing'][:1].cpu().numpy()
        identity=dict(version='native-reference-sdf-v1',source=sample['sample_sha256'],qc=sample['qc_sha256'],
            template=sample['template_sha256'],norm_to_mm=float(sample['norm_to_mm']),clip_mm=self.clip_mm,
            spacing_sha256=hashlib.sha256(spacing.tobytes()).hexdigest(),
            masks_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
            implementation=_sha256(ROOT/'data/mask_transform.py'))
        path=self.root/(digest(identity)+'.pth');side=path.with_suffix('.json')
        with path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists() or not side.exists():
                def phase(value,static_ndim):
                    return value[:1].cpu().numpy() if value.ndim>static_ndim else value.cpu().numpy()
                sdf=mask_to_sdf(native,spacing,view_present=phase(sample['view_present'],1),
                    annotated=phase(sample['annotated'],2),clip_mm=self.clip_mm,normalize=False)
                value=torch.from_numpy(sdf)
                temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
                metadata=side.with_name(side.name+'.'+uuid.uuid4().hex+'.partial')
                try:
                    torch.save(value,temporary)
                    metadata.write_text(json.dumps(dict(identity=identity,sha256=_sha256(temporary))))
                    os.replace(temporary,path);os.replace(metadata,side)
                finally:
                    temporary.unlink(missing_ok=True);metadata.unlink(missing_ok=True)
            else:
                stamp=(path.stat().st_mtime_ns,path.stat().st_size,side.stat().st_mtime_ns)
                if self.verified.get(str(path))!=stamp:
                    meta=json.loads(side.read_text())
                    if meta['identity']!=identity or meta['sha256']!=_sha256(path):raise ValueError('Corrupt native SDF cache')
                    self.verified[str(path)]=stamp
                value=torch.load(path,map_location='cpu',weights_only=True)
        if value.shape!=native.shape or value.dtype!=torch.float32:raise ValueError('Native SDF cache shape/dtype differs')
        return dict(sample,native_sdf=value,native_sdf_units='mm')


def refinement_collate(samples):
    # Encoder still sees24 phases and all acquisition metadata. Native targets
    # alone retain T=1; pose/pixel metadata must stay24-phase for lifting.
    result=[]
    for sample in samples:
        item={k:v for k,v in sample.items() if k not in ('clipped_foreground','center_mm')}
        item['native_masks']=sample['native_masks'][:1]
        result.append(item)
    return default_collate(result)


def reference_drift_prior(output,anchor,unknown_mask):
    """Optional future prior; [B,N] unknown mask, detached frozen S1 anchor.

    Returned mm² terms are NOT enabled in A/B/C; never silently add a prior.
    """
    p=output['vertices_ref'];q=anchor['vertices_ref'].detach().to(p)
    if p.shape!=q.shape or unknown_mask.shape!=p.shape[:-1] or unknown_mask.dtype!=torch.bool:
        raise ValueError('Matching reference meshes and bool[B,N] Unknown mask required')
    error=(p-q).square().sum(-1)
    unknown=error[unknown_mask].mean() if unknown_mask.any() else error.sum()*0
    grids=output['control_grids'];old=anchor['control_grids']
    control=torch.stack([(v-old[k].detach().to(v)).square().mean() for k,v in grids.items()]).mean()
    return dict(unknown_vertex_mse_mm2=unknown,control_delta_mse_mm2=control)


def make_refinement_slicer(model,protocol,arm):
    if arm=='A':return None
    if arm not in ('B','C'):raise ValueError('Refinement arm must be A/B/C')
    return MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,
        **protocol['slice'],supervision_grid='native_grid',
        reduction='cell_mean' if arm=='B' else 'chamber_mean')


def refinement_objective(model,batch,criterion,slicer,*,shape_weight=1.,slice_weight=0.):
    from train_staged_heartffd import phase0
    if shape_weight<=0 or slice_weight<0:raise ValueError('Retain positive3D supervision')
    output=model.reference_output(batch)
    base=reference_loss(criterion,output,batch['vertices_gt'],model.control_spacings_mm)
    sliced=None if slicer is None else slicer(output['vertices_4d'],phase0(batch))
    if slicer is None and slice_weight:raise ValueError('Nonzero slice weight without a slicer')
    total=shape_weight*base['total']+(0 if sliced is None else slice_weight*sliced['total'])
    if not torch.isfinite(total):raise ValueError('Nonfinite S2 objective')
    return output,dict(total=total,shape=base,slice=sliced)


def reference_gradient_report(model):
    """Same six group norms and fail-fast checks, with one host transfer.

    Per-parameter finite checks stay on-device. Only their grouped flags and
    norms cross to the CPU, avoiding a synchronization for each parameter.
    """
    groups = {'cnn':model.encoder,'lifting':model.lifting,'unet':model.unet,
              'joint_fine':model.heads.deform3,'coarse':model.heads.deform1,'medium':model.heads.deform2}
    values=[]
    for key,module in groups.items():
        grads=[p.grad.detach() for p in module.parameters() if p.grad is not None]
        if not grads:raise ValueError('Missing/nonfinite real parameter gradients: '+key)
        finite=torch.stack([torch.isfinite(g).all() for g in grads]).all()
        norm=sum(g.square().sum() for g in grads).sqrt()
        values.append(torch.stack((finite.to(norm.dtype),norm)))
    result={}
    for key,(finite,norm) in zip(groups,torch.stack(values).cpu().tolist()):
        if not finite or not math.isfinite(norm):
            raise ValueError('Missing/nonfinite real parameter gradients: '+key)
        if norm==0:raise ValueError('Zero real parameter gradients: '+key)
        result[key]=norm
    return result


def report_metrics(report):
    """Hard source-rendered Dice tails and paired3D/Unknown guard statistics."""
    if not report.get('complete') or not report.get('cases'):raise ValueError('Complete full evaluation required')
    cases=report['cases'];cells=[c for case in cases for c in case['exact']['cells']]
    scores=np.asarray([c['exact_dice'] for c in cells]);metrics=dict(report['summary'])
    regions=[case['metrics'][c]['regions']['unknown'] for case in cases for c in CHAMBERS]
    count=sum(x['count'] for x in regions)
    metrics.update(minimum_dice=float(scores.min()),p01_dice=float(np.quantile(scores,.01)),
        p05_dice=float(np.quantile(scores,.05)),mean_dice=float(scores.mean()),
        cells_dice_ge098=int((scores>=.98).sum()),cells_below095=int((scores<.95).sum()),
        all_cells_ge098_cases=sum(all(c['exact_dice']>=.98 for c in case['exact']['cells']) for case in cases),
        unknown_rmse_mm=math.sqrt(sum(x['rmse_mm']**2*x['count'] for x in regions if x['count'])/count) if count else 0.,
        worst_unknown_rmse_mm=max((x['rmse_mm'] for x in regions if x['count']),default=0.),
        worst_reference_rmse_mm=max(case['metrics'][c]['reference_vertex_rmse_mm'] for case in cases for c in CHAMBERS),
        worst_surface_hd95_mm=max(case['metrics'][c]['surface_hd95_mm'] for case in cases for c in CHAMBERS))
    safe=True
    for case in cases:
        g=case['geometry']
        safe &= g.get('status')=='pass' and g.get('actual_mesh_self_intersecting_pairs',1)==0 and g.get('degenerate_triangles',1)==0
        safe &= g.get('min_sampled_composed_jacobian',-1)>0
        cross=g.get('inter_chamber',{})
        safe &= cross.get('status')=='pass' and len(cross.get('records',[]))==6 and all(not (r['remote_face_pairs'] or r['noncap_penetrating_vertices'] or r['gross_containment']) for r in cross.get('records',[]))
        safe &= case['exact']['summary'].get('gt_native_pixel_mismatches',1)==0
    metrics['geometry_safe']=bool(safe)
    return metrics


def refinement_guard(candidate,baseline,protocol):
    """Strict frozen-before-training gates; means never compensate for tails."""
    ids=lambda r:sorted(x['case_id'] for x in r.get('cases',[]))
    if not candidate.get('complete') or ids(candidate)!=ids(baseline):
        return dict(eligible=False,hard_target_met=False,failures=['incomplete or different validation cohort'])
    def cells(report):return {(case['case_id'],c['frame'],c['view'],c['chamber']) for case in report['cases'] for c in case['exact']['cells']}
    if cells(candidate)!=cells(baseline):return dict(eligible=False,hard_target_met=False,failures=['different native-cell coverage'])
    for a,b in zip(sorted(candidate['cases'],key=lambda x:x['case_id']),sorted(baseline['cases'],key=lambda x:x['case_id'])):
        if a['exact']['renderer']!=b['exact']['renderer']:
            return dict(eligible=False,hard_target_met=False,failures=['different authoritative renderer identity'])
    now,old=report_metrics(candidate),report_metrics(baseline);failed=[]
    eps=protocol['guards']['numeric_tolerance']
    for key,tolerance in protocol['guards']['maximum_degradation'].items():
        if now[key]>old[key]+tolerance+eps:failed.append(key)
    for key in ('mean_dice','minimum_dice','p01_dice','p05_dice'):
        if now[key]+eps<old[key]:failed.append(key)
    for key in ('cells_dice_ge098','all_cells_ge098_cases'):
        if now[key]<old[key]:failed.append(key)
    if not now['geometry_safe']:failed.append('geometry_safe')
    cells=sum(len(c['exact']['cells']) for c in candidate['cases'])
    return dict(eligible=not failed,hard_target_met=not failed and now['cells_dice_ge098']==cells,
        failures=failed,baseline=old,candidate=now,
        improvement=dict(cells_ge098=now['cells_dice_ge098']-old['cells_dice_ge098'],
                         complete_cases=now['all_cells_ge098_cases']-old['all_cells_ge098_cases']))


def tail_rank(report):
    cells=[c['exact_dice'] for case in report['cases'] for c in case['exact']['cells']]
    return (min(cells),float(np.quantile(cells,.01)),float(np.quantile(cells,.05)),
        sum(v>=.98 for v in cells),sum(all(c['exact_dice']>=.98 for c in x['exact']['cells']) for x in report['cases']),float(np.mean(cells)))


def load_stage1_initialization(protocol,identity,device):
    from train_geometry_heartffdnet import load_checkpoint
    from train_staged_heartffd import configure_execution,code_identity
    path=workspace_path(protocol['stage1_checkpoint'])
    if _sha256(path)!=protocol['stage1_checkpoint_sha256']:raise ValueError('Pinned S1 checkpoint identity differs')
    model,saved=load_checkpoint(path,device,identity=identity)
    if saved['step']!=32000 or saved['runtime'].get('stage_contract')!='goal04s-reference-only-v1':
        raise ValueError('Completed S1 reference checkpoint required')
    # This is explicit new-stage initialization, never a relaxed strict resume.
    previous=saved['runtime']['code_identity'];current=code_identity()
    allowed={'train_staged_heartffd.py','tests/test_staged_heartffdnet.py','utils/slice_supervision.py',
             'utils/geometry_joint_checks.py'}
    if previous.get('train_staged_heartffd.py')!=S1_TRAINER_SHA256:
        raise ValueError('Unrecognized S1 source identity')
    if previous.get('utils/geometry_joint_checks.py') != '1dd2ed13294cb09cbfbc60293c230e925f67fa4b47f52cdb055f9935959678ec':
        raise ValueError('Unrecognized S1 geometry audit source identity')
    for name,value in previous.items():
        archived=path.parent/'source'/name
        if not archived.exists() or _sha256(archived)!=value:raise ValueError('Archived S1 source differs: '+name)
        if name not in allowed and current.get(name)!=value:raise ValueError('S1 model/loss/data/cache source differs: '+name)
    model.__class__=StagedHeartFFDNet;configure_execution(model,saved=saved)
    return model,saved


def _gradient_stats(loss,parameters):
    grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return grads,math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))


def calibrate(protocol,config,data,args):
    from train_geometry_heartffdnet import data_identity,write_json
    from train_neural_ffd import dataset_from_config,to_device
    from train_staged_heartffd import code_identity
    from utils.neural_ffd_losses import NeuralFFDLoss
    model,_=load_stage1_initialization(protocol,data_identity(data),args.device);model.eval()
    base=CachedStageSamples(dataset_from_config(data),protocol['calibration_indices'],args.cache_dir)
    samples=NativeReferenceSamples(base,args.cache_dir/'native_reference_sdf',protocol['native_sdf_clip_mm'])
    criterion=NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(args.device)
    params=[p for p in model.parameters() if p.requires_grad];rows=[];ratios=[]
    for sample in samples:
        batch=to_device(refinement_collate([sample]),args.device)
        output=model.reference_output(batch)
        three=reference_loss(criterion,output,batch['vertices_gt'],model.control_spacings_mm)['total']
        g3,n3=_gradient_stats(three,params)
        for arm in ('B','C'):
            from train_staged_heartffd import phase0
            slicer=make_refinement_slicer(model,protocol,arm).to(args.device)
            sliced=slicer(output['vertices_4d'],phase0(batch))['total']
            g2,n2=_gradient_stats(sliced,params)
            if not math.isfinite(n2) or min(n2,n3)<=0:raise ValueError('Missing/nonfinite calibration gradient')
            dot=sum(float((a*b).sum()) for a,b in zip(g3,g2) if a is not None and b is not None)
            rows.append(dict(case_id=sample['case_id'],arm=arm,shape_loss=float(three),slice_loss=float(sliced),
                shape_gradient_l2=n3,slice_gradient_l2=n2,gradient_cosine=dot/(n3*n2),norm_ratio=n3/n2))
            ratios.append(protocol['shape_weight']*n3/n2)
    frozen=copy.deepcopy(protocol)
    frozen['slice_weight']=protocol['gradient_target_ratio']*float(np.median(ratios))
    result=dict(version='goal04s-refinement-calibration-v1',draft_protocol_sha256=digest(protocol),
        frozen_protocol=frozen,common_slice_weight=frozen['slice_weight'],
        code_identity=code_identity(),data_identity=data_identity(data),rows=rows,
        calibration_device=args.device,torch_version=str(torch.__version__),
        note='TRAIN-only, no optimizer updates; same lambda2D for B/C to isolate reduction')
    write_json(args.output/'calibration.json',result)
    print(json.dumps(dict(common_slice_weight=result['common_slice_weight'],cases=len(samples),gradient_rows=len(rows))),flush=True)


def validate_protocol(protocol,base_protocol):
    if protocol.get('version')!='goal04s-refinement-v1':raise ValueError('Unknown S2 protocol')
    if (protocol['shape_weight']<=0 or protocol['prior_weight']!=0 or protocol['epochs']<1
            or protocol['updates']!=protocol['epochs']*len(base_protocol['train_indices'])
            or base_protocol['batch_size']!=1 or protocol['evaluate_every']!=len(base_protocol['train_indices'])
            or not 0<protocol['minimum_learning_rate']<=protocol['learning_rate']<1):
        raise ValueError('A/B/C requires same fixed batch1 budget, positive3D loss, no additional prior and epoch validation')
    if protocol['slice']['target_mode']!='binary_native_v1':raise ValueError('S2 must supervise original binary native masks')
    schedule=protocol['stride_schedule']
    if (not schedule or schedule[0]['start_step']!=0 or any(x['pixel_stride'] not in (1,2,4) for x in schedule)
            or [x['start_step'] for x in schedule]!=sorted(set(x['start_step'] for x in schedule))
            or schedule[-1]['start_step']>=protocol['updates']):raise ValueError('Invalid fixed pixel-stride schedule')
    if (not math.isfinite(protocol['guards']['numeric_tolerance']) or protocol['guards']['numeric_tolerance']<0
            or any(not math.isfinite(x) or x<0 for x in protocol['guards']['maximum_degradation'].values())):
        raise ValueError('Finite nonnegative explicit guard tolerances required')
    if not set(protocol['calibration_indices']).issubset(base_protocol['train_indices']):raise ValueError('Calibration must use TRAIN only')


def cached_full_report(path,state_sha256,baseline_sha256,source_identity):
    if not Path(path).exists():return None
    try:report=json.loads(Path(path).read_text())
    except (ValueError,OSError):return None
    if (report.get('complete') and report.get('model_state_sha256')==state_sha256
            and report.get('baseline_report_sha256')==baseline_sha256
            and report.get('code_identity')==source_identity):return report
    return None


def run_refinement(args):
    """Unified CLI implementation. S1 warm start != S2 strict resume.

    No optimizer update occurs in calibration or evaluation. A/B/C use the same
    S1 weights, fresh Adam, seed, sample order, updates and fixed LR schedule.
    """
    from train_geometry_heartffdnet import (load_config as load_base,data_identity,write_json,
        save_checkpoint,load_checkpoint)
    from train_neural_ffd import dataset_from_config,to_device,restore_rng
    from train_staged_heartffd import evaluate,code_identity,configure_execution
    from utils.neural_ffd_losses import NeuralFFDLoss
    from utils.geometry_joint_checks import validate_template
    deadline=None if args.max_wall_seconds is None else time.monotonic()+args.max_wall_seconds
    if args.schedule!='cosine':raise ValueError('Fixed S2 A/B/C protocol requires cosine')
    draft=yaml.safe_load(workspace_path(args.refine_config).read_text())
    base_protocol=yaml.safe_load(workspace_path(draft['base_protocol']).read_text())
    config,data=load_base(workspace_path(base_protocol['base_config']))
    validate_protocol(draft,base_protocol)
    torch.set_num_threads(4);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    random.seed(draft['seed']);np.random.seed(draft['seed']);torch.manual_seed(draft['seed'])
    args.output.mkdir(parents=True,exist_ok=True)
    if args.calibrate_refinement and (args.resume or args.evaluate_refinement or args.prepare_cache):raise ValueError('Calibration is a separate no-update action')
    if args.calibrate_refinement:
        calibrate(draft,config,data,args);return
    identity=data_identity(data)
    if args.evaluate_refinement:
        model,saved=load_checkpoint(args.evaluate_refinement,args.device,identity=identity)
        if (saved['runtime'].get('stage_contract')!='goal04s-refinement-v1'
                or saved['runtime']['code_identity']!=code_identity()):raise ValueError('Strict S2 evaluation source/contract differs')
        protocol=saved['config']['refinement'];arm=saved['runtime']['arm']
        model.__class__=StagedHeartFFDNet;configure_execution(model,saved=saved)
    else:
        if not args.refinement_calibration:raise ValueError('Freeze TRAIN-gradient calibration before starting any A/B/C arm')
        calibration_path=workspace_path(args.refinement_calibration)
        calibration=json.loads(calibration_path.read_text())
        if (calibration['draft_protocol_sha256']!=digest(draft) or calibration['code_identity']!=code_identity()
                or calibration['data_identity']!=identity):raise ValueError('Calibration source/data/protocol identity differs')
        protocol=calibration['frozen_protocol'];arm=args.refine_arm
        expected=dict(draft,slice_weight=calibration['common_slice_weight'])
        if protocol!=expected or not math.isfinite(protocol['slice_weight']) or protocol['slice_weight']<=0:
            raise ValueError('Frozen calibration protocol/weight differs from the declared draft')
        if arm not in ('A','B','C'):raise ValueError('Choose one refinement arm A/B/C')
    baseline_path=workspace_path(protocol['baseline_report'])
    if _sha256(baseline_path)!=protocol['baseline_report_sha256']:raise ValueError('S1 validation baseline identity differs')
    baseline=json.loads(baseline_path.read_text())
    reference_cache={}  # Fixed TRAIN32 + VAL100; GT only, retained across epochs.
    if baseline.get('checkpoint_sha256')!=protocol['stage1_checkpoint_sha256']:raise ValueError('Baseline belongs to different S1 weights')
    train=dataset_from_config(data);val=dataset_from_config(data,'val')
    raw_train=CachedStageSamples(train,base_protocol['train_indices'],args.cache_dir)
    raw_val=CachedStageSamples(val,base_protocol['val_indices'],args.cache_dir)
    samples=(raw_train if arm=='A' else NativeReferenceSamples(raw_train,args.cache_dir/'native_reference_sdf',protocol['native_sdf_clip_mm']))
    # Evaluation reuses existing native masks; no SDF is needed by the hard renderer.
    options=dict(workers=args.workers,prefetch=args.prefetch,seed=protocol['seed'],pin_memory=torch.device(args.device).type=='cuda')
    val_loader=make_loader(raw_val,**options)
    train_probe=make_loader(CachedStageSamples(train,base_protocol['train_indices'][:32],args.cache_dir),**options)
    if args.prepare_cache:
        started=time.perf_counter()
        for label,source in (('train',samples),('val',NativeReferenceSamples(raw_val,args.cache_dir/'native_reference_sdf',protocol['native_sdf_clip_mm']))):
            for index,_ in enumerate(make_loader(source,collate_fn=refinement_collate,**options)):
                if index==0 or (index+1)%50==0:print(f'S2 cache {label} {index+1}/{len(source)}',flush=True)
        write_json(args.output/'cache_prepared.json',dict(seconds=time.perf_counter()-started,stage='S2',test_loaded=False))
        return
    if args.evaluate_refinement:
        validate_template(model)
        report=evaluate(model,val_loader,args.device,full=True,surface_samples=base_protocol['surface_samples'],
                        prediction_dir=args.output/'reference_predictions',deadline=deadline,reference_cache=reference_cache)
        guard=refinement_guard(report,baseline,protocol)
        report.update(arm=arm,step=saved['step'],checkpoint_sha256=_sha256(args.evaluate_refinement),guard=guard)
        write_json(args.output/'val_evaluation.json',report)
        print(json.dumps(dict(arm=arm,guard_eligible=guard['eligible'],hard_target_met=guard['hard_target_met'])),flush=True)
        return
    checkpoint_config=dict(config,refinement=protocol,refinement_arm=arm)
    runtime=dict(stage_contract='goal04s-refinement-v1',arm=arm,code_identity=code_identity(),device=args.device,
        source_stage1_sha256=protocol['stage1_checkpoint_sha256'],source_stage1_step=32000,
        calibration_sha256=_sha256(calibration_path),data_identity=identity,
        input_pipeline=dict(cache_dir=str(args.cache_dir),workers=args.workers,prefetch=args.prefetch),
        selection='Hard-native minimum/P01/P05, then threshold counts; full3D/safety guards required before promotion')
    if args.resume:
        model,saved=load_checkpoint(args.resume,args.device,identity=identity,config=checkpoint_config)
        if saved['runtime'].get('stage_contract')!='goal04s-refinement-v1' or saved['runtime']['code_identity']!=code_identity():
            raise ValueError('Strict S2 resume source differs')
        model.__class__=StagedHeartFFDNet;configure_execution(model,saved=saved)
        runtime['execution']=saved['runtime']['execution']
        if saved['runtime']!=runtime:raise ValueError('Strict S2 resume runtime differs')
        step=saved['step']
    else:
        if (args.output/'protocol.json').exists():raise ValueError('Fresh S2 output required; S1 artifacts must not be overwritten')
        model,_=load_stage1_initialization(protocol,identity,args.device)
        runtime['execution']=configure_execution(model,'off');step=0
    validate_template(model)
    criterion=NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(args.device)
    slicer=make_refinement_slicer(model,protocol,arm)
    if slicer is not None:slicer=slicer.to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    if args.resume:
        optimizer.load_state_dict(saved['optimizer_state']);restore_rng({'rng':saved['rng_state']},args.device)
    else:
        write_json(args.output/'protocol.json',dict(config=checkpoint_config,runtime=runtime,data_identity=identity))
        for name in runtime['code_identity']:
            dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            import shutil
            shutil.copy2(ROOT/name,dest)
    def save(name):save_checkpoint(args.output/name,model,optimizer,config=checkpoint_config,identity=identity,
                                   step=step,epoch=step//len(samples),sample_offset=step%len(samples),training_stage='shape_refinement',runtime=runtime)
    selection_path=args.output/'candidate.json'
    best_rank=json.loads(selection_path.read_text())['rank'] if args.resume and selection_path.exists() else None
    if not args.resume:save('initial.pth')
    sampler=EpochOrder(len(samples),protocol['seed'])
    loader=make_loader(samples,sampler=sampler,collate_fn=refinement_collate,**options);iterator=None
    start=time.monotonic()
    with (args.output/'history.jsonl').open('a') as stream:
        while step<protocol['updates']:
            import train_staged_heartffd as entry
            if entry.STOP_REQUESTED or (deadline is not None and time.monotonic()>=deadline):break
            epoch,offset=divmod(step,len(samples))
            if iterator is None or offset==0:sampler.set_position(epoch,offset);iterator=iter(loader)
            tick=time.perf_counter();batch=to_device(next(iterator),args.device,non_blocking=True)
            wait=time.perf_counter()-tick
            stride=next(x['pixel_stride'] for x in reversed(protocol['stride_schedule']) if step>=x['start_step'])
            if slicer is not None:slicer.pixel_stride=stride
            lr=protocol['minimum_learning_rate']+(protocol['learning_rate']-protocol['minimum_learning_rate'])*(1+math.cos(math.pi*step/max(1,protocol['updates']-1)))/2
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);model.train()
            out,loss=refinement_objective(model,batch,criterion,slicer,shape_weight=protocol['shape_weight'],
                                        slice_weight=0. if arm=='A' else protocol['slice_weight'])
            loss['total'].backward();grads=reference_gradient_report(model);optimizer.step();step+=1
            row=dict(step=step,case_id=batch['case_id'][0],arm=arm,lr=lr,pixel_stride=stride,
                total=float(loss['total'].detach()),shape=float(loss['shape']['total'].detach()),
                slice=None if loss['slice'] is None else float(loss['slice']['total'].detach()),
                gradients=grads,data_wait_seconds=wait,update_seconds=time.perf_counter()-tick)
            if step%protocol['evaluate_every']==0 or step==protocol['updates']:
                probe=evaluate(model,train_probe,args.device,native_exact=True,deadline=deadline,reference_cache=reference_cache)
                row['train_probe']=probe['summary'];row['train_probe_complete']=probe['complete']
                row['train_scope']='TRAIN_prefix32'
                report=evaluate(model,val_loader,args.device,native_exact=True,deadline=deadline,reference_cache=reference_cache)
                directory=args.output/'evaluations';directory.mkdir(exist_ok=True)
                write_json(directory/f'step{step:05d}.json',report)
                row['validation']=report['summary'];row['validation_complete']=report['complete']
                if report['complete']:
                    rank=list(tail_rank(report))
                    if best_rank is None or tuple(rank)>tuple(best_rank):
                        best_rank=rank;save('candidate.pth');write_json(selection_path,dict(step=step,rank=rank,
                            note='Native-tail shortlist only; NOT a protected/accepted checkpoint until full3D guard passes'))
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if step%16==0:print(f'S2 {arm} step={step}/{protocol["updates"]} loss={row["total"]:.6f}',flush=True)
            if step%protocol['checkpoint_every']==0:save('latest.pth')
        save('latest.pth')
        name='endpoint.pth' if step==protocol['updates'] else 'time_limit.pth';save(name)
    write_json(args.output/'training_status.json',dict(step=step,target_updates=protocol['updates'],complete=step==protocol['updates'],
         note='S2 checkpoint is unaccepted until full validation and baseline guards are evaluated'))
    if step!=protocol['updates']:return
    # Evaluate at most two distinct shortlisted model states once. S1 full metrics
    # are reused; no re-evaluation of the baseline or per-epoch surface sweeps.
    signatures={};eligible=[]
    for name in ('candidate.pth','endpoint.pth'):
        path=args.output/name
        if not path.exists():continue
        candidate,saved=load_checkpoint(path,args.device,identity=identity)
        candidate.__class__=StagedHeartFFDNet;validate_template(candidate)
        state_hash=hashlib.sha256()
        for key,v in sorted(saved['model_state'].items()):state_hash.update(key.encode());state_hash.update(v.cpu().contiguous().numpy().tobytes())
        signature=state_hash.hexdigest()
        prior=cached_full_report(args.output/(path.stem+'_full_validation.json'),signature,
                                 protocol['baseline_report_sha256'],runtime['code_identity'])
        if signature in signatures:
            report,guard=signatures[signature]
        elif prior is not None:
            report=prior;guard=refinement_guard(report,baseline,protocol);signatures[signature]=(report,guard)
        else:
            report=evaluate(candidate,val_loader,args.device,full=True,surface_samples=base_protocol['surface_samples'],
                            prediction_dir=args.output/(path.stem+'_predictions'),deadline=deadline,reference_cache=reference_cache)
            guard=refinement_guard(report,baseline,protocol);signatures[signature]=(report,guard)
        payload=dict(report,guard=guard,step=saved['step'],checkpoint_sha256=_sha256(path),
                     model_state_sha256=signature,baseline_report_sha256=protocol['baseline_report_sha256'],code_identity=runtime['code_identity'])
        write_json(args.output/(path.stem+'_full_validation.json'),payload)
        if guard['eligible']:eligible.append((tail_rank(report),path,guard))
    if eligible:
        _,path,guard=max(eligible,key=lambda x:x[0])
        import shutil
        shutil.copy2(path,args.output/'best.pth')
        write_json(args.output/'promotion.json',dict(guard_eligible=True,hard_target_met=guard['hard_target_met'],
            source_checkpoint=path.name,checkpoint_sha256=_sha256(path),motion_reference_accepted=guard['hard_target_met']))
    else:write_json(args.output/'promotion.json',dict(guard_eligible=False,hard_target_met=False,motion_reference_accepted=False,
                              reason='No shortlisted checkpoint passed all frozen3D, tail and safety guards; retain S1'))
