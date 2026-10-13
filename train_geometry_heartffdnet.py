"""Independent known-pose Geometry HeartFFDNet training, acceptance and evaluation.

--acceptance runs only the first fixed TRAIN cases and configured short schedule.
Formal training requires --train; no long run is started by default.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate
import yaml
from data.mask_transform import CHAMBERS, VIEWS
from data.synthetic_4d import _sha256
from net.geometry_heartffdnet import GeometryHeartFFDNet, ARCHITECTURE, geometry_digest, loss_output_adapter
from utils.geometry_joint_checks import validate_template, validate_prediction
from utils.neural_ffd_losses import NeuralFFDLoss, reconstruction_metrics
from utils.slice_supervision import MeshSliceSupervisor
# Read-only reuse of verified data, metric, RNG and best-selection helpers.
from train_neural_ffd import (dataset_from_config, data_identity as baseline_data_identity,
    repository_path, to_device, restore_rng, exact_metrics, validation_summary, best_checkpoint_candidates)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 'geometry-ma-joint-checkpoint-v1'
EXPECTED_DATA_IDENTITY = {
    'train_manifest':'c3c0c1e021f0cf5c4f734f1588ea722e908c69ba0a8c2241c7f78e3d54a0e990',
    'val_manifest':'5f96244274a9556654bd6506ad924880bb6668ac065d8d2e032bdc1d9715913d',
    'test_manifest':'1bbbbc67f530824c18848305e9fae38d57e712b00006427e98678a4b333f78f3',
    'asset_path':'29a44d5682a338f2ba52ccdbee21ab831c213b15546e73ce114f185d300cced5',
    'template_path':'8b3a6a0f443df42ee2b8338df9656970963eb86496ad9784753b5620aa4dbd46',
}



def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if set(config) != {'data_config','pose_mode','encoder','lifting','decoder','slice','loss','training','acceptance'}:
        raise ValueError('Geometry configuration sections must match version 1')
    data = yaml.safe_load(repository_path(config['data_config']).read_text())
    d = data['data']
    if data_identity(data) != EXPECTED_DATA_IDENTITY:
        raise ValueError('Immutable 800 TRAIN / 100 VAL / 100 TEST data identity differs')
    if (config['pose_mode'] != 'known_pose' or config['decoder']['frames'] != 24
            or tuple(d['views']) != VIEWS or tuple(d['chambers']) != CHAMBERS
            or d['expected_phases'] != 24 or not d['remove_duplicate_endpoint']
            or d['phase_semantics'] != 'relative_observed_cycle_time'
            or not data['mask']['use_sdf'] or data['mask']['sdf_normalize']
            or config['slice']['target_mode'] != 'binary_native_v1'):
        raise ValueError('Known-pose 24-phase physical SDF/binary-native slice contract required')
    # Historical formal nine-term supervision is fixed for this architecture goal.
    expected = dict(shape=1.,motion=1.,velocity=.2,volume=.01,edge=.05,laplacian=.05,normal=.01,control=.001,slice=.5)
    if config['loss']['weights'] != expected or config['loss']['huber_beta_mm'] != 1.:
        raise ValueError('Goal 04-A must preserve formal nine-term loss weights/units')
    tr,ac = config['training'],config['acceptance']
    if (min(tr[k] for k in ('epochs','batch_size','cpu_threads','validation_cases','checkpoint_every_steps')) < 1
            or tr['workers'] < 0 or min(tr['shape_epochs'],tr['motion_epochs']) < 0
            or tr['shape_epochs']+tr['motion_epochs'] >= tr['epochs']
            or not 0 < tr['learning_rate'] < 1 or not 0 < ac['learning_rate'] < 1
            or not 4 <= ac['cases'] <= 8 or min(ac[k] for k in ('shape_updates','motion_updates','joint_updates')) < 1):
        raise ValueError('Invalid training/short acceptance dimensions')
    return config,data


def data_identity(config):
    return dict(baseline_data_identity(config),test_manifest=_sha256(repository_path(config['data']['test_manifest'])))


def training_collate(samples):
    # Geometry retained in EVERY stage. No GT or generating coefficients enter
    # the encoder: the model itself only accesses its explicit observation keys.
    unused = {'native_masks','clipped_foreground','center_mm'}
    return default_collate([{k:v for k,v in s.items() if k not in unused} for s in samples])


def build_model(dataset, config):
    return GeometryHeartFFDNet.from_dataset_geometry(dataset.shared_geometry,norm_to_mm=dataset.norm_to_mm,
        asset_sha256=dataset.asset_sha256,encoder_config=config['encoder'],lifting_config=config['lifting'],
        decoder_config=config['decoder'],pose_mode=config['pose_mode'])


def geometry_identity(model):
    return dict(geometry_sha256=model.geometry_sha256,template_sha256=model.template_sha256,
        asset_sha256=model.asset_sha256,norm_to_mm=model.norm_to_mm,
        grid_shapes=[list(x) for x in model.grid_shapes],lower_mm=model.lower_mm.tolist(),
        upper_mm=model.upper_mm.tolist(),coordinates='centered physical mm; row-vector source renderer',
        recurrence='coarse -> medium -> fine0 -> sequential fine1..23; independent endpoint',
        feature_coordinates='ResNet stride8/16 adaptive-bin receptive centroids; align_corners=True v1',
        backbone='existing FFDNet conv2 stride1; conv3..5/deconv2..5/deform1..3 reused')


def save_checkpoint(path, model, optimizer, *, config, identity, epoch=0, sample_offset=0,
                    step=0, training_stage='shape', runtime=None):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np_rng = np.random.get_state()
    payload = dict(version=CHECKPOINT_VERSION,architecture=ARCHITECTURE,pose_mode=model.pose_mode,
        model_state=model.state_dict(),optimizer_state=None if optimizer is None else optimizer.state_dict(),
        epoch=epoch,sample_offset=sample_offset,step=step,training_stage=training_stage,
        config=config,data_identity=identity,geometry_identity=geometry_identity(model),runtime=runtime,
        specification=dict(encoder=model.encoder_config,lifting=model.lifting_config,decoder=model.decoder_config),
        rng_state=dict(torch=torch.get_rng_state(),python=random.getstate(),
            numpy=(np_rng[0],np_rng[1].tolist(),*np_rng[2:]),
            cuda=torch.cuda.get_rng_state(model.template_vertices_mm.device) if model.template_vertices_mm.is_cuda else None))
    fd,temporary = tempfile.mkstemp(dir=path.parent,prefix=path.name,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(payload,stream)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def load_checkpoint(path, device='cpu', *, config=None, identity=None):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if (saved.get('version') != CHECKPOINT_VERSION or saved.get('architecture') != ARCHITECTURE
            or saved.get('pose_mode') != 'known_pose'):
        raise ValueError('Unsupported geometry joint architecture/checkpoint/pose mode')
    if config is not None and saved['config'] != config:
        raise ValueError('Strict resume configuration differs')
    if identity is not None and saved['data_identity'] != identity:
        raise ValueError('Strict checkpoint data identity differs')
    state,spec,geo = saved['model_state'],saved['specification'],saved['geometry_identity']
    faces = {c:state['faces_'+c] for c in CHAMBERS}
    if geometry_digest(state['template_vertices_mm'],state['chamber_ranges'],faces) != geo['geometry_sha256']:
        raise ValueError('Physical geometry digest differs')
    model = GeometryHeartFFDNet(state['template_vertices_mm'],state['chamber_ranges'],faces,
        norm_to_mm=geo['norm_to_mm'],template_sha256=geo['template_sha256'],asset_sha256=geo['asset_sha256'],
        encoder_config=spec['encoder'],lifting_config=spec['lifting'],decoder_config=spec['decoder'],pose_mode=saved['pose_mode'])
    if geometry_identity(model) != geo:
        raise ValueError('Control/template/coordinate contract differs')
    parameters = dict(model.named_parameters())
    for key,value in model.state_dict().items():
        if key not in parameters and not torch.equal(value,state[key]):
            raise ValueError('Fixed physical geometry buffer differs: '+key)
    model.load_state_dict(state,strict=True)
    return model.to(device),saved


def compute_objective(model,batch,criterion,slicer,stage):
    output = model(batch)
    geometry = validate_prediction(model,output)
    slices = slicer(output['vertices_4d'],batch) if stage == 'joint' else None
    losses = criterion(loss_output_adapter(output),batch['vertices_gt'],slice_result=slices,
                       control_spacings_mm=model.control_spacings_mm,stage=stage)
    return output,losses,geometry


@torch.no_grad()
def evaluate(model,dataset,device,*,cases,exact=False,cached_samples=None,check_intersections=False,progress=False):
    model.eval();rows=[]
    check_intersections = check_intersections or exact
    samples = cached_samples if cached_samples is not None else [dataset[i] for i in range(min(cases,len(dataset)))]
    for case_index,sample in enumerate(samples):
        started = time.perf_counter()
        if progress:
            label = 'full 24-frame intersection/Jacobian checks' if check_intersections else 'metrics/Jacobian checks'
            print(f'Evaluate {case_index+1}/{len(samples)}: {label}',flush=True)
        batch = to_device(default_collate([sample]),device)
        output = model(batch)
        geometry = validate_prediction(model,output,sample_jacobian=True,check_intersections=check_intersections)
        item = dict(case_id=sample['case_id'],geometry=geometry,
            metrics=reconstruction_metrics(output['vertices_4d'],batch['vertices_gt'],model.chamber_ranges,model.faces),
            control_rms_mm={k:float(v.square().mean().sqrt()) for k,v in output['control_grids'].items()},
            observation_support_fraction=output['observation_support_fraction'].cpu().tolist())
        if exact:item['exact'] = exact_metrics(model,output['vertices_4d'],batch)
        rows.append(item)
        if progress:print(f'Evaluate {case_index+1}/{len(samples)} complete in {time.perf_counter()-started:.1f}s',flush=True)
    report = dict(cases=rows,case_count=len(rows),manifest_sha256=dataset.manifest_sha256,
        pose_mode=model.pose_mode,architecture=ARCHITECTURE,
        reference_semantics='synthetic source frame0, not per-chamber clinical ED',
        motion_semantics='24 independent phases, sequential FFD; 23->0 is a true closing time step')
    report['summary'] = validation_summary(report)
    return report


def finite_gradient_report(model):
    groups = {'cnn':model.encoder,'lifting':model.lifting,'unet':model.unet,'joint_fine':model.heads.deform3,
              'coarse':model.heads.deform1,'medium':model.heads.deform2}
    result = {}
    for key,module in groups.items():
        grads = [p.grad.detach() for p in module.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError('Missing/nonfinite real parameter gradients: '+key)
        result[key] = float(sum(g.square().sum() for g in grads).sqrt())
        if result[key] == 0:
            raise ValueError('Zero real parameter gradients: '+key)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def stage_for_step(step,config):
    a = config['acceptance']
    if step < a['shape_updates']:return 'shape'
    if step < a['shape_updates']+a['motion_updates']:return 'motion'
    return 'joint'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/geometry_heartffd_1k.yaml')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--acceptance',action='store_true')
    action.add_argument('--train',action='store_true')
    action.add_argument('--evaluate',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--device')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--val-cases',type=int)
    parser.add_argument('--exact',action='store_true')
    parser.add_argument('--slice-cell-chunk',type=int,default=8)
    args = parser.parse_args()
    if args.resume and args.evaluate:parser.error('Evaluation is not resume')
    if any(x is not None and x < 1 for x in (args.max_steps,args.max_cases,args.val_cases)):
        parser.error('Explicit limits must be positive')
    config,data_config = load_config(args.config)
    tr,ac = config['training'],config['acceptance']
    torch.set_num_threads(tr['cpu_threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    random.seed(tr['seed']);np.random.seed(tr['seed']);torch.manual_seed(tr['seed'])
    device = args.device or tr['device']
    output_dir = args.output or repository_path(tr['output']);output_dir.mkdir(parents=True,exist_ok=True)
    identity = data_identity(data_config)
    if args.evaluate:
        dataset = dataset_from_config(data_config,'val')
        model,saved = load_checkpoint(args.evaluate,device,identity=identity)
        if model.template_sha256 != dataset.template_sha256 or model.asset_sha256 != dataset.asset_sha256:
            raise ValueError('Evaluation template/asset mismatch')
        validate_template(model)
        report = evaluate(model,dataset,device,cases=args.val_cases or tr['validation_cases'],exact=args.exact,check_intersections=True,progress=True)
        report['checkpoint_sha256'] = _sha256(args.evaluate)
        write_json(output_dir/'evaluation.json',report)
        print(json.dumps(report['summary']),flush=True);return
    dataset = dataset_from_config(data_config)
    count = ac['cases'] if args.acceptance else min(args.max_cases or len(dataset),len(dataset))
    if args.acceptance and args.max_cases:parser.error('Acceptance subset is fixed in config')
    runtime = dict(mode='acceptance' if args.acceptance else 'train',cases=count,slice_cell_chunk=args.slice_cell_chunk)
    step,epoch,offset = 0,0,0
    if args.resume:
        model,saved = load_checkpoint(args.resume,device,config=config,identity=identity)
        if saved['runtime'] != runtime:raise ValueError('Strict resume runtime/subset differs')
        step,epoch,offset = saved['step'],saved['epoch'],saved['sample_offset']
    else:model = build_model(dataset,config).to(device)
    template_check = validate_template(model)
    write_json(output_dir/'template_geometry.json',template_check)
    (output_dir/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    rate = ac['learning_rate'] if args.acceptance else tr['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(),lr=rate)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer_state'])
        restore_rng({'rng':saved['rng_state']},device)
    criterion = NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(device)
    slicer = MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,**config['slice'],
        execution_backend='batched',cell_chunk_size=args.slice_cell_chunk).to(device)
    samples = [dataset[i] for i in range(count)] if args.acceptance else None
    initial_path = output_dir/'initial.json'
    if args.acceptance:
        if args.resume:
            if not initial_path.exists():raise ValueError('Acceptance resume requires original initial metrics')
            initial = json.loads(initial_path.read_text())
        else:
            initial = evaluate(model,dataset,device,cases=count,cached_samples=samples,check_intersections=True,progress=True)
            write_json(initial_path,initial)
        budget = sum(ac[k] for k in ('shape_updates','motion_updates','joint_updates'))
    else:budget = math.ceil(count/tr['batch_size'])*tr['epochs']
    stop = min(budget,args.max_steps) if args.max_steps is not None else budget
    best_path = output_dir/'best_metrics.json'
    best = json.loads(best_path.read_text()) if args.resume and best_path.exists() else {}
    best = {k:v for k,v in best.items() if v['step'] <= step}
    previous_stage = stage_for_step(step-1,config) if args.acceptance and step else None
    start = time.perf_counter()
    with (output_dir/'history.jsonl').open('a') as history:
        while step < stop:
            generator = torch.Generator().manual_seed(tr['seed']+epoch)
            order = torch.randperm(count,generator=generator).tolist()
            subset = Subset(samples if samples is not None else dataset,order[offset:])
            loader = DataLoader(subset,batch_size=tr['batch_size'],num_workers=tr['workers'],
                collate_fn=training_collate,generator=generator)
            for batch in loader:
                stage = (stage_for_step(step,config) if args.acceptance else
                    ('shape' if epoch < tr['shape_epochs'] else 'motion' if epoch < tr['shape_epochs']+tr['motion_epochs'] else 'joint'))
                if previous_stage is not None and stage != previous_stage:
                    report = evaluate(model,dataset,device,cases=count if args.acceptance else tr['validation_cases'],
                        cached_samples=samples,progress=True)
                    write_json(output_dir/('after_'+previous_stage+'.json'),report)
                    print(previous_stage,report['summary'],flush=True)
                previous_stage = stage
                model.train();batch = to_device(batch,device)
                optimizer.zero_grad(set_to_none=True)
                tick = time.perf_counter()
                decoded,losses,geometry = compute_objective(model,batch,criterion,slicer,stage)
                losses['total'].backward()
                gradients = finite_gradient_report(model)
                metrics = reconstruction_metrics(decoded['vertices_4d'].detach(),batch['vertices_gt'],model.chamber_ranges,model.faces)
                optimizer.step();step += 1;offset += len(batch['masks'])
                record = dict(step=step,epoch=epoch,stage=stage,pose_mode=model.pose_mode,
                    total=float(losses['total'].detach()),terms={k:float(v.detach()) for k,v in losses['terms'].items()},
                    metrics=metrics,control_rms_mm={k:float(v.detach().square().mean().sqrt()) for k,v in decoded['control_grids'].items()},
                    parameter_gradient_l2=gradients,geometry=geometry,seconds=time.perf_counter()-tick)
                if torch.device(device).type == 'cuda':
                    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                history.write(json.dumps(record,allow_nan=False)+'\n');history.flush()
                if step == 1 or step % 10 == 0:
                    print(f"step={step}/{stop} stage={stage} loss={record['total']:.5f} seconds={record['seconds']:.2f}",flush=True)
                next_epoch,next_offset = (epoch+1,0) if offset == count else (epoch,offset)
                if step % tr['checkpoint_every_steps'] == 0 or step == stop:
                    save_checkpoint(output_dir/'latest.pth',model,optimizer,config=config,identity=identity,
                        epoch=next_epoch,sample_offset=next_offset,step=step,training_stage=stage,runtime=runtime)
                if step == stop:break
            complete = offset == count
            if complete:
                epoch += 1;offset = 0
            if (args.acceptance and (step == stop or step % ac['evaluation_every_updates'] == 0)) or (not args.acceptance and complete):
                eval_dataset = dataset if args.acceptance else dataset_from_config(data_config,'val')
                report = evaluate(model,eval_dataset,device,cases=count if args.acceptance else tr['validation_cases'],cached_samples=samples,progress=True,exact=args.exact if not args.acceptance else False)
                write_json(output_dir/'progress.json',report)
                with (output_dir/'evaluation_history.jsonl').open('a') as summaries:
                    summaries.write(json.dumps(dict(step=step,epoch=epoch,stage=previous_stage,
                        split='TRAIN engineering subset' if args.acceptance else 'VAL',
                        metrics=report['summary']),allow_nan=False)+'\n')
                print(f"Validation at step {step} [{previous_stage}]: {report['summary']}",flush=True)
                summary = dict(complete=True,validation=report['summary'],stage=previous_stage)
                for name,candidate in best_checkpoint_candidates(summary).items():
                    if name not in best or candidate['score'] < best[name]['score']:
                        best[name] = dict(candidate,step=step)
                        save_checkpoint(output_dir/(name+'.pth'),model,optimizer,config=config,identity=identity,
                            epoch=epoch,sample_offset=offset,step=step,training_stage=previous_stage,runtime=runtime)
                write_json(best_path,best)
    if args.acceptance:
        final = evaluate(model,dataset,device,cases=count,cached_samples=samples,check_intersections=True,progress=True)
        write_json(output_dir/'final.json',final)
        restored,_ = load_checkpoint(output_dir/'latest.pth',device,config=config,identity=identity)
        model.eval();restored.eval()
        with torch.no_grad():
            probe = to_device(default_collate([samples[0]]),device)
            torch.testing.assert_close(model(probe)['vertices_4d'],restored(probe)['vertices_4d'],rtol=0,atol=0)
        keys = ('reference_vertex_rmse_mm','trajectory_vertex_rmse_mm','relative_motion_rmse_mm')
        reductions = {k:1-final['summary'][k]/initial['summary'][k] for k in keys}
        active = all(item['metrics'][c]['motion_amplitude_ratio'] > .05 for item in final['cases'] for c in CHAMBERS)
        passed = (step == budget and reductions[keys[0]] >= .1 and reductions[keys[2]] >= .1 and reductions[keys[1]] > 0 and active)
        result = dict(status='PASS' if passed else 'FAIL',scope='Fixed TRAIN engineering acceptance; no VAL100 claim',
            completed_updates=step,required_updates=budget,initial=initial['summary'],final=final['summary'],
            relative_reductions=reductions,all_chambers_active=active,checkpoint_bitwise_equal=True,
            parameters=sum(p.numel() for p in model.parameters()),device=str(device),seconds=time.perf_counter()-start,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None)
        write_json(output_dir/'acceptance.json',result);print(json.dumps(result,indent=2),flush=True)
        if not passed:raise SystemExit(2)


if __name__ == '__main__':main()
