"""Independent fixed-budget v2/G1 mechanism experiments; no long training mode.

This entry handles data, objective, checkpoint and diagnostics. Spatial lifting,
view fusion, temporal conditioning and controls live exclusively in the model.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
import numpy as np
import torch
from torch.utils.data import default_collate
import yaml
from data.mask_transform import CHAMBERS,VIEWS
from data.synthetic_4d import _sha256
from net.geometry_heartffdnet import GeometryHeartFFDNet,ARCHITECTURE as G1_ARCH,geometry_digest,loss_output_adapter
from net.geometry_heartffdnet_v2 import GeometryHeartFFDNetV2,ARCHITECTURE as G2_ARCH
from train_geometry_heartffdnet import (dataset_from_config,data_identity,geometry_identity as geometry_v1_identity,
    training_collate,to_device,write_json,EXPECTED_DATA_IDENTITY)
from train_neural_ffd import repository_path,restore_rng,validation_summary,exact_metrics
from utils.neural_ffd_losses import NeuralFFDLoss,reconstruction_metrics
from utils.slice_supervision import MeshSliceSupervisor
from utils.geometry_joint_checks import validate_template,validate_prediction
from utils.geometry_v2_checks import (GeometrySafetyError,require_strict_template_safety,
                                    validate_four_chamber_prediction,validate_observed_sections)
from utils.geometry_motion_metrics import observed_vertex_regions,detailed_motion_metrics
from utils.geometry_pair_policy import (bind_geometry_policy,SETTINGS as GEOMETRY_SETTINGS,
    validate_source_geometry,validate_patient_geometry,validate_output_geometry)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 'geometry-goal04a2-checkpoint-v1'
CODE_FILES = ('net/geometry_heartffdnet_v2.py','net/geometry_observation_lifting.py',
    'net/geometry_view_fusion.py','net/local_temporal_motion.py','utils/geometry_v2_checks.py',
    'utils/geometry_motion_metrics.py','utils/geometry_pair_policy.py','train_geometry_heartffd_v2.py',
    'net/geometry_heartffdnet.py','net/geometry_spatial_encoder.py','net/geometry_lifting.py',
    'net/multiview_encoder.py','net/module.py','net/ffdnet.py','utils/ffd_physical.py',
    'utils/neural_ffd_losses.py','utils/slice_supervision.py','utils/slice_evaluation.py',
    'utils/geometry_joint_checks.py','data/mask_transform.py','data/synthetic_4d.py',
    'train_geometry_heartffdnet.py','docs/goal04a2/protocol.yaml')


def code_identity():
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return dict(commit=commit,sha256={x:_sha256(ROOT/x) for x in CODE_FILES})


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    sections = {'data_config','pose_mode','encoder','lifting','decoder','motion','slice','loss','training','geometry_checks'}
    if set(config) != sections:raise ValueError('Unknown/missing v2 configuration section')
    frozen = yaml.safe_load((ROOT/'configs/geometry_heartffd_1k.yaml').read_text())
    if config['geometry_checks']!=GEOMETRY_SETTINGS:raise ValueError('Pinned AV cap geometry contract differs')
    for key in ('data_config','pose_mode','slice','loss'):
        if config[key] != frozen[key]:raise ValueError('Frozen formal contract differs: '+key)
    if config['encoder']['width'] != 128 or tuple(config['encoder']['spatial_sides']) != (14,7):
        raise ValueError('Formal 128-channel 14/7 CNN maps required')
    if config['decoder'] != frozen['decoder']:raise ValueError('Frozen physical FFD domain/limits differ')
    data = yaml.safe_load(repository_path(config['data_config']).read_text())
    if data_identity(data) != EXPECTED_DATA_IDENTITY:raise ValueError('Immutable dataset identity differs')
    d=data['data']
    if tuple(d['views'])!=VIEWS or tuple(d['chambers'])!=CHAMBERS or d['expected_phases']!=24:
        raise ValueError('Full-cycle view/chamber contract differs')
    training=config['training']
    if set(training) != {'device','seed','batch_size','cpu_threads','checkpoint_every_steps','learning_rate',
        'shape_updates','motion_updates','joint_updates','output','slice_cell_chunk','protocol'}:
        raise ValueError('Unknown/missing fixed-budget training settings')
    if min(training[k] for k in ('batch_size','cpu_threads','checkpoint_every_steps','shape_updates','motion_updates','joint_updates','slice_cell_chunk'))<1:
        raise ValueError('Positive dimensions and stage updates required')
    protocol=yaml.safe_load(repository_path(training['protocol']).read_text())
    if training['learning_rate']!=protocol['learning_rate'] or training['batch_size']!=protocol['batch_size']:
        raise ValueError('Preregistered learning/exposure budget differs')
    if any(training[k+'_updates']!=protocol['updates'][k] for k in ('shape','motion','joint')):
        raise ValueError('Preregistered stage budget differs')
    return config,data,protocol


def build_model(dataset,config,variant='A4'):
    options=dict(norm_to_mm=dataset.norm_to_mm,asset_sha256=dataset.asset_sha256,
        encoder_config=config['encoder'],lifting_config=config['lifting'],decoder_config=config['decoder'],pose_mode=config['pose_mode'])
    if variant=='A0':model=GeometryHeartFFDNet.from_dataset_geometry(dataset.shared_geometry,**options)
    else:
        motion=dict(config['motion'],variant=variant)
        model=GeometryHeartFFDNetV2.from_dataset_geometry(dataset.shared_geometry,motion_config=motion,**options)
    bind_geometry_policy(model,config['geometry_checks'])
    return model


def architecture(model):return G2_ARCH if isinstance(model,GeometryHeartFFDNetV2) else G1_ARCH


def geometry_identity(model):
    result=geometry_v1_identity(model)
    if isinstance(model,GeometryHeartFFDNetV2):
        result['backbone']='frozen v1 spatial layers/reference heads; explicit local temporal grouped joint head v2'
        result['local_query_sha256']=hashlib.sha256(model.medium_query_points_mm.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if hasattr(model,'_source_av_cap_policy'):
        result['inter_chamber_policy']=model._source_av_cap_policy.identity
    return result


def save_checkpoint(path,model,optimizer,*,config,identity,step,runtime):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.get_state()
    payload=dict(version=CHECKPOINT_VERSION,architecture=architecture(model),pose_mode=model.pose_mode,
        model_state=model.state_dict(),optimizer_state=None if optimizer is None else optimizer.state_dict(),
        config=config,data_identity=identity,geometry_identity=geometry_identity(model),
        model_spec=dict(encoder=model.encoder_config,lifting=model.lifting_config,decoder=model.decoder_config,
            motion=getattr(model,'motion_config',None)),step=step,training_stage=stage_for_step(max(0,step-1),config),
        runtime=runtime,epoch=step//((len(runtime.get('train_indices',[0]))+config['training']['batch_size']-1)//config['training']['batch_size']),
        sample_offset=(step*config['training']['batch_size'])%len(runtime.get('train_indices',[0])),code_identity=code_identity(),
        rng_state=dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=(rng[0],rng[1].tolist(),*rng[2:]),
            cuda=torch.cuda.get_rng_state(model.template_vertices_mm.device) if model.template_vertices_mm.is_cuda else None))
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:torch.save(payload,stream)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_checkpoint(path,device='cpu',*,config=None,identity=None,strict_code=True):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('version')!=CHECKPOINT_VERSION or saved.get('architecture') not in (G1_ARCH,G2_ARCH) or saved.get('pose_mode')!='known_pose':
        raise ValueError('Unsupported A2 architecture/checkpoint/pose contract')
    if config is not None and saved['config']!=config:raise ValueError('Strict resume configuration differs')
    if identity is not None and saved['data_identity']!=identity:raise ValueError('Strict resume data identity differs')
    if strict_code and saved['code_identity']!=code_identity():raise ValueError('Strict resume code identity differs')
    state,spec,geo=saved['model_state'],saved['model_spec'],saved['geometry_identity']
    faces={c:state['faces_'+c] for c in CHAMBERS}
    if geometry_digest(state['template_vertices_mm'],state['chamber_ranges'],faces)!=geo['geometry_sha256']:
        raise ValueError('Fixed geometry hash differs')
    cls=GeometryHeartFFDNetV2 if saved['architecture']==G2_ARCH else GeometryHeartFFDNet
    extra=dict(motion_config=spec['motion']) if cls is GeometryHeartFFDNetV2 else {}
    model=cls(state['template_vertices_mm'],state['chamber_ranges'],faces,
        template_sha256=geo['template_sha256'],asset_sha256=geo['asset_sha256'],norm_to_mm=geo['norm_to_mm'],
        encoder_config=spec['encoder'],lifting_config=spec['lifting'],decoder_config=spec['decoder'],**extra)
    if 'geometry_checks' in saved['config']:bind_geometry_policy(model,saved['config']['geometry_checks'])
    if geometry_identity(model)!=geo:raise ValueError('Physical grid/coordinate contract differs')
    parameters=dict(model.named_parameters())
    for key,value in model.state_dict().items():
        if key not in parameters and not torch.equal(value,state[key]):raise ValueError('Fixed geometry buffer differs: '+key)
    model.load_state_dict(state,strict=True)
    return model.to(device),saved


def stage_for_step(step,config):
    t=config['training']
    return 'shape' if step<t['shape_updates'] else 'motion' if step<t['shape_updates']+t['motion_updates'] else 'joint'


def gradient_report(model):
    modules=dict(cnn=model.encoder,lifting=model.lifting,unet=model.unet,
                 joint=model.heads.deform3,coarse=model.heads.deform1,medium=model.heads.deform2)
    if isinstance(model,GeometryHeartFFDNetV2):
        modules['local_temporal']=model.local_temporal
        if model.use_fusion:modules['view_gate']=model.view_fusion
        if model.use_fine_skip:modules['fine_skip']=model.fine_motion_skip
    report={}
    for key,module in modules.items():
        gradients=[p.grad.detach() for p in module.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Missing/nonfinite actual gradients: '+key)
        report[key]=float(sum(g.square().sum() for g in gradients).sqrt())
        if not report[key]>0:raise ValueError('Zero actual gradients: '+key)
    return report


@torch.no_grad()
def evaluate(model,samples,device,*,exact=False,full_geometry=False):
    model.eval();rows=[]
    for index,sample in enumerate(samples):
        print(f'Evaluate {index+1}/{len(samples)}: geometry={full_geometry}, exact={exact}',flush=True)
        batch=to_device(default_collate([sample]),device)
        output=model(batch)
        geometry=validate_output_geometry(model,output)
        geometry['observed_sections']=validate_observed_sections(model,output,batch)
        row=dict(case_id=sample['case_id'],candidate_id=sample['candidate_id'],geometry=geometry,
            metrics=reconstruction_metrics(output['vertices_4d'],batch['vertices_gt'],model.chamber_ranges,model.faces),
            control_rms_mm={k:float(v.square().mean().sqrt()) for k,v in output['control_grids'].items()})
        regions=observed_vertex_regions(batch['vertices_gt'].to(output['vertices_4d']),batch,
            model.chamber_ranges,model.norm_to_mm,
            direct_half_thickness_mm=getattr(model,'motion_config',{}).get('direct_half_thickness_mm',.75),
            propagated_distance_mm=model.lifting.sigma_mm*model.lifting.cutoff_sigma)
        row['detailed_motion']=detailed_motion_metrics(output['vertices_4d'],batch['vertices_gt'],
            model.chamber_ranges,model.faces,batch['phases'],regions=regions)
        if exact:row['exact']=exact_metrics(model,output['vertices_4d'],batch)
        if 'constraint_statistics' in output:
            row['constraints']={k:v.cpu().tolist() for k,v in output['constraint_statistics'].items()}
        rows.append(row)
    report=dict(cases=rows,case_count=len(rows),pose_mode='known_pose',architecture=architecture(model))
    report['summary']=validation_summary(report)
    return report


def run_short(config,data,protocol,args):
    device=args.device or config['training']['device']
    seed=args.seed if args.seed is not None else config['training']['seed']
    if seed not in protocol['seeds']:raise ValueError('Seed outside preregistered experiment')
    indices=protocol['train_indices']
    runtime=dict(variant=args.variant,seed=seed,train_indices=indices,batch_size=config['training']['batch_size'])
    output=args.output or repository_path(config['training']['output'])/(args.variant+'_seed'+str(seed))
    output.mkdir(parents=True,exist_ok=True)
    dataset=dataset_from_config(data)
    identity=data_identity(data)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(config['training']['cpu_threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.resume:
        model,saved=load_checkpoint(args.resume,device,config=config,identity=identity)
        if saved['runtime']!=runtime:raise ValueError('Strict resume sample/variant/seed budget differs')
        step=saved['step']
    else:
        if (output/'history.jsonl').exists():raise ValueError('Fresh run requires an empty output directory or explicit resume')
        model=build_model(dataset,config,args.variant).to(device);step=0
    validate_template(model)
    try:
        source_geometry = validate_source_geometry(model)
    except GeometrySafetyError as error:
        write_json(output/'source_geometry.json',error.report)
        write_json(output/'run_status.json',dict(status='rejected_source_geometry',updates=0,reason=str(error)))
        raise
    write_json(output/'source_geometry.json',source_geometry)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['training']['learning_rate'])
    if args.resume:
        optimizer.load_state_dict(saved['optimizer_state']);restore_rng({'rng':saved['rng_state']},device)
    criterion=NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(device)
    slicer=MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,**config['slice'],
        execution_backend='batched',cell_chunk_size=config['training']['slice_cell_chunk']).to(device)
    samples=[dataset[i] for i in indices]
    total=sum(config['training'][k+'_updates'] for k in ('shape','motion','joint'))
    stop=min(total,args.max_steps) if args.max_steps is not None else total
    batch_size=config['training']['batch_size'];per_epoch=(len(indices)+batch_size-1)//batch_size
    manifest=dict(config=config,runtime=runtime,data_identity=identity,code_identity=code_identity(),
        parameters=sum(p.numel() for p in model.parameters()),initialization='random; no inherited O/G1 weights',
        exposure_cases=[dict(candidate_id=s['candidate_id'],case_id=s['case_id']) for s in samples])
    write_json(output/'experiment.json',manifest)
    started=time.perf_counter()
    with (output/'history.jsonl').open('a') as history:
        while step<stop:
            stage=stage_for_step(step,config)
            epoch,offset=divmod(step,per_epoch)
            order=torch.randperm(len(samples),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            selected=order[offset*batch_size:(offset+1)*batch_size]
            batch=to_device(training_collate([samples[i] for i in selected]),device)
            model.train();optimizer.zero_grad(set_to_none=True)
            tick=time.perf_counter();decoded=model(batch)
            geometry=validate_output_geometry(model,decoded)
            geometry['observed_sections']=validate_observed_sections(model,decoded,batch)
            sliced=slicer(decoded['vertices_4d'],batch) if stage=='joint' else None
            losses=criterion(loss_output_adapter(decoded),batch['vertices_gt'],slice_result=sliced,
                control_spacings_mm=model.control_spacings_mm,stage=stage)
            losses['total'].backward();gradients=gradient_report(model);optimizer.step();step+=1
            metrics=reconstruction_metrics(decoded['vertices_4d'].detach(),batch['vertices_gt'],model.chamber_ranges,model.faces)
            record=dict(step=step,stage=stage,total=float(losses['total'].detach()),
                terms={k:float(v.detach()) for k,v in losses['terms'].items()},active_weights=losses['active_weights'],
                gradients=gradients,geometry=geometry,metrics=metrics,seconds=time.perf_counter()-tick,
                candidate_ids=[samples[i]['candidate_id'] for i in selected])
            if 'constraint_statistics' in decoded:
                record['constraints']={k:v.detach().cpu().tolist() for k,v in decoded['constraint_statistics'].items()}
            record['parameters_with_gradients']=sum(p.numel() for p in model.parameters() if p.grad is not None)
            if torch.device(device).type=='cuda':record['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
            history.write(json.dumps(record,allow_nan=False)+'\n');history.flush()
            if step==1 or step%8==0:print(f'{args.variant} seed={seed} {step}/{stop} [{stage}] loss={record["total"]:.5f} seconds={record["seconds"]:.2f}',flush=True)
            if step%config['training']['checkpoint_every_steps']==0 or step==stop:
                save_checkpoint(output/'latest.pth',model,optimizer,config=config,identity=identity,step=step,runtime=runtime)
            if step in (config['training']['shape_updates'],config['training']['shape_updates']+config['training']['motion_updates']):
                write_json(output/('after_'+stage+'.json'),evaluate(model,samples[:4],device))
    write_json(output/'run_status.json',dict(status='budget_complete' if step==total else 'partial',step=step,budget=total,seconds=time.perf_counter()-started))
    print('Training stopped at fixed budget:',step,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/geometry_heartffd_v2.yaml')
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--short-run',action='store_true');action.add_argument('--evaluate',type=Path)
    parser.add_argument('--variant',choices=('A0','A1','A2','A3','A4'),default='A4')
    parser.add_argument('--seed',type=int);parser.add_argument('--device');parser.add_argument('--resume',type=Path)
    parser.add_argument('--max-steps',type=int);parser.add_argument('--output',type=Path);parser.add_argument('--exact',action='store_true')
    args=parser.parse_args()
    config,data,protocol=load_config(args.config)
    if args.max_steps is not None and args.max_steps<1:parser.error('Positive explicit step limit required')
    if args.evaluate:
        if args.resume:parser.error('Evaluation is not resume')
        device=args.device or config['training']['device'];torch.set_num_threads(config['training']['cpu_threads'])
        model,saved=load_checkpoint(args.evaluate,device,identity=data_identity(data));validate_template(model)
        validate_source_geometry(model)
        val=dataset_from_config(data,'val');samples=[val[i] for i in protocol['val_indices']]
        report=evaluate(model,samples,device,exact=args.exact,full_geometry=True)
        report.update(checkpoint_sha256=_sha256(args.evaluate),split='VAL',selection='preregistered fixed cases')
        output=args.output or args.evaluate.parent/'validation';output.mkdir(parents=True,exist_ok=True)
        write_json(output/'evaluation.json',report);print(report['summary'],flush=True)
    else:run_short(config,data,protocol,args)


if __name__=='__main__':main()
