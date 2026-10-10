"""One training/evaluation entry for Neural HeartFFDNet.

Examples (run from any cwd):
  python train_neural_ffd.py --config configs/neural_ffd.yaml
  python train_neural_ffd.py --resume outputs/neural_ffd/last.pth
  python train_neural_ffd.py --evaluate outputs/neural_ffd/last.pth --exact

--max-steps/--max-cases are explicit engineering checks, not formal training.
Source frame 0 is the synthetic reference, not independently selected EDs.
"""
import argparse
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data.mask_transform import CHAMBERS,VIEWS,MaskTransform
from data.synthetic_4d import Synthetic4DDataset,_sha256
from net.neural_ffdnet import NeuralFFDNet,ARCHITECTURE,geometry_digest
from net.multiview_encoder import observation_validity
from utils.neural_ffd_losses import NeuralFFDLoss,reconstruction_metrics
from utils.slice_supervision import MeshSliceSupervisor,exact_section_mask

ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 'neural-heartffd-checkpoint-v1'


def repository_path(value):
    return ROOT/Path(value)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    data_config = yaml.safe_load(repository_path(config['data_config']).read_text())
    data = data_config['data']
    if (tuple(data['views'])!=VIEWS or tuple(data['chambers'])!=CHAMBERS
            or data['phase_semantics']!='relative_observed_cycle_time'
            or not data['remove_duplicate_endpoint']
            or data['expected_phases']!=config['decoder']['frames']
            or data_config['mask']['sdf_normalize'] or not data_config['mask']['use_sdf']):
        raise ValueError('Canonical full-cycle masks, physical SDF and synchronized fixed decoder phases required')
    training = config['training']
    if (min(training[k] for k in ('epochs','batch_size','cpu_threads','validation_cases','checkpoint_every_steps'))<1
            or min(training['shape_epochs'],training['motion_epochs'],training['workers'])<0
            or training['shape_epochs']+training['motion_epochs']>=training['epochs']):
        raise ValueError('Positive training dimensions and at least one final joint epoch required')
    return config,data_config


def dataset_from_config(config,split='train'):
    data = config['data']
    return Synthetic4DDataset(repository_path(data[split+'_manifest']),repository_path(data['asset_path']),
        MaskTransform(**config['mask']),verify=config['check']['verify_source_hash'],
        path_map={old:str(repository_path(new)) for old,new in data['path_map'].items()},
        expected_phases=data['expected_phases'],remove_duplicate_endpoint=data['remove_duplicate_endpoint'],
        template=repository_path(data['template_path']),template_file_sha256=data['template_file_sha256'])


def data_identity(config):
    data = config['data']
    return {key:_sha256(repository_path(data[key])) for key in
            ('train_manifest','val_manifest','asset_path','template_path')}


def to_device(value,device):
    if isinstance(value,torch.Tensor):
        return value.to(device)
    if isinstance(value,dict):
        return {key:to_device(item,device) for key,item in value.items()}
    return value


def build_model(dataset,config):
    return NeuralFFDNet.from_dataset_geometry(dataset.shared_geometry,norm_to_mm=dataset.norm_to_mm,
        asset_sha256=dataset.asset_sha256,encoder_config=config['encoder'],decoder_config=config['decoder'])


def save_checkpoint(path,model,optimizer,*,config,identity,epoch=0,sample_offset=0,step=0,runtime=None):
    """Atomic save with embedded fixed geometry; independent decoder reload."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    rng_numpy = np.random.get_state()
    payload = dict(version=CHECKPOINT_VERSION,architecture=ARCHITECTURE,
        specification=dict(encoder=model.encoder_config,decoder=model.decoder_config,
            template_sha256=model.template_sha256,asset_sha256=model.asset_sha256,
            norm_to_mm=model.norm_to_mm,geometry_sha256=model.geometry_sha256),
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
        config=config,data_identity=identity,epoch=epoch,sample_offset=sample_offset,step=step,runtime=runtime,
        rng=dict(torch=torch.get_rng_state(),python=random.getstate(),
                 numpy=(rng_numpy[0],rng_numpy[1].tolist(),*rng_numpy[2:]),
                 cuda=torch.cuda.get_rng_state(model.template_vertices_mm.device) if model.template_vertices_mm.is_cuda else None))
    descriptor,temporary = tempfile.mkstemp(dir=path.parent,prefix=path.name,suffix='.tmp')
    try:
        with os.fdopen(descriptor,'wb') as stream:
            torch.save(payload,stream)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path,device='cpu'):
    """Construct a self-contained decoder; validate every fixed geometry buffer."""
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved['version']!=CHECKPOINT_VERSION or saved['architecture']!=ARCHITECTURE:
        raise ValueError('Unsupported Neural HeartFFDNet checkpoint')
    state,spec = saved['model'],saved['specification']
    faces = {c:state['faces_'+c] for c in CHAMBERS}
    if geometry_digest(state['template_vertices_mm'],state['chamber_ranges'],faces)!=spec['geometry_sha256']:
        raise ValueError('Checkpoint physical geometry identity differs')
    model = NeuralFFDNet(state['template_vertices_mm'],state['chamber_ranges'],faces,
        template_sha256=spec['template_sha256'],asset_sha256=spec['asset_sha256'],norm_to_mm=spec['norm_to_mm'],
        encoder_config=spec['encoder'],decoder_config=spec['decoder'])
    fixed = ('template_vertices_mm','chamber_ranges','lower_mm','upper_mm','control_origins_mm','control_spacings_mm',
             *['faces_'+c for c in CHAMBERS])
    if model.regional_shape:
        fixed += ('regional_origin_mm','regional_spacing_mm')
    for key in fixed:
        if not torch.equal(model.state_dict()[key],state[key]):
            raise ValueError('Checkpoint fixed control/topology buffer differs: '+key)
    model.load_state_dict(state,strict=True)
    return model.to(device),saved


def restore_rng(saved,device):
    state = saved['rng']
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0],np.asarray(numpy_state[1],dtype=np.uint32),*numpy_state[2:]))
    if state['cuda'] is not None:
        if torch.device(device).type!='cuda':
            raise ValueError('CUDA training resume requires CUDA to preserve RNG semantics')
        torch.cuda.set_rng_state(state['cuda'],device)


def stage_for_epoch(epoch,config):
    training = config['training']
    if epoch<training['shape_epochs']:
        return 'shape'
    if epoch<training['shape_epochs']+training['motion_epochs']:
        return 'motion'
    return 'joint'


def compute_objective(model,batch,criterion,slicer,stage):
    output = model(batch)
    slices = slicer(output['vertices_4d'],batch) if stage=='joint' else None
    losses = criterion(output,batch['vertices_gt'],slice_result=slices,
        control_spacings_mm=model.control_spacings_mm,
        regional_spacing_mm=model.regional_spacing_mm if model.regional_shape else None,stage=stage)
    return output,losses,slices


def surface_metrics(predicted,truth,faces,samples=5000):
    """Deterministic sampled triangle-surface ASSD/HD95; no voxel conversion."""
    import trimesh
    from scipy.spatial import cKDTree
    p = trimesh.sample.sample_surface(trimesh.Trimesh(predicted,faces,process=False),samples,seed=2026)[0]
    g = trimesh.sample.sample_surface(trimesh.Trimesh(truth,faces,process=False),samples,seed=2026)[0]
    distances = np.concatenate((cKDTree(g).query(p)[0],cKDTree(p).query(g)[0]))
    return dict(sampled_surface_assd_mm=float(distances.mean()),sampled_surface_hd95_mm=float(np.quantile(distances,.95)),surface_samples=samples)


def boundary_metrics(prediction,truth,spacing):
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    boundaries = [np.argwhere(x & ~binary_erosion(x)).astype(float)*spacing for x in (prediction.astype(bool),truth.astype(bool))]
    p,g = boundaries
    if not len(p) and not len(g):
        return dict(boundary_assd_mm=0.,boundary_hd95_mm=0.,empty_boundary='both')
    if not len(p) or not len(g):
        return dict(boundary_assd_mm=None,boundary_hd95_mm=None,empty_boundary='one')
    distances = np.concatenate((cKDTree(g).query(p)[0],cKDTree(p).query(g)[0]))
    return dict(boundary_assd_mm=float(distances.mean()),boundary_hd95_mm=float(np.quantile(distances,.95)),empty_boundary=None)


def exact_metrics(model,predicted,batch):
    """All available full-resolution phase/view/chamber cells; pose only here."""
    p = predicted.detach().cpu().numpy()
    g = batch['vertices_gt'].cpu().numpy()
    pose = {k:v.cpu().numpy() for k,v in batch['pose_gt'].items()}
    spacing = batch['pixel_spacing'].cpu().numpy()
    masks = batch['masks'].cpu().numpy()
    valid = observation_validity(batch,len(p),p.shape[1],predicted.device).cpu().numpy()
    rows = []
    for patient,frame,view,chamber in np.argwhere(valid):
        c = CHAMBERS[chamber]
        left,right = model.chamber_ranges[chamber].tolist()
        f = model.faces[c].cpu().numpy()
        arguments = (f,pose['probe_R'][patient,frame,view],pose['probe_t'][patient,frame,view]*model.norm_to_mm,
                     spacing[patient,frame,view],masks.shape[-2:])
        pm = exact_section_mask(p[patient,frame,left:right],*arguments)
        gm = exact_section_mask(g[patient,frame,left:right],*arguments)
        target = masks[patient,frame,view,chamber]
        def dice(a,b):
            mass = int(a.sum())+int(b.sum())
            return float(2*(a.astype(bool)&b.astype(bool)).sum()/mass) if mass else 1.
        rows.append(dict(patient=int(patient),frame=int(frame),view=VIEWS[view],chamber=c,
            exact_dice=dice(pm,target),paired_exact_dice=dice(pm,gm),gt_saved_mask_dice=dice(gm,target),
            **boundary_metrics(pm,target,spacing[patient,frame,view])))
    values = np.array([r['exact_dice'] for r in rows])
    if not len(values):
        raise ValueError('No valid exact evaluation cells')
    summary = dict(cells=len(rows),mean_exact_dice=float(values.mean()),p05_exact_dice=float(np.quantile(values,.05)),
                   fraction_dice_ge_095=float((values>=.95).mean()),fraction_dice_ge_098=float((values>=.98).mean()),
                   minimum_gt_saved_mask_dice=min(r['gt_saved_mask_dice'] for r in rows))
    def group(selected):
        scores = np.array([r['exact_dice'] for r in selected])
        valid_distances = [r for r in selected if r['boundary_assd_mm'] is not None]
        return dict(cells=len(selected),mean_exact_dice=float(scores.mean()),
            p05_exact_dice=float(np.quantile(scores,.05)),fraction_dice_ge_095=float((scores>=.95).mean()),
            fraction_dice_ge_098=float((scores>=.98).mean()),
            mean_boundary_assd_mm=float(np.mean([r['boundary_assd_mm'] for r in valid_distances])) if valid_distances else None,
            mean_boundary_hd95_mm=float(np.mean([r['boundary_hd95_mm'] for r in valid_distances])) if valid_distances else None,
            missing_one_boundary_cells=sum(r['empty_boundary']=='one' for r in selected))
    summary['per_view'] = {v:group([r for r in rows if r['view']==v]) for v in VIEWS if any(r['view']==v for r in rows)}
    summary['per_chamber'] = {c:group([r for r in rows if r['chamber']==c]) for c in CHAMBERS if any(r['chamber']==c for r in rows)}
    return dict(summary=summary,cells=rows)


@torch.no_grad()
def evaluate(model,dataset,device,*,cases,exact=False):
    model.eval()
    rows = []
    for index in range(min(cases,len(dataset))):
        batch = to_device(next(iter(DataLoader(Subset(dataset,[index]),batch_size=1))),device)
        output = model(batch)
        metrics = reconstruction_metrics(output['vertices_4d'],batch['vertices_gt'],model.chamber_ranges,model.faces)
        item = dict(case_id=batch['case_id'][0],metrics=metrics)
        if exact:
            item['exact'] = exact_metrics(model,output['vertices_4d'],batch)
            for c,(left,right) in zip(CHAMBERS,model.chamber_ranges.tolist()):
                metrics[c].update(surface_metrics(output['vertices_ref'][0,left:right].cpu().numpy(),
                    batch['vertices_gt'][0,0,left:right].cpu().numpy(),model.faces[c].cpu().numpy()))
        rows.append(item)
    return dict(cases=rows,case_count=len(rows),manifest_sha256=dataset.manifest_sha256,
                reference_semantics='synthetic source frame 0; not independently selected clinical ED',
                motion_semantics='24 independent phases, sequential increments; closing gap is a true 1/24 step')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/neural_ffd.yaml')
    parser.add_argument('--device')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--val-cases',type=int)
    parser.add_argument('--stage',choices=('shape','motion','joint'))
    parser.add_argument('--evaluate',type=Path)
    parser.add_argument('--split',choices=('val','test'),default='val')
    parser.add_argument('--exact',action='store_true')
    args = parser.parse_args()
    config,data_config = load_config(args.config)
    training = config['training']
    device = args.device or training['device']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(training['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(training['seed']);np.random.seed(training['seed']);torch.manual_seed(training['seed'])
    for value in (args.max_steps,args.max_cases,args.val_cases):
        if value is not None and value<1:
            raise ValueError('Explicit check limits must be positive')
    output_dir = args.output or repository_path(training['output'])
    output_dir.mkdir(parents=True,exist_ok=True)
    validation = dataset_from_config(data_config,args.split if args.evaluate else 'val')
    if args.evaluate:
        model,saved = load_checkpoint(args.evaluate,device)
        if (model.template_sha256!=validation.template_sha256 or model.asset_sha256!=validation.asset_sha256
                or model.norm_to_mm!=validation.norm_to_mm):
            raise ValueError('Evaluation dataset differs from checkpoint geometry')
        report = evaluate(model,validation,device,cases=args.val_cases or training['validation_cases'],exact=args.exact)
        report.update(checkpoint_sha256=_sha256(args.evaluate),split=args.split)
        (output_dir/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('Evaluation saved:',output_dir/'evaluation.json',flush=True)
        return
    train = dataset_from_config(data_config)
    identity = data_identity(data_config)
    runtime = dict(max_cases=args.max_cases,stage_override=args.stage)
    epoch,offset,step = 0,0,0
    if args.resume:
        model,saved = load_checkpoint(args.resume,device)
        if saved['config']!=config or saved['data_identity']!=identity or saved['runtime']!=runtime:
            raise ValueError('Strict resume requires identical config, dataset and training subset/stage')
        epoch,offset,step = saved['epoch'],saved['sample_offset'],saved['step']
    else:
        model = build_model(train,config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=training['learning_rate'])
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved,device)
    criterion = NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(device)
    slicer = MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,**config['slice']).to(device)
    count = min(args.max_cases or len(train),len(train))
    if offset<0 or offset>=count or epoch<0 or epoch>training['epochs']:
        raise ValueError('Checkpoint training position outside selected dataset')
    with (output_dir/'history.jsonl').open('a') as history:
        while epoch<training['epochs']:
            stage = args.stage or stage_for_epoch(epoch,config)
            generator = torch.Generator().manual_seed(training['seed']+epoch)
            order = torch.randperm(count,generator=generator).tolist()
            loader = DataLoader(Subset(train,order[offset:]),batch_size=training['batch_size'],
                                num_workers=training['workers'],generator=generator,
                                multiprocessing_context='spawn' if training['workers'] else None)
            model.train()
            for batch in loader:
                start = time.perf_counter()
                batch = to_device(batch,device)
                optimizer.zero_grad(set_to_none=True)
                decoded,losses,slices = compute_objective(model,batch,criterion,slicer,stage)
                losses['total'].backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite network gradient')
                optimizer.step()
                offset += batch['masks'].shape[0]
                step += 1
                record = dict(epoch=epoch,step=step,sample_offset=offset,stage=stage,total=float(losses['total']),
                    terms={key:float(value.detach()) for key,value in losses['terms'].items()},
                    active_weights=losses['active_weights'],units=losses['units'],seconds=time.perf_counter()-start)
                if slices is not None:
                    record['slice_diagnostics'] = slices['diagnostics']
                history.write(json.dumps(record,allow_nan=False)+'\n');history.flush()
                print(f"epoch={epoch} step={step} stage={stage} loss={record['total']:.6f} seconds={record['seconds']:.2f}",flush=True)
                next_epoch,next_offset = (epoch+1,0) if offset==count else (epoch,offset)
                stopping = args.max_steps is not None and step>=args.max_steps
                if stopping or step%training['checkpoint_every_steps']==0:
                    save_checkpoint(output_dir/'last.pth',model,optimizer,config=config,identity=identity,
                        epoch=next_epoch,sample_offset=next_offset,step=step,runtime=runtime)
                if stopping:
                    report = evaluate(model,validation,device,cases=args.val_cases or training['validation_cases'],exact=args.exact)
                    report.update(scope='limited engineering run; no formal performance claim',step=step)
                    (output_dir/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                    return
            epoch,offset = epoch+1,0
            save_checkpoint(output_dir/'last.pth',model,optimizer,config=config,identity=identity,
                            epoch=epoch,sample_offset=offset,step=step,runtime=runtime)
            report = evaluate(model,validation,device,cases=args.val_cases or training['validation_cases'],exact=args.exact)
            report.update(epoch=epoch,step=step)
            (output_dir/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
