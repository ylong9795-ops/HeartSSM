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
import math
from collections import deque
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate
import yaml
from tqdm.auto import tqdm

from data.mask_transform import CHAMBERS,VIEWS,MaskTransform
from data.synthetic_4d import Synthetic4DDataset,_sha256
from net.neural_ffdnet import NeuralFFDNet,ARCHITECTURE,geometry_digest
from net.multiview_encoder import observation_validity,EncoderDiagnostics
from utils.neural_ffd_losses import NeuralFFDLoss,reconstruction_metrics
from utils.slice_supervision import MeshSliceSupervisor
from utils.slice_evaluation import render_source_slice,source_renderer_identity
from utils.validation_runtime import ValidationRuntime,reference_signature
from utils.training_execution import configure_execution,StepProfiler

ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 'neural-heartffd-checkpoint-v1'
NATIVE_JOINT_CRITERIA = dict(dice_min=.98, boundary_assd_max_mm=.5, boundary_hd95_max_mm=1.)


def repository_path(value):
    return ROOT/Path(value)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    unknown = set(config)-{'data_config','encoder','decoder','slice','loss','training'}
    if unknown:
        raise ValueError('Unknown configuration sections: '+str(sorted(unknown)))
    training = config['training']
    unknown = set(training)-{'device','seed','epochs','shape_epochs','motion_epochs','batch_size',
        'learning_rate','workers','cpu_threads','validation_cases','checkpoint_every_steps',
        'output','joint_min_learning_rate','save_best','freeze'}
    if unknown:
        raise ValueError('Unknown training settings: '+str(sorted(unknown)))
    if training.get('freeze','none') not in ('none','dynamic_only','motion_head_only'):
        raise ValueError('Unknown explicit training freeze scope')
    data_config = yaml.safe_load(repository_path(config['data_config']).read_text())
    data = data_config['data']
    if (tuple(data['views'])!=VIEWS or tuple(data['chambers'])!=CHAMBERS
            or data['phase_semantics']!='relative_observed_cycle_time'
            or not data['remove_duplicate_endpoint']
            or data['expected_phases']!=config['decoder']['frames']
            or data_config['mask']['sdf_normalize'] or not data_config['mask']['use_sdf']):
        raise ValueError('Canonical full-cycle masks, physical SDF and synchronized fixed decoder phases required')
    if (min(training[k] for k in ('epochs','batch_size','cpu_threads','validation_cases','checkpoint_every_steps'))<1
            or min(training['shape_epochs'],training['motion_epochs'],training['workers'])<0
            or training['shape_epochs']+training['motion_epochs']>=training['epochs']):
        raise ValueError('Positive training dimensions and at least one final joint epoch required')
    if (not math.isfinite(training['learning_rate']) or training['learning_rate']<=0
            or ('joint_min_learning_rate' in training and
                not 0<training['joint_min_learning_rate']<=training['learning_rate'])):
        raise ValueError('Positive learning rate and joint minimum no greater than base rate required')
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


def to_device(value,device,non_blocking=False):
    if isinstance(value,torch.Tensor):
        return value.to(device,non_blocking=non_blocking)
    if isinstance(value,dict):
        return {key:to_device(item,device,non_blocking) for key,item in value.items()}
    return value


def training_collate(samples,stage):
    """Exclude evaluation-only arrays before IPC and pinned-memory copies."""
    unused = {'native_masks','model_to_native','native_pixel_spacing','clipped_foreground'}
    if stage == 'joint':
        # Paired boundary supervision queries the source physical grid.
        unused.difference_update(('native_masks', 'native_pixel_spacing'))
    # Physical acquisition geometry is needed only for joint slice supervision.
    if stage != 'joint':
        unused.update(('pose_gt','pixel_spacing','norm_to_mm'))
    return default_collate([{key:value for key,value in sample.items() if key not in unused}
                            for sample in samples])


def training_loader(dataset,order,*,batch_size,workers,prefetch,device,generator,stage):
    options = dict(batch_size=batch_size,num_workers=workers,generator=generator,
                   collate_fn=partial(training_collate,stage=stage),
                   pin_memory=torch.device(device).type=='cuda')
    if workers:
        options.update(multiprocessing_context='spawn',prefetch_factor=prefetch)
    return DataLoader(Subset(dataset,order),**options)


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
        initialization=getattr(model,'initialization_metadata',None),
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
    return model_from_checkpoint(saved,device),saved


def model_from_checkpoint(saved,device='cpu'):
    """Rebuild the original model from an embedded, validated checkpoint."""
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
    if model.regional_motion:
        fixed += ('regional_motion_lower_mm','regional_motion_upper_mm',
                  'regional_motion_origins_mm','regional_motion_spacings_mm')
    for key in fixed:
        if not torch.equal(model.state_dict()[key],state[key]):
            raise ValueError('Checkpoint fixed control/topology buffer differs: '+key)
    model.load_state_dict(state,strict=True)
    model.initialization_metadata = saved.get('initialization')
    return model.to(device)


def initialize_from_checkpoint(model,path,identity):
    """Explicit inherited weights with a precise new-module allowlist.

    Never use strict=False. Reject missing old weights, altered geometry,
    encoder dimensions or decoder settings before constructing a complete state.
    Optimizer and update counters start afresh for the matched-budget experiment.
    """
    source,saved = load_checkpoint(path)
    if saved['data_identity'] != identity or source.geometry_sha256 != model.geometry_sha256:
        raise ValueError('Initialization requires identical source data and fixed topology')
    old_encoder = dict(source.encoder_config)
    new_encoder = dict(model.encoder_config)
    old_decoder,new_decoder = dict(source.decoder_config),dict(model.decoder_config)
    if model.regional_motion and not source.regional_motion:
        for settings in (old_decoder,new_decoder):
            for key in tuple(settings):
                if key.startswith('regional_motion'):
                    settings.pop(key)
    if model.dynamic_conditioner is not None and source.dynamic_conditioner is None:
        for settings in (old_decoder,new_decoder):
            for key in tuple(settings):
                if key=='dynamic_spatial' or key.startswith('dynamic_spatial_'):
                    settings.pop(key)
    if old_encoder != new_encoder or old_decoder != new_decoder:
        raise ValueError('Initialization changes existing encoder/decoder configuration')
    current,previous = model.state_dict(),source.state_dict()
    allowed = set()
    if model.regional_motion and not source.regional_motion:
        allowed.update(key for key in current if key.startswith('regional_motion_heads.') or key in
            ('regional_motion_lower_mm','regional_motion_upper_mm','regional_motion_origins_mm','regional_motion_spacings_mm'))
    if model.dynamic_conditioner is not None and source.dynamic_conditioner is None:
        allowed.update(key for key in current if key.startswith('dynamic_conditioner.'))
    if set(current)-set(previous) != allowed or set(previous)-set(current):
        raise ValueError('Initialization weight changes differ from explicit new-module allowlist')
    for key,value in previous.items():
        if current[key].shape != value.shape or current[key].dtype != value.dtype:
            raise ValueError('Incompatible inherited weight: '+key)
        current[key] = value
    model.load_state_dict(current,strict=True)
    model.initialization_metadata = dict(checkpoint_sha256=_sha256(path),source_epoch=saved['epoch'],
        source_step=saved['step'],inherited_keys=sorted(previous),new_keys=sorted(allowed),
        optimizer='new Adam; update counter excludes inherited training')
    return model.initialization_metadata


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


def learning_rate_for_epoch(epoch,config,stage):
    """Constant warmup rates, optional joint cosine decay; resume needs no state."""
    training = config['training']
    base = training['learning_rate']
    minimum = training.get('joint_min_learning_rate')
    if stage!='joint' or minimum is None:
        return base
    start = training['shape_epochs']+training['motion_epochs']
    duration = training['epochs']-start
    fraction = min(1.,max(0.,(epoch-start)/max(1,duration-1)))
    return minimum+(base-minimum)*(1+math.cos(math.pi*fraction))/2


def configure_trainable(model,training):
    """Explicit Goal03 activation scope, also applied before strict resume."""
    scope=training.get('freeze','none')
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(scope=='none' or
            (scope=='dynamic_only' and name.startswith('dynamic_conditioner.')) or
            (scope=='motion_head_only' and name.startswith('deform3.') and not name.startswith('deform3.reference.')))
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze scope leaves no trainable parameters')


def make_optimizer(model,training):
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=training['learning_rate'])


def best_checkpoint_candidates(summary):
    """Select paired shape/motion errors before they can cancel in trajectories."""
    if not summary['complete']:
        return {}
    metrics,stage = summary['validation'],summary['stage']
    reference,motion = metrics['reference_vertex_rmse_mm'],metrics['relative_motion_rmse_mm']
    score = reference if stage=='shape' else math.hypot(reference,motion)
    candidates = {'best_'+stage:dict(score=score,criterion=(
        'reference_vertex_rmse_mm' if stage=='shape' else 'sqrt(reference_rmse_mm^2+relative_motion_rmse_mm^2)'))}
    if stage=='joint' and metrics['mean_exact_dice'] is not None:
        candidates['best_dice'] = dict(score=-metrics['mean_exact_dice'],criterion='negative_mean_exact_dice')
    return candidates


def compute_objective(model,batch,criterion,slicer,stage):
    with torch.profiler.record_function('joint/base_forward'):
        output = model(batch)
    with torch.profiler.record_function('joint/slice_supervision'):
        slices = slicer(output['vertices_4d'],batch) if stage=='joint' else None
    with torch.profiler.record_function('joint/gt_and_geometry_loss'):
        losses = criterion(output,batch['vertices_gt'],slice_result=slices,
            control_spacings_mm=model.control_spacings_mm,
            regional_spacing_mm=model.regional_spacing_mm if model.regional_shape else None,
            regional_motion_spacings_mm=model.regional_motion_spacings_mm if model.regional_motion else None,stage=stage)
    return output,losses,slices


def regional_motion_diagnostics(output):
    """Per-phase/chamber mm statistics; no independent vertex motion interface."""
    grids = output['control_grids']['regional_motion'].detach()
    return dict(chambers=list(CHAMBERS),
        increment_displacement_rms_mm=output['regional_motion_rms_mm'].detach().cpu().tolist(),
        control_rms_mm=grids.square().mean((3,4,5,6)).sqrt().cpu().tolist(),
        adjacent_control_change_rms_mm=grids.diff(dim=1).square().mean((3,4,5,6)).sqrt().cpu().tolist(),
        first_last_increment_control_change_rms_mm=(grids[:,1]-grids[:,-1]).square().mean((2,3,4,5)).sqrt().cpu().tolist(),
        reference_control='exact zero; reference Shape is independent of regional Motion')


def surface_metrics(predicted,truth,faces,samples=5000,reference_cache=None):
    """Deterministic sampled triangle-surface ASSD/HD95; no voxel conversion."""
    import trimesh
    from scipy.spatial import cKDTree
    p = trimesh.sample.sample_surface(trimesh.Trimesh(predicted,faces,process=False),samples,seed=2026)[0]
    key = reference_signature((truth,faces,np.asarray(samples)))
    cached = None if reference_cache is None else reference_cache.get(key)
    if cached is None:
        g = trimesh.sample.sample_surface(trimesh.Trimesh(truth,faces,process=False),samples,seed=2026)[0]
        cached = (g,cKDTree(g))
        if reference_cache is not None:
            reference_cache[key] = cached
    g,tree = cached
    distances = np.concatenate((tree.query(p)[0],cKDTree(p).query(g)[0]))
    return dict(sampled_surface_assd_mm=float(distances.mean()),sampled_surface_hd95_mm=float(np.quantile(distances,.95)),surface_samples=samples)


def boundary_reference(truth,spacing):
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    mask = truth.astype(bool)
    points = np.argwhere(mask & ~binary_erosion(mask)).astype(float)*spacing
    return points,cKDTree(points) if len(points) else None


def boundary_metrics(prediction,truth,spacing,reference=None):
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    mask = prediction.astype(bool)
    p = np.argwhere(mask & ~binary_erosion(mask)).astype(float)*spacing
    g,tree = boundary_reference(truth,spacing) if reference is None else reference
    if not len(p) and not len(g):
        return dict(boundary_assd_mm=0.,boundary_hd95_mm=0.,empty_boundary='both')
    if not len(p) or not len(g):
        return dict(boundary_assd_mm=None,boundary_hd95_mm=None,empty_boundary='one')
    distances = np.concatenate((tree.query(p)[0],cKDTree(p).query(g)[0]))
    return dict(boundary_assd_mm=float(distances.mean()),boundary_hd95_mm=float(np.quantile(distances,.95)),empty_boundary=None)


def native_cell_criteria(exact_dice, boundary):
    """Predeclared Exact goals; boundary flags are diagnostics, not rejection.

    Empty/empty has Dice=1 but no boundary to evaluate. One absent boundary
    is a failed boundary diagnostic, never a zero-distance success.
    """
    empty = boundary['empty_boundary']
    if empty == 'both':
        assd_pass = hd95_pass = None
    elif empty == 'one':
        assd_pass = hd95_pass = False
    else:
        assd_pass = bool(boundary['boundary_assd_mm'] <= NATIVE_JOINT_CRITERIA['boundary_assd_max_mm'])
        hd95_pass = bool(boundary['boundary_hd95_mm'] <= NATIVE_JOINT_CRITERIA['boundary_hd95_max_mm'])
    return dict(native_dice_pass=bool(exact_dice >= NATIVE_JOINT_CRITERIA['dice_min']),
                native_assd_diagnostic_pass=assd_pass, native_hd95_diagnostic_pass=hd95_pass,
                native_boundary_diagnostic_pass=(None if assd_pass is None else assd_pass and hd95_pass))


def exact_metrics(model,predicted,batch,reference_cache=None):
    """Predicted mesh + original pose -> source renderer -> original-label Dice.

    Rerendered GT must match the saved original native labels pixel-for-pixel.
    Network-input resizing and canonical-FOV warps do not enter evaluation.
    """
    p = predicted.detach().cpu().numpy()
    g = batch['vertices_gt'].cpu().numpy()
    pose = {k:v.cpu().numpy() for k,v in batch['pose_gt'].items()}
    spacing = batch['native_pixel_spacing'].cpu().numpy()
    native_masks = batch['native_masks'].cpu().numpy()
    if (native_masks.shape[:4] != (len(p),p.shape[1],len(VIEWS),len(CHAMBERS))
            or spacing.shape != (*native_masks.shape[:3],2)):
        raise ValueError('Original native labels and native pixel spacing required for exact evaluation')
    valid = observation_validity(batch,len(p),p.shape[1],predicted.device).cpu().numpy()
    faces = {c:model.faces[c].cpu().numpy() for c in CHAMBERS}
    renderer = source_renderer_identity()
    verified = {}
    if reference_cache is not None:
        signature = reference_signature((g,native_masks,spacing,valid,model.chamber_ranges.cpu().numpy(),
            np.asarray(model.norm_to_mm),*pose.values(),*faces.values()),renderer)
        if reference_cache.get('signature')!=signature:
            reference_cache.clear()
            reference_cache.update(signature=signature,cells={})
        verified = reference_cache['cells']
    rows = []
    for patient,frame,view,chamber in np.argwhere(valid):
        c = CHAMBERS[chamber]
        left,right = model.chamber_ranges[chamber].tolist()
        f = faces[c]
        arguments = (f,pose['probe_R'][patient,frame,view],pose['probe_t'][patient,frame,view],
                     pose['probe_s'][patient,frame,view],model.norm_to_mm,native_masks.shape[-2:])
        pm = render_source_slice(p[patient,frame,left:right],*arguments)
        target = native_masks[patient,frame,view,chamber]
        cell = (int(patient),int(frame),int(view),int(chamber))
        reference = verified.get(cell)
        if reference is None:
            gm = render_source_slice(g[patient,frame,left:right],*arguments)
            native_mismatch = int(np.count_nonzero(gm != target))
            if native_mismatch:
                identities = batch.get('case_id',())
                case_id = identities[patient] if identities else 'unknown'
                raise ValueError(f'GT renderer/label mismatch: case_id={case_id}, patient={patient}, frame={frame}, '
                                 f'view={VIEWS[view]}, chamber={c}, native_pixels={native_mismatch}; '
                                 'exact Dice evaluation rejected')
            if reference_cache is not None:
                reference = boundary_reference(target,spacing[patient,frame,view])
                verified[cell] = reference
        def dice(a,b):
            mass = int(a.sum())+int(b.sum())
            return float(2*(a.astype(bool)&b.astype(bool)).sum()/mass) if mass else 1.
        exact_dice = dice(pm,target)
        boundary = boundary_metrics(pm,target,spacing[patient,frame,view],reference)
        rows.append(dict(patient=int(patient),frame=int(frame),view=VIEWS[view],chamber=c,
            exact_dice=exact_dice,paired_exact_dice=exact_dice,gt_saved_mask_dice=1.,
            gt_native_pixel_mismatches=0, **boundary, **native_cell_criteria(exact_dice, boundary)))
    values = np.array([r['exact_dice'] for r in rows])
    if not len(values):
        raise ValueError('No valid exact evaluation cells')
    summary = dict(cells=len(rows),mean_exact_dice=float(values.mean()),p05_exact_dice=float(np.quantile(values,.05)),
                   minimum_exact_dice=float(values.min()), all_native_dice_pass=bool((values>=.98).all()),
                   fraction_dice_ge_095=float((values>=.95).mean()),fraction_dice_ge_098=float((values>=.98).mean()),
                   native_dice_pass_cells=sum(r['native_dice_pass'] for r in rows),
                   native_dice_fail_cells=sum(not r['native_dice_pass'] for r in rows),
                   native_boundary_diagnostic_pass_cells=sum(r['native_boundary_diagnostic_pass'] is True for r in rows),
                   native_boundary_diagnostic_fail_cells=sum(r['native_boundary_diagnostic_pass'] is False for r in rows),
                   native_boundary_not_applicable_cells=sum(r['native_boundary_diagnostic_pass'] is None for r in rows),
                   minimum_gt_saved_mask_dice=min(r['gt_saved_mask_dice'] for r in rows),
                   gt_native_pixel_mismatches=sum(r['gt_native_pixel_mismatches'] for r in rows))
    def group(selected):
        scores = np.array([r['exact_dice'] for r in selected])
        valid_distances = [r for r in selected if r['boundary_assd_mm'] is not None]
        return dict(cells=len(selected),mean_exact_dice=float(scores.mean()),
            minimum_exact_dice=float(scores.min()), all_native_dice_pass=bool((scores>=.98).all()),
            native_dice_fail_cells=int((scores<.98).sum()),
            p05_exact_dice=float(np.quantile(scores,.05)),fraction_dice_ge_095=float((scores>=.95).mean()),
            fraction_dice_ge_098=float((scores>=.98).mean()),
            mean_boundary_assd_mm=float(np.mean([r['boundary_assd_mm'] for r in valid_distances])) if valid_distances else None,
            mean_boundary_hd95_mm=float(np.mean([r['boundary_hd95_mm'] for r in valid_distances])) if valid_distances else None,
            missing_one_boundary_cells=sum(r['empty_boundary']=='one' for r in selected))
    summary['per_view'] = {v:group([r for r in rows if r['view']==v]) for v in VIEWS if any(r['view']==v for r in rows)}
    summary['per_chamber'] = {c:group([r for r in rows if r['chamber']==c]) for c in CHAMBERS if any(r['chamber']==c for r in rows)}
    return dict(renderer=renderer,summary=summary,cells=rows,
                criteria=dict(NATIVE_JOINT_CRITERIA, boundary_role='diagnostic_only', proxy_threshold='disabled_pending_TRAIN_calibration'))


def validation_summary(report):
    """Pool squared errors with equal case/chamber weight; Dice by valid cell."""
    metrics = [case['metrics'][c] for case in report['cases'] for c in CHAMBERS]
    summary = {key:float(np.sqrt(np.mean([item[key]**2 for item in metrics]))) for key in
               ('reference_vertex_rmse_mm','trajectory_vertex_rmse_mm','relative_motion_rmse_mm')}
    summary['volume_mae_ml'] = float(np.mean([item['volume_mae_ml'] for item in metrics]))
    exact = [case['exact']['summary'] for case in report['cases'] if 'exact' in case]
    summary['mean_exact_dice'] = (sum(item['cells']*item['mean_exact_dice'] for item in exact)
                                  /sum(item['cells'] for item in exact)) if exact else None
    exact_cells = [cell for case in report['cases'] if 'exact' in case for cell in case['exact']['cells']]
    exact_scores = [cell['exact_dice'] for cell in exact_cells]
    summary.update(native_exact_cells=len(exact_cells),
        minimum_exact_dice=min(exact_scores) if exact_scores else None,
        native_dice_fail_cells=sum(score < .98 for score in exact_scores) if exact_scores else None,
        all_native_dice_pass=all(score >= .98 for score in exact_scores) if exact_scores else None)
    reference_cells = [cell for case in report['cases'] if 'exact' in case
                       for cell in case['exact']['cells'] if cell['frame']==0]
    summary['reference_exact_dice'] = (float(np.mean([cell['exact_dice'] for cell in reference_cells]))
                                      if reference_cells else None)
    summary['validation_cases'] = report['case_count']
    return summary


def recover_epoch_loss(path,epoch,offset):
    """Recover completed batches up to the checkpoint, ignoring later history.

    A resumed run may append repeated offsets; use its latest record per offset.
    Legacy histories have no batch_cases field, so infer it from adjacent offsets.
    """
    records = {}
    if offset and Path(path).exists():
        with Path(path).open() as stream:
            for line in stream:
                record = json.loads(line)
                if record['epoch']==epoch and 0<record['sample_offset']<=offset:
                    records[record['sample_offset']] = record
    total,samples,previous = 0.,0,0
    for end,record in sorted(records.items()):
        size = record.get('batch_cases',end-previous)
        total += record['total']*size
        samples += size
        previous = end
    return total,samples


def format_epoch_summary(record,epochs):
    metrics = record['validation']
    dice = 'n/a' if metrics['mean_exact_dice'] is None else f"{metrics['mean_exact_dice']:.4f}"
    partial = ' (partial)' if not record['complete'] else ''
    return (f"Epoch {record['epoch']}/{epochs} [{record['stage']}]{partial} | "
            f"Train loss={record['train_mean_loss']:.4f} | "
            f"Val Shape RMSE={metrics['reference_vertex_rmse_mm']:.3f} mm | "
            f"Motion RMSE={metrics['relative_motion_rmse_mm']:.3f} mm | Dice={dice}")


@torch.no_grad()
def evaluate(model,dataset,device,*,cases,exact=False,progress=False,runtime=None):
    model.eval()
    with EncoderDiagnostics(model) as diagnostics:
        if runtime is not None:
            report = evaluate_with_runtime(model,dataset,device,cases=cases,exact=exact,progress=progress,runtime=runtime)
        else:
            report = evaluate_serial(model,dataset,device,cases=cases,exact=exact,progress=progress)
        summary = diagnostics.summary()
        if summary is not None:
            report['encoder_diagnostics'] = summary
        return report


def evaluate_serial(model,dataset,device,*,cases,exact=False,progress=False):
    rows = []
    for index in tqdm(range(min(cases,len(dataset))),desc='Validation',leave=False,
                      dynamic_ncols=True,mininterval=1.,disable=not progress):
        batch = to_device(next(iter(DataLoader(Subset(dataset,[index]),batch_size=1))),device)
        output = model(batch)
        metrics = reconstruction_metrics(output['vertices_4d'],batch['vertices_gt'],model.chamber_ranges,model.faces)
        item = dict(case_id=batch['case_id'][0],metrics=metrics)
        if model.regional_motion:
            item['regional_motion'] = regional_motion_diagnostics(output)
        if exact:
            item['exact'] = exact_metrics(model,output['vertices_4d'],batch)
            for c,(left,right) in zip(CHAMBERS,model.chamber_ranges.tolist()):
                metrics[c].update(surface_metrics(output['vertices_ref'][0,left:right].cpu().numpy(),
                    batch['vertices_gt'][0,0,left:right].cpu().numpy(),model.faces[c].cpu().numpy()))
        rows.append(item)
    return dict(cases=rows,case_count=len(rows),manifest_sha256=dataset.manifest_sha256,
                reference_semantics='synthetic source frame 0; not independently selected clinical ED',
                motion_semantics='24 independent phases, sequential increments; closing gap is a true 1/24 step')


def evaluate_with_runtime(model,dataset,device,*,cases,exact,progress,runtime):
    start = time.perf_counter()
    runtime.bind(model,dataset)
    count = min(cases,len(dataset))
    rows,pending = [],deque()
    counters = dict(gt_cells_verified=0,gt_cells_reused=0)
    compute_seconds,wait_seconds = 0.,0.
    network_keys = ('masks','sdf','phases','view_present','annotated')
    with tqdm(total=count,desc='Validation',leave=False,dynamic_ncols=True,
              mininterval=1.,disable=not progress) as bar:
        def finish():
            nonlocal wait_seconds
            item,future = pending.popleft()
            t0 = time.perf_counter()
            result = future.result()
            wait_seconds += time.perf_counter()-t0
            item['exact'] = result['exact']
            for c in CHAMBERS:
                item['metrics'][c].update(result['surfaces'][c])
            for key in counters:
                counters[key] += result[key]
            rows.append(item)
            bar.update()
        for index,(batch,source,key) in runtime.batches(count):
            t0 = time.perf_counter()
            # Pose and native labels stay on CPU for source-native evaluation.
            observations = {key:to_device(value,device) if key in network_keys else value
                            for key,value in batch.items()}
            output = model(observations)
            metrics = reconstruction_metrics(output['vertices_4d'],batch['vertices_gt'],model.chamber_ranges,model.faces)
            item = dict(case_id=batch['case_id'][0],metrics=metrics)
            if model.regional_motion:
                item['regional_motion'] = regional_motion_diagnostics(output)
            if exact:
                prediction = output['vertices_4d'].detach().cpu().numpy()
                compute_seconds += time.perf_counter()-t0
                pending.append((item,runtime.submit(index,prediction,source,key)))
                if len(pending)>=runtime.max_pending:
                    finish()
            else:
                compute_seconds += time.perf_counter()-t0
                rows.append(item)
                bar.update()
            del output,observations
        while pending:
            finish()
    return dict(cases=rows,case_count=len(rows),manifest_sha256=dataset.manifest_sha256,
        reference_semantics='synthetic source frame 0; not independently selected clinical ED',
        motion_semantics='24 independent phases, sequential increments; closing gap is a true 1/24 step',
        performance=dict(seconds=time.perf_counter()-start,workers=runtime.workers,cache_cases=runtime.cache_cases,
            input_cache_hits=runtime.input_hits,input_cache_misses=runtime.input_misses,
            inference_and_3d_metrics_seconds=compute_seconds,cpu_wait_seconds=wait_seconds,**counters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/neural_ffd.yaml')
    parser.add_argument('--device')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--initialize',type=Path,help='Audited inherited weights for matched-budget experiments; new optimizer')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--val-cases',type=int)
    parser.add_argument('--stage',choices=('shape','motion','joint'))
    parser.add_argument('--evaluate',type=Path)
    parser.add_argument('--split',choices=('val','test'),default='val')
    parser.add_argument('--exact',action='store_true')
    parser.add_argument('--eval-workers',type=int,default=4,
                        help='Spawned original-renderer CPU workers; 0 runs inline with caching')
    parser.add_argument('--eval-cache-cases',type=int,default=32,
                        help='Maximum cached preprocessed validation cases in host memory')
    parser.add_argument('--train-workers',type=int,
                        help='Override training reader workers without changing checkpoint configuration')
    parser.add_argument('--train-prefetch',type=int,default=2,
                        help='Prefetched batches per training reader worker')
    parser.add_argument('--slice-backend',choices=('serial','batched'),default='batched',
                        help='Equivalent differentiable slice execution; batched groups cells on GPU')
    parser.add_argument('--slice-cell-chunk',type=int,default=None,
                        help='Maximum section cells per batched slice group; controls temporary GPU memory')
    parser.add_argument('--memory-profile',choices=('default','throughput'),default='default',
                        help='Throughput: larger CNN/slice batches, no CNN recomputation, equivalent boundary-only objective')
    parser.add_argument('--cnn-chunk-size',type=int)
    parser.add_argument('--checkpoint-cnn',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--slice-query-chunk',type=int)
    parser.add_argument('--boundary-only',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--batch-size',type=int,help='Explicit optimisation batch change; recorded separately from execution tuning')
    parser.add_argument('--profile-steps',type=int,default=0,help='Bounded CPU/CUDA profiler trace; zero disables')
    args = parser.parse_args()
    if sum(value is not None for value in (args.resume,args.initialize,args.evaluate)) > 1:
        parser.error('--resume, --initialize and --evaluate are mutually exclusive')
    if args.eval_workers<0 or args.eval_cache_cases<1:
        parser.error('Evaluation workers must be nonnegative and cache size positive')
    if (args.train_workers is not None and args.train_workers<0) or args.train_prefetch<1:
        parser.error('Training reader workers must be nonnegative and prefetch positive')
    if any(v is not None and v<1 for v in (args.slice_cell_chunk,args.cnn_chunk_size,args.slice_query_chunk,args.batch_size)) or args.profile_steps<0:
        parser.error('Execution chunks/batch must be positive; profile steps nonnegative')
    config,data_config = load_config(args.config)
    training = config['training']
    batch_size = args.batch_size or training['batch_size']
    reader_workers = training['workers'] if args.train_workers is None else args.train_workers
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
        with ValidationRuntime(args.eval_workers,args.eval_cache_cases) as evaluator:
            report = evaluate(model,validation,device,cases=args.val_cases or training['validation_cases'],
                              exact=args.exact,progress=True,runtime=evaluator)
        report.update(checkpoint_sha256=_sha256(args.evaluate),split=args.split,
                      summary=validation_summary(report))
        (output_dir/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        summary = report['summary']
        dice = f"{summary['mean_exact_dice']:.4f}" if summary['mean_exact_dice'] is not None else 'n/a'
        print(f"Evaluation [{args.split}] {report['case_count']} cases | "
              f"Shape RMSE {summary['reference_vertex_rmse_mm']:.3f} mm | "
              f"Motion RMSE {summary['relative_motion_rmse_mm']:.3f} mm | "
              f"Dice {dice} | Volume MAE {summary['volume_mae_ml']:.3f} mL",flush=True)
        print('Evaluation saved:',output_dir/'evaluation.json',flush=True)
        return
    train = dataset_from_config(data_config)
    identity = data_identity(data_config)
    runtime = dict(max_cases=args.max_cases,stage_override=args.stage)
    if batch_size != training['batch_size']:
        runtime['batch_size'] = batch_size
    epoch,offset,step = 0,0,0
    if args.resume:
        model,saved = load_checkpoint(args.resume,device)
        if saved['config']!=config or saved['data_identity']!=identity or saved['runtime']!=runtime:
            raise ValueError('Strict resume requires identical config, dataset and training subset/stage')
        epoch,offset,step = saved['epoch'],saved['sample_offset'],saved['step']
    else:
        model = build_model(train,config).to(device)
        if args.initialize:
            initialization = initialize_from_checkpoint(model,args.initialize,identity)
            (output_dir/'initialization.json').write_text(json.dumps(initialization,indent=2)+'\n')
    (output_dir/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    configure_trainable(model,training)
    optimizer = make_optimizer(model,training)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved,device)
    criterion = NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(device)
    slicer = MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,**config['slice'],
        execution_backend=args.slice_backend,cell_chunk_size=args.slice_cell_chunk or 16).to(device)
    execution = configure_execution(model,slicer,objective_mode=criterion.objective_mode,
        memory_profile=args.memory_profile,cnn_chunk_size=args.cnn_chunk_size,checkpoint_cnn=args.checkpoint_cnn,
        slice_cell_chunk=args.slice_cell_chunk,slice_query_chunk=args.slice_query_chunk,boundary_only=args.boundary_only)
    execution['batch_size'] = batch_size
    (output_dir/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    print('Training execution:',json.dumps(execution),flush=True)
    count = min(args.max_cases or len(train),len(train))
    if offset<0 or offset>=count or epoch<0 or epoch>training['epochs']:
        raise ValueError('Checkpoint training position outside selected dataset')
    best_path = output_dir/'best_metrics.json'
    best = json.loads(best_path.read_text()) if args.resume and best_path.exists() else {}
    best = {key:value for key,value in best.items() if value['step']<=step}
    with ValidationRuntime(args.eval_workers,args.eval_cache_cases) as evaluator, (output_dir/'history.jsonl').open('a') as history, StepProfiler(args.profile_steps,output_dir,device) as profiler:
        while epoch<training['epochs']:
            epoch_start = time.perf_counter()
            stage = args.stage or stage_for_epoch(epoch,config)
            if stage == 'joint':
                joint_start = 0 if args.stage == 'joint' else training['shape_epochs']+training['motion_epochs']
                progress = (epoch-joint_start)/max(1, training['epochs']-joint_start-1)
                slicer.set_joint_progress(min(1., max(0., progress)))
            rate = learning_rate_for_epoch(epoch,config,stage)
            for group in optimizer.param_groups:
                group['lr'] = rate
            generator = torch.Generator().manual_seed(training['seed']+epoch)
            order = torch.randperm(count,generator=generator).tolist()
            loader = training_loader(train,order[offset:],batch_size=batch_size,
                workers=reader_workers,prefetch=args.train_prefetch,device=device,generator=generator,stage=stage)
            model.train()
            if training.get('freeze','none')!='none':
                # Frozen GroupNorm CNN/Transformer use their inherited eval
                # numerical path; the new dynamic module stays in train mode.
                model.encoder.eval()
            loss_sum,loss_samples = recover_epoch_loss(output_dir/'history.jsonl',epoch,offset)
            stopping = False
            io_seconds,compute_seconds,processed_samples = 0.,0.,0
            data_wait_start = time.perf_counter()
            with tqdm(loader,total=math.ceil(count/batch_size),
                      initial=math.ceil(offset/batch_size),
                      desc=f"Epoch {epoch+1}/{training['epochs']} [{stage}]",
                      dynamic_ncols=True,mininterval=1.,leave=False) as progress:
                for batch in progress:
                    start = time.perf_counter()
                    io_wait = start-data_wait_start
                    io_seconds += io_wait
                    batch = to_device(batch,device,non_blocking=torch.device(device).type=='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    decoded,losses,slices = compute_objective(model,batch,criterion,slicer,stage)
                    with torch.profiler.record_function('joint/backward'):
                        losses['total'].backward()
                    finite_gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                    if finite_gradients and not torch.stack(finite_gradients).all():
                        raise ValueError('Nonfinite network gradient')
                    dynamic_probe=None
                    if model.dynamic_conditioner is not None and (step==0 or (step+1)%200==0):
                        # Same already encoded tensors and same pre-update
                        # weights. No second CNN and no optimizer action in the
                        # paired diagnostic. Recursive downstream effects remain.
                        with torch.no_grad():
                            off=model.decode_4d(decoded['z_shape'].detach(),decoded['z_motion'].detach(),return_details=True)
                            field_delta=decoded['control_grids']['fine_motion'][:,1:].detach()-off['control_grids']['fine_motion'][:,1:]
                            mesh_delta=decoded['vertices_4d'].detach()-off['vertices_4d']
                            center=field_delta-field_delta.mean((2,3,4),keepdim=True)
                            prediction=decoded['vertices_4d'].detach()
                            truth=batch['vertices_gt'].to(prediction)
                            motion_error=(prediction-prediction[:,:1])-(truth-truth[:,:1])
                            dynamic_probe=dict(measured_before_update=step+1,
                                fine_control_vector_rms_mm=float(field_delta.square().sum(-1).mean().sqrt()),
                                fine_control_centered_energy_fraction=float(center.square().sum()/(field_delta.square().sum()+1e-20)),
                                mesh_4d_vector_rms_mm=float(mesh_delta.square().sum(-1).mean().sqrt()),
                                reference_max_abs_mm=float(mesh_delta[:,0].abs().max()),
                                chamber_mesh_delta_rms_mm={c:float(mesh_delta[:,:,a:b].square().sum(-1).mean().sqrt()) for c,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist())},
                                chamber_motion_rmse_mm={c:float(motion_error[:,:,a:b].square().sum(-1).mean().sqrt()) for c,(a,b) in zip(CHAMBERS,model.chamber_ranges.tolist())})
                            del off,field_delta,mesh_delta,center,motion_error
                    with torch.profiler.record_function('joint/optimizer'):
                        optimizer.step()
                    batch_cases = batch['masks'].shape[0]
                    offset += batch_cases
                    step += 1
                    term_keys = list(losses['terms'])
                    objective_keys = ('L_S', 'L_M', 'L_O')
                    loss_scalars = torch.stack([losses['total'].detach()]+
                        [losses[key].detach() for key in objective_keys]+
                        [losses['terms'][key].detach() for key in term_keys]).cpu().tolist()
                    record = dict(epoch=epoch,step=step,sample_offset=offset,batch_cases=batch_cases,
                        stage=stage,total=loss_scalars[0],
                        objectives=dict(zip(objective_keys, loss_scalars[1:4])),
                        objective_mode=losses['objective_mode'], objective_weights=losses['objective_weights'],
                        terms=dict(zip(term_keys,loss_scalars[4:])),
                        active_weights=losses['active_weights'],units=losses['units'],seconds=time.perf_counter()-start)
                    for source, target in (('motion_diagnostics', 'gt_motion'), ('observation_summary', 'observation')):
                        summary = losses[source]
                        if summary is not None:
                            record[target] = {key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                                              for key, value in summary.items()}
                    record['io_wait_seconds'] = io_wait
                    if torch.device(device).type=='cuda':
                        record['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                        record['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
                    if model.dynamic_conditioner is not None:
                        record['dynamic_spatial'] = dict(
                            phase_statistics=decoded['dynamic_statistics'].detach().mean(0).cpu().tolist(),
                            columns=['query_change_rms','centered_medium_rms','fine_feature_rms','medium_centered_energy_fraction'],
                            output_weight_l2=float(model.dynamic_conditioner.output.weight.detach().norm()),
                            output_gradient_l2=float(model.dynamic_conditioner.output.weight.grad.norm()),
                            attention_gradient_l2=float(sum((p.grad.detach().square().sum() for p in
                                model.dynamic_conditioner.attention.parameters() if p.grad is not None),
                                model.dynamic_conditioner.output.weight.new_zeros(())).sqrt()))
                        if dynamic_probe is not None:record['dynamic_spatial']['paired_branch_probe']=dynamic_probe
                    if model.regional_motion:
                        grids = decoded['control_grids']['regional_motion'].detach()
                        record['regional_motion'] = dict(
                            increment_displacement_rms_mm=decoded['regional_motion_rms_mm'].detach().square().mean((0,1)).sqrt().cpu().tolist(),
                            control_rms_mm=grids.square().mean((0,1,3,4,5,6)).sqrt().cpu().tolist(),
                            adjacent_control_change_rms_mm=grids.diff(dim=1).square().mean((0,1,3,4,5,6)).sqrt().cpu().tolist())
                    compute_seconds += record['seconds']
                    processed_samples += batch_cases
                    if slices is not None:
                        record['slice_diagnostics'] = slices['diagnostics']
                        record['slice_execution'] = dict(backend=args.slice_backend,cell_chunk_size=slicer.cell_chunk_size,
                            query_chunk_size=slicer.chunk_size,boundary_only=slicer.boundary_only)
                    record['execution'] = execution
                    history.write(json.dumps(record,allow_nan=False)+'\n');history.flush()
                    profiler.step()
                    loss_sum += record['total']*batch_cases
                    loss_samples += batch_cases
                    next_epoch,next_offset = (epoch+1,0) if offset==count else (epoch,offset)
                    stopping = args.max_steps is not None and step>=args.max_steps
                    if stopping or step%training['checkpoint_every_steps']==0:
                        save_checkpoint(output_dir/'last.pth',model,optimizer,config=config,identity=identity,
                            epoch=next_epoch,sample_offset=next_offset,step=step,runtime=runtime)
                    if stopping:
                        break
                    data_wait_start = time.perf_counter()
            complete = offset==count
            epoch_number = epoch+1
            if complete:
                epoch,offset = epoch+1,0
            if not stopping:
                save_checkpoint(output_dir/'last.pth',model,optimizer,config=config,identity=identity,
                                epoch=epoch,sample_offset=offset,step=step,runtime=runtime)
            report = evaluate(model,validation,device,cases=args.val_cases or training['validation_cases'],
                              exact=args.exact,progress=True,runtime=evaluator)
            summary = dict(epoch=epoch_number,step=step,stage=stage,complete=complete,
                train_mean_loss=loss_sum/loss_samples,train_samples=loss_samples,
                validation=validation_summary(report),seconds=time.perf_counter()-epoch_start)
            summary['learning_rate'] = rate
            summary['optimizer_group_learning_rates'] = [group['lr'] for group in optimizer.param_groups]
            if training.get('save_best',False):
                for name,candidate in best_checkpoint_candidates(summary).items():
                    if name not in best or candidate['score']<best[name]['score']:
                        save_checkpoint(output_dir/(name+'.pth'),model,optimizer,config=config,identity=identity,
                            epoch=epoch,sample_offset=offset,step=step,runtime=runtime)
                        best[name] = dict(candidate,epoch=epoch_number,step=step,
                                          validation=summary['validation'])
                best_path.write_text(json.dumps(best,indent=2,allow_nan=False)+'\n')
            summary['validation_performance'] = report['performance']
            if 'encoder_diagnostics' in report:
                summary['encoder_diagnostics'] = report['encoder_diagnostics']
            summary['training_performance'] = dict(reader_workers=reader_workers,prefetch=args.train_prefetch,
                slice_backend=args.slice_backend,slice_cell_chunk=slicer.cell_chunk_size,execution=execution,
                processed_samples=processed_samples,io_wait_seconds=io_seconds,compute_seconds=compute_seconds,
                timing_scope='current invocation segment; includes reader startup')
            report.update(epoch=epoch_number,step=step,epoch_summary=summary)
            if stopping:
                report['scope'] = 'limited engineering run; no formal performance claim'
            rendered_report = json.dumps(report,indent=2,allow_nan=False)+'\n'
            (output_dir/'validation.json').write_text(rendered_report)
            archive = output_dir/'validation'
            archive.mkdir(exist_ok=True)
            name = f'epoch_{epoch_number:04d}.json' if complete else f'partial_step_{step:08d}.json'
            (archive/name).write_text(rendered_report)
            with (output_dir/'epoch_history.jsonl').open('a') as epoch_history:
                epoch_history.write(json.dumps(summary,allow_nan=False)+'\n')
            tqdm.write(format_epoch_summary(summary,training['epochs']))
            if stopping:
                return


if __name__=='__main__':
    main()
