"""Real-data 24-phase engineering audit and representative scientific figures.

This reads existing TRAIN cases and a checkpoint, never generates data or
uses TEST. Numerical tests remain in tests/test_neural_ffd.py.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/heartffd_matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from data.mask_transform import CHAMBERS,VIEWS
from net.neural_ffdnet import geometry_digest
from net.multiview_encoder import observation_validity
from train_neural_ffd import (load_config,dataset_from_config,load_checkpoint,to_device,
                              compute_objective,exact_metrics)
from utils.neural_ffd_losses import NeuralFFDLoss,mesh_volumes_ml,reconstruction_metrics
from utils.slice_supervision import MeshSliceSupervisor,section_soft_mask,exact_section_mask
from utils.slice_evaluation import render_source_slice


def figures(model,trajectory,batch,destination):
    colors = ('#e53935','#ff9800','#2196f3','#43a047')
    p = trajectory.detach().cpu().numpy()[0]
    g = batch['vertices_gt'].cpu().numpy()[0]
    poses = {k:v.cpu().numpy()[0] for k,v in batch['pose_gt'].items()}
    masks = batch['native_masks'].cpu().numpy()[0]
    volumes = mesh_volumes_ml(trajectory[:1],model.chamber_ranges,model.faces).detach().cpu().numpy()[0]
    target_volumes = mesh_volumes_ml(batch['vertices_gt'][:1].to(trajectory),model.chamber_ranges,model.faces).cpu().numpy()[0]
    contraction = int(target_volumes[:,0].argmin())
    fig = plt.figure(figsize=(17,8))
    for column,(mesh,frame,title) in enumerate(((g,0,'GT reference'),(p,0,'Prediction reference'),
                                             (g,contraction,'GT LV Vmin'),(p,contraction,'Prediction LV Vmin'))):
        ax = fig.add_subplot(2,4,column+1,projection='3d')
        for c,(left,right),color in zip(CHAMBERS,model.chamber_ranges.tolist(),colors):
            v,f = mesh[frame,left:right],model.faces[c].cpu().numpy()
            ax.plot_trisurf(v[:,0],v[:,1],v[:,2],triangles=f,color=color,alpha=.7,linewidth=0,rasterized=True)
        ax.set_title(title+f' f{frame}')
        ax.set(xlim=(-80,80),ylim=(-80,80),zlim=(-80,80),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
        ax.set_box_aspect((1,1,1));ax.view_init(20,-65)
    for chamber in range(4):
        ax = fig.add_subplot(2,4,5+chamber)
        time = np.arange(len(p))/len(p)
        ax.plot(time,target_volumes[:,chamber],label='GT',color='black')
        ax.plot(time,volumes[:,chamber],label='prediction',color=colors[chamber])
        ax.set(title=CHAMBERS[chamber],xlabel='relative cycle time',ylabel='volume (mL)');ax.legend()
    fig.suptitle('Checkpoint reconstruction: source reference and LV minimum-volume phase')
    fig.tight_layout();fig.savefig(destination/'mesh_volume.png',dpi=140);plt.close(fig)
    fig,axes = plt.subplots(3,2,figsize=(9,12))
    for view in range(3):
        for column,frame in enumerate((0,contraction)):
            ax = axes[view,column]
            ax.imshow(np.zeros(masks.shape[-2:]),cmap='gray',vmin=0,vmax=1)
            for chamber,(c,(left,right),color) in enumerate(zip(CHAMBERS,model.chamber_ranges.tolist(),colors)):
                if not observation_validity(batch,len(batch['masks']),len(p),trajectory.device)[0,frame,view,chamber]:
                    continue
                predicted = render_source_slice(p[frame,left:right],model.faces[c].cpu().numpy(),
                    poses['probe_R'][frame,view],poses['probe_t'][frame,view],poses['probe_s'][frame,view],
                    model.norm_to_mm,masks.shape[-2:])
                target = masks[frame,view,chamber]
                if target.any(): ax.contour(target,levels=[.5],colors=[color],linestyles='solid',linewidths=1)
                if predicted.any(): ax.contour(predicted,levels=[.5],colors=[color],linestyles='dashed',linewidths=1)
            ax.set_title(f'{VIEWS[view]} f{frame}: original grid; solid GT, dashed prediction');ax.axis('off')
    fig.tight_layout();fig.savefig(destination/'sections.png',dpi=140);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/neural_ffd.yaml')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/neural_ffd_check')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config,data_config = load_config(args.config)
    dataset = dataset_from_config(data_config)
    model,saved = load_checkpoint(args.checkpoint,args.device)
    if model.template_sha256!=dataset.template_sha256 or model.asset_sha256!=dataset.asset_sha256:
        raise ValueError('Checkpoint/dataset geometry mismatch')
    batch = to_device(next(iter(DataLoader(Subset(dataset,[0,1]),batch_size=2))),args.device)
    model.train()
    criterion = NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(args.device)
    slicer = MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,**config['slice']).to(args.device)
    output,losses,slices = compute_objective(model,batch,criterion,slicer,'joint')
    gradients = {}
    routes = [('shape','coarse',model.deform1.weight),('shape','medium',model.deform2.weight),
              ('shape','fine_reference',model.deform3.reference.weight),('shape','regional',model.regional_heads[0].weight),
              ('motion','motion_head',model.deform3.motion.weight),('motion','z_motion',output['z_motion']),
              ('slice','shape_head',model.deform3.reference.weight),('slice','motion_head',model.deform3.motion.weight),
              ('slice','encoder',model.encoder.stem[0].weight)]
    for term,name,parameter in routes:
        gradient = torch.autograd.grad(losses['terms'][term],parameter,retain_graph=True)[0]
        if not torch.isfinite(gradient).all() or gradient.norm()==0:
            raise ValueError(f'Invalid {term} gradient to {name}')
        gradients[term+' -> '+name] = float(gradient.norm())
    report = dict(scope='Two existing TRAIN cases, B=2, all 24 phases; engineering only',
        checkpoint_step=saved['step'],case_ids=batch['case_id'],
        shapes={key:list(value.shape) for key,value in output.items() if isinstance(value,torch.Tensor)},
        template_sha256=model.template_sha256,asset_sha256=model.asset_sha256,
        physical_geometry_sha256=geometry_digest(model.template_vertices_mm,model.chamber_ranges,model.faces),
        norm_to_mm=model.norm_to_mm,chamber_ranges=model.chamber_ranges.cpu().tolist(),
        lower_mm=model.lower_mm.cpu().tolist(),upper_mm=model.upper_mm.cpu().tolist(),
        control_origins_mm=model.control_origins_mm.cpu().tolist(),control_spacings_mm=model.control_spacings_mm.cpu().tolist(),
        slice_diagnostics=slices['diagnostics'],gradient_norms=gradients,
        losses={key:float(value.detach()) for key,value in losses['terms'].items()},
        metrics=reconstruction_metrics(output['vertices_4d'],batch['vertices_gt'],model.chamber_ranges,model.faces))
    del losses,slices
    model.eval()
    with torch.no_grad():
        latents = model.encode(batch)
        first = model.decode_4d(**latents)
        second = model.decode_4d(latents['z_shape'],latents['z_motion']+1)
        if not torch.equal(first[:,0],second[:,0]):
            raise ValueError('Motion latent changed reference shape')
        report['motion_reference_max_difference_mm'] = float((first[:,0]-second[:,0]).abs().max())
        report['motion_trajectory_difference_mm'] = float((first[:,1:]-second[:,1:]).abs().max())
        report['exact'] = exact_metrics(model,first,batch)
        comparisons,area_bias = [],[]
        # Compare the same GT geometry through independent soft/hard sections.
        valid = observation_validity(batch,2,24,first.device)
        for patient,frame,view,chamber in valid.nonzero().tolist():
            left,right = model.chamber_ranges[chamber].tolist()
            vertices = batch['vertices_gt'][patient,frame,left:right].to(first)
            rotation = batch['pose_gt']['probe_R'][patient,frame,view].to(first)
            translation = batch['pose_gt']['probe_t'][patient,frame,view].to(first)*model.norm_to_mm
            spacing = batch['pixel_spacing'][patient,frame,view].to(first)
            rendered = section_soft_mask(vertices,model.faces[CHAMBERS[chamber]],rotation,translation,spacing,
                                         batch['masks'].shape[-2:],sigma_mm=1.,chunk_size=1024)
            soft_hard = (rendered['soft_mask']>.5).cpu().numpy()
            exact = exact_section_mask(vertices.cpu().numpy(),model.faces[CHAMBERS[chamber]].cpu().numpy(),
                rotation.cpu().numpy(),translation.cpu().numpy(),spacing.cpu().numpy(),batch['masks'].shape[-2:])
            mass = soft_hard.sum()+exact.sum()
            comparisons.append(2*(soft_hard&exact.astype(bool)).sum()/mass if mass else 1.)
            area_bias.append((float(soft_hard.sum())-float(exact.sum()))/max(int(exact.sum()),1))
        report['gt_soft_threshold_vs_independent_exact'] = dict(cells=len(comparisons),
            mean_dice=float(np.mean(comparisons)),minimum_dice=float(np.min(comparisons)),
            mean_relative_pixel_area_bias=float(np.mean(area_bias)),
            distinction='Continuous point-in-section vs integer-truncated fillPoly; record rasterization bias, no straight-through gradient')
        args.output.mkdir(parents=True,exist_ok=True)
        figures(model,first,batch,args.output)
    del output,first,second
    model.requires_grad_(False)
    shape = latents['z_shape'][:1].detach().clone().requires_grad_()
    motion = latents['z_motion'][:1].detach().clone().requires_grad_()
    optimizer = torch.optim.Adam((shape,motion),lr=.05)
    trajectory_loss = []
    for _ in range(3):
        optimizer.zero_grad()
        prediction = model.decode_4d(shape,motion)
        loss = (prediction-batch['vertices_gt'][:1].to(prediction)).square().mean()
        trajectory_loss.append(float(loss))
        loss.backward()
        if not torch.isfinite(shape.grad).all() or not torch.isfinite(motion.grad).all() or shape.grad.norm()==0 or motion.grad.norm()==0:
            raise ValueError('Frozen decoder lost latent gradients')
        optimizer.step()
    if trajectory_loss[-1]>=trajectory_loss[0]:
        raise ValueError('Frozen decoder latent optimization did not decrease real-case GT error')
    report['frozen_decoder_latent_optimization'] = dict(loss_mm2=trajectory_loss,steps=3)
    (args.output/'engineering_check.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('shapes','slice_diagnostics','gradient_norms',
        'motion_reference_max_difference_mm','gt_soft_threshold_vs_independent_exact','frozen_decoder_latent_optimization')},indent=2),flush=True)


if __name__=='__main__':
    main()
