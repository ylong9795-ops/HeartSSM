"""Read-only soft/native diagnostics. Neither renderer nor loss is changed."""
import numpy as np
import torch
from data.mask_transform import CHAMBERS,VIEWS
from net.multiview_encoder import observation_validity
from utils.slice_supervision import section_soft_mask,signed_segment_distance
from train_neural_ffd import exact_metrics


@torch.no_grad()
def paired_slice_cells(model,vertices,batch,slicer):
    """Same mesh/pose/targets: per-cell soft terms and authoritative hard Dice.

    Also report a continuous section-to-pixel-boundary distance; its native
    raster counterpart remains the exact evaluator's independent ASSD.
    """
    from scipy.ndimage import binary_erosion
    exact=exact_metrics(model,vertices,batch)
    native={(r['patient'],r['frame'],r['view'],r['chamber']):r for r in exact['cells']}
    valid=observation_validity(batch,len(vertices),24,vertices.device)
    rows=[]
    for patient,phase,view,chamber in valid.nonzero().tolist():
        a,b=model.chamber_ranges[chamber].tolist()
        rotation=batch['pose_gt']['probe_R'][patient,phase,view].to(vertices)
        translation=batch['pose_gt']['probe_t'][patient,phase,view].to(vertices)*model.norm_to_mm
        spacing=batch['pixel_spacing'][patient,phase,view].to(vertices)
        target=batch['masks'][patient,phase,view,chamber].to(vertices)
        sdf=batch['sdf'][patient,phase,view,chamber].to(vertices)
        rendered=section_soft_mask(vertices[patient,phase,a:b],model.faces[CHAMBERS[chamber]],
            rotation,translation,spacing,target.shape,sigma_mm=slicer.sigma_mm,
            pixel_stride=slicer.pixel_stride,chunk_size=slicer.chunk_size,
            require_intersection=bool(target.any()))
        hard=native[patient,phase,VIEWS[view],CHAMBERS[chamber]]
        if rendered['logits'] is None:
            terms={key:0. for key in slicer.weights};continuous=None
        else:
            terms={key:float(value[0]) for key,value in slicer._terms(
                rendered['logits'].flatten()[None],target[::slicer.pixel_stride,::slicer.pixel_stride].flatten()[None],
                sdf[::slicer.pixel_stride,::slicer.pixel_stride].flatten()[None]).items()}
            boundary=np.argwhere(target.cpu().numpy().astype(bool)&~binary_erosion(target.cpu().numpy().astype(bool)))
            if len(boundary):
                rc=torch.tensor(boundary,device=vertices.device,dtype=vertices.dtype)
                points=((rc-target.new_tensor(target.shape)/2)*spacing).flip(-1)
                continuous=float(signed_segment_distance(points,rendered['segments_mm'],256).abs().mean())
            else:continuous=None
        rows.append(dict(**hard,soft_terms=terms,soft_total=sum(slicer.weights[k]*v for k,v in terms.items()),
            gt_pixel_boundary_to_continuous_section_mm=continuous))
    soft=np.asarray([r['soft_total'] for r in rows]);dice=np.asarray([r['exact_dice'] for r in rows])
    correlation=float(np.corrcoef(soft,1-dice)[0,1]) if soft.std()>0 and dice.std()>0 else None
    return dict(cells=rows,summary=dict(valid_cells=len(rows),mean_soft_total=float(soft.mean()),
        mean_exact_native_dice=float(dice.mean()),minimum_exact_dice=float(dice.min()),
        p01_exact_dice=float(np.quantile(dice,.01)),p05_exact_dice=float(np.quantile(dice,.05)),
        cells_below_095=int((dice<.95).sum()),cells_ge_098=int((dice>=.98).sum()),
        all_cells_ge_098=bool((dice>=.98).all()),soft_loss_hard_error_correlation=correlation,
        gt_native_pixel_mismatches=sum(r['gt_native_pixel_mismatches'] for r in rows)))


def compare_slice_updates(baseline,candidate):
    if len(baseline['cells'])!=len(candidate['cells']):raise ValueError('Cell identities differ')
    rows=[]
    for a,b in zip(baseline['cells'],candidate['cells']):
        keys=('patient','frame','view','chamber')
        if any(a[k]!=b[k] for k in keys):raise ValueError('Cell order differs')
        rows.append(dict(**{k:a[k] for k in keys},soft_change=b['soft_total']-a['soft_total'],
                         exact_dice_change=b['exact_dice']-a['exact_dice']))
    improved=[r for r in rows if r['soft_change'] < -1e-8]
    return dict(cells=rows,soft_improved_cells=len(improved),
        soft_improved_hard_decreased_cells=sum(r['exact_dice_change'] < -1e-8 for r in improved),
        soft_improved_hard_unchanged_cells=sum(abs(r['exact_dice_change'])<=1e-8 for r in improved),
        disagreement_fraction=sum(r['exact_dice_change']<=1e-8 for r in improved)/len(improved) if improved else None,
        interpretation='Continuous target mismatch, raster jumps, sampling and topology must be investigated separately; this is not a hard-Dice gradient error')
