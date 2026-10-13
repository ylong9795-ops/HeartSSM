"""Real-data B=1/2, once-computed CNN/head, gradients, timing and memory check."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.utils.data import default_collate
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_geometry_heartffdnet import (load_config,dataset_from_config,build_model,to_device,
    data_identity,finite_gradient_report)
from utils.geometry_joint_checks import validate_template,validate_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/geometry_heartffd_1k.yaml')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/geometry_goal04_final/engineering.json')
    args = parser.parse_args()
    config,data = load_config(args.config)
    torch.set_num_threads(config['training']['cpu_threads']);torch.manual_seed(config['training']['seed'])
    dataset = dataset_from_config(data)
    samples = [dataset[i] for i in range(2)]
    batch = to_device(default_collate(samples),args.device)
    # Supply exactly the documented measured observation fields, no GT mesh.
    measured = {k:batch[k] for k in ('masks','sdf','phases','view_present','annotated','pose_gt',
        'model_to_native','pixel_spacing','native_pixel_spacing','asset_sha256','template_sha256','norm_to_mm','sdf_units')}
    model = build_model(dataset,config).to(args.device).eval();validate_template(model)
    info = {};cnn_counts=[];head_calls=[]
    handles = [model.encoder.stem.register_forward_pre_hook(lambda m,a:cnn_counts.append(len(a[0]))),
        model.encoder.register_forward_hook(lambda m,a,y:info.update(spatial_shapes={k:list(y[k].shape) for k in ('scale14','scale7','validity')})),
        model.lifting.register_forward_hook(lambda m,a,y:info.update(lifted_shape=list(y['features'].shape),
            lifted_spatial_std=float(y['features'].std((-3,-2,-1)).mean()))),
        model.heads.deform3.register_forward_hook(lambda m,a,y:head_calls.append(list(y.shape)))]
    if torch.device(args.device).type == 'cuda':torch.cuda.reset_peak_memory_stats(args.device)
    tick = time.perf_counter()
    with torch.no_grad():output = model(measured)
    if torch.device(args.device).type == 'cuda':torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter()-tick
    for handle in handles:handle.remove()
    assert sum(cnn_counts) == 2*24*3 and len(head_calls) == 1 and head_calls[0][1] == 72
    assert output['vertices_4d'].shape == (2,24,10631,3)
    assert info['lifted_spatial_std'] > 0
    geometry = validate_prediction(model,output,sample_jacobian=True)
    one = {k:(v[:1] if isinstance(v,torch.Tensor) else {a:b[:1] for a,b in v.items()} if isinstance(v,dict) else v[:1] if isinstance(v,list) else v) for k,v in measured.items()}
    # Per-case independence with the real full-size topology and image grid.
    with torch.no_grad():single = model(one)
    torch.testing.assert_close(single['vertices_4d'],output['vertices_4d'][:1],rtol=1e-5,atol=1e-6)
    gradients={}
    model.train()
    for kind in ('shape','motion'):
        model.zero_grad(set_to_none=True);prediction = model(one)
        p = prediction['vertices_4d']
        loss = (p[:,0]-(model.template_vertices_mm[None]+.3)).square().mean() if kind == 'shape' else ((p[:,-1]-p[:,0])-.2).square().mean()
        loss.backward();gradients[kind] = finite_gradient_report(model)
    report = dict(status='PASS',architecture='geometry-ma-joint-ffd-v1',pose_mode='known_pose',
        parameters=sum(p.numel() for p in model.parameters()),batch2_forward_seconds=elapsed,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if torch.device(args.device).type=='cuda' else None,
        gpu_memory_status='measured' if torch.device(args.device).type=='cuda' else 'unavailable; CPU-only environment',
        device=args.device,cnn_observations_evaluated=sum(cnn_counts),joint_head_calls=len(head_calls),
        joint_head_shape=head_calls[0],mesh_shape=list(output['vertices_4d'].shape),**info,
        parameter_gradients=gradients,geometry=geometry,data_identity=data_identity(data),
        versions=dict(torch=torch.__version__,python=sys.version.split()[0]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':main()
