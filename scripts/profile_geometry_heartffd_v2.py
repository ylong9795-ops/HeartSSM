"""Actual GPU forward/backward, BF16/checkpointing and component measurements.

Numerical profiling is independent of reconstruction effectiveness.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch
from torch.utils.data import default_collate
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train_geometry_heartffd_v2 import load_config,dataset_from_config,build_model,to_device,gradient_report
from utils.slice_supervision import MeshSliceSupervisor
from utils.geometry_joint_checks import validate_template,validate_prediction
from utils.geometry_v2_checks import four_chamber_audit
from utils.geometry_pair_policy import validate_source_geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--output',type=Path,default=ROOT/'outputs/goal04a2/gpu_profile.json')
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();config,data,protocol=load_config(ROOT/'configs/geometry_heartffd_v2.yaml')
    if args.warmup<1 or args.repeats<2:parser.error('At least one warmup and two measured repeats required')
    torch.set_num_threads(4);torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA device required; no CPU replacement')
    torch.cuda.set_device(args.device)
    dataset=dataset_from_config(data);samples=[dataset[i] for i in (0,1)]
    source=build_model(dataset,config).to(args.device)
    geometry=validate_source_geometry(source)
    del source;torch.cuda.empty_cache()
    rows=[]
    for batch_size in (1,2):
        for checkpointing in (False,True):
            for precision in ('float32','bf16'):
                if precision=='bf16' and not torch.cuda.is_bf16_supported():
                    rows.append(dict(batch=batch_size,checkpointing=checkpointing,precision=precision,status='unsupported'));continue
                torch.manual_seed(2026)
                cfg=copy.deepcopy(config);cfg['encoder']['checkpoint_cnn']=checkpointing
                model=build_model(dataset,cfg).to(args.device).train();validate_template(model)
                batch=to_device(default_collate(samples[:batch_size]),args.device)
                repeats=[]
                for repeat in range(args.warmup+args.repeats):
                    model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    tick=time.perf_counter()
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                        measured=model(batch)
                    torch.cuda.synchronize();fwd=time.perf_counter()-tick
                    objective=((measured['vertices_4d'][:,-1]-measured['vertices_ref'])-.2).square().mean()
                    tick=time.perf_counter();objective.backward();torch.cuda.synchronize()
                    bwd=time.perf_counter()-tick
                    if repeat>=args.warmup:
                        repeats.append(dict(forward_seconds=fwd,backward_seconds=bwd,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                    del measured,objective
                model.zero_grad(set_to_none=True)
                components={};handles=[]
                modules=dict(cnn=model.encoder,legacy_lifting=model.lifting,view_lifting=model.observation_lifting,
                    view_fusion=model.view_fusion,temporal=model.local_temporal,fine_skip=model.fine_motion_skip,
                    unet=model.unet,joint=model.heads.deform3)
                def before(name):
                    def callback(m,a):
                        torch.cuda.synchronize();m._profile_started=time.perf_counter()
                        m._profile_memory=torch.cuda.memory_allocated()
                    return callback
                def after(name):
                    def callback(m,a,y):
                        torch.cuda.synchronize();entry=components.setdefault(name,dict(seconds=0.,calls=0,net_allocated_bytes=0))
                        entry['seconds']+=time.perf_counter()-m._profile_started;entry['calls']+=1
                        entry['net_allocated_bytes']+=torch.cuda.memory_allocated()-m._profile_memory
                    return callback
                for name,module in modules.items():
                    handles.extend((module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name))))
                original_deform=model._deform
                def measured_deform(vertices,controls,level):
                    torch.cuda.synchronize();tick=time.perf_counter();allocation=torch.cuda.memory_allocated()
                    result=original_deform(vertices,controls,level)
                    torch.cuda.synchronize()
                    entry=components.setdefault('physical_ffd',dict(seconds=0.,calls=0,net_allocated_bytes=0))
                    entry['seconds']+=time.perf_counter()-tick;entry['calls']+=1
                    entry['net_allocated_bytes']+=torch.cuda.memory_allocated()-allocation
                    return result
                model._deform=measured_deform
                torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                    output=model(batch)
                torch.cuda.synchronize();forward_seconds=time.perf_counter()-started
                forward_peak=torch.cuda.max_memory_allocated()
                for h in handles:h.remove()
                model._deform=original_deform
                # Computational objective only: source geometry is explicitly
                # unaccepted, so do not publish formal reconstruction metrics.
                loss=((output['vertices_4d'][:,-1]-output['vertices_ref'])-.2).square().mean()+.01*(output['vertices_ref']-model.template_vertices_mm-.2).square().mean()
                tick=time.perf_counter();loss.backward();torch.cuda.synchronize()
                backward_seconds=time.perf_counter()-tick;gradients=gradient_report(model)
                geom=validate_prediction(model,output,sample_jacobian=True)
                # Differentiable slice compute measured without any optimizer.
                model.zero_grad(set_to_none=True)
                sliced_prediction=model(batch)
                slicer=MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,**cfg['slice'],execution_backend='batched',cell_chunk_size=8).to(args.device)
                torch.cuda.synchronize();tick=time.perf_counter();slices=slicer(sliced_prediction['vertices_4d'],batch)
                torch.cuda.synchronize();slice_forward=time.perf_counter()-tick
                tick=time.perf_counter();slices['total'].backward();torch.cuda.synchronize();slice_backward=time.perf_counter()-tick
                row=dict(batch=batch_size,checkpointing=checkpointing,precision=precision,status='numerical_pass',
                    source_geometry_accepted=geometry['status']=='pass',forward_seconds=forward_seconds,
                    backward_seconds=backward_seconds,forward_peak_bytes=forward_peak,full_peak_bytes=torch.cuda.max_memory_allocated(),
                    slice_forward_seconds=slice_forward,slice_backward_through_model_seconds=slice_backward,
                    components=components,gradient_l2=gradients,intra_chamber_geometry=geom,
                    mesh_shape=list(output['vertices_4d'].shape),field_shape=list(output['control_grids']['fine'].shape),
                    warmup_steps=args.warmup,steady_state_measurements=repeats,
                    parameters=sum(p.numel() for p in model.parameters()))
                rows.append(row);print(json.dumps({k:row[k] for k in ('batch','checkpointing','precision','status','forward_seconds','backward_seconds','full_peak_bytes')}),flush=True)
                del model,output,sliced_prediction,slices,slicer,batch,loss;torch.cuda.empty_cache()
    report=dict(device=str(args.device),name=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        shared_gpu=True,source_geometry=geometry,measurements=rows,scope='numerical GPU engineering; fixed source AV policy; effectiveness assessed separately')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
