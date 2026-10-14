"""Measure strict state restoration and actual next-update CUDA differences.

The computational objective is solely for numerical auditing, never formal
training evidence. Do not hide parameter drift with output-only assertions.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
from torch.utils.data import default_collate
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train_geometry_heartffd_v2 import (load_config,dataset_from_config,build_model,to_device,
                                       save_checkpoint,load_checkpoint)
from train_neural_ffd import restore_rng


def backward(model, batch, optimizer):
    optimizer.zero_grad(set_to_none=True)
    vertices=model(batch)['vertices_4d']
    loss=((vertices[:,-1]-vertices[:,0])-.2).square().mean()
    loss.backward()
    return float(loss.detach())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/goal04a2/gpu_resume_reproducible.json')
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4);torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    config,data,_=load_config(ROOT/'configs/geometry_heartffd_v2.yaml')
    dataset=dataset_from_config(data)
    batch=to_device(default_collate([dataset[0]]),args.device)
    model=build_model(dataset,config).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    backward(model,batch,optimizer);optimizer.step()
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
        path=Path(directory)/'numerical.pth'
        save_checkpoint(path,model,optimizer,config=config,identity={'scope':'numerical audit'},
                        step=1,runtime={'train_indices':[0]})
        restored,saved=load_checkpoint(path,args.device,config=config,identity={'scope':'numerical audit'})
        other=torch.optim.Adam(restored.parameters(),lr=.001);other.load_state_dict(saved['optimizer_state'])
        left,right=dict(model.named_parameters()),dict(restored.named_parameters())
        if list(left)!=list(right):raise AssertionError('Optimizer parameter order differs')
        torch.testing.assert_close(model.state_dict(),restored.state_dict(),rtol=0,atol=0)
        # Compare loaded moments/step before taking either next update.
        for name,parameter in left.items():
            a,b=optimizer.state.get(parameter,{}),other.state.get(right[name],{})
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        model.eval();restored.eval()
        with torch.no_grad():
            a,b=model(batch),restored(batch)
        torch.testing.assert_close(a['vertices_4d'],b['vertices_4d'],rtol=0,atol=0)
        restore_rng({'rng':saved['rng_state']},args.device);loss_left=backward(model,batch,optimizer)
        restore_rng({'rng':saved['rng_state']},args.device);loss_right=backward(restored,batch,other)
        gradients={name:float((p.grad-right[name].grad).abs().max()) for name,p in left.items()
                   if p.grad is not None and right[name].grad is not None}
        optimizer.step();other.step()
        parameters={name:float((value-restored.state_dict()[name]).abs().max())
                    for name,value in model.state_dict().items() if value.is_floating_point()}
        with torch.no_grad():
            next_left,next_right=model(batch),restored(batch)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):bf16=model(batch)
        report=dict(scope='GPU numerical restoration audit; effectiveness is independent',
            torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
            initial_state_and_optimizer_bitwise_equal=True,reloaded_forward_bitwise_equal=True,
            next_losses=[loss_left,loss_right],gradient_max_abs=gradients,
            next_parameter_max_abs=parameters,
            maximum_next_parameter_abs_difference=max(parameters.values()),
            next_parameters_allclose_1e_5=all(torch.allclose(p,right[name],atol=1e-5,rtol=1e-5)
                                            for name,p in left.items()),
            next_mesh_max_abs_mm=float((next_left['vertices_4d']-next_right['vertices_4d']).abs().max()),
            next_fine_control_max_abs_mm=float((next_left['control_grids']['fine']-next_right['control_grids']['fine']).abs().max()),
            bf16_mesh_max_abs_mm=float((next_left['vertices_4d']-bf16['vertices_4d']).abs().max()),
            bf16_control_max_abs_mm=float((next_left['control_grids']['fine']-bf16['control_grids']['fine']).abs().max()),
            conclusion='Parameter-level next-update consistency is unproven; output differences measured separately')
        try:
            torch.use_deterministic_algorithms(True)
            backward(model,batch,optimizer)
            report['deterministic_backward_supported']=True
        except RuntimeError as error:
            report['deterministic_backward_supported']=False
            report['deterministic_backward_error']=str(error)
        finally:
            torch.use_deterministic_algorithms(False)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('gradient_max_abs','next_parameter_max_abs')},indent=2),flush=True)


if __name__=='__main__':main()
