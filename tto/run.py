"""python -m tto.run --config configs/real_tto_joint_v1.yaml --stage all --device auto"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .config import load_config


def run_joint(args, config):
    from .config import resolve
    from .gpu import select_gpu, WaitingForGPUResources
    from .io import write_json
    if args.arm:config['experiment']['arm']=args.arm
    requested=args.device or config['runtime']['device']
    output=resolve(config['output']['root']);output.mkdir(parents=True,exist_ok=True)
    if args.internal_gpu_probe:
        import torch
        config['runtime']['device']='cuda:0'
        config['runtime']['probe_source']=args.probe_source
        if args.probe_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.probe_memory_fraction,0)
        from .joint_pipeline import RealTTOPipeline
        result=RealTTOPipeline(config).run(args.patients,stage='resource_probe')
        # Driver/context memory is not counted by torch's allocator.
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
        for line in query.stdout.splitlines():
            values=[v.strip() for v in line.split(',')]
            if len(values)==2 and values[0]==str(os.getpid()) and values[1].isdigit():
                result['context_mib']=max(0.,float(values[1])-torch.cuda.memory_reserved()/2**20)
        write_json(args.probe_output,result)
        return 0
    def probe(device):
        with tempfile.TemporaryDirectory(prefix='heartssm-tto-probe-') as directory:
            result_path=Path(directory)/'measurement.json'
            command=[sys.executable,'-m','tto.run','--config',args.config,'--stage','preflight',
                     '--internal-gpu-probe','--probe-output',str(result_path),'--output',directory,
                     '--probe-memory-fraction',str(min(.9,.85*device['free_mib']/device['total_mib']))]
            if args.patients:command+=['--patients',*args.patients]
            if args.batch_size is not None:command+=['--batch-size',str(args.batch_size)]
            if args.checkpoint:command+=['--checkpoint',args.checkpoint]
            if args.data_root:command+=['--data-root',args.data_root]
            if args.arm:command+=['--arm',args.arm]
            if args.stage=='synthetic_oracle':command+=['--probe-source','synthetic']
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'])
            print('GPU_WORKLOAD_PROBE',device['uuid'],device['free_mib'],flush=True)
            result=subprocess.run(command,env=environment,capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError('Actual workload probe failed: '+result.stderr[-2500:]+result.stdout[-500:])
            return json.loads(result_path.read_text())
    try:
        selection=select_gpu(requested,probe,margin_fraction=config['runtime']['gpu_margin_fraction'],
            retries=config['runtime']['gpu_retries'],retry_seconds=config['runtime']['gpu_retry_seconds'])
    except WaitingForGPUResources as error:
        write_json(output/'gpu_selection.json',dict(status='WAITING_FOR_GPU_RESOURCES',attempts=getattr(error,'attempts',[])))
        print('WAITING_FOR_GPU_RESOURCES',flush=True);return 75
    # Imports above must not initialize CUDA in the parent before this mapping.
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CUDA context initialized before GPU resource selection')
    os.environ['CUDA_VISIBLE_DEVICES']=selection['cuda_visible_devices']
    config['runtime']['device']=selection['torch_device']
    torch.cuda.set_per_process_memory_fraction(min(.9,.85*selection['selected']['free_mib']/selection['selected']['total_mib']),0)
    write_json(output/'gpu_selection.json',selection)
    history=output/'run_history'/f'{time.time_ns()}_{os.getpid()}'
    write_json(history/'gpu_selection.json',selection)
    write_json(history/'effective_config.json',config)
    from .joint_pipeline import RealTTOPipeline
    started=time.monotonic()
    try:
        pipeline=RealTTOPipeline(config)
        result=pipeline.run(args.patients,stage=args.stage,resume=args.resume)
    except Exception as error:
        write_json(history/'failure.json',dict(error=repr(error),stage=args.stage))
        raise
    finally:
        timing=dict(device=selection['selected'],torch_device=selection['torch_device'],
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            elapsed_seconds=time.monotonic()-started)
        write_json(output/'runtime.json',timing);write_json(history/'runtime.json',timing)
    return 1 if result.get('failures') or (result.get('synthetic_gate') is not None and not result['synthetic_gate']['passed']) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/real_tto_joint_v1.yaml')
    parser.add_argument('--patients', nargs='+')
    parser.add_argument('--stage', choices=('all', 'preflight', 'synthetic_oracle', 'initial', 'pose_init',
        'pose', 'shape', 'motion_lock', 'joint', 'evaluate', 'export'), default='all')
    parser.add_argument('--device', help='auto or cuda:<visible index>')
    parser.add_argument('--gpu', help='Legacy visible CUDA index; prefer --device for the joint protocol')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir')
    parser.add_argument('--output')
    parser.add_argument('--data-root')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--arm', choices=('frozen_prior','pose_only','shape_only','alternate','joint','original_motion','ed_only'))
    parser.add_argument('--internal-gpu-probe', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--probe-output', help=argparse.SUPPRESS)
    parser.add_argument('--probe-memory-fraction', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--probe-source', choices=('real','synthetic'), default='real', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.gpu:config['runtime']['device'] = 'cpu' if args.gpu == 'cpu' else f'cuda:{args.gpu}'
    if args.batch_size is not None:
        if args.batch_size < 1:parser.error('--batch-size must be positive')
        config['runtime']['batch_size'] = args.batch_size
    if args.checkpoint:config['prior']['checkpoint'] = args.checkpoint
    if args.output_dir:config['output']['root'] = args.output_dir
    if args.output:config['output']['root'] = args.output
    if args.data_root:config['data']['root'] = args.data_root
    if config['version']=='real-tto-joint-v1':return run_joint(args,config)
    from .pipeline import RealTTOPipeline
    if not args.patients:parser.error('--patients is required for the legacy protocol')
    if args.device:config['runtime']['device']=args.device
    result = RealTTOPipeline(config).run(args.patients, stage=args.stage, resume=args.resume)
    return 1 if result['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
