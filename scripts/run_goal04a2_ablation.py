"""Execute only preregistered small-budget runs. There is no full-1K mode."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train_geometry_heartffd_v2 import load_config,write_json,code_identity,data_identity


def run_one(job,root,device):
    variant,seed=job;out=root/(variant+'_seed'+str(seed));out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment.json').exists():raise ValueError('Fresh ablation directory required: '+str(out))
    started=time.perf_counter()
    command=[sys.executable,str(ROOT/'train_geometry_heartffd_v2.py'),'--short-run',
        '--variant',variant,'--seed',str(seed),'--device',device,'--output',str(out)]
    with (out/'training.log').open('w') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    record=dict(variant=variant,seed=seed,command=command,training_exit_code=result.returncode,
        seconds=time.perf_counter()-started,status='training_rejected' if result.returncode else 'trained')
    if result.returncode==0:
        command=[sys.executable,str(ROOT/'train_geometry_heartffd_v2.py'),'--evaluate',str(out/'latest.pth'),
            '--device',device,'--exact','--output',str(out/'validation')]
        with (out/'validation.log').open('w') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record.update(validation_exit_code=result.returncode,status='complete' if not result.returncode else 'validation_rejected')
    write_json(out/'orchestration.json',record)
    print(variant,seed,record['status'],flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=parser.parse_args();config,data,protocol=load_config(ROOT/'configs/geometry_heartffd_v2.yaml')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'protocol_snapshot.json',dict(protocol=protocol,code_identity=code_identity(),
        data_identity=data_identity(data),concurrent_processes=args.workers,shared_gpu=True,
        full_1k_started=False,selection='Final fixed-budget checkpoint only'))
    jobs=[(variant,protocol['seeds'][0]) for variant in protocol['variants']]
    jobs.extend((variant,seed) for seed in protocol['seeds'][1:] for variant in ('A0','A4'))
    results=[]
    # Separate processes keep model, optimizer, data ordering and RNG isolated.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(lambda job:run_one(job,args.output,args.device),jobs):
            results.append(record);write_json(args.output/'runs.json',results)


if __name__=='__main__':main()
