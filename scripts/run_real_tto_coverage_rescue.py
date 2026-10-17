"""Queue the predeclared input-coverage experiment after the primary arms finish.

This launcher never changes or signals the primary process. The actual experiment
still runs through tto.experiments, including its fresh oracle and Smoke gates.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live(pid,start):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
        return fields[19]==str(start) and fields[0] not in ('Z','X')
    except FileNotFoundError:
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-output',required=True)
    parser.add_argument('--primary-pid',type=int,required=True)
    parser.add_argument('--primary-start-ticks',required=True)
    parser.add_argument('--poll-seconds',type=float,default=30)
    args=parser.parse_args()
    if not 1<=args.poll_seconds<=60:raise ValueError('Polling interval must be1..60 seconds')
    primary=(ROOT/args.primary_output).resolve()
    folder=primary/'independent_audit'
    plan=read(folder/'coverage_rescue_predeclaration.json')
    config=(ROOT/plan['rescue_config']).resolve()
    state_path=folder/'coverage_rescue_queue.json'
    protocol=read(primary/'protocol.json')
    lock=(folder/'coverage_rescue_queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(pid=os.getpid(),start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().split(') ',1)[1].split()[19])

    def write(status,**extra):
        value=dict(updated=time.time(),status=status,launcher=identity,
                   primary_pid=args.primary_pid,primary_start_ticks=args.primary_start_ticks,
                   rescue_config=str(config),affected_patients=plan['affected_patients'],**extra)
        temporary=state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(state_path)
        return value

    def verify():
        if sha(config)!=plan['rescue_config_sha256']:raise ValueError('Predeclared rescue config changed')
        if sha(ROOT/plan['primary_config'])!=plan['primary_config_sha256']:raise ValueError('Primary config changed')
        if sha(folder/'native_stride_coverage.json')!=plan['coverage_audit_sha256']:raise ValueError('Input coverage evidence changed')
        for name,digest in protocol['sources'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen scientific source changed: '+name)

    try:
        verify();previous=None
        while True:
            comparison=primary/'metrics/ablation_comparison.json'
            complete=comparison.exists() and read(comparison).get('complete') is True
            running=live(args.primary_pid,args.primary_start_ticks)
            if complete and not running:break
            status='WAITING_FOR_PRIMARY_COMPLETION' if running else 'WAITING_FOR_PRIMARY_REPAIR'
            write(status,primary_all_arms_processed=complete)
            if status!=previous:print(status,flush=True);previous=status
            time.sleep(args.poll_seconds)
        verify()
        command=[sys.executable,'-m','tto.experiments','--config',str(config),'--resume']
        log=folder/'coverage_rescue_experiments.log'
        with log.open('ab') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            write('RUNNING',command=command,experiment_pid=child.pid,log=str(log))
            code=child.wait()
        result=(ROOT/plan['rescue_output'])/'metrics/ablation_comparison.json'
        complete=code==0 and result.exists() and read(result).get('complete') is True
        write('COMPLETE' if complete else 'WAITING_FOR_GPU_RESOURCES' if code==75 else 'FAILED',
              command=command,returncode=code,all_rescue_arms_processed=complete,log=str(log))
        return 0 if complete else code or 1
    except Exception as error:
        write('FAILED',error=repr(error));raise


if __name__=='__main__':raise SystemExit(main())
