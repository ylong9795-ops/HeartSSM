"""Bounded subprocess entry for the unchanged strict GeometryGuard."""
import argparse
import time
import torch
from ..io import load_state,write_json
from ..prior import FrozenNeuralFFDPrior
from .guard import GeometryGuard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    value=load_state(args.candidate);trajectory=value['trajectory'].numpy()
    prior=FrozenNeuralFFDPrior(args.checkpoint)
    guard=GeometryGuard(prior.model);started=time.monotonic()
    report=dict(passed=False,status='RUNNING',shape=guard.discrete(trajectory[0]),
        saved_phases=dict(status='UNVERIFIED',passed=False),continuous=dict(status='UNVERIFIED',passed=False),
        inter_chamber=dict(status='UNVERIFIED',passed=False))
    write_json(args.output,report)
    if not report['shape']['passed']:
        report.update(status='REJECTED_REFERENCE_GEOMETRY',elapsed_seconds=time.monotonic()-started)
        write_json(args.output,report);return
    report['saved_phases']=guard.discrete(trajectory)
    cross=report['saved_phases'].get('cross_chamber',[])
    ok=len(cross)==24 and all(r['passed'] for r in cross)
    report['inter_chamber']=dict(status='PASS' if ok else 'FAIL',passed=ok,records=cross)
    write_json(args.output,report)
    if not report['saved_phases']['passed']:
        report.update(status='REJECTED_SAVED_PHASE_GEOMETRY',elapsed_seconds=time.monotonic()-started)
        write_json(args.output,report);return
    report['continuous']=guard.continuous(trajectory)
    report.update(passed=report['continuous']['passed'],status=report['continuous']['status'],
        elapsed_seconds=time.monotonic()-started,
        scope='unchanged guard: reference, all24, fixed AV cap IDs, and every chord including23->0')
    write_json(args.output,report)


if __name__=='__main__':main()
