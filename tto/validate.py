"""Inspect artifacts from formal CLI runs; never substitutes for actual execution."""
import argparse
from pathlib import Path
import numpy as np
import torch

from .io import load_state, read_json, sha256, write_json


def compare_runs(batch_root, single_root, patients, *, atol=3e-3, rtol=2e-4):
    reports = []
    for patient in patients:
        a = Path(batch_root)/'patients'/patient
        b = Path(single_root)/'patients'/patient
        comparisons = []
        for stage in ('initial','pose','shape','motion_lock','joint','evaluate','export'):
            x = load_state(a/'states'/(stage+'.pth'))['state']
            y = load_state(b/'states'/(stage+'.pth'))['state']
            if x['patient'] != patient or y['patient'] != patient:raise AssertionError('Mixed patient state')
            for key in ('z_shape','z_motion','fields','prior4d','trajectory'):
                error = float((x[key]-y[key]).abs().max())
                if not torch.allclose(x[key],y[key],atol=atol,rtol=rtol):
                    raise AssertionError(f'{patient}/{stage}/{key} batch-single max error={error}')
                comparisons.append(dict(stage=stage,variable=key,max_absolute_error=error))
            for key in ('R','center_mm'):
                error = float((x['pose'][key]-y['pose'][key]).abs().max())
                if not torch.allclose(x['pose'][key],y['pose'][key],atol=atol,rtol=rtol):
                    raise AssertionError(f'{patient}/{stage}/Pose {key} max error={error}')
                comparisons.append(dict(stage=stage,variable='pose_'+key,max_absolute_error=error))
        delta = float((x['fields']-load_state(a/'states/initial.pth')['state']['fields']).abs().max())
        if delta != 0:raise AssertionError('Motion fields changed')
        pose_history = x['pose_result']['history'];shape_history = x['shape_result']['history']
        if not pose_history or not all(r['translation_gradient_norm']>0 and r['rotation_gradient_norm']>0 for r in pose_history):
            raise AssertionError('No measured Pose gradients')
        if x['shape_status']=='COMPLETE' and (not shape_history or not all(r['gradient_norm']>0 for r in shape_history)):
            raise AssertionError('No measured Shape gradients')
        reports.append(dict(patient=patient,passed=True,comparisons=comparisons,motion_field_delta_exact=delta,
            status=x['status'],shape_updates=len(shape_history),joint_updates=len(x['joint_result']['history'])))
    return dict(passed=True,atol=atol,rtol=rtol,patients=reports)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root',required=True);parser.add_argument('--single-root',required=True)
    parser.add_argument('--patients',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    report=compare_runs(args.batch_root,args.single_root,args.patients)
    write_json(args.output,report)


if __name__=='__main__':main()
