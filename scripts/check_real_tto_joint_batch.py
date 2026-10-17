"""CPU-only actual-checkpoint batch parity; no optimizer or clinical TTO endpoint."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from tto.config import load_config,resolve
from tto.dataset import RealTTODataset,collate_patients
from tto.io import read_json,sha256,write_json
from tto.joint_pipeline import split_batch
from tto.losses import RealTTOLoss
from tto.motion import MotionPropagation
from tto.pose import estimated_pose
from tto.prior import FrozenNeuralFFDPrior


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/real_tto_joint_v1.yaml')
    parser.add_argument('--patients',nargs='+',default=['C01','C02'])
    parser.add_argument('--output',default='outputs/real_tto_joint_v1_20261016/independent_audit/actual_batch_parity.json')
    args=parser.parse_args();torch.set_num_threads(2)
    if len(args.patients)<2:raise ValueError('At least two independent patients are required')
    cfg=load_config(resolve(args.config));torch.manual_seed(cfg['runtime']['seed']);began=time.time()
    checks=[];errors=[]

    def compare(name,first,second,*,atol,rtol):
        maximum=float((first-second).abs().max())
        try:torch.testing.assert_close(first,second,atol=atol,rtol=rtol);passed=True
        except AssertionError as error:passed=False;errors.append(dict(check=name,error=str(error)))
        checks.append(dict(name=name,passed=passed,max_absolute_difference=maximum,atol=atol,rtol=rtol))

    def compare_tree(name,first,second):
        if isinstance(first,torch.Tensor):compare(name,first,second,atol=1e-4,rtol=1e-5)
        elif isinstance(first,dict):
            if first.keys()!=second.keys():errors.append(dict(check=name,error='Condition/control keys differ'))
            for key in first.keys() & second.keys():compare_tree(name+'.'+key,first[key],second[key])
        else:
            passed=first==second
            checks.append(dict(name=name,passed=passed,comparison='exact metadata or disabled field'))
            if not passed:errors.append(dict(check=name,error='Condition/control metadata differ'))

    print('BATCH_PARITY_STAGE load_actual_data_and_frozen_checkpoint',flush=True)
    patients=[RealTTODataset(resolve(cfg['data']['root']),[name],config=cfg['data'])[0] for name in args.patients]
    prior=FrozenNeuralFFDPrior(resolve(cfg['prior']['checkpoint']),device='cpu',expected_sha256=cfg['prior']['sha256'])
    calibration=read_json(resolve(cfg['prior']['calibration']))
    initial=prior.initialize(collate_patients(patients)['observations'])
    for owner,patient in enumerate(patients):
        print('BATCH_PARITY_STAGE independent_initialization',patient.patient_id,flush=True)
        single=prior.initialize(collate_patients([patient])['observations'])
        for key in ('z_shape_initial','z_motion_initial','vertices_reference_initial','vertices_4d_initial'):
            compare(f'{patient.patient_id}.batch_vs_independent_encoder.{key}',initial[key][owner:owner+1],single[key],
                    atol=1e-4 if key.startswith('vertices') else 1e-5,rtol=1e-5)
        for key in initial['control_grids_initial']:
            compare_tree(f'{patient.patient_id}.batch_vs_independent_encoder.control.{key}',
                         split_batch(initial['control_grids_initial'][key],owner),single['control_grids_initial'][key])
        compare_tree(f'{patient.patient_id}.batch_vs_independent_encoder.conditions',split_batch(initial['conditions'],owner),single['conditions'])

    def objective(subset,starting):
        motion=MotionPropagation(prior,starting,cfg['motion_limits'])
        z=starting['z_shape_initial'].clone().requires_grad_()
        delta=torch.zeros(len(subset),3,6,requires_grad=True)
        base={key:torch.tensor(value,dtype=z.dtype)[None].expand(len(subset),*torch.tensor(value).shape).clone()
              for key,value in calibration['initial_pose'].items()}
        supervisor=RealTTOLoss(prior.model,subset,cfg['loss'],cfg['supervision'])
        result=supervisor(motion.decode(z),estimated_pose(delta,base))
        motion.assert_frozen()
        return result,z,delta,supervisor

    print('BATCH_PARITY_STAGE full_cycle_batch_gradients',flush=True)
    result,z,delta,supervisor=objective(patients,initial)
    values=result['slice'].detach().clone();components={k:v.detach().clone() for k,v in result['components'].items()}
    gradients=[];coverage=[]
    for owner,patient in enumerate(patients):
        dz,dp=torch.autograd.grad(result['slice'][owner],(z,delta),retain_graph=owner<len(patients)-1)
        other=[i for i in range(len(patients)) if i!=owner]
        compare(f'{patient.patient_id}.cross_patient_shape_gradient',dz[other],torch.zeros_like(dz[other]),atol=0.,rtol=0.)
        compare(f'{patient.patient_id}.cross_patient_pose_gradient',dp[other],torch.zeros_like(dp[other]),atol=0.,rtol=0.)
        gradients.append((dz[owner:owner+1].clone(),dp[owner:owner+1].clone()))
        active=[row for row in supervisor.active if row['owner']==owner]
        norms=dict(shape=float(dz[owner].norm()),translation=float(dp[owner,:,:3].norm()),rotation=float(dp[owner,:,3:].norm()))
        if not all(value>0 for value in norms.values()):errors.append(dict(check=patient.patient_id,error='Missing actual full-cycle parameter gradient'))
        coverage.append(dict(patient=patient.patient_id,input_sha256=patient.identity,cells=len(active),
            mesh_lower_phases=sorted({row['phase'] for row in active}),gradient_norms=norms))
    del result,z,delta,supervisor

    for owner,patient in enumerate(patients):
        print('BATCH_PARITY_STAGE identical_initial_state_single_gradients',patient.patient_id,flush=True)
        # Isolate batching arithmetic from the separately measured encoder roundoff.
        single,z,delta,_=objective([patient],split_batch(initial,owner))
        dz,dp=torch.autograd.grad(single['slice'].sum(),(z,delta))
        compare(f'{patient.patient_id}.slice_loss',values[owner:owner+1],single['slice'].detach(),atol=2e-6,rtol=2e-5)
        for key in components:
            compare(f'{patient.patient_id}.component.{key}',components[key][owner:owner+1],single['components'][key].detach(),atol=2e-6,rtol=2e-5)
        compare(f'{patient.patient_id}.shape_gradient',gradients[owner][0],dz,atol=2e-5,rtol=2e-4)
        compare(f'{patient.patient_id}.pose_gradient',gradients[owner][1],dp,atol=2e-5,rtol=2e-4)
    prior.assert_frozen();assert not torch.cuda.is_initialized()
    report=dict(passed=not errors,observed=time.time(),elapsed_s=time.time()-began,device='cpu',
        optimizer_updates=0,clinical_tto_endpoint_generated=False,geometry_status='NOT_CHECKED_BY_THIS_CHECK',
        checkpoint_sha256=prior.checkpoint_sha256,script_sha256=sha256(Path(__file__)),coverage=coverage,checks=checks,errors=errors,
        scope='Actual checkpoint and native C01/C02 data by default. Batch versus separate encoder/decoder initialization; pure full-cycle slice loss and Shape/Pose gradients at identical frozen initialization; exact zero cross-patient gradients. No clinical optimization, candidate selection, geometry gate or endpoint-resume claim.')
    write_json(resolve(args.output),report)
    print(dict(passed=report['passed'],checks=len(checks),errors=errors,elapsed_s=report['elapsed_s']),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
