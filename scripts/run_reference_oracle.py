"""Patient-specific control-only A/B diagnostic; no network or Motion updates.

A receives no GT vertices in its objective. Both arms use predicted-anchor local
geometry and original control regularization; B adds full corresponding GT
vertex supervision. Native binary slice objective, FFD and safety code reused.
Only fully audited block endpoints are accepted; failed trial blocks roll back.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TMPDIR', str(ROOT/'outputs/.runtime'))
import numpy as np
import torch
import yaml
from data.synthetic_4d import _sha256
from net.staged_heartffdnet import reference_loss
from train_geometry_heartffdnet import load_config, data_identity, write_json
from train_neural_ffd import dataset_from_config, to_device
from train_staged_heartffd import phase0, evaluate, code_identity
from utils.reference_refinement import NativeReferenceSamples, refinement_collate, load_stage1_initialization, make_refinement_slicer
from utils.staged_data import CachedStageSamples, configure_runtime_temp, workspace_path
from utils.neural_ffd_losses import NeuralFFDLoss
from utils.geometry_joint_checks import validate_template, validate_prediction
from utils.geometry_pair_policy import SourceAVCapPolicy, SETTINGS


class ReferenceControls(torch.nn.Module):
    """Raw physical controls -> original bounded coarse/medium/fine0 -> V0."""
    def __init__(self, model, initial):
        super().__init__()
        # Deliberately avoid registering the frozen encoder as an optimizer member.
        object.__setattr__(self, 'model', model)
        limit = model.heads.reference_limit_mm
        self.raw = torch.nn.ParameterDict({k: torch.nn.Parameter(limit*torch.atanh(v.detach()/limit))
                                          for k,v in initial['control_grids'].items()})
    def forward(self):
        m = self.model
        grids = {k:m.heads._bound(self.raw[k],m.heads.reference_limit_mm,m.control_spacings_mm[i])
                 for i,k in enumerate(('coarse','medium','fine'))}
        v = m.template_vertices_mm[None]
        for i,k in enumerate(('coarse','medium','fine')):
            v = m._deform(v, grids[k][:,0] if k=='fine' else grids[k], i)
        return dict(vertices_ref=v, vertices_4d=v[:,None], control_grids=grids)


def fit_objective(controls, observations, anchor, prior, slicer, weight, gt_criterion=None, gt=None):
    # A's contract rejects even accidentally passing GT through the observation map.
    if 'vertices_gt' in observations: raise ValueError('Oracle observations must exclude GT vertices')
    if (gt is None) != (gt_criterion is None): raise ValueError('GT and GT criterion must appear together')
    out = controls()
    regular = reference_loss(prior,out,anchor,controls.model.control_spacings_mm)['total']
    sliced = slicer(out['vertices_4d'], observations)['total']
    supervised = out['vertices_ref'].sum()*0 if gt is None else reference_loss(
        gt_criterion,out,gt,controls.model.control_spacings_mm)['total']
    return out, dict(total=regular+weight*sliced+supervised, prior=regular, slice=sliced, gt=supervised)


@torch.no_grad()
def audit(model, out, policy, full=True):
    result=validate_prediction(model,out,reference_only=True,sample_jacobian=full,check_intersections=full)
    if any(float(v.abs().max())>model.heads.reference_limit_mm+1e-5 for v in out['control_grids'].values()):
        raise ValueError('Control displacement limit exceeded')
    if full: result['inter_chamber']=policy.audit_frames(out['vertices_4d'][0].cpu().numpy())
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',default='cuda:7');ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--block',type=int,default=50)
    ap.add_argument('--cases',type=int,nargs='+',default=[233,306,848]);ap.add_argument('--lr',type=float,default=.03)
    args=ap.parse_args()
    if args.steps<1 or args.block<1: raise ValueError('Positive budgets required')
    configure_runtime_temp(ROOT/'outputs/.runtime');torch.set_num_threads(4)
    torch.manual_seed(2026);np.random.seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=workspace_path(args.output);output.mkdir(parents=True,exist_ok=False)
    protocol=yaml.safe_load((ROOT/'configs/staged_heartffd_refine.yaml').read_text())
    base=yaml.safe_load((ROOT/protocol['base_protocol']).read_text())
    config,data=load_config(ROOT/base['base_config']);identity=data_identity(data)
    protocol['slice']=dict(protocol['slice'],pixel_stride=2)
    run=dict(device=args.device,cases=args.cases,steps=args.steps,block=args.block,lr=args.lr,seed=2026,
        slice_weight=2.1242689150749596,slice=protocol['slice'],reduction='cell_mean',
        prior_weights=dict(shape=0.,edge=.05,laplacian=.05,normal=.01,control=.001),
        B_addition='Existing shape Huber over ALL corresponding GT frame0 vertices, weight1',
        selection='minimum surrogate among fully safety-audited block endpoints; no GT selection for A',
        scope='Selected VAL tail cases; diagnostic optimization, not validation generalization or theoretical ceiling',
        checkpoint_sha256=protocol['stage1_checkpoint_sha256'],data_identity=identity,source=code_identity(),
        runner_sha256=_sha256(Path(__file__)),test_loaded=False,network_trainable=False)
    write_json(output/'protocol.json',run)
    (output/'source.py').write_text(Path(__file__).read_text())
    print(json.dumps(dict(status='loading',device=args.device,output=str(output))),flush=True)
    model,_=load_stage1_initialization(protocol,identity,args.device);model.freeze_reference()
    validate_template(model);policy=SourceAVCapPolicy(model,SETTINGS)
    slicer=make_refinement_slicer(model,protocol,'B').to(args.device)
    prior=NeuralFFDLoss(model.chamber_ranges,model.faces,weights=run['prior_weights']).to(args.device)
    gtcriterion=NeuralFFDLoss(model.chamber_ranges,model.faces,weights=dict(
        shape=1.,edge=0.,laplacian=0.,normal=0.,control=0.)).to(args.device)
    ds=dataset_from_config(data,'val')
    rows={int(row['candidate_id']):i for i,row in enumerate(ds.sources)}
    original=model.reference_output
    results=[]
    for candidate in args.cases:
        cache=CachedStageSamples(ds,[rows[candidate]],ROOT/'outputs/goal04s/preprocessed_cache')
        sample=NativeReferenceSamples(cache,ROOT/'outputs/goal04s/preprocessed_cache/native_reference_sdf')[0]
        batch=to_device(refinement_collate([sample]),args.device)
        observations=phase0({k:v for k,v in batch.items() if k!='vertices_gt'})
        with torch.no_grad(): initial=original({k:v for k,v in batch.items() if k!='vertices_gt'})
        anchor=initial['vertices_4d'].detach()
        completed={}
        for arm in ('A','B'):
            dest=output/str(candidate)/arm;dest.mkdir(parents=True)
            control=ReferenceControls(model,initial)
            with torch.no_grad(): error=float((control()['vertices_ref']-initial['vertices_ref']).abs().max())
            if error>1e-4: raise ValueError(f'Initialization differs: {error}')
            audit(model,control(),policy)
            opt=torch.optim.Adam(control.parameters(),lr=args.lr)
            kw={} if arm=='A' else dict(gt_criterion=gtcriterion,gt=batch['vertices_gt'][:,:1])
            fn=lambda:fit_objective(control,observations,anchor,prior,slicer,run['slice_weight'],**kw)
            # Exercise A with no GT key and require informative finite control gradients.
            _,first=fn();first['total'].backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in control.parameters()):
                raise ValueError('Invalid control gradient')
            opt.zero_grad(set_to_none=True)
            best=float(first['total'].detach());beststate=copy.deepcopy(control.state_dict());beststep=0
            accepted=copy.deepcopy(control.state_dict());acceptedopt=copy.deepcopy(opt.state_dict())
            history=[];rejected=0;scale=1.;start=time.monotonic()
            print(json.dumps(dict(status='optimizing',case=candidate,arm=arm,initial_loss=best,initialization_max_abs_mm=error)),flush=True)
            for step in range(1,args.steps+1):
                lr=scale*(.001+(args.lr-.001)*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
                for group in opt.param_groups: group['lr']=lr
                try:
                    opt.zero_grad(set_to_none=True);out,loss=fn();audit(model,out,policy,full=False)
                    loss['total'].backward()
                    norm=torch.nn.utils.clip_grad_norm_(control.parameters(),10.,error_if_nonfinite=True)
                    opt.step()
                    if step%args.block and step!=args.steps: continue
                    out,values=fn();safety=audit(model,out,policy)
                    row=dict(step=step,seconds=time.monotonic()-start,lr=lr,grad_norm=float(norm),
                             **{k:float(v.detach()) for k,v in values.items()},accepted=True)
                    accepted=copy.deepcopy(control.state_dict());acceptedopt=copy.deepcopy(opt.state_dict())
                    if row['total']<best:best=row['total'];beststate=copy.deepcopy(accepted);beststep=step
                except (ValueError,RuntimeError) as exc:
                    if isinstance(exc,torch.cuda.OutOfMemoryError): raise
                    control.load_state_dict(accepted);opt.load_state_dict(acceptedopt);scale*=.5;rejected+=1
                    row=dict(step=step,seconds=time.monotonic()-start,accepted=False,error=str(exc),lr_scale=scale)
                    if rejected>10: raise RuntimeError('Repeated unsafe/invalid trials; abort without relaxing constraints') from exc
                history.append(row)
                with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                torch.save(dict(step=step,raw_controls=accepted,optimizer=acceptedopt,best_raw_controls=beststate,
                    best_step=beststep,best_loss=best,protocol=run,case_id=sample['case_id'],arm=arm),dest/'latest.pth')
                print(json.dumps(dict(case=candidate,arm=arm,**row)),flush=True)
            torch.save(dict(raw_controls=accepted,step=args.steps),dest/'endpoint.pth')
            control.load_state_dict(beststate)
            with torch.no_grad(): chosen=control();safety=audit(model,chosen,policy)
            torch.save(dict(raw_controls=beststate,step=beststep,output=chosen,safety=safety),dest/'selected.pth')
            completed[arm]=chosen
            write_json(dest/'fit.json',dict(case=candidate,arm=arm,initial_loss=float(first['total'].detach()),
                selected_loss=best,selected_step=beststep,attempted_steps=args.steps,rejected_blocks=rejected,
                seconds=time.monotonic()-start,history=history,safety=safety,
                converged=False,convergence_note='Finite budget; inspect late loss/gradient history. Not a theoretical upper bound.'))
        # GT3D evaluation only AFTER both fitting jobs for this case have finished.
        clean={k:v for k,v in sample.items() if k not in ('clipped_foreground','center_mm')}
        reference_cache={}
        for label,out in [('initial',initial),*completed.items()]:
            model.reference_output=lambda _batch,value=out:value
            report=evaluate(model,[clean],args.device,full=True,reference_cache=reference_cache,
                            prediction_dir=output/str(candidate)/label/'predictions')
            write_json(output/str(candidate)/label/'evaluation.json',report)
            results.append(dict(candidate=candidate,arm=label,**report['summary']))
            print(json.dumps(dict(status='evaluated',**results[-1])),flush=True)
        model.reference_output=original
        write_json(output/'results.json',dict(complete=False,results=results))
    write_json(output/'results.json',dict(complete=True,results=results,scope=run['scope']))
    print('ORACLE_COMPLETE',flush=True)


if __name__=='__main__':main()
