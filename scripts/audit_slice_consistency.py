"""Frozen G1 soft/native audit with bounded physically valid FFD candidates."""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import default_collate
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train_geometry_heartffd_v2 import (load_config,dataset_from_config,to_device,write_json,code_identity)
from train_geometry_heartffdnet import load_checkpoint
from net.geometry_heartffdnet_v2 import GeometryHeartFFDNetV2
from utils.geometry_pair_policy import bind_geometry_policy,validate_source_geometry,validate_output_geometry
from utils.slice_supervision import MeshSliceSupervisor
from utils.slice_consistency_audit import paired_slice_cells,compare_slice_updates
from utils.geometry_v2_checks import validate_observed_sections,GeometrySafetyError
from data.synthetic_4d import _sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/geometry_goal04_final/latest.pth')
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    config,data,protocol=load_config(ROOT/'configs/geometry_heartffd_v2.yaml')
    model,saved=load_checkpoint(args.checkpoint,args.device);model.eval()
    bind_geometry_policy(model,config['geometry_checks']);validate_source_geometry(model)
    slicer=MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,**config['slice'],
        execution_backend='batched',cell_chunk_size=8).to(args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'experiment.json',dict(checkpoint_sha256=_sha256(args.checkpoint),
        code_identity=code_identity(),scope='Frozen G1 TRAIN0/VAL0 diagnostics; no model updates or loss changes'))
    for split in ('train','val'):
        sample=dataset_from_config(data,split)[0];batch=to_device(default_collate([sample]),args.device)
        with torch.no_grad():prediction=model(batch)
        baseline=paired_slice_cells(model,prediction['vertices_4d'],batch,slicer)
        base_safety=validate_output_geometry(model,prediction)
        final=prediction['control_grids']['fine'].detach()
        limit=final.new_full((1,24,1,1,1,1),model.heads.increment_limit_mm);limit[:,0]=model.heads.reference_limit_mm
        raw=(limit*torch.atanh((final/limit).clamp(-.999999,.999999))).requires_grad_()
        def output(field):
            controls=dict(prediction['control_grids'],fine=model.heads._bound(field,limit,model.control_spacings_mm[2]))
            vertices=GeometryHeartFFDNetV2.apply_controls(model,controls)
            return dict(vertices_4d=vertices,vertices_ref=vertices[:,0],control_grids=controls)
        reproduced=output(raw)
        torch.testing.assert_close(reproduced['vertices_4d'],prediction['vertices_4d'],atol=2e-5,rtol=1e-6)
        loss=slicer(reproduced['vertices_4d'],batch)['total'];gradient=torch.autograd.grad(loss,raw)[0]
        direction=-gradient/gradient.abs().max().clamp_min(1e-20)
        candidates=[]
        for step_mm in (.01,.03,.1,.3):
            with torch.no_grad():candidate=output(raw+step_mm*direction)
            try:
                safety=validate_output_geometry(model,candidate)
                topology=validate_observed_sections(model,candidate,batch)
                cells=paired_slice_cells(model,candidate['vertices_4d'],batch,slicer)
                candidates.append(dict(step_max_raw_mm=step_mm,status='safe',summary=cells['summary'],
                    comparison=compare_slice_updates(baseline,cells),safety=safety,sections=topology))
            except (GeometrySafetyError,ValueError) as error:
                candidates.append(dict(step_max_raw_mm=step_mm,status='rejected_geometry',reason=str(error)))
        report=dict(case_id=sample['case_id'],baseline=baseline,baseline_safety=base_safety,candidates=candidates,
            scope='Separate supervision diagnosis; benefits never counted as architecture benefits',
            topology_scope='Actual components checked per candidate; matching counts do not prove identical contour shape')
        write_json(args.output/(split+'.json'),report)
        print(split,baseline['summary'],flush=True)


if __name__=='__main__':main()
