"""Quantify rejected/partial predictions without declaring them acceptable.

Same 48-update G1/G2 prefix, fixed VAL8, original metrics and exact renderer.
Geometry failures are retained per patient; raw scores do not override them.
"""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import default_collate
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train_geometry_heartffd_v2 import (load_config,load_checkpoint,data_identity,dataset_from_config,
    to_device,write_json,evaluate,code_identity)
from train_neural_ffd import exact_metrics,validation_summary
from utils.neural_ffd_losses import reconstruction_metrics
from utils.geometry_pair_policy import validate_source_geometry,validate_output_geometry
from utils.geometry_v2_checks import validate_observed_sections,GeometrySafetyError
from utils.geometry_motion_metrics import detailed_motion_metrics,observed_vertex_regions
from data.synthetic_4d import _sha256


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/goal04a2/diagnostic_prefix48')
    parser.add_argument('--device',default='cuda:3');args=parser.parse_args();torch.set_num_threads(4)
    config,data,protocol=load_config(ROOT/'configs/geometry_heartffd_v2.yaml')
    ds=dataset_from_config(data,'val');samples=[ds[i] for i in protocol['val_indices']]
    for variant in ('A0','A4'):
        path=args.root/(variant+'_seed2026')/'latest.pth'
        model,saved=load_checkpoint(path,args.device,identity=data_identity(data))
        if saved['step']!=48:raise ValueError('Exactly48 equal updates required for this diagnostic')
        validate_source_geometry(model);model.eval();rows=[]
        for index,sample in enumerate(samples):
            batch=to_device(default_collate([sample]),args.device);prediction=model(batch)
            geometry={};accepted=True
            try:geometry['physical']=validate_output_geometry(model,prediction)
            except (GeometrySafetyError,ValueError) as error:
                accepted=False;geometry['physical']=dict(status='rejected',reason=str(error),detail=getattr(error,'report',None))
            try:geometry['sections']=validate_observed_sections(model,prediction,batch)
            except GeometrySafetyError as error:
                accepted=False;geometry['sections']=dict(status='rejected',reason=str(error),detail=error.report)
            regions=observed_vertex_regions(batch['vertices_gt'].to(prediction['vertices_4d']),batch,model.chamber_ranges,model.norm_to_mm)
            rows.append(dict(case_id=sample['case_id'],candidate_id=sample['candidate_id'],geometry_accepted=accepted,
                geometry=geometry,metrics=reconstruction_metrics(prediction['vertices_4d'],batch['vertices_gt'],model.chamber_ranges,model.faces),
                detailed_motion=detailed_motion_metrics(prediction['vertices_4d'],batch['vertices_gt'],model.chamber_ranges,model.faces,batch['phases'],regions=regions),
                exact=exact_metrics(model,prediction['vertices_4d'],batch)))
            print(variant,index+1,'geometry accepted:',accepted,flush=True)
        report=dict(cases=rows,case_count=len(rows),summary=validation_summary(dict(cases=rows,case_count=len(rows))),
            scope='Equal48-update diagnostic only; not preregistered144-update effect acceptance',
            eligible_for_formal_effect_pass=False,geometry_accepted_patients=sum(r['geometry_accepted'] for r in rows),
            checkpoint_sha256=_sha256(path),code_identity=code_identity(),actual_step=saved['step'],
            diagnostic_choice='Common completed early motion prefix after144-update runs failed; no best-checkpoint selection',
            pose_mode='known_pose',test_read=False)
        write_json(path.parent/'diagnostic_val8.json',report);print(variant,report['summary'],flush=True)


if __name__=='__main__':main()
