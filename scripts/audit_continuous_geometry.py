"""Independently certify every cyclic chord and observed section of an oracle.

The appended endpoint exists only for CCD, never as a25th network phase.
Continuous interpolation here is piecewise linear between sampled vertices.
"""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import default_collate
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train_geometry_heartffd_v2 import (load_config,dataset_from_config,build_model,
    data_identity,to_device,write_json,code_identity)
from utils.geometry_pair_policy import validate_source_geometry
from utils.geometry_v2_checks import validate_observed_sections,GeometrySafetyError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    config,data,protocol=load_config(ROOT/'configs/geometry_heartffd_v2.yaml')
    ds=dataset_from_config(data);model=build_model(ds,config).to(args.device);validate_source_geometry(model)
    args.output.mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        saved=torch.load(ROOT/f'outputs/goal04a2/motion_oracle/{split}_oracle.pth',map_location='cpu',weights_only=True)
        if saved['data_identity']!=data_identity(data):raise ValueError('Oracle dataset identity differs')
        controls={k:v.to(args.device) for k,v in saved['control_grids'].items()}
        reference=saved['vertices_ref'].to(controls['fine'])
        with torch.no_grad():
            values=[reference]
            for phase in range(1,24):values.append(model._deform(values[-1],controls['fine'][:,phase],2))
            vertices=torch.stack(values,1)
        batch=to_device(default_collate([dataset_from_config(data,split)[0]]),args.device)
        report=dict(code_identity=code_identity(),scope='bounded oracle; all24 frames and closing23->0 chord',
            interpolation='piecewise linear sampled vertex trajectories; not a cubic continuous model claim')
        try:report['sections']=validate_observed_sections(model,{'vertices_4d':vertices},batch)
        except GeometrySafetyError as error:report['sections']=dict(status='rejected',reason=str(error),detail=error.report)
        write_json(args.output/(split+'.json'),report)
        print(split,'sections',report['sections']['status'],flush=True)
        report['continuous']=model._source_av_cap_policy.certify(vertices[0].cpu().numpy(),continuous=True)
        report['overall_passed']=bool(report['continuous']['passed'] and report['sections']['status']=='pass')
        write_json(args.output/(split+'.json'),report);print(split,'continuous',report['continuous']['passed'],flush=True)


if __name__=='__main__':main()
