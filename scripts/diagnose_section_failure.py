"""Reproduce a rejected short-run trajectory; independent section topology.

Exact native renderer is untouched. Indexed mesh edges give an independent
generic-plane contour graph, exposing native endpoint-merging artefacts.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train_geometry_heartffd_v2 import (load_config,load_checkpoint,data_identity,dataset_from_config,
    to_device,training_collate,stage_for_step,write_json)
from train_neural_ffd import restore_rng
from net.geometry_heartffdnet import loss_output_adapter
from utils.neural_ffd_losses import NeuralFFDLoss
from utils.slice_supervision import MeshSliceSupervisor
from utils.geometry_v2_checks import validate_observed_sections,GeometrySafetyError
from data.mask_transform import CHAMBERS


def indexed_section_topology(vertices,faces,rotation,translation_mm):
    """Build cycles using source edge IDs, without distance-based merging.

    Exact on-plane vertices are critical configurations, reported separately.
    Edge IDs are valid only for generic planes; no topology verdict is hidden.
    """
    import networkx as nx
    local=(np.asarray(vertices)-translation_mm)@rotation
    signed=local[:,2]
    crossing=(signed[faces]>0)
    graph=nx.MultiGraph()
    for face,side in zip(faces,crossing):
        endpoints=[]
        for i,j in ((0,1),(1,2),(2,0)):
            if side[i]!=side[j]:endpoints.append(tuple(sorted((int(face[i]),int(face[j])))))
        if endpoints:graph.add_edge(*endpoints)
    components=list(nx.connected_components(graph))
    return dict(closed_components=len(components),open_or_branch_vertices=sum(d!=2 for _,d in graph.degree()),
        edge_nodes=graph.number_of_nodes(),segments=graph.number_of_edges(),
        exact_on_plane_vertices=int((signed==0).sum()),minimum_abs_plane_vertex_distance_mm=float(abs(signed).min()),
        construction='source indexed edge intersections; no geometric endpoint merging')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config,data,protocol=load_config(ROOT/'configs/geometry_heartffd_v2.yaml');torch.set_num_threads(4)
    model,saved=load_checkpoint(args.checkpoint,args.device,identity=data_identity(data))
    optimizer=torch.optim.Adam(model.parameters(),lr=config['training']['learning_rate']);optimizer.load_state_dict(saved['optimizer_state'])
    restore_rng({'rng':saved['rng_state']},args.device)
    criterion=NeuralFFDLoss(model.chamber_ranges,model.faces,**config['loss']).to(args.device)
    slicer=MeshSliceSupervisor(model.chamber_ranges,model.faces,model.norm_to_mm,**config['slice'],execution_backend='batched',cell_chunk_size=8).to(args.device)
    ds=dataset_from_config(data);samples=[ds[i] for i in protocol['train_indices']]
    seed=saved['runtime']['seed'];step=saved['step'];history=[]
    args.output.mkdir(parents=True,exist_ok=True)
    while step<sum(protocol['updates'].values()):
        epoch,offset=divmod(step,len(samples));order=torch.randperm(len(samples),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        selected=order[offset];batch=to_device(training_collate([samples[selected]]),args.device)
        model.train();optimizer.zero_grad(set_to_none=True);output=model(batch)
        try:validate_observed_sections(model,output,batch)
        except GeometrySafetyError as error:
            import re
            patient,phase,view=[int(x) for x in re.findall(r'(?:patient|phase|view) (\d+)',str(error))]
            chamber=next(name for name in CHAMBERS if ('chamber '+name) in str(error));a,b=model.chamber_ranges[CHAMBERS.index(chamber)].tolist()
            vertices=output['vertices_4d'][patient,phase,a:b].detach().cpu().numpy()
            truth=batch['vertices_gt'][patient,phase,a:b].cpu().numpy()
            rotation=batch['pose_gt']['probe_R'][patient,phase,view].cpu().numpy()
            translation=batch['pose_gt']['probe_t'][patient,phase,view].cpu().numpy()*model.norm_to_mm
            faces=model.faces[chamber].cpu().numpy()
            report=dict(step_before_rejected_update=step,case_id=samples[selected]['case_id'],candidate_id=samples[selected]['candidate_id'],
                phase=phase,view=view,chamber=chamber,native_section=error.report,
                indexed_prediction=indexed_section_topology(vertices,faces,rotation,translation),
                indexed_truth=indexed_section_topology(truth,faces,rotation,translation),
                scope='diagnostic replay, excluded from formal sample/update budgets',history=history)
            np.savez_compressed(args.output/'failed_section.npz',vertices=vertices,truth=truth,faces=faces,rotation=rotation,translation_mm=translation)
            write_json(args.output/'diagnosis.json',report);print(report,flush=True);return
        stage=stage_for_step(step,config);sliced=slicer(output['vertices_4d'],batch) if stage=='joint' else None
        loss=criterion(loss_output_adapter(output),batch['vertices_gt'],slice_result=sliced,
            control_spacings_mm=model.control_spacings_mm,stage=stage)['total']
        loss.backward();optimizer.step();step+=1
        history.append(dict(step=step,loss=float(loss.detach())))
    write_json(args.output/'diagnosis.json',dict(status='not_reproduced',history=history))


if __name__=='__main__':main()
