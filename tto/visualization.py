"""Native overlays and fixed-camera 4D mesh previews; never used for scoring."""
from pathlib import Path

import cv2
import numpy as np

from data.mask_transform import CHAMBERS,VIEWS
from .evaluator import native_exact


def export_visualizations(model,patient,state,folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    state.validate()
    if state.observation_pose is None:
        raise ValueError('Native overlays require registered observation-plane poses')
    trajectory=state.patient_trajectory().detach().cpu().numpy()
    pose={key:value.detach().cpu().numpy() for key,value in state.observation_pose.items()}
    faces={c:value.cpu().numpy() for c,value in model.faces.items()}
    ranges=dict(zip(CHAMBERS,model.chamber_ranges.tolist()))
    for phase in (0,6,12,18):
        native=patient.native(phase)
        for view,name in enumerate(VIEWS):
            h,w=native['masks'][view].shape[-2:]
            if patient.images:
                raw=patient.images[view][phase]
                if raw.ndim==2:raw=np.repeat(raw[...,None],3,-1)
                canvas=raw[...,:3].astype(np.uint8).copy()
            else:canvas=np.zeros((h,w,3),np.uint8)
            for chamber,c in enumerate(CHAMBERS):
                if not native['annotated'][view,chamber]:continue
                a,b=ranges[c]
                predicted=native_exact(trajectory[phase,a:b],faces[c],pose['R'][view],pose['center_mm'][view],
                                       native['spacing'][view],(h,w))
                for mask,color in ((native['masks'][view][chamber],(0,255,0)),(predicted,(255,0,255))):
                    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas,contours,-1,color,1)
            cv2.imwrite(str(folder/f'{name}_phase_{phase:02d}_native.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
    lower=trajectory.min((0,1));upper=trajectory.max((0,1));center=(lower+upper)/2;radius=max(upper-lower)/2*1.05
    colors=('#d94b55','#eda849','#5085d9','#54bbbf');frames=[]
    figure=plt.figure(figsize=(5,5),dpi=80)
    for phase in range(24):
        figure.clear();axis=figure.add_subplot(111,projection='3d')
        for chamber,color in zip(CHAMBERS,colors):
            a,b=ranges[chamber]
            axis.add_collection3d(Poly3DCollection(trajectory[phase,a:b][faces[chamber]],facecolors=color,edgecolors='none',alpha=.65))
        axis.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
                 zlim=(center[2]-radius,center[2]+radius),title=f'{patient.patient_id}  phase {phase}/24')
        axis.set_box_aspect((1,1,1));axis.view_init(elev=20,azim=35);axis.set_axis_off()
        figure.canvas.draw();frame=Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[...,:3].copy())
        frames.append(frame)
        if phase==0:frame.save(folder/'reference_mesh.png')
    plt.close(figure)
    frames[0].save(folder/'mesh_24phases.gif',save_all=True,append_images=frames[1:],duration=90,loop=0)
