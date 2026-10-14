"""Standalone scientific figures from saved numerical audit artifacts."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/heartssm-goal04a2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig,path):
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/goal04a2'))
    parser.add_argument('--response',type=Path)
    args=parser.parse_args();out=args.root/'figures';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for split,color in (('train','#176e9c'),('val','#d26935')):
        path=args.root/'motion_oracle'/f'{split}.json'
        if not path.exists():continue
        d=json.loads(path.read_text());hist=d['history']
        axes[0].plot([0]+[r['step'] for r in hist],[d['initial_rmse_mm']]+[r['rmse_mm'] for r in hist],label=split,color=color)
        m=d['metrics']['cases'][0]['chambers']
        axes[1].plot(range(4),[m[c]['relative_motion_rmse_mm'] for c in ('LV','LA','RV','RA')],marker='o',label=split,color=color)
    axes[0].set(xlabel='Accepted/rejected optimizer attempts',ylabel='Motion RMSE (mm)',title='Restricted FFD motion oracle')
    axes[1].set(xticks=range(4),xticklabels=['LV','LA','RV','RA'],ylabel='Motion RMSE (mm)',title='Final chamber errors')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,out/'motion_oracle')
    if args.response:
        for split in ('train','val'):
            path=args.response/f'{split}.json'
            if not path.exists():continue
            d=json.loads(path.read_text());fig,axes=plt.subplots(4,2,figsize=(9,13))
            for row,target in enumerate(d['targets']):
                for col,(key,label) in enumerate((('direct_control_response_output_by_input','Control component RMS (mm)'),
                    ('accumulated_mesh_response_output_by_input','Relative mesh vector RMS (mm)'))):
                    values=np.asarray(target[key]);im=axes[row,col].imshow(values,origin='lower',aspect='equal',cmap='magma')
                    axes[row,col].set(xlabel='Input phase',ylabel='Output phase',title=target['view']+'/'+target['chamber']+' '+label)
                    fig.colorbar(im,ax=axes[row,col],fraction=.046)
            fig.suptitle(split.upper()+' localized contour response; diagnostic checkpoint',y=1.01)
            fig.tight_layout();save(fig,out/('local_response_'+split))
    section=args.root/'section_failure_a0'/'failed_section.npz'
    if section.exists():
        import trimesh
        d=np.load(section);fig,axes=plt.subplots(1,2,figsize=(10,4))
        contours={}
        for name,color in (('truth','#176e9c'),('vertices','#bb3939')):
            mesh=trimesh.Trimesh(vertices=d[name],faces=d['faces'],process=False)
            cut=mesh.section(plane_origin=d['translation_mm'],plane_normal=d['rotation'][:,2])
            lines=[(curve-d['translation_mm'])@d['rotation'] for curve in cut.discrete]
            contours[name]=lines
            for line in lines:
                for ax in axes:ax.plot(line[:,0],line[:,1],color=color,lw=1.5,label=name)
        small=min(contours['vertices'],key=len)
        center=small[:,:2].mean(0);span=max(np.ptp(small[:,:2],axis=0).max(),1.)*1.2
        axes[1].set_xlim(center[0]-span,center[0]+span);axes[1].set_ylim(center[1]-span,center[1]+span)
        axes[0].set_title('Reproduced rejection: LV, phase16, A2C')
        axes[1].set_title('Additional closed contour (magnified)')
        for ax in axes:ax.set_aspect('equal');ax.set_xlabel('Plane x (mm)');ax.set_ylabel('Plane y (mm)');ax.grid(alpha=.2)
        save(fig,out/'illegal_section_reproduction')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,('train','val')):
        path=args.root/'slice_audit'/f'{split}.json'
        if not path.exists():continue
        d=json.loads(path.read_text());rows=d['baseline']['cells']
        ax.scatter([r['soft_total'] for r in rows],[1-r['exact_dice'] for r in rows],s=12,alpha=.7)
        ax.set(xlabel='Binary native soft total loss',ylabel='1 - Exact native Dice',title=split.upper()+' frozen G1; 192 cells');ax.grid(alpha=.2)
    save(fig,out/'soft_native_consistency')


if __name__=='__main__':main()
