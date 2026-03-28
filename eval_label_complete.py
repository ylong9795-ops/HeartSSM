import os
import glob
import argparse
import time
import nibabel as nib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.unet import UNet
from utils.utils import (
    grid_sample_2d,
    find_mutli_view_plane
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='UK Biobank CMR label completion')
    parser.add_argument('--data_dir', default='YOUR_UKBB_DIR', type=str, help='dataset directory')
    parser.add_argument('--save_dir', default='./data/ukbb/', type=str, help='saving directory')
    parser.add_argument('--n_frame', default=50, type=int, help='number of time frames')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    n_frame = args.n_frame
    device = args.device

    # --------------------
    # initialize grids
    # --------------------
    
    # initialize 2D grid for grid sampling
    width = 64
    grid_2d = [torch.arange(0, 2*width), torch.arange(0, 2*width)]
    grid_2d = torch.stack(torch.meshgrid(grid_2d, indexing='ij')).permute(1,2,0)
    grid_2d = grid_2d - torch.tensor([width-0.5, width-0.5])
    grid_2d = grid_2d[None].to(device)

    # initialize 3D grid for grid sampling
    voxel_size = 2.0
    grid_size_3d = [96,96,128]
    # affine matrix
    affine_3d = np.eye(4) * voxel_size
    affine_3d[-1,-1] = 1
    affine_3d[:3,-1] = - np.array(grid_size_3d) + 1
    affine_3d_ = torch.Tensor(affine_3d).to(device)
    # grid for 3d sampling
    grid_3d = [torch.arange(0, s) for s in grid_size_3d]
    grid_3d = torch.stack(torch.meshgrid(
        grid_3d, indexing='ij')).permute(1,2,3,0).to(device).float()
    grid_3d = grid_3d[None] @ affine_3d_[:3,:3].T + affine_3d_[:3,-1]

    # load template 3D seg to compute center of mass
    seg_atlas_nii = nib.load('./template/whs_template_seg.nii.gz')
    seg_atlas = seg_atlas_nii.get_fdata()
    seg_atlas = torch.LongTensor(seg_atlas[None]).to(device)
    seg_atlas = F.one_hot(seg_atlas, num_classes=6).permute(0,4,1,2,3).float()
    seg_atlas = F.interpolate(
        seg_atlas, size=grid_size_3d, mode='trilinear', align_corners=True)
    seg_atlas = seg_atlas.argmax(1)[0].cpu().numpy().astype(np.uint8)
    center_atlas = np.stack(np.where(seg_atlas>0)).mean(-1)
    volume_atlas = (seg_atlas>0).sum()

    # --------------------
    # load nn model
    # --------------------
    unet = UNet(
        dim_in=6,
        dim_hid=[32,64,128,256,256],
        dim_out=6,
        drop_rate=0.2).to(device)

    unet.load_state_dict(torch.load(
        './ckpts/model_label_complete_lcunet.pt', map_location=device))
    unet.eval();
    
    # --------------------
    # 4d seg prediction
    # --------------------
    # data directory
    subj_list = sorted(glob.glob(save_dir+'*'))
    runtime = []
    
    for i in tqdm(range(len(subj_list))):
        subj_id = subj_list[i].split('/')[-1]

        # --------------------
        # load 2d seg data
        # --------------------
        seg_2ch_nii = nib.load(data_dir+subj_id+'/visit_2/seg_la_2ch.nii.gz')
        seg_4ch_nii = nib.load(data_dir+subj_id+'/visit_2/seg4_la_4ch.nii.gz')
        seg_sax_nii = nib.load(data_dir+subj_id+'/visit_2/seg_sa.nii.gz')

        seg_2ch = seg_2ch_nii.get_fdata().transpose(2,3,0,1)[:,:n_frame]
        seg_4ch = seg_4ch_nii.get_fdata().transpose(2,3,0,1)[:,:n_frame]
        seg_sax = seg_sax_nii.get_fdata().transpose(2,3,0,1)[:,:n_frame]

        n_slice = seg_sax.shape[0]

        seg_2ch_tensor = torch.Tensor(seg_2ch).to(device)
        seg_4ch_tensor = torch.Tensor(seg_4ch).to(device)
        seg_sax_tensor = torch.Tensor(seg_sax).to(device)

        # load affine matrix
        affine_all = np.load(save_dir+subj_id+'/affine.npy')
        affine_2ch = affine_all[0]
        affine_4ch = affine_all[1]
        affine_sax = affine_all[2]

        # load center and motion
        shift_all = np.load(save_dir+subj_id+'/shift.npy')
        shift_2ch = torch.Tensor(shift_all[0]).to(device)
        shift_4ch = torch.Tensor(shift_all[1]).to(device)
        shift_sax = torch.Tensor(shift_all[2:]).to(device)

        t_start = time.time()
        # --------------------
        # 2d seg resample
        # --------------------
        # grid sample 2d segmentations
        grid_2ch = grid_2d + shift_2ch
        grid_4ch = grid_2d + shift_4ch
        grid_sax = grid_2d + shift_sax[:,None,None]

        seg_2ch_proc = grid_sample_2d(seg_2ch_tensor, grid_2ch, mode='nearest')
        seg_4ch_proc = grid_sample_2d(seg_4ch_tensor, grid_4ch, mode='nearest')
        seg_sax_proc = grid_sample_2d(seg_sax_tensor, grid_sax, mode='nearest')

        seg_2d_2ch = seg_2ch_proc.long().permute(1,2,3,0).repeat(1,1,1,2)
        seg_2d_4ch = seg_4ch_proc.long().permute(1,2,3,0).repeat(1,1,1,2)
        seg_2d_sax = seg_sax_proc.long().permute(1,2,3,0)

        # compute inverse affine transformation
        inv_affine_2ch = torch.linalg.inv(torch.Tensor(affine_2ch).to(device))[None]
        inv_affine_4ch = torch.linalg.inv(torch.Tensor(affine_4ch).to(device))[None]
        inv_affine_sax = torch.linalg.inv(torch.Tensor(affine_sax).to(device))[None]

        # one-hot segmentations
        seg_2d_2ch = F.one_hot(seg_2d_2ch, num_classes=2).float().permute(0,4,1,2,3)
        seg_2d_4ch = F.one_hot(seg_2d_4ch, num_classes=6).float().permute(0,4,1,2,3)
        seg_2d_sax = F.one_hot(seg_2d_sax, num_classes=4).float().permute(0,4,1,2,3)

        # --------------------
        # 2d --> 3d seg 
        # --------------------
        # find multi-view cutting plane
        mask_2ch, mask_4ch, mask_sax = find_mutli_view_plane(
            grid_3d[0].cpu().numpy(), affine_2ch, affine_4ch, affine_sax, n_slice, voxel_size)
        mask_2ch = torch.LongTensor(mask_2ch[None]).to(device)
        mask_4ch = torch.LongTensor(mask_4ch[None]).to(device)
        mask_sax = torch.LongTensor(mask_sax[None]).to(device)

        # inverse affine transform the 3D grid
        grid_2ch = grid_3d @ inv_affine_2ch[:,:3,:3].transpose(2,1) + inv_affine_2ch[:,:3,-1]
        grid_4ch = grid_3d @ inv_affine_4ch[:,:3,:3].transpose(2,1) + inv_affine_4ch[:,:3,-1]
        grid_sax = grid_3d @ inv_affine_sax[:,:3,:3].transpose(2,1) + inv_affine_sax[:,:3,-1]

        grid_size_2ch = torch.Tensor([seg_2d_2ch.shape[2:]]).to(device)
        grid_size_4ch = torch.Tensor([seg_2d_4ch.shape[2:]]).to(device)
        grid_size_sax = torch.Tensor([seg_2d_sax.shape[2:]]).to(device)

        grid_2ch = (2 * grid_2ch / (grid_size_2ch - 1) - 1).flip(-1)
        grid_4ch = (2 * grid_4ch / (grid_size_4ch - 1) - 1).flip(-1)
        grid_sax = (2 * grid_sax / (grid_size_sax - 1) - 1).flip(-1)

        # perform grid sampling from 2d to 3d
        seg_3d_2ch = F.grid_sample(
            seg_2d_2ch.reshape(1,-1,*seg_2d_2ch.shape[2:]),
            grid_2ch, mode='bilinear', align_corners=True
        ).reshape(n_frame,-1,*grid_2ch.shape[1:-1]).argmax(1).long() * 4

        seg_3d_4ch = F.grid_sample(
            seg_2d_4ch.reshape(1,-1,*seg_2d_4ch.shape[2:]),
            grid_4ch, mode='bilinear', align_corners=True
        ).reshape(n_frame,-1,*grid_4ch.shape[1:-1]).argmax(1).long()

        seg_3d_sax = F.grid_sample(
            seg_2d_sax.reshape(1,-1,*seg_2d_sax.shape[2:]),
            grid_sax, mode='bilinear', align_corners=True
        ).reshape(n_frame,-1,*grid_sax.shape[1:-1]).argmax(1).long()

        seg_3d_2ch = seg_3d_2ch * mask_2ch
        seg_3d_4ch = seg_3d_4ch * mask_4ch
        seg_3d_sax = seg_3d_sax * mask_sax

        # merge multi-view segmentations
        seg_3d = seg_3d_sax.clone()
        seg_3d = seg_3d_2ch + seg_3d * (1-(seg_3d_2ch > 0).long())
        seg_3d = seg_3d_4ch + seg_3d * (1-(seg_3d_4ch > 0).long())

        # --------------------
        # 3d shape completion
        # --------------------
        seg_4d = np.zeros([n_frame, *grid_size_3d])
        for t in range(n_frame):
            # shape completion for each time frame
            seg_3d_t = F.one_hot(seg_3d[t:t+1], num_classes=6).permute(0,4,1,2,3).float()

            with torch.no_grad():
                seg_4d_t = unet(seg_3d_t)

            seg_3d_t = seg_3d_t.argmax(1)[0].cpu().numpy()
            seg_4d_t = seg_4d_t.argmax(1)[0].cpu().numpy()
            seg_4d[t] = seg_4d_t
        # torch.cuda.empty_cache()

        # --------------------
        # scale and shift
        # --------------------
        # compute center of mass and volume
        center_4d = np.stack(np.where(seg_4d[0]>0)).mean(-1)
        volume_4d = (seg_4d[0]>0).sum()
        scale_factor = (volume_4d / volume_atlas)**(1/3)

        # for inverse affine transformation
        inv_affine_align = np.eye(4) / scale_factor
        inv_affine_align[-1,-1] = 1
        trans_4d = affine_3d[:3,:3] @ center_4d + affine_3d[:3,-1]
        trans_atlas = affine_3d[:3,:3] @ center_atlas + affine_3d[:3,-1]
        inv_affine_align[:3,-1] = - trans_4d / scale_factor + trans_atlas
        affine_align = np.linalg.inv(inv_affine_align)
        affine_align_ = torch.Tensor(affine_align).to(device)

        # grid sampling for scale and shift
        grid_align = grid_3d @ affine_align_[:3,:3].T + affine_align_[:3,-1]
        grid_size_align = torch.Tensor(grid_size_3d).to(device)
        grid_align = (grid_align / (grid_size_align - 1)).flip(-1)

        seg_4d_align = torch.LongTensor(seg_4d).to(device)
        seg_4d_align = F.one_hot(
            seg_4d_align, num_classes=6).permute(0,4,1,2,3).float() #.to(device)
        seg_4d_align = F.grid_sample(
            seg_4d_align.reshape(1,-1,*seg_4d_align.shape[2:]),
            grid_align, mode='bilinear', align_corners=True
        ).reshape(n_frame,-1,*grid_align.shape[1:-1]).argmax(1).long()
        seg_4d_align = seg_4d_align.cpu().numpy().astype(np.uint8)

        t_end = time.time()
        runtime.append(t_end - t_start)

        # save nifti file
        affine_4d = affine_align @ affine_3d
        seg_4d_nii = nib.Nifti1Image(
            seg_4d_align.transpose(1,2,3,0).astype(np.uint8), affine=affine_4d)
        nib.save(seg_4d_nii, save_dir+subj_id+'/seg_4d.nii.gz')
        torch.cuda.empty_cache()

    print('average runtime: {} sec'.format(np.mean(runtime)))
    