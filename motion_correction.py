import os
import glob
import argparse
import time
import nibabel as nib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.metrics import dice_score, correlation
from utils.utils import (
    find_plane_from_affine,
    affine_transform,
    rotate_matrix,
    grid_sample_2d
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='UK Biobank data preprocessing and motion correction')
    parser.add_argument('--data_dir', default='YOUR_UKBB_DIR', type=str, help='dataset directory')
    parser.add_argument('--save_dir', default='./data/ukbb/', type=str, help='saving directory')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    device = args.device
    
    width = 64  # width for 2D images
    # initialize grid for 2D sampling
    grid_2d = [torch.arange(0, 2*width), torch.arange(0, 2*width)]
    grid_2d = torch.stack(torch.meshgrid(grid_2d, indexing='ij')).permute(1,2,0)
    grid_2d = grid_2d - torch.tensor([width-0.5, width-0.5])
    grid_2d = grid_2d[None].to(device)

    # directory of ukbb data
    subj_list = sorted(glob.glob(data_dir+'*'))

    dice_orig_list = []
    dice_mc_list = []
    pcc_orig_list = []
    pcc_mc_list = []
    rmsd_orig_list = []
    rmsd_mc_list = []
    runtime = []

    for i in tqdm(range(len(subj_list))):
        subj_id = subj_list[i].split('/')[-1]

        # ------------------------
        # load UK biobank data
        # ------------------------
        # check if there are missing data
        no_missing_data = os.path.exists(data_dir+subj_id+'/visit_2/la_2ch.nii.gz') & \
                          os.path.exists(data_dir+subj_id+'/visit_2/la_4ch.nii.gz') & \
                          os.path.exists(data_dir+subj_id+'/visit_2/sa.nii.gz') & \
                          os.path.exists(data_dir+subj_id+'/visit_2/seg_la_2ch.nii.gz') & \
                          os.path.exists(data_dir+subj_id+'/visit_2/seg4_la_4ch.nii.gz') & \
                          os.path.exists(data_dir+subj_id+'/visit_2/seg_sa.nii.gz')
        if not no_missing_data:
            continue

        img_2ch_nii = nib.load(data_dir+subj_id+'/visit_2/la_2ch.nii.gz')
        img_4ch_nii = nib.load(data_dir+subj_id+'/visit_2/la_4ch.nii.gz')
        img_sax_nii = nib.load(data_dir+subj_id+'/visit_2/sa.nii.gz')
        seg_2ch_nii = nib.load(data_dir+subj_id+'/visit_2/seg_la_2ch.nii.gz')
        seg_4ch_nii = nib.load(data_dir+subj_id+'/visit_2/seg4_la_4ch.nii.gz')
        seg_sax_nii = nib.load(data_dir+subj_id+'/visit_2/seg_sa.nii.gz')

        # (L,W,H,T) --> (H,T,L,W)
        img_2ch = img_2ch_nii.get_fdata().transpose(2,3,0,1)
        img_4ch = img_4ch_nii.get_fdata().transpose(2,3,0,1)
        img_sax = img_sax_nii.get_fdata().transpose(2,3,0,1)
        seg_2ch = seg_2ch_nii.get_fdata().transpose(2,3,0,1)
        seg_4ch = seg_4ch_nii.get_fdata().transpose(2,3,0,1)
        seg_sax = seg_sax_nii.get_fdata().transpose(2,3,0,1)

        n_slice = seg_sax.shape[0]

        affine_2ch = seg_2ch_nii.affine
        affine_4ch = seg_4ch_nii.affine
        affine_sax = seg_sax_nii.affine

        seg_2ch_la = (seg_2ch==1).astype(np.float32)
        seg_4ch_lv = (seg_4ch==1).astype(np.float32)
        seg_4ch_my = (seg_4ch==2).astype(np.float32)
        seg_4ch_rv = (seg_4ch==3).astype(np.float32)
        seg_4ch_la = (seg_4ch==4).astype(np.float32)
        seg_4ch_ra = (seg_4ch==5).astype(np.float32)
        seg_sax_lv = (seg_sax==1).astype(np.float32)
        seg_sax_my = (seg_sax==2).astype(np.float32)
        seg_sax_rv = (seg_sax==3).astype(np.float32)

        # check if there are missing segmentation labels at each frame
        no_missing_seg = seg_2ch_la.sum(axis=(0,2,3)).all() &\
                         seg_4ch_lv.sum(axis=(0,2,3)).all() &\
                         seg_4ch_my.sum(axis=(0,2,3)).all() &\
                         seg_4ch_rv.sum(axis=(0,2,3)).all() &\
                         seg_4ch_la.sum(axis=(0,2,3)).all() &\
                         seg_4ch_ra.sum(axis=(0,2,3)).all() &\
                         seg_sax_lv.sum(axis=(0,2,3)).all() &\
                         seg_sax_my.sum(axis=(0,2,3)).all() &\
                         seg_sax_rv.sum(axis=(0,2,3)).all()
        if not no_missing_seg:
            continue

        if not os.path.exists(save_dir+subj_id):
            os.makedirs(save_dir+subj_id)

        t_start = time.time()
        # ------------------------
        # correct the image center in 3D space
        # ------------------------
        # compute three viewing planes to obtain the intersection points of 3 planes
        n_2ch, p_2ch = find_plane_from_affine(affine_2ch)
        n_4ch, p_4ch = find_plane_from_affine(affine_4ch)
        n_sax, p_sax = find_plane_from_affine(affine_sax)
        isect_pt = np.linalg.inv(np.vstack([n_2ch, n_4ch, n_sax])) @ \
                   np.array([p_2ch, p_4ch, p_sax])

        # move the intersection point along the intersection line
        # such that it is located in the centroid of the heart
        d_2ch_4ch = 25  # direction towards the LV
        n_2ch_4ch = np.cross(n_2ch, n_4ch)
        isect_pt = isect_pt + d_2ch_4ch * n_2ch_4ch
        d_4ch_sax = 25  # direciton towards the right heart
        n_4ch_sax = np.cross(n_4ch, n_sax)
        isect_pt = isect_pt + d_4ch_sax * n_4ch_sax
        d_2ch_sax = 15  # direciton along the 2ch viewing plane
        n_2ch_sax = np.cross(n_2ch, n_sax)
        isect_pt = isect_pt + d_2ch_sax * n_2ch_sax

        # the center points are no longer in the original view planes (2D spaces)
        # it should be projected from the real-world space to the 2D space
        center_2ch = affine_transform(isect_pt, np.linalg.inv(affine_2ch))
        center_4ch = affine_transform(isect_pt, np.linalg.inv(affine_4ch))
        center_sax = affine_transform(isect_pt, np.linalg.inv(affine_sax))
        center_2ch[2] = 0
        center_4ch[2] = 0
        center_sax[2] = 0

        # ------------------------
        # correct the image center and clip 2D images
        # ------------------------
        # we manipulate the inverse affine matrix such that center (x0,y0)
        # is mapped to (width-1,width-1), which is the center of the clipped image
        affine_2ch_proc = affine_2ch.copy()
        inv_affine_2ch_proc = np.linalg.inv(affine_2ch_proc)
        inv_affine_2ch_proc[:2,-1] -= center_2ch[:2] - width + 0.5
        affine_2ch_proc = np.linalg.inv(inv_affine_2ch_proc)

        affine_4ch_proc = affine_4ch.copy()
        inv_affine_4ch_proc = np.linalg.inv(affine_4ch_proc)
        inv_affine_4ch_proc[:2,-1] -= center_4ch[:2] - width + 0.5
        affine_4ch_proc = np.linalg.inv(inv_affine_4ch_proc)

        affine_sax_proc = affine_sax.copy()
        inv_affine_sax_proc = np.linalg.inv(affine_sax_proc)
        inv_affine_sax_proc[:2,-1] -= center_sax[:2] - width + 0.5
        affine_sax_proc = np.linalg.inv(inv_affine_sax_proc)

        # resample the image such that the projected center is on the image center
        # the resampeld image has the size of (128,128)
        center_2ch = torch.Tensor(center_2ch[:2]).to(device)
        center_4ch = torch.Tensor(center_4ch[:2]).to(device)
        center_sax = torch.Tensor(center_sax[:2]).to(device)

        img_2ch_tensor = torch.Tensor(img_2ch).to(device)
        img_4ch_tensor = torch.Tensor(img_4ch).to(device)
        img_sax_tensor = torch.Tensor(img_sax).to(device)
        seg_2ch_tensor = torch.Tensor(seg_2ch).to(device)
        seg_4ch_tensor = torch.Tensor(seg_4ch).to(device)
        seg_sax_tensor = torch.Tensor(seg_sax).to(device)

        grid_2ch = grid_2d + center_2ch
        grid_4ch = grid_2d + center_4ch
        grid_sax = (grid_2d + center_sax).repeat(n_slice,1,1,1)

        img_2ch_proc = grid_sample_2d(img_2ch_tensor, grid_2ch, mode='bilinear')
        img_4ch_proc = grid_sample_2d(img_4ch_tensor, grid_4ch, mode='bilinear')
        img_sax_proc = grid_sample_2d(img_sax_tensor, grid_sax, mode='bilinear')

        seg_2ch_proc = grid_sample_2d(seg_2ch_tensor, grid_2ch, mode='nearest')
        seg_4ch_proc = grid_sample_2d(seg_4ch_tensor, grid_4ch, mode='nearest')
        seg_sax_proc = grid_sample_2d(seg_sax_tensor, grid_sax, mode='nearest')

        # move the center of cardiac shapes to the intersection points of 3 planes
        affine_2ch_proc[:3,-1] -= isect_pt
        affine_4ch_proc[:3,-1] -= isect_pt
        affine_sax_proc[:3,-1] -= isect_pt

        # ------------------------
        # rotate the image in 3D space
        # ------------------------
        # we rotate the heart in the world space to make sure that 
        # the cardiac shapes have similar orientations

        # i) we first rotate the sax viewing plane to normal direction (0,0,1)
        n_z = np.array([0,0,1])  # target direction
        rot_z = np.zeros([4,4])
        rot_z[:3,:3] = rotate_matrix(n_sax, n_z)
        rot_z[-1,-1] = 1

        # ii) the 4ch viewing plane is almost perpendicular to the sax plane
        # we then rotate the 4ch viewing plane to normal direction (0,1,0) 
        n_y = np.array([0,1,0])  # target direction
        rot_y = np.zeros([4,4])
        rot_y[:3,:3] = rotate_matrix(affine_transform(n_4ch, rot_z), n_y)
        rot_y[-1,-1] = 1

        # apply rotation matrix to original affine matrix
        affine_2ch_proc = rot_y @ rot_z @ affine_2ch_proc
        affine_4ch_proc = rot_y @ rot_z @ affine_4ch_proc
        affine_sax_proc = rot_y @ rot_z @ affine_sax_proc

        # provide inverse affine matrix for reconstructing 3D from 2D
        inv_affine_2ch_proc = np.linalg.inv(affine_2ch_proc)
        inv_affine_4ch_proc = np.linalg.inv(affine_4ch_proc)
        inv_affine_sax_proc = np.linalg.inv(affine_sax_proc)

        # ------------------------
        # extract intersections lines
        # ------------------------
        # compute three viewing planes
        n_2ch, p_2ch = find_plane_from_affine(affine_2ch_proc)
        n_4ch, p_4ch = find_plane_from_affine(affine_4ch_proc)
        n_sax, p_sax = find_plane_from_affine(affine_sax_proc)

        # extract sax & lax intersections
        p_sax_0 = p_sax.copy()
        len_4ch_sax = 72  # 144
        len_2ch_sax = 60  # 120
        res = 1.0
        len_2ch_sax = int(len_2ch_sax/res)
        len_4ch_sax = int(len_4ch_sax/res)

        # num_slice = img_sax_proc.shape[-1]
        line_2ch_sax = np.zeros([n_slice, 2*len_2ch_sax+1, 3])
        line_4ch_sax = np.zeros([n_slice, 2*len_4ch_sax+1, 3])
        grid_2ch_sax = np.arange(-len_2ch_sax, len_2ch_sax+1, 1)[:,None]
        grid_4ch_sax = np.arange(-len_4ch_sax, len_4ch_sax+1, 1)[:,None]

        # for each sax slice
        for k in range(n_slice):
            # plane of the k-th sax slice (10mm thickness)
            p_sax = p_sax_0 + k * n_sax @ affine_sax_proc[:3,2]
            # direction of the intersection line
            n_4ch_sax = np.cross(n_4ch, n_sax)
            n_2ch_sax = np.cross(n_2ch, n_sax)

            # find the intersection point among 4ch, sax, and an auxiliary plane
            isect_4ch_sax = np.linalg.inv(
                np.vstack([n_4ch, n_sax, n_4ch_sax])
            ) @ np.array([p_4ch, p_sax, 0])

            isect_2ch_sax = np.linalg.inv(
                np.vstack([n_2ch, n_sax, n_2ch_sax])
            ) @ np.array([p_2ch, p_sax, 0])

            line_4ch_sax[k] = isect_4ch_sax + res * grid_4ch_sax * n_4ch_sax
            line_2ch_sax[k] = isect_2ch_sax + res * grid_2ch_sax * n_2ch_sax

        # extract 2ch & 4ch intersections
        length = 100
        line_grid = np.arange(-int(length/res)+1, int(length/res), 1)[:,None]
        n_2ch_4ch = np.cross(n_2ch, n_4ch)

        # find the intersection point among 4ch, sax, and an auxiliary plane
        isect_2ch_4ch = np.linalg.inv(
            np.vstack([n_2ch, n_4ch, n_2ch_4ch])
        ) @ np.array([p_2ch, p_4ch, 0])

        line_2ch_4ch = (isect_2ch_4ch + res * line_grid * n_2ch_4ch)[None]

        line_4ch_sax = torch.Tensor(line_4ch_sax).to(device)
        line_2ch_sax = torch.Tensor(line_2ch_sax).to(device)
        line_2ch_4ch = torch.Tensor(line_2ch_4ch).to(device)

        invA_2ch = torch.Tensor(inv_affine_2ch_proc[None]).to(device)
        invA_4ch = torch.Tensor(inv_affine_4ch_proc[None]).to(device)
        invA_sax = torch.Tensor(inv_affine_sax_proc[None]).to(device)

        # project 3D intersection lines onto 2D slices
        line_4s_4ch = line_4ch_sax @ invA_4ch[:,:3,:3].transpose(1,2) + invA_4ch[:,:3,-1]
        line_4s_sax = line_4ch_sax @ invA_sax[:,:3,:3].transpose(1,2) + invA_sax[:,:3,-1]
        line_4s_4ch = line_4s_4ch[...,:2]
        line_4s_sax = line_4s_sax[...,:2]

        line_2s_2ch = line_2ch_sax @ invA_2ch[:,:3,:3].transpose(1,2) + invA_2ch[:,:3,-1]
        line_2s_sax = line_2ch_sax @ invA_sax[:,:3,:3].transpose(1,2) + invA_sax[:,:3,-1]
        line_2s_2ch = line_2s_2ch[...,:2]
        line_2s_sax = line_2s_sax[...,:2]

        line_24_2ch = line_2ch_4ch @ invA_2ch[:,:3,:3].transpose(1,2) + invA_2ch[:,:3,-1]
        line_24_4ch = line_2ch_4ch @ invA_4ch[:,:3,:3].transpose(1,2) + invA_4ch[:,:3,-1]
        line_24_2ch = line_24_2ch[...,:2]
        line_24_4ch = line_24_4ch[...,:2]

        # ------------------------
        # initial evaluation
        # ------------------------
        grid_4s_4ch = line_4s_4ch[:,:,None]
        grid_4s_sax = line_4s_sax[:,:,None]
        grid_2s_2ch = line_2s_2ch[:,:,None]
        grid_2s_sax = line_2s_sax[:,:,None]
        grid_24_2ch = line_24_2ch[:,:,None]
        grid_24_4ch = line_24_4ch[:,:,None]

        value_4s_4ch = grid_sample_2d(
            img_4ch_proc.repeat(n_slice,1,1,1), grid_4s_4ch, mode='bilinear')[...,0]
        value_4s_sax = grid_sample_2d(
            img_sax_proc, grid_4s_sax, mode='bilinear')[...,0]

        value_2s_2ch = grid_sample_2d(
            img_2ch_proc.repeat(n_slice,1,1,1), grid_2s_2ch, mode='bilinear')[...,0]
        value_2s_sax = grid_sample_2d(
            img_sax_proc, grid_2s_sax, mode='bilinear')[...,0]

        value_24_2ch = grid_sample_2d(
            img_2ch_proc, grid_24_2ch, mode='bilinear')[...,0]
        value_24_4ch = grid_sample_2d(
            img_4ch_proc, grid_24_4ch, mode='bilinear')[...,0]

        label_4s_4ch = grid_sample_2d(
            seg_4ch_proc.repeat(n_slice,1,1,1), grid_4s_4ch, mode='nearest')[...,0]
        label_4s_sax = grid_sample_2d(
            seg_sax_proc, grid_4s_sax, mode='nearest')[...,0]

        label_2s_2ch = grid_sample_2d(
            seg_2ch_proc.repeat(n_slice,1,1,1), grid_2s_2ch, mode='nearest')[...,0]
        label_2s_sax = grid_sample_2d(
            seg_sax_proc, grid_2s_sax, mode='nearest')[...,0]

        label_24_2ch = grid_sample_2d(
            seg_2ch_proc, grid_24_2ch, mode='nearest')[...,0]
        label_24_4ch = grid_sample_2d(
            seg_4ch_proc, grid_24_4ch, mode='nearest')[...,0]

        dice_lv = dice_score(label_4s_4ch==1, label_4s_sax==1).item()
        dice_my = dice_score(label_4s_4ch==2, label_4s_sax==2).item()
        dice_rv = dice_score(label_4s_4ch==3, label_4s_sax==3).item()
        dice_la = dice_score(label_24_2ch==1, label_24_4ch==4).item()
        dice_orig = (dice_lv + dice_my + dice_rv + dice_la) / 4
        dice_orig_list.append(dice_orig)

        pcc_4s = correlation(value_4s_4ch, value_4s_sax).item()
        pcc_2s = correlation(value_2s_2ch, value_2s_sax).item()
        pcc_24 = correlation(value_24_2ch, value_24_4ch).item()
        pcc_orig = (pcc_4s + pcc_2s + pcc_24) / 3
        pcc_orig_list.append(pcc_orig)

        rmsd_4s = ((value_4s_4ch - value_4s_sax)**2).mean().sqrt().item()
        rmsd_2s = ((value_2s_2ch - value_2s_sax)**2).mean().sqrt().item()
        rmsd_24 = ((value_24_2ch - value_24_4ch)**2).mean().sqrt().item()
        rmsd_orig = (rmsd_4s + rmsd_2s + rmsd_24) / 3
        rmsd_orig_list.append(rmsd_orig)

        # ------------------------
        # perform motion correction
        # ------------------------
        # initialize inplane 2D translation
        shift_sax = torch.zeros([n_slice,1,2], requires_grad=True, device=device)
        shift_4ch = torch.zeros([1,1,2], requires_grad=True, device=device)
        shift_2ch = torch.zeros([1,1,2], requires_grad=True, device=device)

        n_iter = 1000
        loss_pre = 0

        optimizer = optim.Adam(
            [shift_sax, shift_4ch, shift_2ch], lr=1e-1)

        for j in range(n_iter):
            optimizer.zero_grad()
            # add inplane shift for motion correciton,
            # and rescale coordinates to [-1,1]
            grid_4s_4ch = (line_4s_4ch + shift_4ch)[:,:,None]
            grid_4s_sax = (line_4s_sax + shift_sax)[:,:,None]
            grid_2s_2ch = (line_2s_2ch + shift_2ch)[:,:,None]
            grid_2s_sax = (line_2s_sax + shift_sax)[:,:,None]
            grid_24_2ch = (line_24_2ch + shift_2ch)[:,:,None]
            grid_24_4ch = (line_24_4ch + shift_4ch)[:,:,None]

            # compute the laplacian loss for the center of sax-view intensity
            grid_lap_sax = grid_2d[:,32:-32,32:-32,:] + shift_sax[:,:,None] + width - 0.5

            value_4s_4ch = grid_sample_2d(
                img_4ch_proc.repeat(n_slice,1,1,1), grid_4s_4ch, mode='bilinear')[...,0]
            value_4s_sax = grid_sample_2d(
                img_sax_proc, grid_4s_sax, mode='bilinear')[...,0]

            value_2s_2ch = grid_sample_2d(
                img_2ch_proc.repeat(n_slice,1,1,1), grid_2s_2ch, mode='bilinear')[...,0]
            value_2s_sax = grid_sample_2d(
                img_sax_proc, grid_2s_sax, mode='bilinear')[...,0]

            value_24_2ch = grid_sample_2d(
                img_2ch_proc, grid_24_2ch, mode='bilinear')[...,0]
            value_24_4ch = grid_sample_2d(
                img_4ch_proc, grid_24_4ch, mode='bilinear')[...,0]

            value_lap_sax = grid_sample_2d(img_sax_proc, grid_lap_sax, mode='bilinear')

            # compute loss function
            loss_4s = ((value_4s_4ch - value_4s_sax).abs()).mean()
            loss_2s = ((value_2s_2ch - value_2s_sax).abs()).mean()
            loss_24 = ((value_24_2ch - value_24_4ch).abs()).mean()

            # laplacian smoothness across slices
            loss_lap = (value_lap_sax[2:] + value_lap_sax[:-2] - \
                        2*value_lap_sax[1:-1]).abs().mean()

            loss = loss_4s + loss_2s + loss_24 + 0.5*loss_lap
            loss.backward()
            optimizer.step()  # gradient descent

            if np.abs(loss.item()-loss_pre) > 1e-5 :
                loss_pre = loss.item()
            else:  # early stop
                break
            # print("iter:{}, loss:{}".format(j, loss.item()))

        # ------------------------
        # image after motion correction
        # ------------------------
        shift_2ch = shift_2ch.detach()
        shift_4ch = shift_4ch.detach()
        shift_sax = shift_sax.detach()

        grid_2ch_mc = grid_2ch + shift_2ch[:,:,None]
        grid_4ch_mc = grid_4ch + shift_4ch[:,:,None]
        grid_sax_mc = grid_sax + shift_sax[:,:,None]

        img_2ch_mc = grid_sample_2d(img_2ch_tensor, grid_2ch_mc, mode='bilinear')
        img_4ch_mc = grid_sample_2d(img_4ch_tensor, grid_4ch_mc, mode='bilinear')
        img_sax_mc = grid_sample_2d(img_sax_tensor, grid_sax_mc, mode='bilinear')

        seg_2ch_mc = grid_sample_2d(seg_2ch_tensor, grid_2ch_mc, mode='nearest')
        seg_4ch_mc = grid_sample_2d(seg_4ch_tensor, grid_4ch_mc, mode='nearest')
        seg_sax_mc = grid_sample_2d(seg_sax_tensor, grid_sax_mc, mode='nearest')

        t_end = time.time()
        runtime.append(t_end - t_start)

        # ------------------------
        # save affine and motion
        # ------------------------
        np.save(save_dir+subj_id+'/affine.npy',
            [affine_2ch_proc.astype(np.float32),
             affine_4ch_proc.astype(np.float32),
             affine_sax_proc.astype(np.float32)])

        np.save(save_dir+subj_id+'/shift.npy',
            np.vstack([(center_2ch + shift_2ch)[:,0].cpu().numpy().astype(np.float32),
                       (center_4ch + shift_4ch)[:,0].cpu().numpy().astype(np.float32),
                       (center_sax + shift_sax)[:,0].cpu().numpy().astype(np.float32)]))

        # ------------------------
        # final evaluation
        # ------------------------
        grid_4s_4ch = line_4s_4ch[:,:,None]
        grid_4s_sax = line_4s_sax[:,:,None]
        grid_2s_2ch = line_2s_2ch[:,:,None]
        grid_2s_sax = line_2s_sax[:,:,None]
        grid_24_2ch = line_24_2ch[:,:,None]
        grid_24_4ch = line_24_4ch[:,:,None]

        value_4s_4ch = grid_sample_2d(
            img_4ch_mc.repeat(n_slice,1,1,1), grid_4s_4ch, mode='bilinear')[...,0]
        value_4s_sax = grid_sample_2d(
            img_sax_mc, grid_4s_sax, mode='bilinear')[...,0]

        value_2s_2ch = grid_sample_2d(
            img_2ch_mc.repeat(n_slice,1,1,1), grid_2s_2ch, mode='bilinear')[...,0]
        value_2s_sax = grid_sample_2d(
            img_sax_mc, grid_2s_sax, mode='bilinear')[...,0]

        value_24_2ch = grid_sample_2d(
            img_2ch_mc, grid_24_2ch, mode='bilinear')[...,0]
        value_24_4ch = grid_sample_2d(
            img_4ch_mc, grid_24_4ch, mode='bilinear')[...,0]

        label_4s_4ch = grid_sample_2d(
            seg_4ch_mc.repeat(n_slice,1,1,1), grid_4s_4ch, mode='nearest')[...,0]
        label_4s_sax = grid_sample_2d(
            seg_sax_mc, grid_4s_sax, mode='nearest')[...,0]

        label_2s_2ch = grid_sample_2d(
            seg_2ch_mc.repeat(n_slice,1,1,1), grid_2s_2ch, mode='nearest')[...,0]
        label_2s_sax = grid_sample_2d(
            seg_sax_mc, grid_2s_sax, mode='nearest')[...,0]

        label_24_2ch = grid_sample_2d(
            seg_2ch_mc, grid_24_2ch, mode='nearest')[...,0]
        label_24_4ch = grid_sample_2d(
            seg_4ch_mc, grid_24_4ch, mode='nearest')[...,0]

        dice_lv = dice_score(label_4s_4ch==1, label_4s_sax==1).item()
        dice_my = dice_score(label_4s_4ch==2, label_4s_sax==2).item()
        dice_rv = dice_score(label_4s_4ch==3, label_4s_sax==3).item()
        dice_la = dice_score(label_24_2ch==1, label_24_4ch==4).item()
        dice_mc = (dice_lv + dice_my + dice_rv + dice_la) / 4
        dice_mc_list.append(dice_mc)

        pcc_4s = correlation(value_4s_4ch, value_4s_sax).item()
        pcc_2s = correlation(value_2s_2ch, value_2s_sax).item()
        pcc_24 = correlation(value_24_2ch, value_24_4ch).item()
        pcc_mc = (pcc_4s + pcc_2s + pcc_24) / 3
        pcc_mc_list.append(pcc_mc)

        rmsd_4s = ((value_4s_4ch - value_4s_sax)**2).mean().sqrt().item()
        rmsd_2s = ((value_2s_2ch - value_2s_sax)**2).mean().sqrt().item()
        rmsd_24 = ((value_24_2ch - value_24_4ch)**2).mean().sqrt().item()
        rmsd_mc = (rmsd_4s + rmsd_2s + rmsd_24) / 3
        rmsd_mc_list.append(rmsd_mc)
        
        np.save(save_dir+subj_id+'/metrics_motion.npy',
                [dice_orig, dice_mc, pcc_orig, pcc_mc, rmsd_orig, rmsd_mc])

    print('--------------------')
    print('average runtime: {} sec'.format(np.mean(runtime)))
    print('before dice:', np.mean(dice_orig_list))
    print(' after dice:', np.mean(dice_mc_list))
    print('before pcc:', np.mean(pcc_orig_list))
    print(' after pcc:', np.mean(pcc_mc_list))
    print('before rmsd:', np.mean(rmsd_orig_list))
    print(' after rmsd:', np.mean(rmsd_mc_list))
    print('--------------------')

