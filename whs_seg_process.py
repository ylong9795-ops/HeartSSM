import os
import glob
import argparse
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes

from tqdm import tqdm
import ants
import torch
import torch.nn.functional as F

from utils.utils import (
    affine_transform,
    rotate_matrix
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CT whole-heart segmentation processing')
    parser.add_argument('--data_dir', default='YOUR_WHS_DIR', type=str, help='dataset directory')
    parser.add_argument('--save_dir', default='./data/whs/', type=str, help='saving directory')
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # --------------------
    # load seg atlas
    # --------------------
    seg_atlas_whs_nii = nib.load('./template/whs_atlas_seg.nii.gz')

    seg_atlas_whs = seg_atlas_whs_nii.get_fdata()
    affine_atlas_whs = seg_atlas_whs_nii.affine

    seg_atlas_whs_ants = ants.image_read('./template/whs_atlas_seg.nii.gz')
    args_atlas_whs_ants = (
        seg_atlas_whs_ants.origin,
        seg_atlas_whs_ants.spacing,
        seg_atlas_whs_ants.direction)

    vert_atlas_lv, face_atlas_lv, _, _ = marching_cubes(seg_atlas_whs == 1)
    vert_atlas_my, face_atlas_my, _, _ = marching_cubes((seg_atlas_whs == 1)|(seg_atlas_whs == 2))
    vert_atlas_rv, face_atlas_rv, _, _ = marching_cubes(seg_atlas_whs == 3)
    vert_atlas_la, face_atlas_la, _, _ = marching_cubes(seg_atlas_whs == 4)
    vert_atlas_ra, face_atlas_ra, _, _ = marching_cubes(seg_atlas_whs == 5)

    vert_atlas_lv = affine_transform(vert_atlas_lv, affine_atlas_whs)
    vert_atlas_my = affine_transform(vert_atlas_my, affine_atlas_whs)
    vert_atlas_rv = affine_transform(vert_atlas_rv, affine_atlas_whs)
    vert_atlas_la = affine_transform(vert_atlas_la, affine_atlas_whs)
    vert_atlas_ra = affine_transform(vert_atlas_ra, affine_atlas_whs)

    # find the directional vector for registration
    n_atlas_right = vert_atlas_rv.mean(0) - vert_atlas_lv.mean(0)
    n_atlas_right = n_atlas_right / np.linalg.norm(n_atlas_right)
    n_atlas_up = vert_atlas_la.mean(0) - vert_atlas_lv.mean(0)
    n_atlas_up = n_atlas_up / np.linalg.norm(n_atlas_up)

    # initialize grid for resampling
    vol_size = np.array([192,192,256])
    vol_center = (vol_size - 1) / 2
    grid_3d = [torch.arange(0, s) for s in vol_size]
    grid_3d = torch.stack(torch.meshgrid(
        grid_3d, indexing='ij')).permute(1,2,3,0)

    # multiple data sources
    dataset_dir = [
        data_dir+'ImageCAS/processed/seg_*.nii.gz',
        data_dir+'TotalSegmentator/processed/s*.nii.gz',
        data_dir+'mmwhs/whspp_ct/Center*/Case*/*_label.nii.gz',
        data_dir+'mmwhs/lacs13_ct/*_whs.nii.gz',
        data_dir+'mmwhs/casdc13/image*.nii.gz',
        data_dir+'mmwhs/ccec08/*/*.nii.gz'
    ]
    dataset_name = [
        'imagecas',
        'totalseg',
        'whspp_ct',
        'lacs13_ct',
        'casdc13',
        'ccec08'
    ]

    idx_0 = [4,0,4,11,5,5]
    idx_1 = [-7,5,8,15,7,7]
    
    # --------------------
    # rigid + scale registration
    # --------------------
    for n in range(len(dataset_dir)):
        subj_list = sorted(glob.glob(dataset_dir[n]))
        print('dataset:', dataset_name[n])

        for subj_dir in tqdm(subj_list):
            subj_id = subj_dir.split('/')[-1][idx_0[n]:idx_1[n]]
            seg_nii = nib.load(subj_dir)
            seg = seg_nii.get_fdata()

            if dataset_name[n] in ['imagecas', 'totalseg']:
                seg_whs = seg.copy()
            else:
                seg_whs = np.zeros_like(seg)
                seg_whs[np.where(seg==500)] = 1  # LV
                seg_whs[np.where((seg==204)|(seg==205))] = 2  # LV Myocardium
                seg_whs[np.where(seg==600)] = 3  # RV
                seg_whs[np.where(seg==420)] = 4  # LA
                seg_whs[np.where(seg==550)] = 5  # RA

            affine = seg_nii.affine
            center = np.stack(np.where(seg_whs)).mean(-1)
            center = center @ affine[:3,:3].T
            affine_proc = affine.copy()
            affine_proc[:3,-1] = - center
            
            # find the centers of chambers
            center_lv = np.vstack(np.where(seg_whs == 1)).mean(-1)
            center_lv = affine_transform(center_lv, affine_proc)
            center_rv = np.vstack(np.where(seg_whs == 3)).mean(-1)
            center_rv = affine_transform(center_rv, affine_proc)
            center_la = np.vstack(np.where(seg_whs == 4)).mean(-1)
            center_la = affine_transform(center_la, affine_proc)

            # find the directional vector for registration
            n_right = center_rv - center_lv
            n_right = n_right / np.linalg.norm(n_right)
            n_up = center_la - center_lv
            n_up = n_up / np.linalg.norm(n_up)

            # compute an initial rotation matrix for registration
            rot_right = rotate_matrix(n_right, n_atlas_right)
            rot_up = rotate_matrix(n_up @ rot_right.T, n_atlas_up)

            rot_mat = np.eye(4)
            rot_mat[:3,:3] = rot_right
            affine_proc = rot_mat @ affine_proc
            rot_mat[:3,:3] = rot_up
            affine_proc = rot_mat @ affine_proc

            # resample the original segmentation
            affine_resample = affine_proc.copy()
            affine_resample[:3,-1] += vol_center
            grid_resample = affine_transform(
                grid_3d[None], np.linalg.inv(affine_resample))
            seg_whs_onehot = torch.Tensor(seg_whs)[None]
            seg_whs_onehot = F.one_hot(
                seg_whs_onehot.long()).permute(0,4,1,2,3).float()

            # rescale to [-1,1]
            scale_orig = torch.Tensor(list(seg_whs_onehot.shape[2:]))
            grid_resample = 2 * grid_resample / (scale_orig - 1) - 1
            grid_resample = grid_resample.flip(-1).float()

            # grid sampling
            seg_whs_resample = F.grid_sample(
                seg_whs_onehot, grid_resample, mode='bilinear', align_corners=True)
            seg_whs_resample = seg_whs_resample[0].cpu().numpy()
            seg_whs_resample = seg_whs_resample.argmax(0).astype(np.uint8)

            # perform rigid+scale registration
            seg_whs_resample_ants = ants.from_numpy(
                seg_whs_resample, *args_atlas_whs_ants)

            transform_rigid = ants.registration(
                fixed=seg_atlas_whs_ants,
                moving=seg_whs_resample_ants,
                type_of_transform='Similarity',
                aff_metric='mattes')  # ['mattes', 'GC', 'meansquares']

            seg_whs_rigid_ants = ants.apply_transforms(
                fixed=seg_atlas_whs_ants,
                moving=seg_whs_resample_ants,
                transformlist=transform_rigid['fwdtransforms'],
                interpolator='genericLabel')

            seg_whs_rigid = seg_whs_rigid_ants.numpy()
            affine_whs = np.eye(4)
            affine_whs[:3,-1] -= vol_center

            # save nifti file
            seg_whs_rigid_nii = nib.Nifti1Image(
                seg_whs_rigid.astype(np.uint8), affine=affine_whs)
            nib.save(seg_whs_rigid_nii, 
                     save_dir+dataset_name[n]+'_'+subj_id+'.nii.gz')
    
