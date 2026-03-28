import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from skimage.measure import marching_cubes
from pytorch3d.structures import Pointclouds


def seg_3d_to_cmr(seg_3d, p_4ch=40, sigma=2.0, voxel_size=1.0):
    """
    convert 3d cardiac segmentation to multi-view CMR labels
    - p_4ch: index of the 4-chamber view position
    - sigma: standard deviation for motion artifacts (mm)
    - voxel_size: spatial resolution
    """

    seg_lv_3d = (seg_3d==1)
    seg_my_3d = (seg_3d==2)
    seg_rv_3d = (seg_3d==3)
    seg_la_3d = (seg_3d==4)
    seg_ra_3d = (seg_3d==5)

    # segmentation of different views
    seg_2ch_cmr = np.zeros_like(seg_3d)
    seg_4ch_cmr = np.zeros_like(seg_3d)
    seg_sax_cmr = np.zeros_like(seg_3d)
    L,W,H = seg_3d.shape
    
    # random gaussian variance for motion artifacts
    sigma = sigma * np.random.rand(1) / voxel_size  # Uniform(0,2)
    
    # ------ segmentation for 2CH view ------
    grid_2d = [torch.arange(0, s) for s in seg_3d.shape[:2]]
    grid_2d = torch.stack(torch.meshgrid(
        grid_2d, indexing='ij')).permute(1,2,0).numpy()
    
    # find the 2CH plane
    centroid_lv_3d = np.stack(np.where(seg_lv_3d)).mean(-1)
    centroid_rv_3d = np.stack(np.where(seg_rv_3d)).mean(-1)
    n_2ch = (centroid_rv_3d - centroid_lv_3d)[:2]
    n_2ch = n_2ch / np.linalg.norm(n_2ch)
    
    # find the cutting plane with 2 voxels width
    dist_2ch = (grid_2d - centroid_lv_3d[:2]) @ n_2ch
    coord_2ch = np.stack(np.where((dist_2ch>=-1) & (dist_2ch<=1)))
    seg_2ch_cmr[coord_2ch[0], coord_2ch[1],:] = 1
    seg_2ch_cmr = seg_2ch_cmr * seg_la_3d * 4  # only consider left atrium

    # ------ segmentation for 4CH view ------
    p_4ch = p_4ch + np.random.randint(2)  # random shift one voxel
    seg_4ch_cmr[:,p_4ch] = 1
    seg_4ch_cmr[:,p_4ch+1] = 1
    seg_4ch_cmr = seg_4ch_cmr * seg_3d

    # ------ segmentation for SAX view------
    mask_sax_3d = seg_lv_3d + seg_my_3d + seg_rv_3d
    seg_sax_3d = mask_sax_3d * seg_3d
    
    thickness = 10 / voxel_size  # 10mm thick slices
    
    coord_valve = np.stack(np.where(seg_my_3d))[-1].max()
    coord_apex = np.stack(np.where(seg_my_3d))[-1].min()
    # random crop the sax segmentation
    coord_sax = coord_apex + np.random.uniform(0, thickness)

    while coord_sax < coord_valve:
        z_floor = int(np.floor(coord_sax))
        seg_sax_cmr_z = seg_sax_3d[:,:,z_floor]
        
        # random 2D in-plane shift (only considere 2D motions)
        shift_x, shift_y = np.round(sigma*np.random.randn(2)).astype(int)
        seg_sax_cmr_z = np.pad(
            seg_sax_cmr_z, 
            ((max(0,shift_x), max(0,-shift_x)),
             (max(0,shift_y), max(0,-shift_y))))
        seg_sax_cmr_z = seg_sax_cmr_z[
            max(0,-shift_x):L+max(0,-shift_x),
            max(0,-shift_y):W+max(0,-shift_y)]
        
        seg_sax_cmr[:,:,z_floor] = seg_sax_cmr_z
        seg_sax_cmr[:,:,z_floor+1] = seg_sax_cmr_z
        coord_sax += thickness

    seg_cmr = merge_multiview(seg_2ch_cmr, seg_4ch_cmr, seg_sax_cmr)
    
    return seg_cmr.astype(np.uint8)

    
def merge_multiview(seg_2ch, seg_4ch, seg_sax):
    """
    merge multiview CMR segmentations
    """
    seg_cmr = seg_sax.copy()
    seg_cmr = seg_2ch + seg_cmr * (1-(seg_2ch > 0))
    seg_cmr = seg_4ch + seg_cmr * (1-(seg_4ch > 0))
    return seg_cmr.astype(np.uint8)


   
class SegDataset(Dataset):
    '''
    Dataset class for label completion
    Args:
    - data_type: {'train','valid,','test'}
    - sigma: standard deviation for motion artifacts (mm)
    - scale: random scaling ratio
    - p_4ch: index of the 4-chamber view position
    - voxel_size: spatial resolution
    - grid_size: grid size for downsampling
    '''
    def __init__(
        self,
        data_dir,
        data_indices,
        data_type,
        seed=12345,
        sigma=2.0,
        scaling=0.2,
        p_4ch=40,
        voxel_size=2.0,
        grid_size=[96,96,128],
        device='cuda:0'
    ):
        super(Dataset, self).__init__()
        
        # ------ load path ------ 
        subj_list = sorted(glob.glob(data_dir+'*.nii.gz'))
        self.data_list = []
        self.data_type = data_type
        self.scaling = scaling
        self.sigma = sigma
        self.p_4ch = p_4ch
        self.voxel_size = voxel_size
        self.device = device
        
        # initialize grid for random scaling
        grid_size_3d = torch.Tensor(grid_size).to(device)
        grid_3d = [torch.arange(0, s) for s in grid_size_3d]
        grid_3d = torch.stack(torch.meshgrid(
            grid_3d, indexing='ij')).permute(1,2,3,0).to(device)
        grid_3d = 2 * grid_3d / (grid_size_3d - 1) - 1
        grid_3d = grid_3d[None].flip(-1)
        self.grid_3d = grid_3d        
        
        if data_type == 'valid' or data_type == 'test':
            # fix the validation and test sets
            np.random.seed(seed)
        
        for i in tqdm(data_indices):
            subj_dir = subj_list[i]
            subj_id = subj_list[i].split('/')[-1]

            seg_3d_nii = nib.load(subj_dir)
            seg_3d = seg_3d_nii.get_fdata().astype(np.uint8)
            
            # downsampling to save GPU memoery
            seg_3d = torch.LongTensor(seg_3d[None]).to(device)
            seg_3d = F.one_hot(seg_3d, num_classes=6).permute(0,4,1,2,3).float()
            seg_3d = F.interpolate(
                seg_3d, size=grid_size, mode='trilinear', align_corners=True)
            seg_3d = seg_3d.argmax(1)[0].cpu().numpy().astype(np.uint8)
            
            if data_type == 'valid' or data_type == 'test':
                # one hot segmentation
                seg_3d = torch.LongTensor(seg_3d[None]).to(device)
                seg_3d = F.one_hot(seg_3d, num_classes=6).permute(0,4,1,2,3).float()

                # random scaling for data augmentation [1-scaling, 1+scaling]
                scaling_factor = 1.0 - scaling + 2 * scaling * np.random.rand()
                seg_3d = F.grid_sample(
                    seg_3d, scaling_factor * grid_3d, mode='bilinear', align_corners=True)
                seg_3d = seg_3d.argmax(1)[0].cpu().numpy().astype(np.uint8)

                # random clipping and motion
                seg_cmr = seg_3d_to_cmr(seg_3d, p_4ch, sigma, voxel_size)
                self.data_list.append((seg_cmr, seg_3d))
            else:
                self.data_list.append(seg_3d)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        if self.data_type == 'train':
            # generate multi-view segs online
            seg_3d = self.data_list[i]
            
            # one hot segmentation
            seg_3d = torch.LongTensor(seg_3d[None]).to(self.device)
            seg_3d = F.one_hot(seg_3d, num_classes=6).permute(0,4,1,2,3).float()

            # random scaling for data augmentation [1-scaling, 1+scaling]
            scaling_factor = 1.0 - self.scaling + 2 * self.scaling * np.random.rand()
            seg_3d = F.grid_sample(
                seg_3d, scaling_factor * self.grid_3d, mode='bilinear', align_corners=True)
            seg_3d = seg_3d.argmax(1)[0].cpu().numpy().astype(np.uint8)
            
            # 3d to multi-view 2d segmentations
            seg_cmr = seg_3d_to_cmr(seg_3d, self.p_4ch, self.sigma, self.voxel_size)
        else:
            seg_cmr, seg_3d = self.data_list[i]
        return seg_cmr, seg_3d
    
    

class SurfDataset(Dataset):
    '''
    Dataset class for 3D+t four-chamber mesh reconstruction
    '''
    def __init__(
        self,
        data_dir,
        data_indices,
        n_frame=50
    ):
        super(Dataset, self).__init__()

        # ------ load path ------ 
        subj_list = sorted(glob.glob(data_dir+'*'))
        self.data_list = []
        
        for i in tqdm(data_indices):
            subj_dir = subj_list[i]

            # load 4d segmentation
            seg_4d_nii = nib.load(subj_dir+'/seg_4d.nii.gz')
            seg_4d = seg_4d_nii.get_fdata().transpose(3,0,1,2)
            # affine_4d = seg_4d_nii.affine

            vert_lv_gt = []
            vert_my_gt = []
            vert_rv_gt = []
            vert_la_gt = []
            vert_ra_gt = []

            for t in range(n_frame):
                seg_4d_t = seg_4d[t]

                # extract gt surface points from segmentations
                vert_lv_gt_t = marching_cubes((seg_4d_t==1).astype(np.float32))[0]
                vert_my_gt_t = marching_cubes(((seg_4d_t==1)|(seg_4d_t==2)).astype(np.float32))[0]
                vert_rv_gt_t = marching_cubes((seg_4d_t==3).astype(np.float32))[0]
                vert_la_gt_t = marching_cubes((seg_4d_t==4).astype(np.float32))[0]
                vert_ra_gt_t = marching_cubes((seg_4d_t==5).astype(np.float32))[0]

                vert_lv_gt.append(torch.Tensor(vert_lv_gt_t.copy()))
                vert_my_gt.append(torch.Tensor(vert_my_gt_t.copy()))
                vert_rv_gt.append(torch.Tensor(vert_rv_gt_t.copy()))
                vert_la_gt.append(torch.Tensor(vert_la_gt_t.copy()))
                vert_ra_gt.append(torch.Tensor(vert_ra_gt_t.copy()))

            vert_lv_gt = Pointclouds(vert_lv_gt)
            vert_my_gt = Pointclouds(vert_my_gt)
            vert_rv_gt = Pointclouds(vert_rv_gt)
            vert_la_gt = Pointclouds(vert_la_gt)
            vert_ra_gt = Pointclouds(vert_ra_gt)

            n_vert_lv_gt = vert_lv_gt.num_points_per_cloud()
            n_vert_my_gt = vert_my_gt.num_points_per_cloud()
            n_vert_rv_gt = vert_rv_gt.num_points_per_cloud()
            n_vert_la_gt = vert_la_gt.num_points_per_cloud()
            n_vert_ra_gt = vert_ra_gt.num_points_per_cloud()

            vert_lv_gt = vert_lv_gt.points_padded()
            vert_my_gt = vert_my_gt.points_padded()
            vert_rv_gt = vert_rv_gt.points_padded()
            vert_la_gt = vert_la_gt.points_padded()
            vert_ra_gt = vert_ra_gt.points_padded()
    
            self.data_list.append((
                seg_4d.astype(np.uint8),
                vert_lv_gt.float(),
                vert_my_gt.float(),
                vert_rv_gt.float(),
                vert_la_gt.float(),
                vert_ra_gt.float(),
                n_vert_lv_gt,
                n_vert_my_gt,
                n_vert_rv_gt,
                n_vert_la_gt,
                n_vert_ra_gt
            ))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        surf_data = self.data_list[i]
        return surf_data
 
