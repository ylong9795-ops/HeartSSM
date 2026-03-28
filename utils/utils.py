import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import find_contours
from skimage.measure import label as connected


def find_plane_from_affine(affine):
    '''
    Given a affine matrix that maps a 2D image to 3D space,
    find the normal n and point p of the 3D plane.
    '''
    n = np.cross(affine[:3,0], affine[:3,1])
    n = n / np.linalg.norm(n)
    p = n @ affine[:3,-1]
    return n, p


def affine_transform(x, affine):
    return x @ affine[:3,:3].T + affine[:3,-1]


def rotate_matrix(n1, n2):
    '''
    Compute rotation matrix rot between n1 and n2
    based on Rodrigues formulate
    '''
    r_axis = np.cross(n1,n2)
    cos_theta = np.dot(n1,n2)
    r_mat = np.zeros([3,3])
    r_mat[0,1] = - r_axis[2]
    r_mat[1,0] = r_axis[2]
    r_mat[0,2] = r_axis[1]
    r_mat[2,0] = -r_axis[1]
    r_mat[1,2] = -r_axis[0]
    r_mat[2,1] = r_axis[0]
    rot = np.eye(3) + r_mat + r_mat @ r_mat/(1+cos_theta)
    return rot


def grid_sample_2d(img, grid, mode='bilinear'):
    '''
    Grid sampling for 2D images
    Args:
    - img  (B,C,L,W)
    - grid (B,D1,D2,2)
    Return:
    - img_out (B,D1,D2,C)
    '''
    # initialize 2D grid
    scale = torch.tensor(img.shape[-2:]).to(img.device)
    
    # rescale coordinates to [-1,1]
    grid = 2 * grid / (scale - 1) - 1
    
    # perform grid sampling
    img_out = F.grid_sample(
        img, grid.flip(-1), mode=mode, align_corners=True)

    return img_out # (B,C,D1,D2)


def find_mutli_view_plane(
    grid_3d,
    affine_2ch,
    affine_4ch,
    affine_sax,
    n_slice,
    voxel_size=1.0
):
    '''
    find multi-view cutting planes
    '''
    
    mask_2ch = np.zeros(grid_3d.shape[:-1])
    mask_4ch = np.zeros(grid_3d.shape[:-1])
    mask_sax = np.zeros(grid_3d.shape[:-1])
    
    p0 = np.array([0,0,0,1])
    p1 = np.array([0,1,0,1])
    p2 = np.array([1,0,0,1])
    
    # 2ch
    p0_2ch = (affine_2ch @ p0)[:3]
    p1_2ch = (affine_2ch @ p1)[:3]
    p2_2ch = (affine_2ch @ p2)[:3]
    normal_2ch = np.cross(p1_2ch-p0_2ch, p2_2ch-p0_2ch)
    normal_2ch = normal_2ch / np.linalg.norm(normal_2ch)
    # find the cutting plane with 2 voxels width
    dist_2ch = (grid_3d - p0_2ch) @ normal_2ch
    coord_2ch = np.stack(np.where(
        (dist_2ch>=-voxel_size) & (dist_2ch<=voxel_size)))
    mask_2ch[coord_2ch[0], coord_2ch[1], coord_2ch[2]] = 1
        
    # 4ch
    p0_4ch = (affine_4ch @ p0)[:3]
    p1_4ch = (affine_4ch @ p1)[:3]
    p2_4ch = (affine_4ch @ p2)[:3]
    normal_4ch = np.cross(p1_4ch-p0_4ch, p2_4ch-p0_4ch)
    normal_4ch = normal_4ch / np.linalg.norm(normal_4ch)
    # find the cutting plane with 2 voxels width
    dist_4ch = (grid_3d - p0_4ch) @ normal_4ch
    coord_4ch = np.stack(np.where(
        (dist_4ch>=-voxel_size) & (dist_4ch<=voxel_size)))
    mask_4ch[coord_4ch[0], coord_4ch[1], coord_4ch[2]] = 1
    
    # sax
    for s in range(n_slice):
        p0_sax = np.array([0,0,s,1])
        p1_sax = np.array([0,1,s,1])
        p2_sax = np.array([1,0,s,1])
        p0_sax = (affine_sax @ p0_sax)[:3]
        p1_sax = (affine_sax @ p1_sax)[:3]
        p2_sax = (affine_sax @ p2_sax)[:3]
        normal_sax = np.cross(p1_sax-p0_sax, p2_sax-p0_sax)
        normal_sax = normal_sax / np.linalg.norm(normal_sax)
        # find the cutting plane with 2 voxels width
        dist_sax = (grid_3d - p0_sax) @ normal_sax
        coord_sax = np.stack(np.where(
            (dist_sax>=-voxel_size) & (dist_sax<=voxel_size)))
        mask_sax[coord_sax[0], coord_sax[1], coord_sax[2]] = 1
        
    return mask_2ch, mask_4ch, mask_sax



def grid_recon_3d(img, grid, invA, shift=False, mode='bilinear'):
    '''
    reconstruct 3D volume based on input 2D images.
    img: (B,C,L,W,H), input 2D images
    grid: (D1,D2,D3,C)  input 3D grid
    invA: (1,4,4), inverse affine matrix
    shift: bool, 
        if shift is True, we shift the grid along the z-axis by (D3-1)/2,
        such that center of the grid is located in the middle of the image stacks.
    '''
    # size of the image
    scale_3d = torch.tensor(img.shape[2:]).float().to(img.device)

    # map the grid from real world space to image space
    grid_3d = grid[None] @ invA[:,:3,:3].transpose(2,1) + invA[:,:3,-1]

    # if shift is True, shift the grid along the z-axis
    # - the z-axis of 2ch or 4ch grid is shifted by 0.5 voxels
    #   to preseve spatial location
    # - the z-axis of feature maps is shifted by (D3-1)/2 voxels
    if shift:
        grid_3d[...,2] -= (scale_3d[-1] - 1)/2
    # scale to [-1,1]
    grid_3d = 2 * grid_3d / (scale_3d - 1) - 1
    grid_3d = grid_3d.flip(-1)  # (1,D1,D2,D3,3)

    # perform grid sampling
    vol = F.grid_sample(
        img, grid_3d, mode=mode, align_corners=True)
    return vol


def extract_contour(seg, affine, level=0.5):
    contour_list = []
    # find the contour of each slice
    for j in range(seg.shape[-1]):
        contour_j = find_contours(seg[...,j], level=level)
        if len(contour_j) > 0:
            contour_j_list = []
            # each segmentation may have multiple contours
            for contour in contour_j:
                # 3D contours
                contour_3d = np.zeros([contour.shape[0]-1, 3])
                contour_3d[:,:2] = contour[:-1]
                contour_3d[:,-1] = j
                # append multiple contours
                contour_j_list.append(contour_3d)                
            # append the contour for each slice
            contour_list.append(np.vstack(contour_j_list))
            
    contour_all = np.vstack(contour_list)
    # transform to world space
    contour_all = affine_transform(contour_all, affine)
    
    return contour_all


def connected_component_filter(seg):
    cc, n_cc = connected(seg, return_num=True)
    # compute the unique values and their counts
    uni, n_uni= np.unique(cc, return_counts=True)
    # remove the background
    uni = uni[1:]
    n_uni = n_uni[1:]
    # preserve the largest connected component apart from the background
    idx = np.argmax(n_uni)
    # only keep the largest connected component
    seg_cc = (cc==uni[idx]).astype(np.float32)

    # also check the reverse image
    cc, n_cc = connected(1-seg_cc, return_num=True)
    # compute the unique values and their counts
    uni, n_uni= np.unique(cc, return_counts=True)
    # remove the background
    uni = uni[1:]
    n_uni = n_uni[1:]
    # preserve the largest connected component apart from the background
    idx = np.argmax(n_uni)
    # only keep the largest connected component
    seg_cc = 1 - (cc==uni[idx]).astype(np.float32)
    return seg_cc
