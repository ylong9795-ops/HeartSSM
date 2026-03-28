import os
import glob
import argparse
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt, gaussian_filter

from tqdm import tqdm
import torch
import torch.nn.functional as F
import trimesh
import pymeshlab
import pyvista as pv

from utils.mesh import taubin_smooth, deformable


def remesh_convex_hull(vert, face, length=1.0):
    m = pymeshlab.Mesh(vert, face)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=10, targetlen=pymeshlab.AbsoluteValue(length))
    ms.apply_coord_laplacian_smoothing(
        stepsmoothnum=20, boundary=False, cotangentweight=False)
    vert_remesh = ms[0].vertex_matrix()
    face_remesh = ms[0].face_matrix()
    return vert_remesh, face_remesh


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='create four-chamber mesh template')
    parser.add_argument('--data_dir', default='./data/whs/', type=str, help='dataset directory')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    args = parser.parse_args()
    data_dir = args.data_dir
    device = args.device

    
    # --------------------
    # average all seg
    # --------------------
    subj_list = sorted(glob.glob(data_dir+'*.nii.gz'))
    n_data = len(subj_list)
    n_train = int(0.6*n_data)

    np.random.seed(12345)
    data_indices = np.random.permutation(n_data)
    train_indices = data_indices[:n_train]

    seg_whs_avg = 0

    # only use ED frame data to create template
    for i in tqdm(train_indices):
        subj_dir = subj_list[i]
        seg_whs_nii = nib.load(subj_dir)
        seg_whs = seg_whs_nii.get_fdata()
        affine_whs = seg_whs_nii.affine
        seg_whs_onehot = F.one_hot(
            torch.Tensor(seg_whs[None]).long(), num_classes=6
        ).permute(0,4,1,2,3).numpy()
        seg_whs_avg += seg_whs_onehot[0] / n_train
        
    center = - affine_whs[:3,-1]
    seg_whs_template = seg_whs_avg.copy()
    
    # enhance the LVM and disconnect atria
    seg_whs_template[1] -= 0.15
    seg_whs_template[4] -= 0.3
    seg_whs_template[5] -= 0.3
    seg_whs_template = seg_whs_template.argmax(0)

    # save template segmentation
    seg_whs_template_nii = nib.Nifti1Image(
        seg_whs_template.astype(np.uint8), affine=affine_whs)
    nib.save(seg_whs_template_nii, './template/whs_template_seg.nii.gz')

    
    # --------------------
    # extract convex hull
    # --------------------
    # extract vertices
    seg_lv_avg = (seg_whs_template==1).astype(np.float32)
    seg_my_avg = ((seg_whs_template==1)|(seg_whs_template==2)).astype(np.float32)
    seg_rv_avg = (seg_whs_template==3).astype(np.float32)
    seg_la_avg = (seg_whs_template==4).astype(np.float32)
    seg_ra_avg = (seg_whs_template==5).astype(np.float32)

    vert_lv_avg = marching_cubes(seg_lv_avg)[0]
    vert_my_avg = marching_cubes(seg_my_avg)[0]
    vert_rv_avg = marching_cubes(seg_rv_avg)[0]
    vert_la_avg = marching_cubes(seg_la_avg)[0]
    vert_ra_avg = marching_cubes(seg_ra_avg)[0]
    
    # create convex hull
    convex_lv = trimesh.convex.convex_hull(vert_lv_avg)
    convex_my = trimesh.convex.convex_hull(vert_my_avg)
    convex_rv = trimesh.convex.convex_hull(vert_rv_avg)
    convex_la = trimesh.convex.convex_hull(vert_la_avg)
    convex_ra = trimesh.convex.convex_hull(vert_ra_avg)

    vert_lv_convex, face_lv_convex = convex_lv.vertices, convex_lv.faces
    vert_my_convex, face_my_convex = convex_my.vertices, convex_my.faces
    vert_rv_convex, face_rv_convex = convex_rv.vertices, convex_rv.faces
    vert_la_convex, face_la_convex = convex_la.vertices, convex_la.faces
    vert_ra_convex, face_ra_convex = convex_ra.vertices, convex_ra.faces
    
    # remesh convex hull
    vert_lv_convex, face_lv_convex = remesh_convex_hull(vert_lv_convex, face_lv_convex, length=1.65)
    vert_my_convex, face_my_convex = remesh_convex_hull(vert_my_convex, face_my_convex, length=2.0)
    vert_rv_convex, face_rv_convex = remesh_convex_hull(vert_rv_convex, face_rv_convex, length=2.0)
    vert_la_convex, face_la_convex = remesh_convex_hull(vert_la_convex, face_la_convex, length=1.6)
    vert_ra_convex, face_ra_convex = remesh_convex_hull(vert_ra_convex, face_ra_convex, length=1.6)

    print('Euler number LV:', vert_lv_convex.shape[0] - face_lv_convex.shape[0]/2)
    print('Euler number MY:', vert_my_convex.shape[0] - face_my_convex.shape[0]/2)
    print('Euler number RV:', vert_rv_convex.shape[0] - face_rv_convex.shape[0]/2)
    print('Euler number LA:', vert_la_convex.shape[0] - face_la_convex.shape[0]/2)
    print('Euler number RA:', vert_ra_convex.shape[0] - face_ra_convex.shape[0]/2)

    # ensure the similar number of vertices, as the two atria will also inflate
    print('num of vertices LV:', vert_lv_convex.shape[0])
    print('num of vertices MY:', vert_my_convex.shape[0]) 
    print('num of vertices RV:', vert_rv_convex.shape[0]) 
    print('num of vertices LA:', vert_la_convex.shape[0]) 
    print('num of vertices RA:', vert_ra_convex.shape[0]) 

    face_lv_convex_vtk = np.hstack([3*np.ones([face_lv_convex.shape[0],1]), face_lv_convex])
    face_my_convex_vtk = np.hstack([3*np.ones([face_my_convex.shape[0],1]), face_my_convex])
    face_rv_convex_vtk = np.hstack([3*np.ones([face_rv_convex.shape[0],1]), face_rv_convex])
    face_la_convex_vtk = np.hstack([3*np.ones([face_la_convex.shape[0],1]), face_la_convex])
    face_ra_convex_vtk = np.hstack([3*np.ones([face_ra_convex.shape[0],1]), face_ra_convex])

    
    # --------------------
    # create SDF
    # --------------------
    # transform segmentation to SDF, interior < 0 and exterior > 0
    sdf_lv_avg = distance_transform_edt(1-seg_lv_avg) -\
        distance_transform_edt(seg_lv_avg)
    sdf_my_avg = distance_transform_edt(1-seg_my_avg) -\
        distance_transform_edt(seg_my_avg)
    sdf_rv_avg = distance_transform_edt(1-seg_rv_avg) -\
        distance_transform_edt(seg_rv_avg)
    sdf_la_avg = distance_transform_edt(1-seg_la_avg) -\
        distance_transform_edt(seg_la_avg)
    sdf_ra_avg = distance_transform_edt(1-seg_ra_avg) -\
        distance_transform_edt(seg_ra_avg)

    sdf_lv_avg = gaussian_filter(sdf_lv_avg, 1.0)
    sdf_my_avg = gaussian_filter(sdf_my_avg, 1.0)
    sdf_rv_avg = gaussian_filter(sdf_rv_avg, 1.0)
    sdf_la_avg = gaussian_filter(sdf_la_avg, 1.0)
    sdf_ra_avg = gaussian_filter(sdf_ra_avg, 1.0)

    sdf_lv_avg = torch.Tensor(sdf_lv_avg[None,None]).to(device)
    sdf_my_avg = torch.Tensor(sdf_my_avg[None,None]).to(device)
    sdf_rv_avg = torch.Tensor(sdf_rv_avg[None,None]).to(device)
    sdf_la_avg = torch.Tensor(sdf_la_avg[None,None]).to(device)
    sdf_ra_avg = torch.Tensor(sdf_ra_avg[None,None]).to(device)

    vert_lv_init = torch.Tensor(vert_lv_convex[None]).to(device)
    face_lv_init = torch.LongTensor(face_lv_convex[None]).to(device)
    vert_rv_init = torch.Tensor(vert_rv_convex[None]).to(device)
    face_rv_init = torch.LongTensor(face_rv_convex[None]).to(device)
    vert_la_init = torch.Tensor(vert_la_convex[None]).to(device)
    face_la_init = torch.LongTensor(face_la_convex[None]).to(device)
    vert_ra_init = torch.Tensor(vert_ra_convex[None]).to(device)
    face_ra_init = torch.LongTensor(face_ra_convex[None]).to(device)

    
    # --------------------
    # deformable model
    # --------------------
    # perform deformable model
    n_iters = 500
    h = 0.02  # step size
    gamma = 2.0  # smoothness

    vert_lv_deform, face_lv_deform = deformable(
        vert_lv_init, face_lv_init, sdf_lv_avg, n_iters, h, gamma)
    vert_lv_deform = taubin_smooth(
        vert_lv_deform, face_lv_deform, n_iters=100)

    # use LV surface as the input to deform LVM surface
    vert_my_deform, face_my_deform = deformable(
        vert_lv_deform, face_lv_deform, sdf_my_avg, n_iters, h, gamma)
    vert_my_deform = taubin_smooth(
        vert_my_deform, face_my_deform, n_iters=100)

    vert_rv_deform, face_rv_deform = deformable(
        vert_rv_init, face_rv_init, sdf_rv_avg, n_iters, h, gamma)
    vert_rv_deform = taubin_smooth(
        vert_rv_deform, face_rv_deform, n_iters=100)

    vert_la_deform, face_la_deform = deformable(
        vert_la_init, face_la_init, sdf_la_avg, n_iters, h, gamma)
    vert_la_deform = taubin_smooth(
        vert_la_deform, face_la_deform, n_iters=100)

    vert_ra_deform, face_ra_deform = deformable(
        vert_ra_init, face_ra_init, sdf_ra_avg, n_iters, h, gamma)
    vert_ra_deform = taubin_smooth(
        vert_ra_deform, face_ra_deform, n_iters=100)

    vert_lv_deform = vert_lv_deform[0].cpu().numpy()
    face_lv_deform = face_lv_deform[0].cpu().numpy()
    vert_my_deform = vert_my_deform[0].cpu().numpy()
    face_my_deform = face_my_deform[0].cpu().numpy()
    vert_rv_deform = vert_rv_deform[0].cpu().numpy()
    face_rv_deform = face_rv_deform[0].cpu().numpy()
    vert_la_deform = vert_la_deform[0].cpu().numpy()
    face_la_deform = face_la_deform[0].cpu().numpy()
    vert_ra_deform = vert_ra_deform[0].cpu().numpy()
    face_ra_deform = face_ra_deform[0].cpu().numpy()

    face_lv_deform_vtk = np.hstack([3*np.ones([face_lv_deform.shape[0],1]), face_lv_deform])
    face_my_deform_vtk = np.hstack([3*np.ones([face_my_deform.shape[0],1]), face_my_deform])
    face_rv_deform_vtk = np.hstack([3*np.ones([face_rv_deform.shape[0],1]), face_rv_deform])
    face_la_deform_vtk = np.hstack([3*np.ones([face_la_deform.shape[0],1]), face_la_deform])
    face_ra_deform_vtk = np.hstack([3*np.ones([face_ra_deform.shape[0],1]), face_ra_deform])

    mesh_lv_deform_vtk = pv.PolyData(vert_lv_deform - center, face_lv_deform_vtk.astype(int))
    mesh_my_deform_vtk = pv.PolyData(vert_my_deform - center, face_my_deform_vtk.astype(int))
    mesh_rv_deform_vtk = pv.PolyData(vert_rv_deform - center, face_rv_deform_vtk.astype(int))
    mesh_la_deform_vtk = pv.PolyData(vert_la_deform - center, face_la_deform_vtk.astype(int))
    mesh_ra_deform_vtk = pv.PolyData(vert_ra_deform - center, face_ra_deform_vtk.astype(int))

    mesh_lv_deform_vtk.save('./template/whs_template_mesh_lv.vtk')
    mesh_my_deform_vtk.save('./template/whs_template_mesh_my.vtk')
    mesh_rv_deform_vtk.save('./template/whs_template_mesh_rv.vtk')
    mesh_la_deform_vtk.save('./template/whs_template_mesh_la.vtk')
    mesh_ra_deform_vtk.save('./template/whs_template_mesh_ra.vtk')

