import numpy as np
import glob
import argparse
import time
import nibabel as nib
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.ffdnet import FFDNet
from utils.utils import affine_transform
from utils.ffd import bspline_ffd
import pyvista as pv


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HeartSSM training')
    parser.add_argument('--data_dir', default='./data/ukbb/', type=str, help='dataset directory')
    parser.add_argument('--n_frame', default=50, type=int, help='number of time frames')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    args = parser.parse_args()
    data_dir = args.data_dir
    n_frame = args.n_frame
    device = args.device
    
    # ------------------------
    # load templates
    # ------------------------    

    # load template 3D seg
    seg_atlas_nii = nib.load('./template/whs_template_seg.nii.gz')
    seg_atlas = seg_atlas_nii.get_fdata()
    affine_atlas = seg_atlas_nii.affine
    voxel_size = 2
    seg_atlas = torch.LongTensor(seg_atlas[None]).to(device)
    seg_atlas = F.one_hot(seg_atlas, num_classes=6).permute(0,4,1,2,3).float()
    seg_atlas = F.interpolate(
        seg_atlas, size=[96,96,128], mode='trilinear', align_corners=True
    ).argmax(1)[None].float()

    # load template mesh
    mesh_lv_in = pv.read('./template/whs_template_mesh_lv.vtk')
    mesh_my_in = pv.read('./template/whs_template_mesh_my.vtk')
    mesh_rv_in = pv.read('./template/whs_template_mesh_rv.vtk')
    mesh_la_in = pv.read('./template/whs_template_mesh_la.vtk')
    mesh_ra_in = pv.read('./template/whs_template_mesh_ra.vtk')

    vert_lv_in, face_lv_in = mesh_lv_in.points, mesh_lv_in.faces.reshape(-1,4)[:,1:]
    vert_my_in, face_my_in = mesh_my_in.points, mesh_my_in.faces.reshape(-1,4)[:,1:]
    vert_rv_in, face_rv_in = mesh_rv_in.points, mesh_rv_in.faces.reshape(-1,4)[:,1:]
    vert_la_in, face_la_in = mesh_la_in.points, mesh_la_in.faces.reshape(-1,4)[:,1:]
    vert_ra_in, face_ra_in = mesh_ra_in.points, mesh_ra_in.faces.reshape(-1,4)[:,1:]

    vert_lv_in = affine_transform(vert_lv_in, np.linalg.inv(affine_atlas)) / voxel_size
    vert_my_in = affine_transform(vert_my_in, np.linalg.inv(affine_atlas)) / voxel_size
    vert_rv_in = affine_transform(vert_rv_in, np.linalg.inv(affine_atlas)) / voxel_size
    vert_la_in = affine_transform(vert_la_in, np.linalg.inv(affine_atlas)) / voxel_size
    vert_ra_in = affine_transform(vert_ra_in, np.linalg.inv(affine_atlas)) / voxel_size

    vert_lv_in = torch.Tensor(vert_lv_in[None]).to(device)
    vert_my_in = torch.Tensor(vert_my_in[None]).to(device)
    vert_rv_in = torch.Tensor(vert_rv_in[None]).to(device)
    vert_la_in = torch.Tensor(vert_la_in[None]).to(device)
    vert_ra_in = torch.Tensor(vert_ra_in[None]).to(device)

    face_lv_in = torch.LongTensor(face_lv_in[None].copy()).to(device)
    face_my_in = torch.LongTensor(face_my_in[None].copy()).to(device)
    face_rv_in = torch.LongTensor(face_rv_in[None].copy()).to(device)
    face_la_in = torch.LongTensor(face_la_in[None].copy()).to(device)
    face_ra_in = torch.LongTensor(face_ra_in[None].copy()).to(device)

    # number of vertices
    n_vert_lv_in = vert_lv_in.shape[1]
    n_vert_my_in = vert_my_in.shape[1]
    n_vert_rv_in = vert_rv_in.shape[1]
    n_vert_la_in = vert_la_in.shape[1]
    n_vert_ra_in = vert_ra_in.shape[1]

    # concatenate the vertices of all chambers
    vert_all_in = torch.cat(
        [vert_lv_in, vert_my_in, vert_rv_in, vert_la_in, vert_ra_in], dim=1)

    # map template mesh into real-world space
    vert_all_atlas = affine_transform(
        vert_all_in[0].cpu().numpy() * voxel_size, affine_atlas)
    center_atlas = vert_all_atlas.mean(0)
    vert_all_atlas_ = vert_all_atlas - center_atlas

    # index to split the tensor of all vertices
    idx_vert_lv_in = n_vert_lv_in
    idx_vert_my_in = idx_vert_lv_in + n_vert_my_in
    idx_vert_rv_in = idx_vert_my_in + n_vert_rv_in
    idx_vert_la_in = idx_vert_rv_in + n_vert_la_in
    idx_vert_all_in = [idx_vert_lv_in, idx_vert_my_in, idx_vert_rv_in, idx_vert_la_in]

    # faces for saving vtk files
    face_lv_pred = face_lv_in[0].cpu().numpy()
    face_my_pred = face_my_in[0].cpu().numpy()
    face_rv_pred = face_rv_in[0].cpu().numpy()
    face_la_pred = face_la_in[0].cpu().numpy()
    face_ra_pred = face_ra_in[0].cpu().numpy()

    face_my_pred += face_lv_pred.max() + 1
    face_rv_pred += face_my_pred.max() + 1
    face_la_pred += face_rv_pred.max() + 1
    face_ra_pred += face_la_pred.max() + 1

    face_all_pred = np.vstack([
        face_lv_pred,
        face_my_pred,
        face_rv_pred,
        face_la_pred,
        face_ra_pred])

    face_all_pred_vtk = np.hstack([
        3*np.ones([face_all_pred.shape[0], 1], dtype=int), face_all_pred])
    
    # ------------------------
    # load neural network model
    # ------------------------
    ffd_net = FFDNet(
        n_frame=n_frame, dim_hid=256, drop_rate=0.0).to(device)
    ffd_net.load_state_dict(torch.load(
        './ckpts/model_heartffd_unet.pt', map_location=device))
    ffd_net.eval();
    
    # ------------------------
    # load SSM
    # ------------------------
    # load ssm model
    with open('./ssm/heartssm_64k.pkl', 'rb') as f:
        ipca = pickle.load(f)
    mean_shape = ipca.mean_  # mean shape
    heart_pc = ipca.components_  # principle components
    
    # ------------------------
    # predict mesh and recon shape descriptor
    # ------------------------

    subj_list = sorted(glob.glob(data_dir+'*'))
    feat_list = []
    runtime = []

    for i in tqdm(range(len(subj_list))):

        subj_dir = subj_list[i]
        
        # load 4d segmentations
        seg_4d_nii = nib.load(subj_dir+'/seg_4d.nii.gz')
        affine_4d = seg_4d_nii.affine
        seg_4d = seg_4d_nii.get_fdata().transpose(3,0,1,2)

        t_start = time.time()
        seg_4d = torch.Tensor(seg_4d[None]).to(device).float()
        seg_in = torch.cat([seg_atlas, seg_4d], dim=1) / 5.

        # predict ffd
        with torch.no_grad():
            ffd_pred = ffd_net(seg_in)

        # apply ffd
        vert_all_pred = []
        vert_all_pred_ed = vert_all_in.clone()

        # ED frame
        for t in range(3):  # multi-scale
            vert_all_pred_ed = vert_all_pred_ed + bspline_ffd(
                vert_all_pred_ed / (2**(4-t)), ffd_pred[t])
        vert_all_pred.append(vert_all_pred_ed)

        # all frames
        for t in range(3,n_frame+2):
            vert_all_pred_t = vert_all_pred[-1] + bspline_ffd(
                vert_all_pred[-1] / 4, ffd_pred[t])
            vert_all_pred.append(vert_all_pred_t)

        vert_all_pred = torch.vstack(vert_all_pred).cpu().numpy()
        vert_all_pred = vert_all_pred @ affine_4d[:3,:3] + affine_4d[:3,-1]

        # find the best rigid transformation
        center_pred = vert_all_pred[0].mean(0)
        vert_all_pred_ = vert_all_pred[0] - center_pred

        # compute optimal rotation matrix via Kabsch algorithm
        U, _, V = np.linalg.svd(
            vert_all_pred_.T @ vert_all_atlas_)
        Rot = V.T @ U.T
        # compute optimal translation
        Trans = center_atlas - Rot @ center_pred
        # apply rigid transformation
        vert_all_pred = vert_all_pred @ Rot.T + Trans

        # compute shape feature for individual
        feat = heart_pc @ (vert_all_pred.reshape(-1) - mean_shape)    
        feat_list.append(feat)
        
        t_end = time.time()
        runtime.append(t_end - t_start)
        
        # save rigid matrix
        rigid = np.eye(4)
        rigid[:3,:3] = Rot
        rigid[:3,-1] = Trans
        np.save(subj_dir+'/rigid.npy', rigid)

        # save shape descriptor as feature vector
        np.save(subj_dir+'/feat_64k.npy', feat)
        
    # runtime for HeartSSM shape fitting
    print('average runtime: {} sec'.format(np.mean(runtime)))

    # save all feature vectors
    np.save('./ssm/feat_64k.npy', np.stack(feat_list))
