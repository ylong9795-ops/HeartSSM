import numpy as np
import argparse
import logging
import glob
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from net.ffdnet import FFDNet
from utils.data import SurfDataset
from utils.utils import affine_transform
from utils.metrics import surface_distance, correlation
from utils.mesh import laplacian, curvature
from utils.ffd import bspline_basis, bspline_ffd
import pyvista as pv


def train_loop(args):
    
    data_dir = args.data_dir
    tag = args.tag
    device = args.device
    lr = args.lr
    n_epoch = args.n_epoch
    n_frame = args.n_frame  # number of frames
    n_train = args.n_train  # number of training data
    n_valid = args.n_valid  # number of validation data
    drop_rate = args.drop_rate
    w_edge = args.w_edge  # weight for edge length loss
    w_curv = args.w_curv  # weight for curvature loss
    w_geod = args.w_geod  # weight for geodesic loss
    w_cycl = args.w_cycl  # weight for cycle consistency loss
    
    # start training logging
    logging.basicConfig(
        filename='./ckpts/log_heartffd_'+tag+'.log',
        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    
    n_data = len(glob.glob(data_dir+'*'))

    np.random.seed(12345)
    data_indices = np.random.permutation(n_data)
    train_indices = data_indices[:n_train]
    valid_indices = data_indices[n_train:n_train+n_valid]

    trainset = SurfDataset(
        data_dir=data_dir, data_indices=train_indices, n_frame=n_frame)
    validset = SurfDataset(
        data_dir=data_dir, data_indices=valid_indices, n_frame=n_frame)

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    
    # ------ load templates ------ 
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

    # edge
    edge_lv_in = torch.cat([face_lv_in[0,:,[0,1]],
                            face_lv_in[0,:,[1,2]],
                            face_lv_in[0,:,[2,0]]], dim=0).T
    edge_my_in = torch.cat([face_my_in[0,:,[0,1]],
                            face_my_in[0,:,[1,2]],
                            face_my_in[0,:,[2,0]]], dim=0).T
    edge_rv_in = torch.cat([face_rv_in[0,:,[0,1]],
                            face_rv_in[0,:,[1,2]],
                            face_rv_in[0,:,[2,0]]], dim=0).T
    edge_la_in = torch.cat([face_la_in[0,:,[0,1]],
                            face_la_in[0,:,[1,2]],
                            face_la_in[0,:,[2,0]]], dim=0).T
    edge_ra_in = torch.cat([face_ra_in[0,:,[0,1]],
                            face_ra_in[0,:,[1,2]],
                            face_ra_in[0,:,[2,0]]], dim=0).T

    # laplacian
    L_lv = laplacian(face_lv_in)
    L_my = laplacian(face_my_in)
    L_rv = laplacian(face_rv_in)
    L_la = laplacian(face_la_in)
    L_ra = laplacian(face_ra_in)

    # curvature
    curv_lv_in = curvature(vert_lv_in, face_lv_in, L_lv)
    curv_my_in = curvature(vert_my_in, face_my_in, L_my)
    curv_rv_in = curvature(vert_rv_in, face_rv_in, L_rv)
    curv_la_in = curvature(vert_la_in, face_la_in, L_la)
    curv_ra_in = curvature(vert_ra_in, face_ra_in, L_ra)

    # number of vertices
    n_vert_lv_in = vert_lv_in.shape[1]
    n_vert_my_in = vert_my_in.shape[1]
    n_vert_rv_in = vert_rv_in.shape[1]
    n_vert_la_in = vert_la_in.shape[1]
    n_vert_ra_in = vert_ra_in.shape[1]

    # concatenate the vertices of all chambers
    vert_all_in = torch.cat(
        [vert_lv_in, vert_my_in, vert_rv_in, vert_la_in, vert_ra_in], dim=1)

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
    
    # ------ initialize model ------ 
    logging.info("initalize model ...")
    ffd_net = FFDNet(
        n_frame=n_frame, dim_hid=256, drop_rate=drop_rate).to(device)
    optimizer = optim.Adam(ffd_net.parameters(), lr=1e-4)
    
    # ------ training loop ------ 
    for epoch in tqdm(range(n_epoch+1)):
        train_recon_loss = []
        train_edge_loss = []
        train_curv_loss = []
        train_geod_loss = []
        train_cycl_loss = []
        
        ffd_net.train()
        
        for idx, data in enumerate(trainloader):
            optimizer.zero_grad()
            
            # load data
            seg_4d, vert_lv_gt, vert_my_gt, vert_rv_gt, vert_la_gt, \
            vert_ra_gt, n_vert_lv_gt, n_vert_my_gt, n_vert_rv_gt, \
            n_vert_la_gt, n_vert_ra_gt = data
                
            seg_4d = seg_4d.to(device).float()
            seg_in = torch.cat([seg_atlas, seg_4d], dim=1) / 5.

            vert_lv_gt = vert_lv_gt[0].float().to(device)
            vert_my_gt = vert_my_gt[0].float().to(device)
            vert_rv_gt = vert_rv_gt[0].float().to(device)
            vert_la_gt = vert_la_gt[0].float().to(device)
            vert_ra_gt = vert_ra_gt[0].float().to(device)
            
            n_vert_lv_gt = n_vert_lv_gt[0].long().to(device)
            n_vert_my_gt = n_vert_my_gt[0].long().to(device)
            n_vert_rv_gt = n_vert_rv_gt[0].long().to(device)
            n_vert_la_gt = n_vert_la_gt[0].long().to(device)
            n_vert_ra_gt = n_vert_ra_gt[0].long().to(device)

            # predict ffd
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
            vert_all_pred = torch.vstack(vert_all_pred)

            vert_lv_pred, vert_my_pred, vert_rv_pred, \
            vert_la_pred, vert_ra_pred = torch.tensor_split(
                vert_all_pred, idx_vert_all_in, dim=1)
            
            # compute reconstruction loss
            recon_loss_lv = surface_distance(vert_lv_pred, vert_lv_gt, y_lengths=n_vert_lv_gt)
            recon_loss_my = surface_distance(vert_my_pred, vert_my_gt, y_lengths=n_vert_my_gt)
            recon_loss_rv = surface_distance(vert_rv_pred, vert_rv_gt, y_lengths=n_vert_rv_gt)
            recon_loss_la = surface_distance(vert_la_pred, vert_la_gt, y_lengths=n_vert_la_gt)
            recon_loss_ra = surface_distance(vert_ra_pred, vert_ra_gt, y_lengths=n_vert_ra_gt)

            recon_loss = recon_loss_lv + recon_loss_my + recon_loss_rv + \
                         recon_loss_la + recon_loss_ra

            # compute edge loss
            vert_lv_i = vert_lv_pred[:,edge_lv_in[0]]
            vert_lv_j = vert_lv_pred[:,edge_lv_in[1]]
            vert_my_i = vert_my_pred[:,edge_my_in[0]]
            vert_my_j = vert_my_pred[:,edge_my_in[1]]
            vert_rv_i = vert_rv_pred[:,edge_rv_in[0]]
            vert_rv_j = vert_rv_pred[:,edge_rv_in[1]]
            vert_la_i = vert_la_pred[:,edge_la_in[0]]
            vert_la_j = vert_la_pred[:,edge_la_in[1]]
            vert_ra_i = vert_ra_pred[:,edge_ra_in[0]]
            vert_ra_j = vert_ra_pred[:,edge_ra_in[1]]

            # standard deviation of edge length
            edge_loss = (vert_lv_i - vert_lv_j).norm(dim=-1).std(dim=1).mean() + \
                        (vert_my_i - vert_my_j).norm(dim=-1).std(dim=1).mean() + \
                        (vert_rv_i - vert_rv_j).norm(dim=-1).std(dim=1).mean() + \
                        (vert_la_i - vert_la_j).norm(dim=-1).std(dim=1).mean() + \
                        (vert_ra_i - vert_ra_j).norm(dim=-1).std(dim=1).mean()

            # compute curvature loss
            curv_lv_pred = curvature(vert_lv_pred, face_lv_in, L_lv)
            curv_my_pred = curvature(vert_my_pred, face_my_in, L_my)
            curv_rv_pred = curvature(vert_rv_pred, face_rv_in, L_rv)
            curv_la_pred = curvature(vert_la_pred, face_la_in, L_la)
            curv_ra_pred = curvature(vert_ra_pred, face_ra_in, L_ra)

            # pcc of curvature
            curv_loss = (1-correlation(curv_lv_pred, curv_lv_in.repeat(n_frame,1))) + \
                        (1-correlation(curv_my_pred, curv_my_in.repeat(n_frame,1))) + \
                        (1-correlation(curv_rv_pred, curv_rv_in.repeat(n_frame,1))) + \
                        (1-correlation(curv_la_pred, curv_la_in.repeat(n_frame,1))) + \
                        (1-correlation(curv_ra_pred, curv_ra_in.repeat(n_frame,1)))

            # geodesic loss
            geod_loss = (vert_lv_pred[1:] - vert_lv_pred[:-1]).norm(dim=-1).mean() + \
                        (vert_my_pred[1:] - vert_my_pred[:-1]).norm(dim=-1).mean() + \
                        (vert_rv_pred[1:] - vert_rv_pred[:-1]).norm(dim=-1).mean() + \
                        (vert_la_pred[1:] - vert_la_pred[:-1]).norm(dim=-1).mean() + \
                        (vert_ra_pred[1:] - vert_ra_pred[:-1]).norm(dim=-1).mean()
            
            # cycle consistency loss
            cycl_loss = (vert_lv_pred[-1] - vert_lv_pred[0]).norm(dim=-1).mean() + \
                        (vert_my_pred[-1] - vert_my_pred[0]).norm(dim=-1).mean() + \
                        (vert_rv_pred[-1] - vert_rv_pred[0]).norm(dim=-1).mean() + \
                        (vert_la_pred[-1] - vert_la_pred[0]).norm(dim=-1).mean() + \
                        (vert_ra_pred[-1] - vert_ra_pred[0]).norm(dim=-1).mean()

            train_recon_loss.append(recon_loss.item())
            train_edge_loss.append(w_edge*edge_loss.item())
            train_curv_loss.append(w_curv*curv_loss.item())
            train_geod_loss.append(w_geod*geod_loss.item())
            train_cycl_loss.append(w_cycl*cycl_loss.item())   

            loss = recon_loss + w_edge*edge_loss + w_curv*curv_loss + w_geod*geod_loss + w_cycl*cycl_loss
            loss.backward()
            optimizer.step()
        
        logging.info("epoch:{}, train recon loss:{}".format(epoch, np.mean(train_recon_loss)))
        logging.info("epoch:{}, train  edge loss:{}".format(epoch, np.mean(train_edge_loss)))
        logging.info("epoch:{}, train  curv loss:{}".format(epoch, np.mean(train_curv_loss)))
        logging.info("epoch:{}, train  geod loss:{}".format(epoch, np.mean(train_geod_loss)))
        logging.info("epoch:{}, train  cycl loss:{}".format(epoch, np.mean(train_cycl_loss)))
        logging.info("------------------------------")

        if epoch % 10 == 0:  # start validation
            ffd_net.eval()
            # save model checkpoints
            torch.save(ffd_net.state_dict(),
                       './ckpts/model_heartffd_'+tag+'_'+str(epoch)+'epochs.pt')

            valid_recon_loss = []
            valid_edge_loss = []
            valid_curv_loss = []
            valid_geod_loss = []
            valid_cycl_loss = []

            for idx, data in enumerate(validloader):
                # load data
                seg_4d, vert_lv_gt, vert_my_gt, vert_rv_gt, vert_la_gt, \
                vert_ra_gt, n_vert_lv_gt, n_vert_my_gt, n_vert_rv_gt, \
                n_vert_la_gt, n_vert_ra_gt = data

                seg_4d = seg_4d.to(device).float()
                seg_in = torch.cat([seg_atlas, seg_4d], dim=1) / 5.

                vert_lv_gt = vert_lv_gt[0].float().to(device)
                vert_my_gt = vert_my_gt[0].float().to(device)
                vert_rv_gt = vert_rv_gt[0].float().to(device)
                vert_la_gt = vert_la_gt[0].float().to(device)
                vert_ra_gt = vert_ra_gt[0].float().to(device)
                
                n_vert_lv_gt = n_vert_lv_gt[0].long().to(device)
                n_vert_my_gt = n_vert_my_gt[0].long().to(device)
                n_vert_rv_gt = n_vert_rv_gt[0].long().to(device)
                n_vert_la_gt = n_vert_la_gt[0].long().to(device)
                n_vert_ra_gt = n_vert_ra_gt[0].long().to(device)

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
                vert_all_pred = torch.vstack(vert_all_pred)

                vert_lv_pred, vert_my_pred, vert_rv_pred, \
                vert_la_pred, vert_ra_pred = torch.tensor_split(
                    vert_all_pred, idx_vert_all_in, dim=1)

                # compute reconstruction loss
                recon_loss_lv = surface_distance(vert_lv_pred, vert_lv_gt, y_lengths=n_vert_lv_gt)
                recon_loss_my = surface_distance(vert_my_pred, vert_my_gt, y_lengths=n_vert_my_gt)
                recon_loss_rv = surface_distance(vert_rv_pred, vert_rv_gt, y_lengths=n_vert_rv_gt)
                recon_loss_la = surface_distance(vert_la_pred, vert_la_gt, y_lengths=n_vert_la_gt)
                recon_loss_ra = surface_distance(vert_ra_pred, vert_ra_gt, y_lengths=n_vert_ra_gt)

                recon_loss = recon_loss_lv + recon_loss_my + recon_loss_rv + \
                             recon_loss_la + recon_loss_ra

                # compute edge loss
                vert_lv_i = vert_lv_pred[:,edge_lv_in[0]]
                vert_lv_j = vert_lv_pred[:,edge_lv_in[1]]
                vert_my_i = vert_my_pred[:,edge_my_in[0]]
                vert_my_j = vert_my_pred[:,edge_my_in[1]]
                vert_rv_i = vert_rv_pred[:,edge_rv_in[0]]
                vert_rv_j = vert_rv_pred[:,edge_rv_in[1]]
                vert_la_i = vert_la_pred[:,edge_la_in[0]]
                vert_la_j = vert_la_pred[:,edge_la_in[1]]
                vert_ra_i = vert_ra_pred[:,edge_ra_in[0]]
                vert_ra_j = vert_ra_pred[:,edge_ra_in[1]]

                # standard deviation of edge length
                edge_loss = (vert_lv_i - vert_lv_j).norm(dim=-1).std(dim=1).mean() + \
                            (vert_my_i - vert_my_j).norm(dim=-1).std(dim=1).mean() + \
                            (vert_rv_i - vert_rv_j).norm(dim=-1).std(dim=1).mean() + \
                            (vert_la_i - vert_la_j).norm(dim=-1).std(dim=1).mean() + \
                            (vert_ra_i - vert_ra_j).norm(dim=-1).std(dim=1).mean()

                # compute curvature loss
                curv_lv_pred = curvature(vert_lv_pred, face_lv_in, L_lv)
                curv_my_pred = curvature(vert_my_pred, face_my_in, L_my)
                curv_rv_pred = curvature(vert_rv_pred, face_rv_in, L_rv)
                curv_la_pred = curvature(vert_la_pred, face_la_in, L_la)
                curv_ra_pred = curvature(vert_ra_pred, face_ra_in, L_ra)

                # pcc of curvature
                curv_loss = (1-correlation(curv_lv_pred, curv_lv_in.repeat(n_frame,1))) + \
                            (1-correlation(curv_my_pred, curv_my_in.repeat(n_frame,1))) + \
                            (1-correlation(curv_rv_pred, curv_rv_in.repeat(n_frame,1))) + \
                            (1-correlation(curv_la_pred, curv_la_in.repeat(n_frame,1))) + \
                            (1-correlation(curv_ra_pred, curv_ra_in.repeat(n_frame,1)))

                # geodesic loss
                geod_loss = (vert_lv_pred[1:] - vert_lv_pred[:-1]).norm(dim=-1).mean() + \
                            (vert_my_pred[1:] - vert_my_pred[:-1]).norm(dim=-1).mean() + \
                            (vert_rv_pred[1:] - vert_rv_pred[:-1]).norm(dim=-1).mean() + \
                            (vert_la_pred[1:] - vert_la_pred[:-1]).norm(dim=-1).mean() + \
                            (vert_ra_pred[1:] - vert_ra_pred[:-1]).norm(dim=-1).mean()

                # cycle consistency loss
                cycl_loss = (vert_lv_pred[-1] - vert_lv_pred[0]).norm(dim=-1).mean() + \
                            (vert_my_pred[-1] - vert_my_pred[0]).norm(dim=-1).mean() + \
                            (vert_rv_pred[-1] - vert_rv_pred[0]).norm(dim=-1).mean() + \
                            (vert_la_pred[-1] - vert_la_pred[0]).norm(dim=-1).mean() + \
                            (vert_ra_pred[-1] - vert_ra_pred[0]).norm(dim=-1).mean()

                valid_recon_loss.append(recon_loss.item())
                valid_edge_loss.append(w_edge*edge_loss.item())
                valid_curv_loss.append(w_curv*curv_loss.item())
                valid_geod_loss.append(w_geod*geod_loss.item())
                valid_cycl_loss.append(w_cycl*cycl_loss.item())   
                        
            logging.info("epoch:{}, valid recon loss:{}".format(epoch, np.mean(valid_recon_loss)))
            logging.info("epoch:{}, valid  edge loss:{}".format(epoch, np.mean(valid_edge_loss)))
            logging.info("epoch:{}, valid  curv loss:{}".format(epoch, np.mean(valid_curv_loss)))
            logging.info("epoch:{}, valid  geod loss:{}".format(epoch, np.mean(valid_geod_loss)))
            logging.info("epoch:{}, valid  cycl loss:{}".format(epoch, np.mean(valid_cycl_loss)))
            logging.info("------------------------------")
            
            for t in [0,9,19,29,39,49]:
                # save 3d gt points
                vert_all_gt_t = np.vstack([
                    vert_lv_gt[t].cpu().numpy(),
                    vert_my_gt[t].cpu().numpy(),
                    vert_rv_gt[t].cpu().numpy(),
                    vert_la_gt[t].cpu().numpy(),
                    vert_ra_gt[t].cpu().numpy(),
                ])
                vert_all_gt_t_vtk = pv.PolyData(vert_all_gt_t)
                vert_all_gt_t_vtk.save(
                    './ckpts/val_vert_all_gt_'+tag+'_t'+('0'+str(t+1))[-2:]+'.vtk')

                # save predicted affine mesh
                mesh_all_pred_t_vtk = pv.PolyData(
                    vert_all_pred[t].cpu().numpy(), face_all_pred_vtk)
                mesh_all_pred_t_vtk.save(
                    './ckpts/val_mesh_all_pred_'+tag+'_t'+('0'+str(t+1))[-2:]+'.vtk')
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeartFFDNet Mesh Reconstruction")
    parser.add_argument('--data_dir', default='./dataset/', type=str, help="data directory")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    
    parser.add_argument('--n_epoch', default=200, type=int, help="number of training epochs")
    parser.add_argument('--n_frame', default=50, type=int, help="number of frames")
    parser.add_argument('--n_train', default=600, type=int, help="number of training data")
    parser.add_argument('--n_valid', default=100, type=int, help="number of validation data")

    parser.add_argument('--w_edge', default=0.1, type=float, help="weight for edge length loss")
    parser.add_argument('--w_curv', default=0.1, type=float, help="weight for curvature loss")
    parser.add_argument('--w_geod', default=0.1, type=float, help="weight for geodesic loss")
    parser.add_argument('--w_cycl', default=0.1, type=float, help="weight for cycle consistency loss")
    parser.add_argument('--drop_rate', default=0.0, type=float, help="dropout rate")

    args = parser.parse_args()
    
    train_loop(args)
    