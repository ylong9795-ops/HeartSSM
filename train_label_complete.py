import numpy as np
import argparse
import logging
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from net.unet import UNet
from utils.data import SegDataset
from utils.metrics import surface_distance, dice_score
from skimage.measure import marching_cubes


def train_loop(args):
    
    data_dir = args.data_dir
    tag = args.tag
    device = args.device
    n_epoch = args.n_epoch
    lr = args.lr
    sigma = args.sigma  # random motion
    scaling = args.scaling  # random scaling
    drop_rate = args.drop_rate
    
    # start training logging
    logging.basicConfig(
        filename='./ckpts/log_label_complete_'+tag+'.log',
        level=logging.INFO, format='%(asctime)s %(message)s')

    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    
    n_data = len(glob.glob(data_dir+'*.nii.gz'))
    n_train = int(0.6*n_data)
    n_valid = int(0.1*n_data)
    n_test = int(0.3*n_data)

    np.random.seed(12345)
    data_indices = np.random.permutation(n_data)
    train_indices = data_indices[:n_train]
    valid_indices = data_indices[n_train:n_train+n_valid]
    test_indices = data_indices[n_train+n_valid:]

    trainset = SegDataset(
        data_dir=data_dir, data_indices=train_indices,
        data_type='train', seed=12345, sigma=sigma, 
        scaling=scaling, p_4ch=40, voxel_size=2.0,
        grid_size=[96,96,128], device=device)
    
    validset = SegDataset(
        data_dir=data_dir, data_indices=valid_indices,
        data_type='valid', seed=12345, sigma=sigma, 
        scaling=scaling, p_4ch=40, voxel_size=2.0,
        grid_size=[96,96,128], device=device)
    
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    
    
    # ------ initialize model ------ 
    logging.info("initalize model ...")
    unet = UNet(
        dim_in=6,
        dim_hid=[32,64,128,256,256],
        dim_out=6,
        drop_rate=drop_rate).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    
    # ------ training loop ------ 
    for epoch in tqdm(range(n_epoch+1)):
        avg_loss = []
        unet.train()
        for idx, data in enumerate(trainloader):
            optimizer.zero_grad()
            seg_cmr, seg_3d = data
            seg_cmr = F.one_hot(
                seg_cmr.long(), num_classes=6
            ).permute(0,4,1,2,3).float().to(device)
            seg_3d = F.one_hot(
                seg_3d.long(), num_classes=6
            ).permute(0,4,1,2,3).float().to(device)

            # learn the residual of segmentations
            seg_pred = unet(seg_cmr)
            loss = nn.MSELoss()(seg_pred, seg_3d)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(avg_loss)))


        if epoch % 10 == 0:  # start validation
            unet.eval()
            # save model checkpoints
            torch.save(unet.state_dict(),
                       './ckpts/model_label_complete_'+tag+'_'+str(epoch)+'epochs.pt')

            valid_loss = []

            # dice score
            valid_dice_all = []
            valid_dice_lv = []
            valid_dice_my = []
            valid_dice_rv = []
            valid_dice_la = []
            valid_dice_ra = []

            valid_assd_all = []
            valid_assd_lv = []
            valid_assd_my = []
            valid_assd_rv = []
            valid_assd_la = []
            valid_assd_ra = []
            
            valid_hd90_all = []
            valid_hd90_lv = []
            valid_hd90_my = []
            valid_hd90_rv = []
            valid_hd90_la = []
            valid_hd90_ra = []

            for idx, data in enumerate(validloader):
                seg_cmr, seg_3d = data
                seg_cmr = F.one_hot(
                    seg_cmr.long(), num_classes=6
                ).permute(0,4,1,2,3).float().to(device)
                seg_3d = F.one_hot(
                    seg_3d.long(), num_classes=6
                ).permute(0,4,1,2,3).float().to(device)

                with torch.no_grad():
                    seg_pred = unet(seg_cmr)
                loss = nn.MSELoss()(seg_pred, seg_3d)
                valid_loss.append(loss.item())

                seg_3d = seg_3d.argmax(1)[0].cpu().numpy()
                seg_cmr = seg_cmr.argmax(1)[0].cpu().numpy()
                seg_pred = seg_pred.argmax(1)[0].cpu().numpy()

                # compute dice score
                dice_lv = dice_score(seg_pred==1, seg_3d==1)
                dice_my = dice_score(seg_pred==2, seg_3d==2)
                dice_rv = dice_score(seg_pred==3, seg_3d==3)
                dice_la = dice_score(seg_pred==4, seg_3d==4)
                dice_ra = dice_score(seg_pred==5, seg_3d==5)
                dice_all = (dice_lv + dice_my + dice_rv + dice_la + dice_ra)/5

                valid_dice_all.append(dice_all)
                valid_dice_lv.append(dice_lv)
                valid_dice_my.append(dice_my)
                valid_dice_rv.append(dice_rv)
                valid_dice_la.append(dice_la)
                valid_dice_ra.append(dice_ra)

                vert_lv_pred = marching_cubes((seg_pred==1).astype(np.float32))[0]
                vert_lv_3d = marching_cubes((seg_3d==1).astype(np.float32))[0]
                vert_my_pred = marching_cubes((seg_pred==2).astype(np.float32))[0]
                vert_my_3d = marching_cubes((seg_3d==2).astype(np.float32))[0]
                vert_rv_pred = marching_cubes((seg_pred==3).astype(np.float32))[0]
                vert_rv_3d = marching_cubes((seg_3d==3).astype(np.float32))[0]
                vert_la_pred = marching_cubes((seg_pred==4).astype(np.float32))[0]
                vert_la_3d = marching_cubes((seg_3d==4).astype(np.float32))[0]
                vert_ra_pred = marching_cubes((seg_pred==5).astype(np.float32))[0]
                vert_ra_3d = marching_cubes((seg_3d==5).astype(np.float32))[0]

                vert_lv_pred = torch.Tensor(vert_lv_pred[None].copy()).to(device)
                vert_my_pred = torch.Tensor(vert_my_pred[None].copy()).to(device)
                vert_rv_pred = torch.Tensor(vert_rv_pred[None].copy()).to(device)
                vert_la_pred = torch.Tensor(vert_la_pred[None].copy()).to(device)
                vert_ra_pred = torch.Tensor(vert_ra_pred[None].copy()).to(device)

                vert_lv_3d = torch.Tensor(vert_lv_3d[None].copy()).to(device)
                vert_my_3d = torch.Tensor(vert_my_3d[None].copy()).to(device)
                vert_rv_3d = torch.Tensor(vert_rv_3d[None].copy()).to(device)
                vert_la_3d = torch.Tensor(vert_la_3d[None].copy()).to(device)
                vert_ra_3d = torch.Tensor(vert_ra_3d[None].copy()).to(device)

                assd_lv, hd90_lv = surface_distance(vert_lv_pred, vert_lv_3d, hausdorff=True, percentile=0.9)
                assd_my, hd90_my = surface_distance(vert_my_pred, vert_my_3d, hausdorff=True, percentile=0.9)
                assd_rv, hd90_rv = surface_distance(vert_rv_pred, vert_rv_3d, hausdorff=True, percentile=0.9)
                assd_la, hd90_la = surface_distance(vert_la_pred, vert_la_3d, hausdorff=True, percentile=0.9)
                assd_ra, hd90_ra = surface_distance(vert_ra_pred, vert_ra_3d, hausdorff=True, percentile=0.9)
                assd_all = (assd_lv + assd_my + assd_rv + assd_la + assd_ra) / 5
                hd90_all = (hd90_lv + hd90_my + hd90_rv + hd90_la + hd90_ra) / 5

                valid_assd_all.append(assd_all.item())
                valid_assd_lv.append(assd_lv.item())
                valid_assd_my.append(assd_my.item())
                valid_assd_rv.append(assd_rv.item())
                valid_assd_la.append(assd_la.item())
                valid_assd_ra.append(assd_ra.item())
                
                valid_hd90_all.append(hd90_all.item())
                valid_hd90_lv.append(hd90_lv.item())
                valid_hd90_my.append(hd90_my.item())
                valid_hd90_rv.append(hd90_rv.item())
                valid_hd90_la.append(hd90_la.item())
                valid_hd90_ra.append(hd90_ra.item())


            logging.info("------------------------------")
            logging.info("epoch:{}, valid error:{}".format(epoch, np.mean(valid_loss)))
            logging.info("epoch:{}, valid Dice all:{}".format(epoch, np.mean(valid_dice_all)))
            logging.info("epoch:{}, valid ASSD all:{}".format(epoch, np.mean(valid_assd_all)))
            logging.info("epoch:{}, valid HD90 all:{}".format(epoch, np.mean(valid_hd90_all)))
            logging.info("")
            logging.info("epoch:{}, Dice LV:{}".format(epoch, np.mean(valid_dice_lv)))
            logging.info("epoch:{}, Dice MY:{}".format(epoch, np.mean(valid_dice_my)))
            logging.info("epoch:{}, Dice RV:{}".format(epoch, np.mean(valid_dice_rv)))
            logging.info("epoch:{}, Dice LA:{}".format(epoch, np.mean(valid_dice_la)))
            logging.info("epoch:{}, Dice RA:{}".format(epoch, np.mean(valid_dice_ra)))
            logging.info("")
            logging.info("epoch:{}, ASSD LV:{}".format(epoch, np.mean(valid_assd_lv)))
            logging.info("epoch:{}, ASSD MY:{}".format(epoch, np.mean(valid_assd_my)))
            logging.info("epoch:{}, ASSD RV:{}".format(epoch, np.mean(valid_assd_rv)))
            logging.info("epoch:{}, ASSD LA:{}".format(epoch, np.mean(valid_assd_la)))
            logging.info("epoch:{}, ASSD RA:{}".format(epoch, np.mean(valid_assd_ra)))
            logging.info("")
            logging.info("epoch:{}, HD90 LV:{}".format(epoch, np.mean(valid_hd90_lv)))
            logging.info("epoch:{}, HD90 MY:{}".format(epoch, np.mean(valid_hd90_my)))
            logging.info("epoch:{}, HD90 RV:{}".format(epoch, np.mean(valid_hd90_rv)))
            logging.info("epoch:{}, HD90 LA:{}".format(epoch, np.mean(valid_hd90_la)))
            logging.info("epoch:{}, HD90 RA:{}".format(epoch, np.mean(valid_hd90_ra)))
            logging.info("------------------------------")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Completion")
    parser.add_argument('--data_dir', default='./dataset/', type=str, help="data directory")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--n_epoch', default=200, type=int, help="number of training epochs")
    parser.add_argument('--sigma', default=2.0, type=float, help="standard deviation of random motion")
    parser.add_argument('--scaling', default=0.2, type=float, help="scaling factor")
    parser.add_argument('--drop_rate', default=0.0, type=float, help="dropout rate")

    args = parser.parse_args()
    
    train_loop(args)