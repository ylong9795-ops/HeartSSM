import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points


def uni_surface_distance(x, y, x_lengths=None, y_lengths=None, hausdorff=False, percentile=0.9):
    '''
    Uni-directional surface distances adapted from PyTorch3D
    https://pytorch3d.readthedocs.io/en/latest/modules/loss.html
    '''
    # nearest neighbors of each ground truth point
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)
    
    # non-squared distances
    cham_y = y_nn.dists[..., 0].sqrt()  # (B,N)
    
    if y_lengths is not None:
        dist_y = cham_y.sum(-1) / y_lengths  # (B)
    else:
        dist_y = cham_y.mean(-1)  # (B)
    assd = dist_y.mean()
    
    if hausdorff:
        if y_lengths is not None:
            hd = []
            for i in range(cham_y.shape[0]):
                hd.append(torch.quantile(cham_y[i,:y_lengths[i]], percentile, dim=-1))  # (B)
            hd = torch.stack(hd).mean()
        else:
            hd = torch.quantile(cham_y, percentile, dim=-1).mean()
        return assd, hd
    else:
        return assd
    

# def uni_surface_distance(x, y, x_lengths=None, y_lengths=None):
#     # nearest neighbors of each ground truth point
#     y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)
    
#     # non-squared distances
#     cham_y = y_nn.dists[..., 0].sqrt()
    
#     if y_lengths is not None:
#         dist_y = cham_y.sum(-1) / y_lengths  # (B)
#     else:
#         dist_y = cham_y.mean(-1)  # (B)
        
#     return dist_y.mean()


def surface_distance(x, y, x_lengths=None, y_lengths=None, hausdorff=False, percentile=0.9):
    '''
    Bi-directional surface distances adapted from PyTorch3D
    https://pytorch3d.readthedocs.io/en/latest/modules/loss.html
    '''
    # find nearest neighbor points
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    # non-squared distances
    cham_x = x_nn.dists[..., 0].sqrt()  # (B, N1)
    cham_y = y_nn.dists[..., 0].sqrt()  # (B, N2)

    if x_lengths is not None:
        dist_x = cham_x.sum(-1) / x_lengths  # (B)
    else:
        dist_x = cham_x.mean(-1)  # (B)
        
    if y_lengths is not None:
        dist_y = cham_y.sum(-1) / y_lengths  # (B)
    else:
        dist_y = cham_y.mean(-1)  # (B)

    # average symmetric surface distance
    assd = (dist_x + dist_y).mean() / 2

    if hausdorff:
        # compute Hausdorff distance
        x_quantile = torch.quantile(cham_x, percentile, dim=-1)  # (B)
        y_quantile = torch.quantile(cham_y, percentile, dim=-1)  # (B)
        hd = torch.maximum(x_quantile, y_quantile).mean()
        return assd, hd
    else:
        return assd


def correlation(x,y):
    '''Pearson Correlation Coefficient'''
    mean_xy = ((x - x.mean(dim=-1, keepdim=True)) * (y - y.mean(dim=-1, keepdim=True))).mean(dim=-1)
    std_xy = x.std(dim=-1) * y.std(dim=-1)
    pcc = mean_xy / (std_xy + 1e-12)
    return pcc.mean()


def dice_score(x, y):
    '''
    Dice score between two binary segmentation maps
    '''
    return (2*(x*y).sum()/(x.sum()+y.sum()+1e-12)).mean()
