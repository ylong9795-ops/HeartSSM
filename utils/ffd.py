import numpy as np
import torch
import torch.nn.functional as F



def bspline_basis(u):
    '''
    Cubic B-spline basis function
    '''
    B0 = ((1 - u) ** 3) / 6
    B1 = (3 * u ** 3 - 6 * u ** 2 + 4) / 6
    B2 = (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6
    B3 = (u ** 3) / 6
    basis = torch.stack([B0, B1, B2, B3], dim=-1)
    return basis  # (B,N,4)


def bspline_ffd(vert, ctrl_grid):
    '''
    B-spline Free-Form Deformation (FFD)
    '''
    L, W, H = ctrl_grid.shape[:3]
    shift = torch.arange(4, device=vert.device)[None,None,:]

    x = vert[...,0]  # (B,N)
    y = vert[...,1]  # (B,N)
    z = vert[...,2]  # (B,N)
    i = torch.floor(x).long().clamp(0, L - 4)
    j = torch.floor(y).long().clamp(0, W - 4)
    k = torch.floor(z).long().clamp(0, H - 4)

    u = x - i.float()
    v = y - j.float()
    w = z - k.float()

    Bu = bspline_basis(u)[...,:,None,None,None]  # (B,N,4,1,1,1)
    Bv = bspline_basis(v)[...,None,:,None,None]  # (B,N,1,4,1,1)
    Bw = bspline_basis(w)[...,None,None,:,None]  # (B,N,1,1,4,1)

    grid_i = i[...,None] + shift  # (B,N,4)
    grid_j = j[...,None] + shift  # (B,N,4)
    grid_k = k[...,None] + shift  # (B,N,4)

    grid_i = grid_i[...,:,None,None].repeat(1,1,1,4,4)  # (B,N,4,4,4)
    grid_j = grid_j[...,None,:,None].repeat(1,1,4,1,4)  # (B,N,4,4,4)
    grid_k = grid_k[...,None,None,:].repeat(1,1,4,4,1)  # (B,N,4,4,4)

    # control points (L,W,H,3) --> (B,N,4,4,4,3)
    ctrl_pts = ctrl_grid[grid_i, grid_j, grid_k]  # (B,N,4,4,3)

    # deformation
    ffd = (Bu * Bv * Bw * ctrl_pts).sum(dim=[2,3,4])  # (B, N, 3)
    return ffd

