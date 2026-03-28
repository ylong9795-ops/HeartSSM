import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """
    3D convolutional block

    Args:
    - dim_in: dim of input channels
    - dim_out: dim of output channels
    - K: kernel size

    Input:
    - x: input features
    - t_emb: time embedding
    """
    def __init__(self, dim_in, dim_out, K=3, stride=1, drop_rate=0.0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=dim_in, out_channels=dim_out,
            kernel_size=K, stride=stride, padding=K//2)
        self.norm = nn.InstanceNorm3d(dim_in, affine=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        # norm -> activation -> dropout -> conv -> norm
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x
