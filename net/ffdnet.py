import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net.module import ConvBlock


class FFDNet(nn.Module):
    """
    HeartFFDNet learning free-form deformations (FFD)
    for 3D+t cardiac mesh reconstruction

    Args:
    - dim_in: dim of input channels
    - dim_hid: dim of hidden channels
    - dim_out: dim of output channels
    - K: kernel size
    - drop_rate: dropout rate

    Inputs:
    - x: input volume (B,C_in,L,W,H)

    Returns:
    - x: output volume (B,C_out,L,W,H)
    """
    
    def __init__(
        self,
        n_frame=50,
        dim_hid=256,
        K=3,
        drop_rate=0.0
    ):
        super(FFDNet, self).__init__()

        self.conv0 = nn.Conv3d(in_channels=n_frame+1, out_channels=dim_hid//4, kernel_size=K, stride=1, padding=K//2)
        self.conv1 = ConvBlock(dim_in=dim_hid//4, dim_out=dim_hid//2, K=K, stride=2, drop_rate=drop_rate)
        self.conv2 = ConvBlock(dim_in=dim_hid//2, dim_out=dim_hid, K=K, stride=2, drop_rate=drop_rate)
        self.conv3 = ConvBlock(dim_in=dim_hid, dim_out=dim_hid, K=K, stride=2, drop_rate=drop_rate)
        self.conv4 = ConvBlock(dim_in=dim_hid, dim_out=dim_hid, K=K, stride=2, drop_rate=drop_rate)
        self.conv5 = ConvBlock(dim_in=dim_hid, dim_out=dim_hid, K=K, stride=1, drop_rate=drop_rate)

        self.deconv5 = ConvBlock(dim_in=dim_hid, dim_out=dim_hid, K=K, stride=1, drop_rate=drop_rate)
        self.deconv4 = ConvBlock(dim_in=dim_hid*2, dim_out=dim_hid, K=K, stride=1, drop_rate=drop_rate)
        self.deconv3 = ConvBlock(dim_in=dim_hid*2, dim_out=dim_hid, K=K, stride=1, drop_rate=drop_rate)
        self.deconv2 = ConvBlock(dim_in=dim_hid*2, dim_out=dim_hid, K=K, stride=1, drop_rate=drop_rate)

        self.deform1 = ConvBlock(dim_in=dim_hid, dim_out=3, K=K, stride=1, drop_rate=drop_rate)
        self.deform2 = ConvBlock(dim_in=dim_hid, dim_out=3, K=K, stride=1, drop_rate=drop_rate)
        self.deform3 = ConvBlock(dim_in=dim_hid, dim_out=3*n_frame, K=K, stride=1, drop_rate=drop_rate)

        nn.init.normal_(self.deform1.conv.weight, 0, 1e-5)
        nn.init.constant_(self.deform1.conv.bias, 0.0)
        nn.init.normal_(self.deform2.conv.weight, 0, 1e-5)
        nn.init.constant_(self.deform2.conv.bias, 0.0)
        nn.init.normal_(self.deform3.conv.weight, 0, 1e-5)
        nn.init.constant_(self.deform3.conv.bias, 0.0)
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        ffd_list = []  # all free-form deformations
        
        # encode
        x = self.conv0(x)    # (96,96,128)
        x = self.conv1(x)    # (48,48,64)
        x2 = self.conv2(x)   # (24,24,32)
        x3 = self.conv3(x2)  # (12,12,16)
        x4 = self.conv4(x3)  # (6,6,8)
        x  = self.conv5(x4)  # (6,6,8)

        # decode
        x = self.deconv5(x)  # (6,6,8)
        x = torch.cat([x, x4], dim=1)
        x = self.deconv4(x)  # (6,6,8)
        ffd1 = F.pad(self.deform1(x), pad=[1]*6)
        ffd1 = ffd1[0].permute(1,2,3,0)
        ffd_list.append(ffd1)
        
        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = self.deconv3(x)   # (12,12,16)
        ffd2 = F.pad(self.deform2(x), pad=[1]*6)
        ffd2 = ffd2[0].permute(1,2,3,0)
        ffd_list.append(ffd2)
        
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = self.deconv2(x)   # (24,24,32)
        ffd3 = F.pad(self.deform3(x), pad=[1]*6)
        ffd3 = ffd3.reshape(-1,3,*ffd3.shape[2:])
        ffd3 = ffd3.permute(0,2,3,4,1)
        ffd_list = ffd_list + list(ffd3.unbind(dim=0))
        
        return ffd_list
