"""Explicit masked Eulerian local temporal conditions on physical 3D grids.

No difference is called tissue velocity. Reference/previous features, overlap
masks and phase identities remain separate inputs. Circular Conv1d mixes time
at each physical node, primarily at medium resolution.
"""
import torch
from torch import nn
from torch.nn import functional as F


def temporal_conditions(features, valid, phases, difference_masks=None):
    if features.ndim != 6 or features.shape != valid.shape or features.shape[1] != 24:
        raise ValueError('Local features/validity must be [B,24,C,Gx,Gy,Gz]')
    valid = valid.bool()
    current = torch.where(valid,features,0.)
    reference,previous = current[:,:1].expand_as(current),current.roll(1,1)
    ref_valid,prev_valid = valid[:,:1].expand_as(valid),valid.roll(1,1)
    ref_overlap,prev_overlap = valid & ref_valid,valid & prev_valid
    if difference_masks is not None:
        ref_overlap = ref_overlap & difference_masks['reference']
        prev_overlap = prev_overlap & difference_masks['previous']
    reference_delta = torch.where(ref_overlap,current-reference,0.)
    previous_delta = torch.where(prev_overlap,current-previous,0.)
    time = torch.stack((torch.sin(2*torch.pi*phases),torch.cos(2*torch.pi*phases)),-1)
    time = time.to(features)[...,None,None,None].expand(-1,-1,-1,*features.shape[-3:])
    flags = torch.cat((ref_overlap,prev_overlap,~valid),2).to(features)
    conditions = torch.cat((current,reference,previous,reference_delta,previous_delta,flags,time),2)
    return dict(conditions=conditions,current=current,reference_delta=reference_delta,
                previous_delta=previous_delta,reference_overlap=ref_overlap,previous_overlap=prev_overlap)


class LocalTemporalMotionConditioner(nn.Module):
    def __init__(self, input_width, width=8):
        super().__init__()
        if min(input_width,width) < 1:
            raise ValueError('Positive temporal feature widths required')
        self.input_width,self.width = input_width,width
        self.project = nn.Conv3d(8*input_width+2,width,1)
        self.temporal = nn.Conv1d(width,width,3,groups=width)
        self.spatial = nn.Conv3d(width,width,3,padding=1)

    def forward(self, features, validity, phases, difference_masks=None):
        pieces = temporal_conditions(features,validity,phases,difference_masks)
        b,t,_,gx,gy,gz = features.shape
        h = F.silu(self.project(pieces['conditions'].flatten(0,1))).reshape(b,t,self.width,gx,gy,gz)
        series = h.permute(0,3,4,5,2,1).reshape(-1,self.width,t)
        series = self.temporal(F.pad(series,(1,1),mode='circular'))
        series = series.reshape(b,gx,gy,gz,self.width,t).permute(0,5,4,1,2,3)
        h = h+series
        h = h+F.silu(self.spatial(h.flatten(0,1))).reshape_as(h)
        return dict(features=h,reference_overlap=pieces['reference_overlap'],previous_overlap=pieces['previous_overlap'])


class FineLocalMotionSkip(nn.Module):
    def __init__(self, input_width, width=8):
        super().__init__()
        self.width = width
        self.project = nn.Conv3d(8*input_width+2,width,1)

    def forward(self, features, valid, phases, difference_masks=None):
        pieces = temporal_conditions(features,valid,phases,difference_masks)
        b,t = features.shape[:2]
        return F.silu(self.project(pieces['conditions'].flatten(0,1))).reshape(b,t,self.width,*features.shape[-3:])


def resample_physical_features(features, target_points_mm, origin_mm, spacing_mm):
    """[B,T,C,Gx,Gy,Gz] -> target physical lattice, XYZ->grid_sample ZYX."""
    b,t,c = features.shape[:3]
    shape = features.shape[-3:]
    point = target_points_mm.to(device=features.device,dtype=torch.float32)
    origin = origin_mm.to(device=features.device,dtype=torch.float32)
    spacing = spacing_mm.to(device=features.device,dtype=torch.float32)
    index = (point-origin)/spacing
    side = torch.tensor(shape,device=features.device,dtype=torch.float32)
    grid = (2*index/(side-1)-1)[...,[2,1,0]]
    return F.grid_sample(features.flatten(0,1),grid[None].expand(b*t,*grid.shape),
                        mode='bilinear',padding_mode='border',align_corners=True).reshape(b,t,c,*target_points_mm.shape[:-1])
