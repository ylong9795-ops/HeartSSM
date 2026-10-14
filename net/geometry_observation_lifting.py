"""V2 per-view physical sampling; coordinate implementation remains frozen v1.

view_features: [B,T,V,2*P+4,Gx,Gy,Gz]. Last four channels are sampled
2D slice SDF conditions, never a recovered 3D SDF. All source identities and
physical query positions remain explicit until view fusion.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from net.geometry_lifting import project_physical_points, feature_sampling_grid


class GeometryObservationLifting(nn.Module):
    def __init__(self, sigma_mm=8., cutoff_sigma=3., direct_half_thickness_mm=.75,
                 sdf_scale_mm=15.):
        super().__init__()
        if (not all(math.isfinite(x) and x > 0 for x in
                    (sigma_mm, cutoff_sigma, direct_half_thickness_mm, sdf_scale_mm))
                or direct_half_thickness_mm >= sigma_mm*cutoff_sigma):
            raise ValueError('Finite positive distance/SDF scales and a thin direct slab required')
        self.sigma_mm, self.cutoff_sigma = sigma_mm, cutoff_sigma
        self.direct_half_thickness_mm, self.sdf_scale_mm = direct_half_thickness_mm, sdf_scale_mm

    def forward(self, spatial, observations, queries_mm, norm_to_mm, projections):
        shape = queries_mm.shape[:-1]
        projection = project_physical_points(queries_mm.reshape(-1,3), observations,
            norm_to_mm, observations['masks'].shape[-2:])
        valid = spatial['validity']
        distance = projection['distance_mm']
        available = (valid.any(-1)[...,None] & projection['inside']
                     & (distance.abs() <= self.sigma_mm*self.cutoff_sigma))
        direct = available & (distance.abs() <= self.direct_half_thickness_mm)
        propagated = available & ~direct
        b,t,v,n = available.shape
        sampled = []
        for key,project,stride in zip(('scale14','scale7'),projections,(8,16)):
            value = project(spatial[key].flatten(0,2))
            grid = feature_sampling_grid(projection['model_pixels'],
                observations['masks'].shape[-2:],value.shape[-2:],stride)
            sample = F.grid_sample(value,grid.reshape(b*t*v,n,1,2),
                align_corners=True,mode='bilinear',padding_mode='border').reshape(b,t,v,-1,n)
            sampled.append(torch.where(available[...,None,:],sample,0.))
        cnn = torch.cat(sampled,3)
        sdf = observations['sdf'].to(queries_mm)
        active = valid[...,None,None]
        sdf = torch.where(active,sdf,0.)
        sampled_sdf = F.grid_sample(sdf.flatten(0,2),projection['grid'].reshape(b*t*v,n,1,2),
            align_corners=False,mode='bilinear',padding_mode='zeros').reshape(b,t,v,4,n)
        chamber_support = available[...,None,:] & valid[...,None]
        sampled_sdf = torch.where(chamber_support,sampled_sdf,0.)
        feature = torch.cat((cnn,sampled_sdf/self.sdf_scale_mm),3)
        feature_validity = torch.cat((available[...,None,:].expand_as(cnn),chamber_support),3)
        confidence = torch.where(available,torch.exp(-.5*(distance/self.sigma_mm).square()),0.)
        def volume(value):return value.reshape(*value.shape[:-1],*shape)
        return dict(view_features=volume(feature),feature_validity=volume(feature_validity),
            sampled_slice_sdf_mm=volume(sampled_sdf),view_support=volume(available),
            direct_support=volume(direct),propagated_support=volume(propagated),
            chamber_support=volume(chamber_support),distance_mm=volume(distance),
            confidence=volume(confidence),view_present=valid.any(-1),channel_annotated=valid,
            query_points_mm=queries_mm)
