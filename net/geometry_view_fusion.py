"""Masked, interpretable view gates with retained A2C/A3C/A4C provenance.

No averaging is the sole evidence path: weighted per-view features and explicit
cross-view conflict variance are retained alongside the old context features.
"""
import torch
from torch import nn


class SupportAwareViewFusion(nn.Module):
    def __init__(self, view_width):
        super().__init__()
        self.view_width = view_width
        self.gate = nn.Conv3d(view_width+7,1,1)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        self.output_width = 4*view_width+8

    def forward(self, lifted, *, learned=True):
        h = lifted['view_features']
        b,t,v,c,gx,gy,gz = h.shape
        if v != 3 or c != self.view_width:
            raise ValueError('Expected [B,T,V=3,C,Gx,Gy,Gz] view features')
        feature_valid=lifted['feature_validity']
        h=torch.where(feature_valid,h,0.)
        support = lifted['view_support']
        chamber = lifted['chamber_support'].to(h)
        metadata = torch.cat((lifted['distance_mm'][:,:,:,None]/24.,
            lifted['direct_support'][:,:,:,None].to(h),
            lifted['propagated_support'][:,:,:,None].to(h),chamber),3)
        logits = self.gate(torch.cat((h,metadata),3).flatten(0,2)).reshape(b,t,v,gx,gy,gz) if learned else torch.zeros_like(lifted['confidence'])
        # Softmax with an explicit all-unknown case, no invalid-view bias leakage.
        logits = logits.float()+lifted['confidence'].float().clamp_min(1e-12).log()
        logits = torch.where(support,logits,torch.full_like(logits,-1e4))
        weights = torch.softmax(logits,dim=2)*support
        weights = weights/weights.sum(2,keepdim=True).clamp_min(1e-8)
        weighted = h*weights[:,:,:,None].to(h)
        # Conflict compares only actually annotated sources for each channel.
        # Missing RV/RA labels must not masquerade as zero/background evidence.
        channel_weights=weights[:,:,:,None].to(h)*feature_valid
        channel_weights=channel_weights/channel_weights.sum(2,keepdim=True).clamp_min(1e-8)
        mean=(h*channel_weights).sum(2)
        variance=((h-mean[:,:,None]).square()*channel_weights).sum(2)
        variance=torch.where(feature_valid.sum(2)>=2,variance,0.)
        direct = lifted['direct_support'].any(2)
        propagated = ~direct & lifted['propagated_support'].any(2)
        unknown = ~support.any(2)
        chamber_union = lifted['chamber_support'].any(2)
        nearest = lifted['distance_mm'].abs().masked_fill(~support,24.).amin(2).clamp_max(24.)/24.
        flags = torch.cat((direct[:,:,None],propagated[:,:,None],unknown[:,:,None],
                           chamber_union),2).to(h)
        # output_width = 3*C provenance + C conflict + 7 flags + distance.
        features = torch.cat((weighted.flatten(2,3),variance,flags,nearest[:,:,None]),2)
        validity = torch.cat((lifted['feature_validity'].flatten(2,3),
            lifted['feature_validity'].any(2),torch.ones_like(flags,dtype=torch.bool),
            torch.ones_like(nearest[:,:,None],dtype=torch.bool)),2)
        return dict(features=features,feature_validity=validity,weights=weights,
                    direct=direct,propagated=propagated,unknown=unknown,conflict=variance)
