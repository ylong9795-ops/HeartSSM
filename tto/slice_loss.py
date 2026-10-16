"""Heterogeneous native-grid supervision via existing differentiable sections."""
import numpy as np
import torch

from data.mask_transform import CHAMBERS
from utils.slice_supervision import batched_plane_segments, batched_signed_segment_distance

DEFAULT_LOSS = dict(sigma_mm=1., boundary_band_mm=1.5, boundary_weight=.25,
    dice_smooth=1., gce_q=.7, gce_weight=.1, sdf_scale_mm=3., sdf_band_mm=12.,
    sdf_weight=.08, tail_cells=2, tail_weight=.25, capture_distance_weight=.02,
    capture_centroid_weight=.1)


class NativeSliceLoss:
    def __init__(self, model, natives, stride=4, config=None):
        self.model, self.natives, self.groups = model, natives, []
        self.config = dict(DEFAULT_LOSS if config is None else config)
        if set(self.config) != set(DEFAULT_LOSS):
            raise ValueError('Complete versioned Native slice loss configuration required')
        if any(not np.isfinite(v) or v <= 0 for v in self.config.values()):
            raise ValueError('Slice loss settings must be positive finite numbers')
        if not isinstance(self.config['tail_cells'], int) or not 1 <= self.config['tail_cells'] <= 12:
            raise ValueError('tail_cells must be an integer from1 to12')
        self.count = len(natives)
        device = model.template_vertices_mm.device
        self.weights = torch.tensor(np.stack([n['weights']*n['annotated'] for n in natives]), device=device)
        if (self.weights.sum((1, 2)) <= 0).any():
            raise ValueError('Each patient needs trusted available observation cells')
        for ci, chamber in enumerate(CHAMBERS):
            entries = [(bi, vi) for bi, native in enumerate(natives) for vi in range(3)
                       if native['annotated'][vi, ci] and native['weights'][vi, ci] > 0]
            if not entries:
                continue
            pixels, targets, fields = [], [], []
            for bi, vi in entries:
                native = natives[bi];mask = native['masks'][vi][ci];h, w = mask.shape
                dy, dx = native['spacing'][vi]
                yy, xx = torch.meshgrid((torch.arange(0, h, stride, device=device)-h/2)*dy,
                    (torch.arange(0, w, stride, device=device)-w/2)*dx, indexing='ij')
                pixels.append(torch.stack([xx, yy], -1).reshape(-1, 2))
                targets.append(torch.tensor(mask[::stride, ::stride].flatten(), device=device, dtype=torch.float32))
                fields.append(torch.tensor(native['sdf'][vi][ci][::stride, ::stride].flatten(), device=device))
            size = max(len(p) for p in pixels)
            points = torch.zeros(len(entries), size, 2, device=device)
            truth = torch.zeros(len(entries), size, device=device)
            sdf = torch.zeros_like(truth);valid = torch.zeros_like(truth)
            for i, (p, target, field) in enumerate(zip(pixels, targets, fields)):
                points[i, :len(p)] = p;truth[i, :len(p)] = target;sdf[i, :len(p)] = field;valid[i, :len(p)] = 1
            self.groups.append(dict(chamber=chamber, ci=ci, owners=torch.tensor([b for b, _ in entries], device=device),
                views=torch.tensor([v for _, v in entries], device=device), points=points, truth=truth, sdf=sdf, valid=valid))

    def __call__(self, vertices, pose):
        cfg = self.config
        values = vertices.new_zeros(self.count, 3, 4)
        missing_cells = []
        for group in self.groups:
            owners, views, ci = group['owners'], group['views'], group['ci']
            a, b = self.model.chamber_ranges[ci].tolist()
            v, r, center = vertices[owners, a:b], pose['R'][owners, views], pose['center_mm'][owners, views]
            segments, present = batched_plane_segments(v, self.model.faces[group['chamber']], r, center)
            distance = batched_signed_segment_distance(group['points'], segments, present, chunk_size=128)
            probability = torch.sigmoid(-distance/cfg['sigma_mm'])
            target, field = group['truth'], group['sdf']
            weight = torch.where(field.abs() <= cfg['boundary_band_mm'], cfg['boundary_weight'], 1.)*group['valid']
            smooth = cfg['dice_smooth']
            dice = 1-(2*(weight*probability*target).sum(-1)+smooth)/((weight*probability).sum(-1)+(weight*target).sum(-1)+smooth)
            correct = probability*target+(1-probability)*(1-target)
            gce = (weight*(1-correct.clamp_min(1e-6).pow(cfg['gce_q']))/cfg['gce_q']).sum(-1)/weight.sum(-1).clamp_min(1)
            band = (field.abs() <= cfg['sdf_band_mm']).to(weight)*weight
            robust = (band*torch.log1p(((distance-field)/cfg['sdf_scale_mm']).square())).sum(-1)/band.sum(-1).clamp_min(1)
            loss = dice+cfg['gce_weight']*gce+cfg['sdf_weight']*robust
            missing = ~present.any(-1)
            empty_observation = target.sum(-1) == 0
            # A valid empty label and absent section agree. Capture applies only
            # when an observed foreground is missed by the current plane.
            loss = torch.where(missing & empty_observation, loss*0., loss)
            missing = missing & ~empty_observation
            if missing.any():
                local = torch.bmm(v-center[:, None], r)
                observed = (group['points']*target[..., None]).sum(1)/target.sum(-1).clamp_min(1)[:, None]
                capture = local[..., 2].abs().amin(-1)+cfg['capture_centroid_weight']*(local[..., :2].mean(1)-observed).norm(dim=-1)
                loss = torch.where(missing, 1+cfg['capture_distance_weight']*capture, loss)
                missing_cells.extend((int(owners[k]), int(views[k]), ci) for k in missing.nonzero().flatten())
            values[owners, views, ci] = loss
        weighted = values*self.weights
        mean = weighted.sum((1, 2))/self.weights.sum((1, 2))
        # Tail term respects trust: no pseudo-label is silently promoted to manual.
        worst = weighted.flatten(1).topk(cfg['tail_cells'], dim=1).values.mean(1)
        return mean+cfg['tail_weight']*worst, values, missing_cells
