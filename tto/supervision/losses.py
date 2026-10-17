"""No-GT native Dice/BCE/boundary supervision, without solver-specific priors."""
import numpy as np
import torch
from data.mask_transform import CHAMBERS
from utils.slice_supervision import MeshSliceSupervisor
from .slicing import batched_plane_segments, batched_signed_segment_distance
from .observations import observation_cells, native_cell


class NativeSupervision:
    def __init__(self, model, patients, config, policy, *, phases=range(24), stride=None):
        self.model, self.patients, self.config = model, patients, dict(config)
        self.rows = observation_cells(patients, policy, phases)
        self.active = [row for row in self.rows if row['used']]
        self.batch_size = len(patients)
        self.stride = int(stride or config['pixel_stride'])
        if self.stride < 1:
            raise ValueError('Positive native pixel stride required')
        if config['target_mode'] != 'binary_native_v1':
            raise ValueError('Real slice targets cannot depend on GT mesh')
        self.supervisor = MeshSliceSupervisor(model.chamber_ranges, model.faces, model.norm_to_mm,
            sigma_mm=config['sigma_mm'], pixel_stride=self.stride, chunk_size=config['chunk_size'],
            dice_weight=config['dice_weight'], bce_weight=config['bce_weight'],
            boundary_weight=config['boundary_weight'], target_mode='binary_native_v1')
        self.groups = []
        device = model.template_vertices_mm.device
        for chamber in range(4):
            entries = [r for r in self.active if r['chamber'] == chamber]
            for start in range(0, len(entries), config['cell_chunk_size']):
                rows = entries[start:start+config['cell_chunk_size']]
                points, truth, sdf = [], [], []
                for row in rows:
                    mask,field,spacing = native_cell(patients[row['owner']],row)
                    h, w = mask.shape
                    dy, dx = spacing
                    yy, xx = torch.meshgrid((torch.arange(0, h, self.stride, device=device)-h/2)*dy,
                        (torch.arange(0, w, self.stride, device=device)-w/2)*dx, indexing='ij')
                    sampled = mask[::self.stride, ::self.stride]
                    if mask.any() and not sampled.any():
                        raise ValueError('Native stride misses annotated foreground; explicitly reduce stride')
                    points.append(torch.stack((xx, yy), -1).reshape(-1, 2))
                    truth.append(torch.as_tensor(sampled.reshape(-1), device=device, dtype=torch.float32))
                    sdf.append(torch.as_tensor(field[::self.stride, ::self.stride].reshape(-1), device=device))
                size = max(len(p) for p in points)
                padded = torch.zeros(len(rows), size, 2, device=device)
                for index, p in enumerate(points):padded[index, :len(p)] = p
                self.groups.append(dict(rows=rows, points=padded, truth=truth, sdf=sdf,
                    owner=torch.tensor([r['owner'] for r in rows], device=device),
                    phase=torch.tensor([r['phase'] for r in rows], device=device),
                    upper=torch.tensor([r['mesh_upper'] for r in rows],device=device),
                    alpha=torch.tensor([r['mesh_alpha'] for r in rows],device=device)[:,None,None],
                    view=torch.tensor([r['view'] for r in rows], device=device), chamber=chamber))
        weights = np.zeros(self.batch_size, np.float32)
        for row in self.active:weights[row['owner']] += row['confidence']
        if (weights <= 0).any():
            raise ValueError('Each patient requires at least one usable phase-mapped observation')
        self.denominator = torch.as_tensor(weights, device=device)

    def __call__(self, trajectory, pose):
        if trajectory.ndim != 4 or trajectory.shape[1] != 24:
            raise ValueError('Full [B,24,N,3] differentiable trajectory required')
        accumulated = {key: trajectory.new_zeros(self.batch_size)
                       for key in ('dice', 'bce', 'boundary_mm')}
        cell_terms, indices, missing,observation_ids = [], [], [],[]
        for group in self.groups:
            owners, phases, views = group['owner'], group['phase'], group['view']
            chamber = group['chamber'];a, b = self.model.chamber_ranges[chamber].tolist()
            vertices = ((1-group['alpha'])*trajectory[owners,phases,a:b]+
                        group['alpha']*trajectory[owners,group['upper'],a:b])
            rotations, centers = pose['R'][owners, views], pose['center_mm'][owners, views]
            segments, present = batched_plane_segments(vertices, self.model.faces[CHAMBERS[chamber]], rotations, centers)
            distances = batched_signed_segment_distance(group['points'], segments, present,
                                                        chunk_size=self.config['chunk_size'])
            for index, row in enumerate(group['rows']):
                truth, sdf = group['truth'][index], group['sdf'][index]
                logits = -distances[index, :len(truth)]/self.config['sigma_mm']
                absent = not bool(present[index].any())
                if absent:
                    # The exact empty section has no local geometric derivative.
                    # Report it for global alignment; invent no capture force.
                    zero = (vertices[index].sum()+centers[index].sum()+rotations[index].sum())*0
                    empty_logits = torch.full_like(logits, -30.)+zero
                    terms = self.supervisor._terms(empty_logits[None], truth[None], sdf[None])
                    if not truth.any():
                        terms = {key: value*0 for key, value in terms.items()}
                else:
                    terms = self.supervisor._terms(logits[None], truth[None], sdf[None])
                terms = {key: value.squeeze(0) for key, value in terms.items()}
                owner = row['owner'];confidence = row['confidence']
                for key,value in terms.items():
                    # index_add preserves independent patient objectives/gradients.
                    accumulated[key] = accumulated[key].index_add(0, owners[index:index+1], (value*confidence).reshape(1))
                cell_terms.append(terms);indices.append((owner,row['phase'],row['view'],chamber));missing.append(absent)
                observation_ids.append((owner,row['native_frame'],row['view'],chamber))
        components = {key:value/self.denominator for key,value in accumulated.items()}
        loss = (self.config['dice_weight']*components['dice']+
                self.config['bce_weight']*components['bce']+
                self.config['boundary_weight']*components['boundary_mm'])
        return dict(slice=loss, components=components,
                    cell_indices=indices, cell_terms=cell_terms, missing_sections=missing,
                    cell_observation_ids=observation_ids,
                    supervision_manifest=self.rows)
