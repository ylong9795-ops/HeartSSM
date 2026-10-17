"""Full-cycle native observations using the ORIGINAL Dice/BCE/Boundary terms.

Only the heterogeneous grid adapter and confidence reduction live here. The
mesh-plane kernels and three component definitions are reused without copying.
"""
import numpy as np
import torch

from data.mask_transform import CHAMBERS
from utils.slice_supervision import (MeshSliceSupervisor, batched_plane_segments,
                                     batched_signed_segment_distance)
from .dataset import sdf_mm


def observation_cells(patients, policy, phases=range(24)):
    """Fixed evidence-based membership; predictions/Dice never enter this policy."""
    rows = []
    for owner, patient in enumerate(patients):
        contract = patient.metadata['contract']
        synthetic = contract.get('acquisition_status') == 'synthetic source phase mapping verified'
        timing_factor = 1. if synthetic else policy['unverified_phase_confidence']
        if not synthetic and all('times' in view for view in patient.views):
            start,duration,anchor=(float(contract[key]) for key in ('cycle_start_s','cycle_duration_s','anchor_time_s'))
            for view_index,view in enumerate(patient.views):
                for native_index,time_s in enumerate(view['times']):
                    if not start<=time_s<start+duration:continue
                    position=float(((time_s-anchor)/duration)%1)*24
                    lower=int(np.floor(position))%24;alpha=position-np.floor(position)
                    if alpha<1e-7:alpha=0.
                    if alpha>1-1e-7:lower=(lower+1)%24;alpha=0.
                    if len(phases)==1 and list(phases)[0]==0:
                        if native_index!=patient.mapping[0][view_index]['nearest_index']:continue
                    elif lower not in phases:continue
                    for chamber in range(4):
                        if not view['annotated'][native_index,chamber]:continue
                        source_confidence=float(view['weights'][native_index,chamber])
                        confidence=source_confidence*timing_factor
                        rows.append(dict(owner=owner,phase=lower,mesh_upper=(lower+1)%24,mesh_alpha=float(alpha),
                            view=view_index,chamber=chamber,native_index=native_index,
                            native_frame=view['frame_ids'][native_index],native_time_s=float(time_s),
                            source=view['sources'][native_index][chamber],confidence=confidence,
                            source_confidence=source_confidence,phase_confidence=timing_factor,used=confidence>0,
                            reason='Unique original frame inside selected RR; mesh interpolated at observed time; source phase alignment estimated'))
            continue
        for phase in phases:
            native = patient.native(phase)
            for view in range(3):
                mapping = patient.mapping[phase][view]
                timing_ok = abs(mapping['nearest_time_error_ms']) <= policy['max_nearest_time_error_ms']
                for chamber in range(4):
                    if not native['annotated'][view, chamber]:
                        continue
                    source = native['sources'][view][chamber]
                    source_confidence = float(native['weights'][view, chamber])
                    confidence = source_confidence*timing_factor if timing_ok else 0.
                    rows.append(dict(owner=owner, phase=phase, view=view, chamber=chamber,
                        mesh_upper=(phase+1)%24,mesh_alpha=0.,native_index=mapping['nearest_index'],
                        native_frame=mapping['nearest_native_frame'],native_time_s=mapping.get('target_time_s',phase/24),
                        source=source, confidence=confidence, source_confidence=source_confidence,
                        phase_confidence=timing_factor, used=confidence > 0,
                        reason=('nearest native time outside declared tolerance' if not timing_ok else
                                'synthetic phase verified' if synthetic else
                                'ED anchoring is an estimate; source frame0 equivalence unverified')))
    return rows


def native_cell(patient,row):
    """One original observation, never a resampled binary mask."""
    view=patient.views[row['view']];index=row['native_index'];chamber=row['chamber']
    mask=view['masks'][index,chamber]
    cache_key=('observed_sdf',row['view'],index,chamber)
    if cache_key not in patient._native_cache:
        patient._native_cache[cache_key]=sdf_mm(mask,view['spacing'])
    return mask,patient._native_cache[cache_key],view['spacing']


def interpolate_cell(trajectory,row):
    alpha=row['mesh_alpha']
    return (1-alpha)*trajectory[row['phase']]+alpha*trajectory[row['mesh_upper']]


class RealTTOLoss:
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
                       for key in ('dice', 'bce', 'boundary_mm', 'capture_prior')}
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
                    # Missing foreground has no local section derivative. The
                    # separate capture prior supplies a pose/shape recovery signal.
                    empty_logits = logits*0-30.
                    terms = self.supervisor._terms(empty_logits[None], truth[None], sdf[None])
                    if truth.any():
                        local = (vertices[index]-centers[index]) @ rotations[index]
                        target_center = (group['points'][index, :len(truth)]*truth[:, None]).sum(0)/truth.sum()
                        capture = local[:, 2].abs().amin()+self.config['capture_centroid_weight']*(local[:, :2].mean(0)-target_center).norm()
                    else:
                        terms = {key:value*0 for key, value in terms.items()}
                        capture = vertices[index].sum()*0
                else:
                    terms = self.supervisor._terms(logits[None], truth[None], sdf[None])
                    capture = vertices[index].sum()*0
                terms = {key:value.squeeze(0) for key,value in terms.items()}
                terms['capture_prior'] = capture
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
        return dict(slice=loss, capture_prior=components['capture_prior'], components=components,
                    cell_indices=indices, cell_terms=cell_terms, missing_sections=missing,
                    cell_observation_ids=observation_ids,
                    supervision_manifest=self.rows)
