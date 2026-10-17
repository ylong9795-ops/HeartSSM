"""Patient-isolated native observations and explicit 24-phase physical mappings.

Supports the repository's exported real_data layout and a portable tto_input.json
contract. Native masks never resize. Only encoder inputs use MaskTransform.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from data.mask_transform import CHAMBERS, VIEWS, MaskTransform
from .io import read_json, sha256, digest_json

PLANE_IDS = (2, 3, 1)
FOLDERS = ('A2C_两腔心', 'A3C_三腔心', 'A4C_四腔心')
DEFAULT_VALID = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], bool)


def sdf_mm(mask, spacing):
    diagonal = float(np.linalg.norm(np.array(mask.shape) * spacing))
    if not mask.any():
        return np.full(mask.shape, diagonal, np.float32)
    if mask.all():
        return np.full(mask.shape, -diagonal, np.float32)
    return (distance_transform_edt(~mask.astype(bool), sampling=spacing) -
            distance_transform_edt(mask.astype(bool), sampling=spacing)).astype(np.float32)


@dataclass
class PatientObservations:
    patient_id: str
    views: list
    mapping: list
    encoder: dict
    metadata: dict
    images: list = field(default_factory=list)
    _native_cache: dict = field(default_factory=dict, repr=False)

    def native(self, phase=0):
        """Original nearest binary labels, per-cell sources/weights, native SDF."""
        if phase not in self._native_cache:
            masks, fields, sources, weights, annotated = [], [], [], [], []
            for vi, view in enumerate(self.views):
                index = self.mapping[phase][vi]['nearest_index']
                mask = view['masks'][index]
                masks.append(mask)
                fields.append(np.stack([sdf_mm(x, view['spacing']) for x in mask]))
                sources.append(view['sources'][index])
                weights.append(view['weights'][index])
                annotated.append(view['annotated'][index])
            self._native_cache[phase] = dict(masks=masks, sdf=fields, sources=sources,
                spacing=[v['spacing'] for v in self.views], weights=np.array(weights, np.float32),
                annotated=np.array(annotated, bool))
        return self._native_cache[phase]

    @property
    def identity(self):
        return digest_json(self.metadata)


class RealTTODataset(Dataset):
    def __init__(self, root, patients=None, *, config):
        self.root = Path(root)
        self.config = config
        self.patients = list(patients) if patients else sorted(
            p.name for p in self.root.iterdir() if p.is_dir())
        if len(set(self.patients)) != len(self.patients):
            raise ValueError('Duplicate patient IDs would alias output states')
        for patient in self.patients:
            if Path(patient).name != patient or patient in ('.', '..'):
                raise ValueError('Patient ID must be a single directory name')
            if not (self.root / patient).is_dir():
                raise FileNotFoundError(self.root / patient)

    def __len__(self):
        return len(self.patients)

    def _repository_contract(self, folder):
        import pydicom
        labels = folder / '患者标签（每一帧）'
        calibration = read_json(labels / '原始切面与像素间距.json')
        manual = read_json(labels / '手工ED_ES源标注.json')
        acquisition = calibration['primary_acquisition_id']
        paths = sorted((folder / '患者原始dicom').glob(acquisition + '_*.dcm'))
        if len(paths) != 1:
            raise ValueError('Need exactly one primary DICOM or an explicit tto_input.json')
        dicom = pydicom.dcmread(paths[0], stop_before_pixels=True)
        dt = float(dicom.FrameTime) / 1000
        delay = float(getattr(dicom, 'FrameDelay', 0)) / 1000
        ed, es = (int(manual['phases'][key]['frame']) for key in ('ED', 'ES'))
        anchor = ed * dt + delay
        waves = np.array(dicom.RWaveTimeVector, float) / 1000
        cycles = [(a, b) for a, b in zip(waves[:-1], waves[1:])
                  if a <= anchor < b and a >= delay and b <= (int(dicom.NumberOfFrames)-1)*dt+delay]
        if len(cycles) != 1:
            raise ValueError('Manual ED must belong to one fully covered reported RR')
        start, end = cycles[0]
        es_time = es*dt+delay
        regions = {r['plane']: r for r in calibration['regions']}
        views = []
        for vi, (name, plane) in enumerate(zip(FOLDERS, PLANE_IDS)):
            index = read_json(labels / name / '帧索引.json')
            region = regions[plane]
            matching = [r for r in dicom.SequenceOfUltrasoundRegions
                        if int(r.RegionLocationMinX0) == region['x0'] and int(r.RegionLocationMinY0) == region['y0']]
            if len(matching) != 1:
                raise ValueError('Ambiguous DICOM calibration region')
            r = matching[0]
            if int(r.PhysicalUnitsXDirection) != 3 or int(r.PhysicalUnitsYDirection) != 3:
                raise ValueError('DICOM region must use centimetres')
            spacing = [abs(float(r.PhysicalDeltaY))*10, abs(float(r.PhysicalDeltaX))*10]
            if not np.allclose(spacing, [region['delta_y_mm'], region['delta_x_mm']], atol=1e-5, rtol=0):
                raise ValueError('Export and DICOM physical spacing disagree')
            frames = index['frames']
            views.append(dict(name=VIEWS[vi], spacing_yx_mm=spacing, region=region,
                mask_files=[str(labels / name / x['mask']) for x in frames],
                times_s=[x['time_s'] for x in frames],
                sources=[[x.get('chamber_sources', {}).get(c, x['source']) for c in CHAMBERS] for x in frames],
                annotated=np.broadcast_to(DEFAULT_VALID[vi], (len(frames), 4)).tolist(),
                channels=list(index['chamber_channels']), frame_ids=[x['frame_index'] for x in frames]))
        return dict(version='real-tto-input-v1', patient_id=folder.name, cycle_start_s=start,
            cycle_duration_s=end-start, anchor_time_s=anchor, ED_frame=ed, ES_frame=es,
            dicom=str(paths[0]), dicom_sha256=sha256(paths[0]), views=views,
            acquisition_status='display-time synchronized; absolute 3D calibration unverified',
            manual_phase_alignment=dict(ED_time_s=anchor, ES_time_s=es_time,
                ES_inside_selected_RR=bool(start<=es_time<end),
                source_frame0_equals_real_ED='UNVERIFIED; source reference phase is not universally ventricular ED'),
            image_layout='DICOM display region crop')

    def __getitem__(self, index):
        patient = self.patients[index]
        folder = self.root / patient
        manifest = folder / 'tto_input.json'
        contract = read_json(manifest) if manifest.exists() else self._repository_contract(folder)
        if contract.get('version') != 'real-tto-input-v1' or contract['patient_id'] != patient:
            raise ValueError('Input version/patient identity mismatch')
        if [v['name'] for v in contract['views']] != list(VIEWS):
            raise ValueError('Views must be ordered A2C/A3C/A4C')
        def path(value):
            p = Path(value)
            return p if p.is_absolute() else folder / p
        views, source_hashes = [], {}
        for spec in contract['views']:
            channels = spec['channels']
            if set(channels) != set(CHAMBERS):
                raise ValueError('Four named independent mask channels required')
            permutation = [channels.index(c) for c in CHAMBERS]
            if 'mask_files' in spec:
                files = [path(p) for p in spec['mask_files']]
                masks = np.stack([np.load(p, allow_pickle=False)['masks'][permutation] for p in files])
            else:
                files = [path(spec['masks_npz'])]
                with np.load(files[0], allow_pickle=False) as z:
                    masks = z['masks'][:, permutation].copy()
            for p in files:
                source_hashes[str(p)] = sha256(p)
            times = np.array(spec['times_s'], float)
            spacing = np.array(spec['spacing_yx_mm'], float)
            annotated = np.array(spec['annotated'], bool)
            sources = np.array(spec['sources'], str)
            if (masks.ndim != 4 or masks.shape[1] != 4 or not np.isin(masks, [0, 1]).all()
                    or masks.shape[0] != len(times) or not np.isfinite(times).all()
                    or (np.diff(times) <= 0).any() or spacing.shape != (2,)
                    or not np.isfinite(spacing).all() or (spacing <= 0).any()
                    or annotated.shape != masks.shape[:2] or sources.shape != annotated.shape):
                raise ValueError('Invalid native mask/time/spacing/availability/source contract')
            policy = self.config['source_weights']
            unknown = set(sources[annotated]) - set(policy)
            if unknown:
                raise ValueError(f'Explicit trust weights required for sources: {sorted(unknown)}')
            weights = np.array([[policy.get(s, 0.) for s in row] for row in sources], np.float32) * annotated
            views.append(dict(masks=masks, times=times, spacing=spacing, annotated=annotated,
                sources=sources.tolist(), weights=weights, frame_ids=spec.get('frame_ids', list(range(len(times))))))
        duration, start, anchor = (float(contract[k]) for k in ('cycle_duration_s', 'cycle_start_s', 'anchor_time_s'))
        if not np.isfinite([duration, start, anchor]).all() or duration <= 0 or not start <= anchor < start+duration:
            raise ValueError('Finite complete RR and reference anchor required')
        mapping = []
        for phase in range(24):
            target = start + ((anchor-start+phase*duration/24) % duration)
            row = []
            for view in views:
                times = view['times']
                if target < times[0] or target > times[-1]:
                    raise ValueError('Target phase outside available native observations')
                hi = int(np.searchsorted(times, target, side='left'))
                lo = max(0, hi-1)
                if abs(times[hi]-target) < 1e-9:
                    lo = hi
                weight = (target-times[lo])/(times[hi]-times[lo]) if hi != lo else 0.
                nearest = lo if weight < .5 else hi
                row.append(dict(model_phase=phase/24, target_time_s=target, lower_index=lo, upper_index=hi,
                    upper_weight=weight, nearest_index=nearest, nearest_native_frame=view['frame_ids'][nearest],
                    nearest_time_error_ms=(times[nearest]-target)*1000))
            mapping.append(row)
        # Normalize each view on its own grid; padding different native sizes would move its centre.
        encoded_masks, encoded_sdf, transforms, pixel_spacing, validity = [], [], [], [], []
        for vi, view in enumerate(views):
            h, w = view['masks'].shape[-2:]
            slab = np.zeros((24, 3, 4, h, w), np.uint8)
            valid = np.zeros((24, 3, 4), bool)
            present = np.zeros(3, bool); present[vi] = True
            cache = {}
            for phase in range(24):
                m = mapping[phase][vi];lo, hi, weight = m['lower_index'], m['upper_index'], m['upper_weight']
                valid[phase, vi] = view['annotated'][lo] & view['annotated'][hi]
                if lo == hi:
                    slab[phase, vi] = view['masks'][lo]
                else:
                    for frame in (lo, hi):
                        if frame not in cache:
                            cache[frame] = np.stack([sdf_mm(x, view['spacing']) for x in view['masks'][frame]])
                    slab[phase, vi] = ((1-weight)*cache[lo]+weight*cache[hi] <= 0)
            transform = MaskTransform(target_size=self.config['model_size'],
                canonical_fov_mm=self.config['canonical_fov_mm'], allow_clipping=False,
                sdf_clip_mm=self.config['sdf_clip_mm'], sdf_normalize=False)
            norm = transform(slab, np.tile(view['spacing'], (3, 1)), view_present=present, annotated=valid)
            encoded_masks.append(norm['masks'][:, vi]);encoded_sdf.append(norm['sdf'][:, vi])
            transforms.append(norm['model_to_native'][:, vi]);pixel_spacing.append(norm['pixel_spacing'][:, vi])
            validity.append(valid[:, vi])
        encoder = dict(masks=torch.tensor(np.stack(encoded_masks, 1), dtype=torch.float32),
            sdf=torch.tensor(np.stack(encoded_sdf, 1), dtype=torch.float32),
            annotated=torch.tensor(np.stack(validity, 1)), view_present=torch.ones(24, 3, dtype=torch.bool),
            phases=torch.arange(24, dtype=torch.float32)/24, sdf_units='mm')
        images = []
        if self.config.get('load_images', True):
            if 'dicom' in contract:
                import pydicom
                pixels = pydicom.dcmread(path(contract['dicom'])).pixel_array
                for vi, spec in enumerate(contract['views']):
                    r = spec['region']
                    indices = [mapping[t][vi]['nearest_index'] for t in range(24)]
                    crop = pixels[indices, r['y0']:r['y0']+r['height'], r['x0']:r['x0']+r['width']].copy()
                    if crop.shape[1:3] != views[vi]['masks'].shape[-2:]:
                        raise ValueError('DICOM crop and native mask grid disagree')
                    images.append(crop)
                del pixels
            else:
                for vi, spec in enumerate(contract['views']):
                    p = path(spec['images_npy'])
                    source_hashes[str(p)] = sha256(p)
                    raw = np.load(p, allow_pickle=False)
                    if raw.shape[0] != len(views[vi]['times']) or raw.shape[1:3] != views[vi]['masks'].shape[-2:]:
                        raise ValueError('Original images and masks must share native grid/time')
                    images.append(raw[[mapping[t][vi]['nearest_index'] for t in range(24)]].copy())
        metadata = dict(contract=contract, source_hashes=source_hashes, mapping=mapping,
            raw_temporal_coverage=[dict(view=VIEWS[vi], original_frames=len(view['times']),
                frames_inside_selected_RR=int(((view['times']>=start)&(view['times']<start+duration)).sum()),
                excluded_frame_reason='Outside the preselected complete RR; independent of prediction/Dice')
                for vi,view in enumerate(views)],
            model_to_native=np.stack(transforms, 1).tolist(), model_spacing_yx_mm=np.stack(pixel_spacing, 1).tolist(),
            model_grid='per-view physical centre W/2,H/2; fixed FOV; no cross-view padding',
            native_evaluation='nearest original binary observation; timing error retained',
            image_status='loaded original native images' if images else 'not loaded (explicit configuration)')
        return PatientObservations(patient, views, mapping, encoder, metadata, images)


def collate_patients(patients):
    """Stack only common model grids; keep heterogeneous native arrays as a list."""
    if len({p.patient_id for p in patients}) != len(patients):
        raise ValueError('Duplicate patient in batch')
    keys = ('masks', 'sdf', 'annotated', 'view_present', 'phases')
    return dict(observations={**{k: torch.stack([p.encoder[k] for p in patients]) for k in keys},
                              'sdf_units': ['mm'] * len(patients)}, patients=patients)
