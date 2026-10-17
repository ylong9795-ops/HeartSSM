"""Identity temporal mapping for previously selected 24-frame observations.

Only spatial MaskTransform is allowed here. Source timestamps are provenance,
never an instruction to select, interpolate, roll or resample frames again.
"""
from pathlib import Path

import numpy as np
import torch

from data.mask_transform import CHAMBERS, VIEWS, MaskTransform
from .io import read_json, sha256

PHASE_SEMANTICS = 'synthetic source frame 0; not independently selected clinical ED'


def _names(names, expected, label):
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f'{label} requires unique names {expected}')
    return [names.index(name) for name in expected]


def packaged_contract(folder):
    """Read the existing package without claiming its phase semantics verified."""
    from .dataset import DEFAULT_VALID, PLANE_IDS
    meta = read_json(folder / 'metadata.json')
    if meta['patient'] != folder.name or meta['frame_count'] != 24:
        raise ValueError('Packaged patient identity/24-frame count mismatch')
    calibration = read_json(folder / '原始切面与像素间距.json')
    if calibration['patient_id'] != folder.name:
        raise ValueError('Packaged calibration patient mismatch')
    regions = {r['plane']: r for r in calibration['regions']}
    channels = list(meta['chamber_channels'])
    permutation = _names(channels, CHAMBERS, 'chambers')
    views = []
    for vi, (name, plane) in enumerate(zip(VIEWS, PLANE_IDS)):
        spec = meta['planes'][str(plane)]
        region = regions[plane]
        spacing = [abs(region['delta_y_mm']), abs(region['delta_x_mm'])]
        if not np.allclose(spacing, spec['pixel_spacing_yx_mm'], atol=1e-8, rtol=0):
            raise ValueError('Packaged mask and calibration spacing disagree')
        # Annotation/source arrays use the DECLARED source channel order too.
        valid = np.broadcast_to(DEFAULT_VALID[vi], (24, 4))[:, np.argsort(permutation)]
        records = spec['frames']
        if len(records) != 24 or any(r['output_frame'] != t or
                r['source_frame'] != meta['source_frame_indices'][t] for t, r in enumerate(records)):
            raise ValueError('Packaged output/source frame mapping disagrees')
        views.append(dict(name=name, channels=channels, masks_npz=spec['file'],
            sha256=spec['sha256'], spacing_yx_mm=spacing, region=region,
            times_s=meta['source_time_s'], frame_ids=meta['source_frame_indices'],
            annotated=valid.tolist(), sources=[[r['chamber_sources'][c] for c in channels] for r in records],
            archive_metadata=True))
    return dict(version='real-tto-input-v1', input_mode='aligned24', patient_id=folder.name,
        phases=(np.arange(24, dtype=float)/24).tolist(), views=views,
        phase_alignment=dict(status='UNVERIFIED',
            evidence='Package records selected native observations and actual times, not synthetic reference equivalence.',
            synthetic_reference=PHASE_SEMANTICS, phase_grid_status='UNVERIFIED',
            phase_grid_evidence='i/24 is a diagnostic model index only; uniform physiological phase is not established.',
            phase0_source_frame=meta['source_frame_indices'][0], manual_output_indices=meta['phase_indices'],
            sampling=meta['sampling'], is_uniform_time_sampling=meta['is_uniform_time_sampling'],
            lv_area_cycle=meta['lv_area_cycle'], coverage_envelope=meta['lv_area_coverage_envelope']),
        acquisition_status='existing selected24 package; temporal order unchanged',
        package_metadata='metadata.json')


def load_aligned24(folder, config):
    from .dataset import PatientObservations
    folder = Path(folder)
    manifest = folder / 'tto_input.json'
    contract = read_json(manifest) if manifest.exists() else packaged_contract(folder)
    if (contract.get('version') != 'real-tto-input-v1' or contract.get('input_mode') != 'aligned24'
            or contract.get('patient_id') != folder.name):
        raise ValueError('Explicit aligned24 version/patient contract required')
    if config['model_size'] != 224 or config['frames'] != 24:
        raise ValueError('aligned24 model grid must be [24,3,4,224,224]')
    phases = np.asarray(contract['phases'], dtype=np.float32)
    if phases.shape != (24,) or not np.allclose(phases, np.arange(24)/24, atol=1e-7, rtol=0):
        raise ValueError('aligned24 phases must match synthetic i/24, with no duplicate closing endpoint')
    alignment = contract['phase_alignment']
    for key in ('status', 'phase_grid_status'):
        if alignment.get(key) not in ('VERIFIED', 'UNVERIFIED'):
            raise ValueError(f'phase_alignment.{key} requires VERIFIED or UNVERIFIED')
    if (not alignment.get('evidence') or not alignment.get('phase_grid_evidence')
            or alignment.get('synthetic_reference') != PHASE_SEMANTICS):
        raise ValueError('Explicit phase0/grid evidence and synthetic reference definition required')
    specs = contract['views']
    view_order = _names([s['name'] for s in specs], VIEWS, 'views')
    specs = [specs[i] for i in view_order]
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else folder / value
    hashes = {str(manifest if manifest.exists() else folder / 'metadata.json'):
              sha256(manifest if manifest.exists() else folder / 'metadata.json')}
    if 'package_metadata' in contract:
        calibration = folder / '原始切面与像素间距.json'
        hashes[str(calibration)] = sha256(calibration)
    views, mapping = [], [[] for _ in range(24)]
    encoded_masks, encoded_sdf, transforms, spacings, validities, presences, audit = [], [], [], [], [], [], []
    images = []
    for vi, spec in enumerate(specs):
        channels = spec['channels']
        permutation = _names(channels, CHAMBERS, 'chambers')
        file = path(spec['masks_npz'])
        hashes[str(file)] = sha256(file)
        if spec.get('sha256') and hashes[str(file)] != spec['sha256']:
            raise ValueError('Packaged mask SHA256 mismatch')
        times = np.asarray(spec['times_s'], float)
        spacing = np.asarray(spec['spacing_yx_mm'], float)
        with np.load(file, allow_pickle=False) as z:
            masks = z['masks'][:, permutation].copy()
            if spec.get('archive_metadata'):
                for key, expected in (('chamber_channels', channels), ('source_frame_indices', spec['frame_ids']),
                        ('source_time_s', times), ('pixel_spacing_yx_mm', spacing), ('chamber_sources', spec['sources'])):
                    if not np.array_equal(z[key], np.asarray(expected)):
                        raise ValueError(f'Packaged archive {key} disagrees with manifest')
        annotated_raw, present_raw = np.asarray(spec['annotated']), np.asarray(spec.get('view_present', [True]*24))
        if (annotated_raw.shape != (24, 4) or present_raw.shape != (24,)
                or not np.isin(annotated_raw, [0, 1]).all() or not np.isin(present_raw, [0, 1]).all()):
            raise ValueError('aligned24 availability must be binary [24,4] and [24]')
        annotated = annotated_raw[:, permutation].astype(bool)
        present = present_raw.astype(bool)
        sources_raw = np.asarray(spec['sources'], str)
        if sources_raw.shape != (24, 4):
            raise ValueError('aligned24 sources require [24,4] in declared channel order')
        sources = sources_raw[:, permutation]
        frames = spec.get('frame_ids', list(range(24)))
        if (masks.ndim != 4 or masks.shape[:2] != (24, 4) or min(masks.shape[-2:]) < 1
                or not np.isin(masks, [0, 1]).all() or times.shape != (24,)
                or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
                or spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any()
                or len(frames) != 24 or len(set(frames)) != 24 or (annotated & ~present[:, None]).any()):
            raise ValueError('Invalid aligned24 native mask/time/spacing/frame/availability contract')
        policy = config['source_weights']
        unknown = set(sources[annotated]) - set(policy)
        if unknown:
            raise ValueError(f'Explicit trust weights required for sources: {sorted(unknown)}')
        weights = np.array([[policy.get(s, 0.) for s in row] for row in sources], np.float32)*annotated
        views.append(dict(masks=masks, times=times, spacing=spacing, annotated=annotated,
            sources=sources.tolist(), weights=weights, frame_ids=frames))
        for t in range(24):
            mapping[t].append(dict(model_phase=float(phases[t]), target_time_s=float(times[t]),
                lower_index=t, upper_index=t, upper_weight=0., nearest_index=t,
                nearest_native_frame=frames[t], nearest_time_error_ms=0.))
        h, w = masks.shape[-2:]
        slab = np.zeros((24, 3, 4, h, w), masks.dtype); slab[:, vi] = masks
        valid = np.zeros((24, 3, 4), bool); valid[:, vi] = annotated
        available = np.zeros((24, 3), bool); available[:, vi] = present
        norm = MaskTransform(target_size=224, canonical_fov_mm=config['canonical_fov_mm'],
            allow_clipping=False, sdf_clip_mm=config['sdf_clip_mm'], sdf_normalize=False)(
                slab, np.tile(spacing, (3, 1)), view_present=available, annotated=valid)
        matrix, pixel_spacing = norm['model_to_native'][:, vi], norm['pixel_spacing'][:, vi]
        centre = np.einsum('tij,j->ti', matrix, [112., 112., 1.])
        if not np.allclose(centre, [w/2, h/2, 1.], atol=1e-8, rtol=0):
            raise ValueError('Model/native physical centres disagree')
        encoded_masks.append(norm['masks'][:, vi]); encoded_sdf.append(norm['sdf'][:, vi])
        transforms.append(matrix); spacings.append(pixel_spacing)
        validities.append(annotated); presences.append(present)
        audit.append(dict(view=VIEWS[vi], source_channels=channels, canonical_channels=list(CHAMBERS),
            channel_permutation=permutation, native_shape=list(masks.shape),
            native_spacing_yx_mm=spacing.tolist(), model_spacing_yx_mm=pixel_spacing[0].tolist(),
            centre_max_error_px=float(np.abs(centre-[w/2, h/2, 1.]).max()), temporal_mapping='identity'))
        if config.get('load_images', False):
            file = path(spec['images_npy']); hashes[str(file)] = sha256(file)
            raw = np.load(file, allow_pickle=False)
            if raw.shape[:3] != (24, h, w):
                raise ValueError('aligned24 images must share original native grid and 24 frame order')
            images.append(raw.copy())
    arrays = dict(masks=np.stack(encoded_masks, 1).astype(np.float32), sdf=np.stack(encoded_sdf, 1),
        annotated=np.stack(validities, 1), view_present=np.stack(presences, 1), phases=phases)
    matrices, pixel_spacing = np.stack(transforms, 1), np.stack(spacings, 1)
    # An already prepared encoder archive is consumed unchanged, after checking
    # against the exact pretrained spatial preprocessing on the retained native grid.
    if 'encoder_npz' in contract:
        file = path(contract['encoder_npz']); hashes[str(file)] = sha256(file)
        with np.load(file, allow_pickle=False) as z:
            po = _names(z['views'].tolist(), VIEWS, 'encoder views')
            co = _names(z['channels'].tolist(), CHAMBERS, 'encoder chambers')
            if str(z['sdf_units'].item()) != 'mm':
                raise ValueError('aligned24 SDF must be millimetres, negative inside/positive outside')
            for key, expected in arrays.items():
                value = z[key].copy()
                if key != 'phases': value = value[:, po]
                if key in ('masks', 'sdf', 'annotated'): value = value[:, :, co]
                if (value.shape != expected.shape or not np.isfinite(value).all()
                        or (key in ('masks', 'annotated', 'view_present') and not np.array_equal(value, expected))
                        or not np.allclose(value, expected, atol=1e-6 if key == 'sdf' else 1e-7, rtol=0)):
                    raise ValueError(f'aligned24 encoder {key} disagrees with native pretrained preprocessing')
                arrays[key] = value
            for key, expected in (('model_to_native', matrices), ('pixel_spacing_yx_mm', pixel_spacing)):
                value = z[key][:, po]
                if value.shape != expected.shape or not np.allclose(value, expected, atol=1e-8, rtol=0):
                    raise ValueError(f'aligned24 encoder {key} disagrees with physical calibration')
    encoder = {k: torch.as_tensor(v, dtype=torch.bool if k in ('annotated', 'view_present') else torch.float32)
               for k, v in arrays.items()}
    encoder['sdf_units'] = 'mm'
    metadata = dict(contract=contract, source_hashes=hashes, mapping=mapping,
        input_mode='aligned24', temporal_resampling_performed=False, phase_order_changed=False,
        input_audit=audit, phase_alignment=alignment,
        model_to_native=matrices.tolist(), model_spacing_yx_mm=pixel_spacing.tolist(),
        model_grid='per-view physical centre W/2,H/2; pretrained fixed FOV; no cross-view padding',
        native_evaluation='original selected24 binary observations; identity model index; phase semantics audited separately',
        image_status='loaded original native images' if images else 'not loaded (explicit configuration)')
    return PatientObservations(folder.name, views, mapping, encoder, metadata, images)
