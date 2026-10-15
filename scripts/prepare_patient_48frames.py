#!/usr/bin/env python3
"""Export synchronized, consecutive 48-frame three-view patient sequences.

No interpolation, temporal stride, duplication, wrapping, or spatial resizing.
Only within-patient alignment is enforced. ED/ES locations may vary by patient.
Qualification means structural consistency, NOT anatomical correctness.

Dependencies: numpy, pydicom>=3 (and a decoder if the DICOM is compressed).
Usage:
  python scripts/prepare_patient_48frames.py --check-only
  python scripts/prepare_patient_48frames.py --output outputs/patient_48frames
  python scripts/prepare_patient_48frames.py --patients C01 C02 --output outputs/demo48

Output per patient:
  A4C_四腔心.npz / A2C_两腔心.npz / A3C_三腔心.npz:
    images: (48,H,W) or (48,H,W,3), native decoded pixels
    masks: (48,4,H,W), original LV/RV/LA/RA masks unchanged
    source_frame_indices, time_s, source_time_s, is_manual_ed_es
  metadata.json: original/new ED/ES indices, timestamps, per-chamber sources
  original_labels/: original metadata, hand annotations, reports, selected contours
  audit.json at output root: selected/rejected patients and exact reasons

By default the whole patient's three label sequences must be complete, including
frames outside the selected window. --patients restricts processing explicitly.
Model/fallback masks remain identified as such; they are not promoted to manual.
Original DICOMs are referenced by absolute path, not copied or modified.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_FRAMES = 48
CHANNELS = ['LV', 'RV', 'LA', 'RA']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def choose_window(n, ed, es):
    """Center on the ED/ES midpoint, then clamp; return consecutive indices."""
    require(n >= N_FRAMES, f'only {n} frames; need {N_FRAMES}')
    require(0 <= ed < n and 0 <= es < n and ed != es, 'invalid ED/ES indices')
    lo, hi = sorted((ed, es))
    require(hi - lo < N_FRAMES, 'ED/ES cannot fit in a consecutive 48-frame window')
    start = max(0, min(n - N_FRAMES, (lo + hi - (N_FRAMES - 1)) // 2))
    indices = np.arange(start, start + N_FRAMES, dtype=np.int32)
    require(ed in indices and es in indices, 'window lost ED/ES')
    return indices


def inspect_patient(root, patient):
    from pydicom import dcmread

    pid = patient['patient']
    base = root / pid
    labels = base / '患者标签（每一帧）'
    meta = read_json(labels / '原始切面与像素间距.json')
    manual_path = labels / '手工ED_ES源标注.json'
    manual = read_json(manual_path)
    require(hashlib.sha256(manual_path.read_bytes()).hexdigest() ==
            patient['manual_annotation_sha256'], 'manual annotation hash mismatch')
    require(meta['patient_id'] == pid and manual['patient'] == pid, 'patient ID mismatch')
    phases = patient['manual_phase_frame_indices']
    require(all(manual['phases'][ph]['frame'] == phases[ph] for ph in ('ED', 'ES')),
            'manual ED/ES frame indices disagree')
    n = int(meta['source_frame_count'])
    indices = choose_window(n, phases['ED'], phases['ES'])
    require(set(patient['planes']) == {'1', '2', '3'}, 'three planes required')
    require(np.isfinite(meta['frame_time_ms']) and meta['frame_time_ms'] > 0,
            'invalid frame time')
    dicom = base / '患者原始dicom' / Path(meta['source_dicom']).name
    require(dicom.is_file(), f'missing DICOM: {dicom.name}')
    declared = next((d for d in patient['dicoms'] if d['name'] == dicom.name), None)
    require(declared is not None and dicom.stat().st_size == declared['bytes'],
            'DICOM missing from manifest or size mismatch')
    ds = dcmread(dicom, stop_before_pixels=True, specific_tags=[
        'NumberOfFrames', 'FrameTime', 'Rows', 'Columns', 'SequenceOfUltrasoundRegions'])
    require(int(ds.NumberOfFrames) == n, 'DICOM frame count mismatch')
    require(np.isclose(float(ds.FrameTime), meta['frame_time_ms']), 'frame time mismatch')
    regions = {str(r['plane']): r for r in meta['regions']}
    require(set(regions) == {'1', '2', '3'}, 'missing crop regions')
    selected = {}
    for key in ('1', '2', '3'):
        plane = patient['planes'][key]
        folder = labels / plane['name']
        idx = read_json(folder / '帧索引.json')
        require(idx['patient'] == pid and str(idx['plane']) == key, 'index identity mismatch')
        require(idx['chamber_channels'] == CHANNELS and idx['frame_index_base'] == 0,
                'unexpected channel order/index base')
        require(idx['frame_count'] == plane['frame_count'] == len(idx['frames']) == n,
                f'{plane["name"]}: frame count mismatch')
        r = regions[key]
        h, w = plane['shape']
        require((r['height'], r['width']) == (h, w) and
                r['x1'] - r['x0'] + 1 == w and r['y1'] - r['y0'] + 1 == h and
                0 <= r['x0'] <= r['x1'] < ds.Columns and
                0 <= r['y0'] <= r['y1'] < ds.Rows, 'invalid crop region')
        dr = ds.SequenceOfUltrasoundRegions[r['dicom_region'] - 1]
        require((dr.RegionLocationMinX0, dr.RegionLocationMinY0,
                 dr.RegionLocationMaxX1, dr.RegionLocationMaxY1) ==
                (r['x0'], r['y0'], r['x1'], r['y1']), 'DICOM crop region mismatch')
        masks, records = [], []
        for i, fr in enumerate(idx['frames']):
            tag = f'{plane["name"]}/frame_{i:04d}'
            require(fr['frame_index'] == i and fr['display_frame'] == i + 1,
                    f'{tag}: nonconsecutive index')
            require(np.isclose(fr['time_s'], i * meta['frame_time_ms'] / 1000),
                    f'{tag}: inconsistent timestamp')
            expected_phase = next((ph for ph, fi in phases.items() if fi == i), None)
            require(fr['phase'] == expected_phase, f'{tag}: phase mismatch')
            require(fr['mask'] == f'frame_{i:04d}/masks.npz' and
                    fr['contours'] == f'frame_{i:04d}/contours.json',
                    f'{tag}: file paths do not match frame index')
            for field in ('mask', 'contours'):
                path = (folder / fr[field]).resolve()
                require(path.is_relative_to(folder.resolve()) and path.is_file(),
                        f'{tag}: missing or invalid {field} file')
            with np.load(folder / fr['mask'], allow_pickle=False) as archive:
                mask = archive['masks']
            require(mask.shape == (4, h, w) and np.isin(mask, [0, 1]).all(),
                    f'{tag}: invalid mask shape/values')
            active = range(4) if key == '1' else (0, 2)
            require(all(mask[c].any() for c in active), f'{tag}: empty active chamber')
            require(key == '1' or not mask[[1, 3]].any(), f'{tag}: nonzero inactive chamber')
            contour = read_json(folder / fr['contours'])
            require(all(contour.get(k) == v for k, v in fr.items()),
                    f'{tag}: contour/index metadata mismatch')
            if expected_phase:
                require(fr['source'] == 'manual_ed_es' and
                        all(fr['chamber_sources'][CHANNELS[c]] == 'manual_ed_es' for c in active),
                        f'{tag}: ED/ES is not fully manual')
            if i in indices:
                masks.append(mask.copy())
                records.append(fr)
        selected[key] = {'name': plane['name'], 'masks': np.stack(masks),
                         'records': records, 'region': r}
    return base, dicom, meta, phases, indices, selected


def export_patient(destination, base, dicom, meta, phases, indices, selected):
    from pydicom.pixels import iter_pixels

    images = {key: [] for key in selected}
    for frame in iter_pixels(dicom, indices=indices.tolist()):
        for key, seq in selected.items():
            r = seq['region']
            images[key].append(frame[r['y0']:r['y1'] + 1, r['x0']:r['x1'] + 1].copy())
    require(all(len(x) == N_FRAMES for x in images.values()), 'DICOM decoded fewer than 48 frames')
    start = int(indices[0])
    manual_flags = np.isin(indices, list(phases.values()))
    source_time = indices.astype(np.float64) * meta['frame_time_ms'] / 1000
    output_meta = {'patient': base.name, 'frame_count': N_FRAMES,
                   'sampling': 'consecutive_native_frames', 'source_dicom': str(dicom.resolve()),
                   'source_frame_indices': indices.tolist(), 'frame_time_ms': meta['frame_time_ms'],
                   'source_phase_indices': phases,
                   'phase_indices': {ph: int(i - start) for ph, i in phases.items()},
                   'chamber_channels': CHANNELS, 'planes': {},
                   'qualification': 'structural checks only; no anatomical review'}
    for key, seq in selected.items():
        name = seq['name']
        image_array = np.stack(images[key])
        require(image_array.shape[1:3] == seq['masks'].shape[2:], 'image/mask size mismatch')
        target = destination / f'{name}.npz'
        np.savez_compressed(target, images=image_array, masks=seq['masks'],
                            source_frame_indices=indices, source_time_s=source_time,
                            time_s=source_time - source_time[0], is_manual_ed_es=manual_flags)
        with np.load(target, allow_pickle=False) as saved:
            require(np.array_equal(saved['masks'], seq['masks']) and
                    np.array_equal(saved['images'], image_array), 'output roundtrip mismatch')
        output_meta['planes'][key] = {
            'file': target.name, 'region': seq['region'],
            'frames': [dict(fr, frame_index=j, display_frame=j + 1,
                            source_frame_index=fr['frame_index'],
                            source_time_s=fr['time_s'], time_s=float(source_time[j] - source_time[0]),
                            mask=f'{name}.npz:masks[{j}]',
                            contours=f'original_labels/{name}/frame_{fr["frame_index"]:04d}/contours.json')
                       for j, fr in enumerate(seq['records'])]}
        for fr in seq['records']:
            target_dir = destination / 'original_labels' / name / f'frame_{fr["frame_index"]:04d}'
            target_dir.mkdir(parents=True)
            shutil.copy2(base / '患者标签（每一帧）' / name / fr['contours'], target_dir / 'contours.json')
    originals = destination / 'original_labels'
    for path in (base / '患者标签（每一帧）').glob('*.json'):
        shutil.copy2(path, originals / path.name)
    if (base / '患者报告').is_dir():
        shutil.copytree(base / '患者报告', destination / '患者报告')
    (destination / 'metadata.json').write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=ROOT / '患者逐帧分割整理_20261015')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/patient_48frames')
    parser.add_argument('--patients', nargs='+', help='Optional patient IDs, e.g. C01 C02')
    parser.add_argument('--check-only', action='store_true', help='Validate and report; do not decode/export images')
    args = parser.parse_args()
    manifest = read_json(args.input / '整理清单.json')
    known = {p['patient'] for p in manifest['patients']}
    require(not args.patients or set(args.patients) <= known, 'unknown patient ID requested')
    require(not args.output.resolve().is_relative_to(args.input.resolve()), 'output must be outside input dataset')
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {'mode': 'check_only' if args.check_only else 'export', 'accepted': [], 'rejected': []}
    report = args.output / ('validation.json' if args.check_only else 'audit.json')
    require(not report.exists(), f'refusing to overwrite existing report: {report}')
    for patient in manifest['patients']:
        pid = patient['patient']
        if args.patients and pid not in args.patients:
            continue
        temporary = None
        try:
            final = args.output / pid
            require(args.check_only or not final.exists(), 'output patient already exists; choose a fresh output directory')
            data = inspect_patient(args.input, patient)
            _, _, _, phases, indices, _ = data
            if not args.check_only:
                temporary = Path(tempfile.mkdtemp(prefix=f'.{pid}-', dir=args.output))
                export_patient(temporary, *data)
                temporary.rename(final)
                temporary = None
            row = {'patient': pid, 'first_source_frame': int(indices[0]),
                   'last_source_frame': int(indices[-1]),
                   'phase_indices': {ph: int(i - indices[0]) for ph, i in phases.items()}}
            audit['accepted'].append(row)
            print(f'OK {pid}: {row}', flush=True)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError, AttributeError, IndexError) as exc:
            audit['rejected'].append({'patient': pid, 'reason': str(exc)})
            print(f'SKIP {pid}: {exc}', flush=True)
        finally:
            if temporary is not None:
                shutil.rmtree(temporary)
        report.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Accepted {len(audit["accepted"])}, rejected {len(audit["rejected"])}; report: {report}')
    return 0 if audit['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
