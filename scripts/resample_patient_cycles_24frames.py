#!/usr/bin/env python3
"""Rescue long LV-area cycles using 24 ED/ES-anchored native observations.

This is temporal downsampling, not a consecutive-original-frame export.
All output images, masks, and contours are exact native observations. Manual
ED/ES, area extrema, and sensitivity envelope endpoints are mandatory samples.
The remaining samples approximate time-uniform spacing within anchor segments.
Actual timestamps and target-time errors are retained; no uniform phase grid
or closed periodic trajectory is implied.

  python scripts/resample_patient_cycles_24frames.py --output outputs/long_cycles24
  python scripts/resample_patient_cycles_24frames.py --patients C11 C15 --check-only --output outputs/check24

Dependencies: numpy, scipy, pydicom>=3; matplotlib for the export preview.
"""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

if __package__:
    from .prepare_patient_48frames import CHANNELS, ROOT, inspect_patient, read_json, require
else:
    from prepare_patient_48frames import CHANNELS, ROOT, inspect_patient, read_json, require


def anchored_downsample(first, last, anchors, count=24):
    """Select distinct native frames, retaining all mandatory anchors exactly."""
    require(all(isinstance(v, (int, np.integer)) for v in (first, last, count)), 'integer bounds required')
    require(first >= 0 and count >= 2 and last-first+1 >= count, 'source interval too short for downsampling')
    require(all(isinstance(v, (int, np.integer)) and first <= v <= last for v in anchors), 'invalid anchor')
    fixed = sorted({int(first), int(last), *(int(v) for v in anchors)})
    require(len(fixed) <= count, 'more mandatory anchors than output frames')
    lengths = np.diff(fixed)
    ideal = lengths.astype(float)*(count-1)/(last-first)
    intervals = np.ones(len(lengths), dtype=int)
    # Allocate integer sample gaps near the proportional quotas, while each
    # anchor segment retains at least one gap and cannot duplicate native frames.
    while int(intervals.sum()) < count-1:
        deficits = np.where(intervals < lengths, ideal-intervals, -np.inf)
        require(np.isfinite(deficits).any(), 'cannot allocate unique native samples')
        intervals[int(np.argmax(deficits))] += 1
    positions = np.concatenate([np.linspace(a, b, int(k)+1)[:-1]
                                for a, b, k in zip(fixed[:-1], fixed[1:], intervals)] + [np.array([last], float)])
    indices = np.floor(positions+.5).astype(np.int32)
    require(len(indices) == count and (np.diff(indices) > 0).all(), 'resampling repeated or reversed a native frame')
    require(set(fixed) <= set(indices.tolist()), 'resampling lost a mandatory anchor')
    return dict(source_frame_indices=indices.tolist(), target_source_frame_positions=positions.tolist(),
                anchor_source_indices=fixed, segment_interval_counts=intervals.tolist())


def make_plan(coverage, phases):
    require(coverage['status'] == 'rejected_native24_too_short', 'only verified long LV-area cycles need this fallback')
    require(all(v['reliable'] for v in coverage['sensitivity']), 'LV cycle boundaries require review')
    envelope, cycle = coverage['coverage_envelope'], coverage['detection']['selected']
    first = min(envelope['first_peak'], *phases.values())
    last = max(envelope['next_peak'], *phases.values())
    labels = {}
    for frame, label in [(first, 'coverage_start'), (last, 'coverage_end'),
                         (envelope['first_peak'], 'area_boundary_envelope_start'),
                         (envelope['next_peak'], 'area_boundary_envelope_end'),
                         (cycle['first_peak'], 'area_first_peak'), (cycle['area_minimum'], 'area_minimum'),
                         (cycle['next_peak'], 'area_next_peak'),
                         *[(value, f'manual_{phase}') for phase, value in phases.items()]]:
        labels.setdefault(frame, []).append(label)
    plan = anchored_downsample(first, last, list(labels))
    indices = plan['source_frame_indices']
    plan.update(first_source_frame=first, last_source_frame=last, source_interval_frame_count=last-first+1,
                frame_count=24, phase_indices={ph: indices.index(i) for ph, i in phases.items()},
                mandatory_anchors=[dict(source_frame=i, output_frame=indices.index(i), labels=labels[i]) for i in sorted(labels)])
    return plan


def mapping_rows(plan, frame_time_ms):
    rows = []
    start = plan['first_source_frame']
    labels = {a['source_frame']: a['labels'] for a in plan.get('mandatory_anchors', [])}
    for j, (native, target) in enumerate(zip(plan['source_frame_indices'], plan['target_source_frame_positions'])):
        lower, upper = int(np.floor(target)), int(np.ceil(target))
        lower_weight = int(native == lower)
        rows.append(dict(output_frame=j, source_frame=native, target_source_frame_position=target,
                         target_lower_source_frame=lower, target_upper_source_frame=upper,
                         nearest_lower_weight=lower_weight, nearest_upper_weight=1-lower_weight,
                         source_time_s=native*frame_time_ms/1000, time_s=(native-start)*frame_time_ms/1000,
                         target_source_time_s=target*frame_time_ms/1000,
                         nearest_time_error_ms=(native-target)*frame_time_ms,
                         anchor_labels=';'.join(labels.get(native, []))))
    return rows


def export_patient(destination, data, plan):
    from pydicom.pixels import iter_pixels

    base, dicom, calibration, phases, original48, selected, coverage = data
    indices = np.asarray(plan['source_frame_indices'], dtype=np.int32)
    dt = float(calibration['frame_time_ms'])
    rows = mapping_rows(plan, dt)
    labels = base / '患者标签（每一帧）'
    images = {key: [] for key in selected}
    for frame in iter_pixels(dicom, indices=indices.tolist()):
        for key, view in selected.items():
            region = view['region']
            images[key].append(frame[region['y0']:region['y1']+1, region['x0']:region['x1']+1].copy())
    require(all(len(frames) == 24 for frames in images.values()), 'DICOM decoded fewer than 24 frames')
    actual_time = indices.astype(float)*dt/1000
    target_time = np.asarray(plan['target_source_frame_positions'])*dt/1000
    manual_flags = np.isin(indices, list(phases.values()))
    metadata = dict(patient=base.name, frame_count=24, frame_index_base=0,
                    sampling='lv_area_anchored_native_temporal_downsample',
                    is_consecutive_native_frames=False, is_uniform_time_sampling=bool((np.diff(indices) == np.diff(indices)[0]).all()),
                    image_and_label_interpolation='none; nearest original frames selected',
                    source_dicom=str(dicom.resolve()), source_native_frame_time_ms=dt,
                    source_phase_indices=phases, phase_indices=plan['phase_indices'],
                    source_frame_indices=indices.tolist(), source_time_s=actual_time.tolist(),
                    time_s=(actual_time-actual_time[0]).tolist(), source_time_intervals_s=np.diff(actual_time).tolist(),
                    time_reference='source frame 0 at time 0, consistent with original label timestamps',
                    target_source_time_s=target_time.tolist(), plan=plan, mapping=rows,
                    cycle_coverage=coverage, chamber_channels=CHANNELS, planes={},
                    endpoint_policy='closed coverage interval; periodic closure is not imposed',
                    manual_annotation_sha256=hashlib.sha256((labels / '手工ED_ES源标注.json').read_bytes()).hexdigest())
    for key, view in selected.items():
        folder = labels / view['name']
        index = read_json(folder / '帧索引.json')
        records = [index['frames'][i] for i in indices]
        masks = []
        for record in records:
            with np.load(folder / record['mask'], allow_pickle=False) as archive:
                masks.append(archive['masks'].copy())
        masks, native_images = np.stack(masks), np.stack(images[key])
        require(native_images.shape[1:3] == masks.shape[2:], 'native image/mask shape mismatch')
        archive_path = destination / f"{view['name']}.npz"
        np.savez_compressed(archive_path, images=native_images, masks=masks, source_frame_indices=indices,
                            source_time_s=actual_time, time_s=actual_time-actual_time[0],
                            target_source_time_s=target_time, nearest_time_error_ms=(actual_time-target_time)*1000,
                            is_manual_ed_es=manual_flags, is_interpolated=np.zeros(24, bool))
        with np.load(archive_path, allow_pickle=False) as saved:
            require(np.array_equal(saved['images'], native_images) and np.array_equal(saved['masks'], masks)
                    and np.array_equal(saved['source_frame_indices'], indices), 'archive roundtrip mismatch')
        phase_checks = {}
        for ph, output_frame in plan['phase_indices'].items():
            record = records[output_frame]
            require(record['frame_index'] == phases[ph] and record['phase'] == ph
                    and record['source'] == 'manual_ed_es', 'manual phase was lost or relabelled')
            phase_checks[ph] = dict(source_frame=phases[ph], output_frame=output_frame,
                                    image_sha256=hashlib.sha256(native_images[output_frame].tobytes()).hexdigest(),
                                    masks_sha256=hashlib.sha256(masks[output_frame].tobytes()).hexdigest(), unchanged=True)
        metadata['planes'][key] = dict(name=view['name'], file=archive_path.name, region=view['region'],
            manual_preservation=phase_checks,
            frames=[dict(output_frame=j, source_frame=int(indices[j]), source=record['source'],
                         chamber_sources=record['chamber_sources'], phase=record['phase'],
                         contours=f"original_labels/{view['name']}/frame_{indices[j]:04d}/contours.json")
                    for j, record in enumerate(records)])
        for record in records:
            target = destination / 'original_labels' / view['name'] / f"frame_{record['frame_index']:04d}"
            target.mkdir(parents=True)
            shutil.copy2(folder / record['contours'], target / 'contours.json')
    for path in labels.glob('*.json'):
        shutil.copy2(path, destination / 'original_labels' / path.name)
    with (destination / 'resampling_map.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (destination / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


def write_report(destination, audit):
    lines = ['较长LV面积周期重采样为24帧', '==========================', '',
             '本目录只处理原生连续24帧不足的病例。基于已确认的LV面积周期，不使用ECG/RR。',
             '采用保留人工ED/ES、LV面积峰谷和敏感性边界的时间下采样，输出24张不同的原始帧。',
             '图像、二值标签与原始轮廓原样保留；没有插值合成标签，也没有复制帧。',
             '输出为时间重采样序列，不是24张相邻原始帧；实际采样间隔可能不同。',
             '所有帧号从0开始，源时间及每个采样间隔在metadata.json、resampling_map.csv及NPZ中保存。', '',
             '.. list-table:: 导出结果', '   :header-rows: 1', '',
             '   * - 患者', '     - 原始闭区间', '     - 原始帧数 -> 输出帧数',
             '     - 原始ED/ES', '     - 输出ED/ES', '     - 最多跳过的相邻帧数']
    for row in audit['accepted']:
        plan = row['plan']; phases = row['source_phase_indices']
        values = [row['patient'], f"{plan['first_source_frame']}..{plan['last_source_frame']}",
                  f"{plan['source_interval_frame_count']} -> 24", f"{phases['ED']}/{phases['ES']}",
                  f"{plan['phase_indices']['ED']}/{plan['phase_indices']['ES']}",
                  int(np.max(np.diff(plan['source_frame_indices'])))-1]
        lines += [f"   {'* -' if i == 0 else '  -'} {v}" for i, v in enumerate(values)]
    lines += ['', '24帧覆盖原周期及人工标注；不代表获得了24个严格均匀的周期相位。',
              '用于运动建模时须使用真实时间映射；本次未启动训练或TTO，也未修改原始数据。', '']
    if audit['rejected']:
        lines += ['失败病例', '--------', '']+[f"* {r['patient']}：{r['reason']}" for r in audit['rejected']]
    (destination / 'report.rst').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=ROOT / 'real_data')
    parser.add_argument('--cycle-audit', type=Path, default=ROOT / 'outputs/patient_lv_area_cycle_20261017/analysis/audit.json')
    parser.add_argument('--patients', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source_audit = read_json(args.cycle_audit)
    manifest = read_json(args.input / '整理清单.json')
    patients = {p['patient']: p for p in manifest['patients']}
    requested = args.patients or source_audit['summary']['rejected']
    require(requested and len(set(requested)) == len(requested) and set(requested) <= set(patients), 'invalid patient selection')
    require(not args.output.resolve().is_relative_to(args.input.resolve()), 'output must be outside original data')
    args.output.mkdir(parents=True, exist_ok=False)
    audit = dict(mode='check_only' if args.check_only else 'export',
                 method='LV_area_anchored_24_native_observations',
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 source_audit_sha256=hashlib.sha256(args.cycle_audit.read_bytes()).hexdigest(), accepted=[], rejected=[])
    for pid in requested:
        temporary = None
        try:
            data = inspect_patient(args.input, patients[pid], require_native24=False)
            phases, coverage = data[3], data[-1]
            plan = make_plan(coverage, phases)
            if not args.check_only:
                temporary = Path(tempfile.mkdtemp(prefix=f'.{pid}-', dir=args.output))
                export_patient(temporary, data, plan)
                temporary.rename(args.output / pid); temporary = None
            audit['accepted'].append(dict(patient=pid, source_phase_indices=phases, plan=plan))
            print(f"OK {pid}: {plan['source_interval_frame_count']} -> 24; ED/ES={plan['phase_indices']}", flush=True)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, AttributeError, IndexError) as exc:
            audit['rejected'].append(dict(patient=pid, reason=str(exc)))
            print(f'SKIP {pid}: {exc}', flush=True)
        finally:
            if temporary is not None:
                shutil.rmtree(temporary)
        (args.output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(args.output, audit)
    return 1 if audit['rejected'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
