#!/usr/bin/env python3
"""Package the verified 17-patient native/downsampled24 masks and reports.

Each patient contains three native-resolution masks NPZs, original patient
reports, all original manual ED/ES annotations, exact manual masks/contours,
and explicit source/output frame and time mappings. No images are required.
"""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['LV', 'RV', 'LA', 'RA']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def package_patient(source, destination, patient, located, resampled_root, cycle_root):
    pid = patient['patient']; base = source / pid; labels = base / '患者标签（每一帧）'
    measurement, analysis = located['measurements'], located['analysis']
    manual_path = labels / '手工ED_ES源标注.json'
    require(digest(manual_path) == patient['manual_annotation_sha256'] == measurement['manual_sha256'],
            f'{pid}: original manual annotation changed')
    manual = read_json(manual_path)
    phases = {phase: manual['phases'][phase]['frame'] for phase in ('ED', 'ES')}
    require(phases == patient['manual_phase_frame_indices'] == analysis['manual_phase_indices'], 'ED/ES index mismatch')
    calibration_path = labels / '原始切面与像素间距.json'
    require(digest(calibration_path) == measurement['calibration_sha256'], 'calibration changed')
    calibration = read_json(calibration_path)
    regions = {str(region['plane']): region for region in calibration['regions']}
    require(manual['patient'] == calibration['patient_id'] == pid, 'patient identity mismatch')
    resampled = None
    if analysis['status'] == 'accepted_lv_area_native24':
        indices = analysis['native24']['source_frame_indices']
        sampling = 'consecutive_native24'
    else:
        require(analysis['status'] == 'rejected_native24_too_short', 'unverified LV area cycle')
        resampled = read_json(resampled_root / pid / 'metadata.json')
        require(resampled['patient'] == pid and resampled['source_phase_indices'] == phases and
                resampled['manual_annotation_sha256'] == digest(manual_path), 'resampled identity mismatch')
        indices = resampled['source_frame_indices']; sampling = 'anchored_native_temporal_downsample24'
    require(len(indices) == 24 and all(isinstance(i, int) for i in indices) and
            (np.diff(indices) > 0).all() and 0 <= indices[0] < indices[-1] < calibration['source_frame_count'],
            'expected 24 distinct chronological source frames')
    if sampling == 'consecutive_native24':
        require((np.diff(indices) == 1).all(), 'native24 is not consecutive')
    require(all(i in indices for i in phases.values()), 'selected24 lost manual ED/ES')
    phase_indices = {phase: indices.index(frame) for phase, frame in phases.items()}
    envelope = analysis['coverage_envelope']
    require(indices[0] <= envelope['first_peak'] and envelope['next_peak'] <= indices[-1], 'LV cycle boundaries lost')
    source_time = np.asarray(indices, float)*calibration['frame_time_ms']/1000
    times = source_time-source_time[0]
    directory = destination / pid; directory.mkdir()
    masks_dir = directory / '24帧mask'; masks_dir.mkdir()
    originals = directory / '人工标注'; originals.mkdir()
    shutil.copy2(manual_path, originals / manual_path.name)
    shutil.copy2(calibration_path, directory / calibration_path.name)
    manual_flags = np.isin(indices, list(phases.values()))
    meta = dict(patient=pid, frame_count=24, frame_index_base=0, sampling=sampling,
                is_consecutive_native_frames=bool((np.diff(indices) == 1).all()),
                is_uniform_time_sampling=bool((np.diff(indices) == np.diff(indices)[0]).all()),
                source_native_frame_time_ms=calibration['frame_time_ms'], source_frame_indices=indices,
                source_time_s=source_time.tolist(), time_s=times.tolist(), time_intervals_s=np.diff(source_time).tolist(),
                source_phase_indices=phases, phase_indices=phase_indices, chamber_channels=CHANNELS,
                manual_annotation='人工标注/手工ED_ES源标注.json',
                original_manual_annotation_sha256=digest(manual_path),
                lv_area_cycle=analysis['detection']['selected'], lv_area_coverage_envelope=envelope,
                lv_area_detection_method=analysis['method'], lv_area_warnings=analysis['warnings'],
                images_included=False, masks_spatially_resized=False, masks_interpolated=False, planes={})
    for key in ('1', '2', '3'):
        plane = patient['planes'][key]; name = plane['name']; folder = labels / name
        index_path = folder / '帧索引.json'
        require(digest(index_path) == measurement['views'][key]['index_sha256'], 'source frame index changed')
        index = read_json(index_path)
        require(index['patient'] == pid and str(index['plane']) == key and index['chamber_channels'] == CHANNELS
                and index['frame_index_base'] == 0, 'plane/channel identity mismatch')
        records, masks = [], []
        active = range(4) if key == '1' else (0, 2)
        for output_frame, source_frame in enumerate(indices):
            record = index['frames'][source_frame]
            require(record['frame_index'] == source_frame, 'source frame mismatch')
            raw = (folder / record['mask']).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == measurement['views'][key]['mask_npz_sha256'][source_frame],
                    f'{pid}/{name}/{source_frame}: mask changed since LV cycle location')
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                mask = archive['masks'].copy()
            require(mask.shape == (4, *plane['shape']) and np.isin(mask, [0, 1]).all()
                    and all(mask[c].any() for c in active), 'invalid native mask')
            require(key == '1' or not mask[[1, 3]].any(), 'inactive chambers must remain zero')
            masks.append(mask)
            expected_phase = next((phase for phase, frame in phases.items() if source_frame == frame), None)
            require(record['phase'] == expected_phase, 'manual phase tag mismatch')
            if expected_phase:
                require(record['source'] == 'manual_ed_es' and
                        all(record['chamber_sources'][CHANNELS[c]] == 'manual_ed_es' for c in active),
                        'an ED/ES active chamber is not manual')
                target = originals / name / expected_phase; target.mkdir(parents=True)
                (target / 'masks.npz').write_bytes(raw)
                shutil.copy2(folder / record['contours'], target / 'contours.json')
            records.append(dict(output_frame=output_frame, source_frame=source_frame, phase=expected_phase,
                                source=record['source'], chamber_sources=record['chamber_sources'],
                                source_time_s=float(source_time[output_frame]), time_s=float(times[output_frame])))
        require(len({mask.dtype.str for mask in masks}) == 1, 'native mask dtypes differ')
        stacked = np.stack(masks)
        if resampled:
            with np.load(resampled_root / pid / resampled['planes'][key]['file'], allow_pickle=False) as saved:
                require(np.array_equal(saved['source_frame_indices'], indices) and
                        np.array_equal(saved['masks'], stacked), 'long-cycle masks differ from verified resampled24')
        region = regions[key]
        spacing = np.array([abs(region['delta_y_mm']), abs(region['delta_x_mm'])], float)
        output = masks_dir / f'{name}.npz'
        np.savez_compressed(output, masks=stacked, source_frame_indices=np.asarray(indices, np.int32),
                            source_time_s=source_time, time_s=times, pixel_spacing_yx_mm=spacing,
                            chamber_channels=np.asarray(CHANNELS), is_manual_ed_es=manual_flags,
                            phase_names=np.asarray(['ED', 'ES']),
                            phase_indices=np.asarray([phase_indices['ED'], phase_indices['ES']], np.int32),
                            frame_sources=np.asarray([r['source'] for r in records]),
                            chamber_sources=np.asarray([[r['chamber_sources'][c] for c in CHANNELS] for r in records]))
        with np.load(output, allow_pickle=False) as saved:
            require(np.array_equal(saved['masks'], stacked) and saved['masks'].shape[0] == 24
                    and int(saved['is_manual_ed_es'].sum()) == 2, 'saved24 mask validation failed')
        meta['planes'][key] = dict(name=name, file=str(output.relative_to(directory)),
                                  shape=list(stacked.shape), dtype=stacked.dtype.str,
                                  pixel_spacing_yx_mm=spacing.tolist(), sha256=digest(output),
                                  frames=records, manual_ED_ES_exactly_preserved=True)
    shutil.copytree(base / '患者报告', directory / '患者报告')
    report_hashes = {}
    for original in sorted((base / '患者报告').rglob('*')):
        if original.is_file():
            relative = original.relative_to(base / '患者报告')
            copied = directory / '患者报告' / relative
            require(digest(original) == digest(copied), 'patient report copy differs')
            report_hashes[str(relative)] = digest(copied)
    require(report_hashes, 'patient reports are missing')
    meta['patient_reports'] = {f'患者报告/{name}': sha for name, sha in report_hashes.items()}
    shutil.copy2(cycle_root / 'figures' / f'{pid}.png', directory / 'LV面积周期定位.png')
    if resampled:
        shutil.copy2(resampled_root / pid / 'lv_area_resampling.png', directory / 'LV面积下采样.png')
        shutil.copy2(resampled_root / pid / 'resampling_map.csv', directory / '下采样时间映射.csv')
    write_json(directory / 'metadata.json', meta)
    write_json(originals / 'ED_ES_24帧映射.json', dict(patient=pid, source_phase_indices=phases,
               output_phase_indices=phase_indices, frame_index_base=0, original_manual_file_unchanged=True))
    mapping_rows = [dict(output_frame=j, source_frame=i, source_time_s=float(source_time[j]),
                        time_s=float(times[j]), phase=next((p for p, fi in phases.items() if fi == i), ''))
                    for j, i in enumerate(indices)]
    with (directory / '帧映射.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0])); writer.writeheader(); writer.writerows(mapping_rows)
    cycle = analysis['detection']['selected']
    report = [f'{pid}：24帧mask与患者报告', '================================', '',
              f'采样方式：{sampling}。原始帧闭区间{indices[0]}..{indices[-1]}，输出24帧。',
              f'人工ED：原始{phases["ED"]} -> 输出{phase_indices["ED"]}；人工ES：原始{phases["ES"]} -> 输出{phase_indices["ES"]}。',
              '全部帧号从0开始；显示帧号为此处帧号+1。三个切面的两个人工时相全部原样保留。',
              f'LV面积峰/谷/下一峰：{cycle["first_peak"]}/{cycle["area_minimum"]}/{cycle["next_peak"]}；边界并集{envelope["first_peak"]}..{envelope["next_peak"]}。',
              '三个NPZ均含masks(24,4,H,W)，通道LV/RV/LA/RA；A2C/A3C的RV、RA维持原始零通道。',
              '原始手工标注JSON和六份ED/ES掩膜及轮廓位于人工标注/；患者报告/为原报告完整复制。',
              'metadata.json和帧映射.csv记录原始帧、输出帧、真实时间、像素间距和逐心腔标签来源。',
              '非人工帧仍标识为其原始来源，不提升为人工标注。C11/C15使用下采样，须保留实际时间间隔。', '']
    (directory / '整理报告.rst').write_text('\n'.join(report), encoding='utf-8')
    return dict(patient=pid, sampling=sampling, frame_count=24, source_first=indices[0], source_last=indices[-1],
                source_ED=phases['ED'], source_ES=phases['ES'], output_ED=phase_indices['ED'], output_ES=phase_indices['ES'],
                metadata=f'{pid}/metadata.json', manual_phase_plane_frames_verified=6,
                metadata_sha256=digest(directory / 'metadata.json'),
                original_manual_annotation_sha256=digest(manual_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'real_data')
    parser.add_argument('--cycle-root', type=Path, default=ROOT / 'outputs/patient_lv_area_cycle_20261017/analysis')
    parser.add_argument('--resampled-root', type=Path, default=ROOT / 'outputs/patient_lv_area_resampled24_20261017')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.input / '整理清单.json'; source_manifest = read_json(manifest_path)
    cycle_path = args.cycle_root / 'audit.json'; cycle_audit = read_json(cycle_path)
    located = {p['patient']: p for p in cycle_audit['patients']}
    require(set(located) == {p['patient'] for p in source_manifest['patients']}, 'cohort mismatch')
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.input.resolve()),
            'choose a new output folder outside the input dataset')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{args.output.name}-', dir=args.output.parent))
    try:
        rows = []
        for patient in source_manifest['patients']:
            row = package_patient(args.input, temporary, patient, located[patient['patient']], args.resampled_root, args.cycle_root)
            rows.append(row)
            print(f"OK {row['patient']}: 24 frames, ED={row['output_ED']} ES={row['output_ES']}; {row['sampling']}", flush=True)
        packaged = dict(format='native24_masks_reports_manual_ED_ES_v1', patient_count=len(rows),
                        plane_sequences=len(rows)*3, plane_frames=len(rows)*3*24,
                        manual_phase_plane_frames_verified=len(rows)*6, channels=CHANNELS, frame_index_base=0,
                        sampling_counts=dict(Counter(r['sampling'] for r in rows)),
                        source_manifest_sha256=digest(manifest_path), cycle_audit_sha256=digest(cycle_path),
                        script_sha256=digest(Path(__file__)), patients=rows)
        write_json(temporary / '整理清单.json', packaged)
        with (temporary / '患者汇总.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            fields = [key for key in rows[0] if 'sha256' not in key]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
        readme = ['患者24帧mask、人工ED/ES与报告', '============================', '',
                  f'共{len(rows)}位患者，每位三个切面，每个切面24帧；保留全部已有人工ED/ES帧和原始标注。',
                  '15例为相邻原始24帧；C11、C15为保留人工时相和LV面积峰谷的24帧时间下采样。',
                  '周期根据LV面积定位，不使用心电RR。掩膜均来自原始帧，保持原始尺寸、通道与0/1值。', '',
                  '目录结构::', '', '    C01/', '      24帧mask/                 A4C_四腔心.npz、A2C_两腔心.npz、A3C_三腔心.npz',
                  '      患者报告/                 原始original_report.txt、metrics.json等完整报告文件',
                  '      人工标注/                 完整手工ED_ES源标注.json及三个切面的ED/ES原始掩膜和轮廓',
                  '      metadata.json             新旧ED/ES索引、真实时间、像素间距、逐帧来源和文件校验值',
                  '      帧映射.csv                输出0..23与原始帧号、时间及ED/ES的对应',
                  '      整理报告.rst              本患者选帧与保留标注的说明',
                  '      LV面积周期定位.png         原全序列LV面积诊断图', '',
                  '读取示例::', '', '    import numpy as np',
                  "    with np.load('C01/24帧mask/A4C_四腔心.npz', allow_pickle=False) as z:",
                  "        masks = z['masks']  # (24,4,H,W)，通道LV,RV,LA,RA",
                  "        ed, es = z['phase_indices']  # 输出帧号，0-based",
                  "        original_frames = z['source_frame_indices']",
                  "        times_s = z['time_s']", '',
                  'A2C/A3C仅LV、LA有效，RV、RA仍为原始零通道；不同切面和患者的H/W可能不同。',
                  '人工ED/ES的全部有效心腔已核验为manual_ed_es；其他帧仍保留模型或原有修正来源。',
                  'C11/C15的时间间隔不完全相等，使用time_s/metadata.json时间映射，不应假定24个等间隔相位。',
                  '原始手工JSON及ED/ES掩膜、轮廓保留原始帧号；输出帧号见ED_ES_24帧映射.json。',
                  '本目录包含mask、标注与报告，不包含超声像素图像或原始DICOM；未修改源数据或启动训练。', '']
        (temporary / 'README.rst').write_text('\n'.join(readme), encoding='utf-8')
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f'PACKAGED {len(rows)} patients: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
