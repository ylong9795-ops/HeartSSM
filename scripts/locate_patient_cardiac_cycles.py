#!/usr/bin/env python3
"""Locate mechanical cycles from native LV mask area, without ECG/RR input.

Three view-specific area curves are median-filtered (3 samples), smoothed
(Gaussian sigma=0.7 frames), and normalized by their own 5th/95th percentiles.
Their pointwise median supplies the consensus area curve. A complete cycle
is two adjacent prominent area maxima with an intervening minimum, supported
by at least two views. Manual ED/ES are immutable anchors, never replacements
for detected extrema. Filtering affects detection only, never images/masks.

The native24 window must cover the detected peak-to-peak cycle AND both manual
frames. Boundary margins are allowed. Small detection parameter variations
are included in the coverage envelope; cycle detection does not use 24 as a
period prior. This estimates mechanical cycles from segmentation, not 3D GT.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(median_size=3, sigma_frames=.7, prominence=.25, min_peak_distance=3,
              view_match_period_fraction=.20, anchor_period_fraction=.25,
              minimum_supporting_views=2)
VARIANTS = ((.7, .25), (.5, .25), (1., .25), (.7, .20), (.7, .30))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def area_trace(values, sigma=.7, prominence=.25):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 3 and np.isfinite(values).all()
            and (values > 0).all(), 'invalid/empty LV area observations')
    smooth = gaussian_filter1d(median_filter(values, size=CONFIG['median_size'], mode='nearest'), sigma)
    lo, hi = np.percentile(smooth, [5, 95])
    usable = bool(hi-lo > max(1e-8, abs(lo)*1e-6))
    normalized = (smooth-lo)/(hi-lo) if usable else np.zeros_like(smooth)
    peaks, _ = find_peaks(normalized, prominence=prominence, distance=CONFIG['min_peak_distance'])
    valleys, _ = find_peaks(-normalized, prominence=prominence, distance=CONFIG['min_peak_distance'])
    return dict(smoothed_area_px=smooth.tolist(), normalized_area=normalized.tolist(), usable=usable,
                peaks=peaks.tolist(), valleys=valleys.tolist(),
                percentile5=float(lo), percentile95=float(hi))


def detect_area_cycles(areas, ed, es, sigma=.7, prominence=.25):
    traces = {key: area_trace(a, sigma, prominence) for key, a in areas.items()}
    usable = [t['normalized_area'] for t in traces.values() if t['usable']]
    consensus = np.median(usable, axis=0) if len(usable) >= 2 else np.zeros(len(areas['1']))
    peaks, _ = find_peaks(consensus, prominence=prominence, distance=CONFIG['min_peak_distance'])
    valleys, _ = find_peaks(-consensus, prominence=prominence, distance=CONFIG['min_peak_distance'])
    period = float(np.median(np.diff(peaks))) if len(peaks) >= 2 else None
    tolerance = max(2., CONFIG['view_match_period_fraction']*period) if period else 2.
    cycles = []
    for index, (start, end) in enumerate(zip(peaks[:-1], peaks[1:])):
        troughs = valleys[(valleys > start) & (valleys < end)]
        if not len(troughs):
            continue
        valley = int(min(troughs, key=lambda i: consensus[i]))
        supporting, matches = [], {}
        for key, trace in traces.items():
            pairs = [(a, b) for a, b in zip(trace['peaks'][:-1], trace['peaks'][1:])
                     if abs(a-start) <= tolerance and abs(b-end) <= tolerance
                     and any(a < v < b and abs(v-valley) <= tolerance for v in trace['valleys'])]
            if pairs:
                a, b = min(pairs, key=lambda pair: abs(pair[0]-start)+abs(pair[1]-end))
                supporting.append(key)
                matches[key] = dict(first_peak=a, next_peak=b)
        cycles.append(dict(cycle_index=index, first_peak=int(start), area_minimum=valley,
                           next_peak=int(end), frame_intervals=int(end-start), frame_count=int(end-start+1),
                           supporting_views=supporting, view_peak_matches=matches,
                           supported=len(supporting) >= CONFIG['minimum_supporting_views']))
    # Associate the mechanical cycle with the annotated systole before testing
    # its length. Never select an unrelated shorter beat just to fit native24.
    candidates = [c for c in cycles if c['first_peak'] < es < c['next_peak']]
    selected = min(candidates, key=lambda c: abs(c['first_peak']-ed)) if candidates else None
    alignment_limit = max(2., CONFIG['anchor_period_fraction']*period) if period else 2.
    reliable = bool(selected and selected['supported'] and
                    abs(selected['first_peak']-ed) <= alignment_limit and
                    abs(selected['area_minimum']-es) <= alignment_limit)
    return dict(traces=traces, consensus=consensus.tolist(), peaks=peaks.tolist(), valleys=valleys.tolist(),
                typical_period_frames=period, view_match_tolerance_frames=tolerance,
                manual_alignment_limit_frames=alignment_limit, cycles=cycles,
                selected=selected, reliable=reliable)


def locate_cycles(areas, ed, es, window, frame_time_ms=1., sources=None):
    """Return native area-defined cycles and a conservative original24 crop."""
    require(set(areas) == {'1', '2', '3'}, 'LV areas from all three views are required')
    n = len(areas['1'])
    require(all(len(a) == n for a in areas.values()), 'LV area sequence lengths differ')
    require(isinstance(ed, (int, np.integer)) and isinstance(es, (int, np.integer))
            and 0 <= ed < n and 0 <= es < n and ed != es, 'invalid manual ED/ES')
    frames = np.asarray(window)
    require(frames.shape == (48,) and np.issubdtype(frames.dtype, np.integer)
            and (np.diff(frames) == 1).all() and frames[0] >= 0 and frames[-1] < n,
            'expected 48 consecutive original frame indices')
    require(ed in frames and es in frames, '48-frame window lost manual ED/ES')
    require(np.isfinite(frame_time_ms) and frame_time_ms > 0, 'invalid frame time')
    detection = detect_area_cycles(areas, ed, es)
    variations = []
    for sigma, prominence in VARIANTS:
        estimate = detection if (sigma, prominence) == VARIANTS[0] else detect_area_cycles(areas, ed, es, sigma, prominence)
        cycle = estimate['selected']
        variations.append(dict(sigma_frames=sigma, prominence=prominence, reliable=estimate['reliable'],
                               first_peak=cycle['first_peak'] if cycle else None,
                               area_minimum=cycle['area_minimum'] if cycle else None,
                               next_peak=cycle['next_peak'] if cycle else None))
    selected = detection['selected']
    reliable = all(v['reliable'] for v in variations)
    available = [v for v in variations if v['first_peak'] is not None]
    envelope = dict(first_peak=min(v['first_peak'] for v in available),
                    next_peak=max(v['next_peak'] for v in available)) if available else None
    required_first = min(ed, es, envelope['first_peak']) if envelope else None
    required_last = max(ed, es, envelope['next_peak']) if envelope else None
    required = required_last-required_first+1 if envelope else None
    crop = None
    if reliable and required <= 24:
        low, high = max(0, required_last-23), min(required_first, n-24)
        require(low <= high, 'invalid native24 bounds')
        start = max(low, min(high, (required_first+required_last-23)//2))
        crop = dict(first_source_frame=start, last_source_frame=start+23,
                    source_frame_indices=list(range(start, start+24)),
                    manual_phase_indices={'ED': int(ed-start), 'ES': int(es-start)},
                    cycle_index=selected['cycle_index'], area_cycle_first=selected['first_peak'],
                    area_cycle_minimum=selected['area_minimum'], area_cycle_next=selected['next_peak'],
                    coverage_envelope=envelope,
                    inside_existing48=bool(frames[0] <= start and start+23 <= frames[-1]))
    warnings = []
    for key, area in areas.items():
        if area[es] >= area[ed]:
            warnings.append(f'view {key}: manual ES LV area >= manual ED LV area; labels unchanged')
        if selected and key not in selected['supporting_views']:
            warnings.append(f'view {key}: area extrema do not support the consensus cycle')
        if sources and any(s not in ('manual_ed_es', 'manual_review') for s in sources[key]):
            warnings.append(f'view {key}: nonmanual segmentation contributes to the area curve')
    complete = [c['cycle_index'] for c in detection['cycles'] if c['supported'] and
                frames[0] <= c['first_peak'] and c['next_peak'] <= frames[-1]]
    return dict(method='native_multiview_lv_area_peak_minimum_peak_v1', config=CONFIG.copy(),
                frame_index_base=0, frame_time_ms=float(frame_time_ms), source_frame_count=n,
                manual_phase_indices={'ED': int(ed), 'ES': int(es)}, detection=detection,
                sensitivity=variations, coverage_envelope=envelope,
                minimum_frames_for_cycle_and_manual=required,
                existing48=dict(first_source_frame=int(frames[0]), last_source_frame=int(frames[-1]),
                                complete_lv_cycle_indices=complete, complete_lv_cycle_count=len(complete),
                                unsupported_peak_pair_count=int(sum(not c['supported'] and frames[0] <= c['first_peak']
                                                                    and c['next_peak'] <= frames[-1] for c in detection['cycles']))),
                native24=crop, warnings=warnings,
                status=('accepted_lv_area_native24' if crop else
                        'rejected_native24_too_short' if reliable else 'review_required_lv_area'),
                coverage_rule='same native24 contains immutable manual ED/ES and complete LV area peak-minimum-peak cycle')


def measure_patient(root, existing, patient):
    pid = patient['patient']
    labels = root / pid / '患者标签（每一帧）'
    calibration_path = labels / '原始切面与像素间距.json'
    calibration = read_json(calibration_path)
    manual_path = labels / '手工ED_ES源标注.json'
    manual = read_json(manual_path)
    require(digest(manual_path) == patient['manual_annotation_sha256'], 'manual annotation hash mismatch')
    export = read_json(existing / pid / 'metadata.json')
    require(calibration['patient_id'] == manual['patient'] == export['patient'] == pid, 'patient identity mismatch')
    phases = {ph: manual['phases'][ph]['frame'] for ph in ('ED', 'ES')}
    require(phases == patient['manual_phase_frame_indices'] == export['source_phase_indices'], 'manual phase mismatch')
    regions = {str(r['plane']): r for r in calibration['regions']}
    row = dict(patient=pid, frame_count=calibration['source_frame_count'], frame_time_ms=calibration['frame_time_ms'],
               manual_phase_indices=phases, manual_sha256=digest(manual_path),
               calibration_sha256=digest(calibration_path), existing48=export['source_frame_indices'], views={})
    for key in ('1', '2', '3'):
        plane = patient['planes'][key]
        folder = labels / plane['name']
        index_path = folder / '帧索引.json'
        index = read_json(index_path)
        require(index['patient'] == pid and str(index['plane']) == key and index['frame_index_base'] == 0,
                'plane identity/index base mismatch')
        require(index['frame_count'] == len(index['frames']) == row['frame_count'], 'frame count mismatch')
        areas, sources, hashes = [], [], []
        channel = index['chamber_channels'].index('LV')
        for i, frame in enumerate(index['frames']):
            require(frame['frame_index'] == i and np.isclose(frame['time_s']*1000, i*row['frame_time_ms']),
                    'frame index/time mismatch')
            path = (folder / frame['mask']).resolve()
            require(path.is_relative_to(folder.resolve()), 'mask path leaves plane folder')
            raw = path.read_bytes()
            hashes.append(hashlib.sha256(raw).hexdigest())
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                mask = archive['masks'][channel]
            require(list(mask.shape) == plane['shape'] and np.isin(mask, [0, 1]).all() and mask.any(),
                    'invalid LV mask')
            areas.append(int(mask.sum()))
            sources.append(frame['chamber_sources']['LV'])
        require(all(sources[i] == 'manual_ed_es' for i in phases.values()), 'manual LV provenance mismatch')
        scale = abs(regions[key]['delta_x_mm']*regions[key]['delta_y_mm'])
        row['views'][key] = dict(name=plane['name'], area_px=areas, area_mm2=(np.asarray(areas)*scale).tolist(),
                                 lv_sources=sources, mask_npz_sha256=hashes, index_sha256=digest(index_path))
    return row


def summary_row(patient):
    r = patient['analysis']; cycle = r['detection']['selected']; crop = r['native24']
    return dict(patient=patient['patient'], ED=r['manual_phase_indices']['ED'], ES=r['manual_phase_indices']['ES'],
                existing48_complete_lv_cycles=r['existing48']['complete_lv_cycle_count'],
                area_first_peak=cycle['first_peak'] if cycle else None,
                area_minimum=cycle['area_minimum'] if cycle else None,
                area_next_peak=cycle['next_peak'] if cycle else None,
                envelope_first=r['coverage_envelope']['first_peak'] if r['coverage_envelope'] else None,
                envelope_last=r['coverage_envelope']['next_peak'] if r['coverage_envelope'] else None,
                minimum_required_frames=r['minimum_frames_for_cycle_and_manual'],
                native24_first=crop['first_source_frame'] if crop else None,
                native24_last=crop['last_source_frame'] if crop else None,
                inside_existing48=crop['inside_existing48'] if crop else None,
                supporting_views=','.join(cycle['supporting_views']) if cycle else '', status=r['status'])


def write_report(destination, audit):
    (destination / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    rows = [summary_row(p) for p in audit['patients']]
    if rows:
        with (destination / 'patients.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = ['按LV面积定位原始机械心动周期', '==============================', '',
            '本报告取代此前RR覆盖结论。周期定位与筛选不读取或使用ECG、R波、心率或RR。',
            'LV面积=原始二值掩膜像素数；mm²面积另由原始像素间距换算。全序列、三切面逐帧测量。',
            '人工ED/ES保持原帧号和标签，不将自动检测极值冒充人工标注。全部帧号从0开始，区间含两端。', '',
            '方法', '----', '',
            '每切面先做3点中值滤波与sigma=0.7帧的高斯平滑，再按各自P5/P95归一化，取三曲线逐帧中位数。',
            '只在分析曲线上滤波；原始帧、图像和标签没有插值、重采样、复制或改写。',
            '相邻显著面积峰及中间谷构成一个机械周期候选；峰显著性0.25、最小峰间距3帧。',
            '至少两切面的相邻峰与中间谷须支持该周期；匹配容差为max(2帧,全序列中位峰间距的20%)。',
            '按人工ES所在的收缩-舒张段关联周期，并检查面积峰/谷与人工ED/ES的偏差。',
            '关联容差为max(2帧,全序列中位峰间距的25%)；不使用24帧长度来寻找更短的周期。',
            '额外检查sigma=0.5/1.0和显著性=0.20/0.30；最终24帧覆盖这些估计边界的并集及人工ED/ES。',
            '24帧允许周期外余量。面积曲线有模型标签参与，结果是LV面积定义的周期估计，不是独立生理真值。', '',
            '.. list-table:: 逐例定位', '   :header-rows: 1', '',
            '   * - 患者', '     - 人工ED/ES', '     - 旧48帧完整面积周期数', '     - 面积峰/谷/下一峰',
            '     - 敏感性边界并集', '     - 连同人工帧最少帧数', '     - 连续24帧 / 结论']
    for row in rows:
        values = [row['patient'], f"{row['ED']}/{row['ES']}", row['existing48_complete_lv_cycles'],
                  f"{row['area_first_peak']}/{row['area_minimum']}/{row['area_next_peak']}",
                  f"{row['envelope_first']}..{row['envelope_last']}", row['minimum_required_frames'],
                  f"{row['native24_first']}..{row['native24_last']}" if row['native24_first'] is not None else row['status']]
        text += [f"   {'* -' if i == 0 else '  -'} {value}" for i, value in enumerate(values)]
    text += ['', '结果与证据', '----------', '', json.dumps(audit['summary'], ensure_ascii=False), '',
             'audit.json保存全部面积曲线、峰谷、跨视图支持和参数敏感性，不因24帧上限截断周期。',
             '旧48帧周期数只统计两端峰均位于窗口内、且至少两视图支持的完整峰-谷-峰；未确认候选单列。',
             'figures/保存每例原始面积与检测曲线；lv_area_overview.png为全队列概览。',
             'manual ES面积不小于ED、视图不支持和非人工标签来源均在audit.json的warnings中保留。',
             '本次只定位与校验，未重新导出图像或启动训练。', '']
    (destination / 'report.rst').write_text('\n'.join(text), encoding='utf-8')


def write_plots(destination, patients):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = destination / 'figures'; figures.mkdir()
    overview, panels = plt.subplots(6, 3, figsize=(21, 24))
    for panel, patient in zip(panels.flat, patients):
        r = patient['analysis']; detection = r['detection']; ed, es = r['manual_phase_indices'].values()
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for key, view in patient['measurements']['views'].items():
            axes[0].plot(view['area_mm2'], label=view['name'].split('_')[0], alpha=.8)
            for axis in (axes[1], panel):
                axis.plot(detection['traces'][key]['normalized_area'], lw=1, alpha=.5)
        for axis in (axes[1], panel):
            z = np.asarray(detection['consensus']); axis.plot(z, color='black', lw=2, label='median LV area')
            axis.scatter(detection['peaks'], z[detection['peaks']], s=15, color='black')
            for peak in detection['peaks']:
                axis.text(peak, z[peak]+.035, str(peak), fontsize=8)
        for axis in (*axes, panel):
            axis.axvspan(r['existing48']['first_source_frame'], r['existing48']['last_source_frame'], color='gray', alpha=.07)
            axis.axvline(ed, color='#005db4', linestyle='--', label=f'manual ED {ed}')
            axis.axvline(es, color='#c93232', linestyle='--', label=f'manual ES {es}')
            cycle = detection['selected']
            if cycle:
                axis.axvspan(cycle['first_peak'], cycle['next_peak'], color='#218348', alpha=.12)
            if r['native24']:
                crop = r['native24']; axis.axvspan(crop['first_source_frame'], crop['last_source_frame'],
                                                  facecolor='none', edgecolor='#7844ac', linewidth=1.5)
            axis.set_xlabel('Original frame (zero-based)')
        axes[0].set_ylabel('Original LV mask area (mm2)'); axes[0].legend(ncol=3, fontsize=8)
        axes[1].set_ylabel('Normalized area (detection only)')
        title = f"{patient['patient']}: ED={ed}, ES={es}; {r['status']}"
        panel.set_title(title, fontsize=9); axes[0].set_title(title)
        figure.tight_layout()
        for suffix in ('png', 'svg'):
            figure.savefig(figures / f"{patient['patient']}.{suffix}", dpi=130)
        plt.close(figure)
    for axis in list(panels.flat)[len(patients):]:
        axis.axis('off')
    overview.tight_layout(); overview.savefig(destination / 'lv_area_overview.png', dpi=130); plt.close(overview)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=ROOT / 'real_data')
    parser.add_argument('--existing48', type=Path, default=ROOT / 'outputs/real_data_48frames')
    parser.add_argument('--areas-json', type=Path, help='Replay saved area measurements; recorded mask hashes remain attached')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.input / '整理清单.json'; manifest = read_json(manifest_path)
    require(not args.output.resolve().is_relative_to(args.input.resolve()) and
            not args.output.resolve().is_relative_to(args.existing48.resolve()), 'output must be outside source datasets')
    args.output.mkdir(parents=True, exist_ok=False)
    measured = read_json(args.areas_json)['patients'] if args.areas_json else [
        measure_patient(args.input, args.existing48, p) for p in manifest['patients']]
    require({p['patient'] for p in measured} == {p['patient'] for p in manifest['patients']}
            and len(measured) == len(manifest['patients']), 'measurement cohort mismatch')
    patients = []
    for measurement in measured:
        pid = measurement['patient']; labels = args.input / pid / '患者标签（每一帧）'
        require(digest(labels / '手工ED_ES源标注.json') == measurement['manual_sha256'], 'manual source changed')
        require(digest(labels / '原始切面与像素间距.json') == measurement['calibration_sha256'], 'calibration changed')
        for view in measurement['views'].values():
            require(digest(labels / view['name'] / '帧索引.json') == view['index_sha256'], 'plane index changed')
        phases = measurement['manual_phase_indices']
        result = locate_cycles({k: v['area_px'] for k, v in measurement['views'].items()}, phases['ED'], phases['ES'],
                               measurement['existing48'], measurement['frame_time_ms'],
                               sources={k: v['lv_sources'] for k, v in measurement['views'].items()})
        row = dict(patient=pid, measurements=measurement, analysis=result); patients.append(row)
        print(json.dumps(summary_row(row), ensure_ascii=False), flush=True)
    audit = dict(version='lv-area-native24-cycle-coverage-v1', script_sha256=digest(Path(__file__)),
                 manifest_sha256=digest(manifest_path), areas_json_sha256=digest(args.areas_json) if args.areas_json else None,
                 measurement_mode='replay_recorded_areas_with_mask_hashes' if args.areas_json else 'measured_original_masks',
                 patients=patients, summary=dict(patient_count=len(patients), statuses=dict(Counter(p['analysis']['status'] for p in patients)),
                     accepted=[p['patient'] for p in patients if p['analysis']['status'] == 'accepted_lv_area_native24'],
                     rejected=[p['patient'] for p in patients if p['analysis']['status'] == 'rejected_native24_too_short'],
                     review_required=[p['patient'] for p in patients if p['analysis']['status'] == 'review_required_lv_area']))
    write_report(args.output, audit); write_plots(args.output, patients)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
