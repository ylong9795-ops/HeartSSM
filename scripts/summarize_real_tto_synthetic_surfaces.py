"""Summarize saved post-endpoint synthetic surface errors without optimization.

No GT is passed to a model or optimizer. Surface distances are the formal
evaluator's saved vertex-to-triangle measurements, not newly recomputed ones.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('reference_vertex_rmse_mm', 'reference_ASSD_mm', 'reference_HD95_mm',
           'trajectory_vertex_rmse_mm', 'relative_motion_rmse_mm')
CHAMBERS = ('LV', 'LA', 'RV', 'RA')
DEFINITION = 'symmetric vertex-sampled point-to-opposite-triangle-surface distance; no GT used for optimization'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def nonnegative(value):
    if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid saved distance: ' + repr(value))


def summarize(values):
    before = [v['before'] for v in values]
    after = [v['after'] for v in values]
    delta = [b - a for a, b in zip(before, after)]
    return dict(patients=len(values), before_median=median(before), after_median=median(after),
                paired_delta_median=median(delta), decreased=sum(x < 0 for x in delta),
                unchanged=sum(x == 0 for x in delta), increased=sum(x > 0 for x in delta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/real_tto_joint_v1_20261016')
    args = parser.parse_args()
    root = Path(args.output)
    if not root.is_absolute():
        root = ROOT / root
    protocol = read(root / 'protocol.json')
    required = protocol['scientific_config']['synthetic']['cases']
    records, rows, pending = [], [], []
    for ordinal in range(required):
        name = f'SYN_{ordinal:03d}'
        folder = root / 'synthetic_hidden_gt/patients' / name
        endpoint = folder / 'states/export.pth'
        completion = folder / 'export/completed.json'
        official_path = folder / 'hidden_gt_evaluation.json'
        independent_path = root / 'independent_audit/synthetic' / name / 'three_stage_hidden_gt.json'
        if not all(p.exists() for p in (endpoint, completion, official_path, independent_path)):
            pending.append(name)
            continue
        official, independent = read(official_path), read(independent_path)
        truth = root / 'synthetic_hidden_gt/hidden_truth' / (name + '.pth')
        endpoint_sha, truth_sha = digest(endpoint), digest(truth)
        if (endpoint_sha != independent['endpoint_sha256'] or truth_sha != independent['hidden_truth_sha256']
                or official['case_id'] != independent['case_id']):
            raise ValueError('Saved evaluation identity changed: ' + name)
        if not (official['status'] == independent['status'] == read(completion)['status'] == 'COMPLETE'):
            raise ValueError('Synthetic endpoint is not admitted: ' + name)
        stages = official['metrics']
        for stage, independent_stage in (('before', 'initial'), ('after', 'joint')):
            for key in METRICS:
                nonnegative(stages[stage][key])
            comparisons = (('reference_vertex_rmse_mm', 'reference_rmse_mm'),
                           ('trajectory_vertex_rmse_mm', 'trajectory_rmse_mm'),
                           ('relative_motion_rmse_mm', 'relative_motion_rmse_mm'))
            for saved_key, independent_key in comparisons:
                if abs(stages[stage][saved_key] - independent['stages'][independent_stage][independent_key]) > 1e-5:
                    raise ValueError('Independent endpoint RMSE disagrees: ' + name)
            for chamber in CHAMBERS:
                surface = stages[stage]['by_chamber'][chamber]
                if surface['definition'] != DEFINITION:
                    raise ValueError('Surface-distance definition changed: ' + name)
                for key in ('reference_vertex_rmse_mm', 'ASSD_mm', 'HD95_mm'):
                    nonnegative(surface[key])
                motion = stages[stage]['gt_trajectory_motion_volume_by_chamber'][chamber]
                for key in ('per_frame_trajectory_rmse_mm', 'per_frame_relative_motion_rmse_mm'):
                    if len(motion[key]) != 24:
                        raise ValueError('Incomplete full-cycle error: ' + name)
                    for value in motion[key]:
                        nonnegative(value)
        row = dict(patient=name)
        for key in METRICS:
            row[key + '_before'] = stages['before'][key]
            row[key + '_after'] = stages['after'][key]
            row[key + '_delta'] = stages['after'][key] - stages['before'][key]
        rows.append(row)
        records.append(dict(patient=name, endpoint_sha256=endpoint_sha, hidden_truth_sha256=truth_sha,
                            saved_evaluation_sha256=digest(official_path),
                            independent_evaluation_sha256=digest(independent_path),
                            metrics={key: dict(before=stages['before'][key], after=stages['after'][key]) for key in METRICS},
                            by_chamber={c: {key: dict(before=stages['before']['by_chamber'][c][key],
                                                     after=stages['after']['by_chamber'][c][key])
                                            for key in ('reference_vertex_rmse_mm', 'ASSD_mm', 'HD95_mm')}
                                        for c in CHAMBERS}))
    aggregate = {key: summarize([r['metrics'][key] for r in records]) for key in METRICS} if records else {}
    chamber_summary = {c: {key: summarize([r['by_chamber'][c][key] for r in records])
                           for key in ('reference_vertex_rmse_mm', 'ASSD_mm', 'HD95_mm')}
                       for c in CHAMBERS} if records else {}
    output = root / 'independent_audit'
    output.mkdir(parents=True, exist_ok=True)
    result = dict(updated=time.time(), summarizer_sha256=digest(Path(__file__)),
                  protocol_sha256=digest(root / 'protocol.json'), source_sha256=protocol['source_sha256'],
                  distance_evaluator_source_sha256=protocol['sources']['tto/synthetic_oracle.py'],
                  independent_rmse_agreement_tolerance_mm=1e-5, required=required,
                  summarized=len(records), pending=pending, all_required_metric_records_available=len(records) == required,
                  aggregate=aggregate, by_chamber=chamber_summary, patients=records,
                  distance_definition=DEFINITION, units='mm; original physical coordinates, no GT alignment',
                  scope='Saved formal post-endpoint surface metrics; current endpoint/truth hashes and independent RMSE checked. '
                        'No new point-to-surface computation, optimization, gate or geometry decision. Synthetic TRAIN only; no real 3D accuracy claim.')
    temporary = output / 'synthetic_surface_summary.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output / 'synthetic_surface_summary.json')
    if rows:
        with (output / 'synthetic_surface_cases.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 合成隐藏GT：参考表面与完整周期误差', '',
             f'已核对{len(records)}/{required}例提交端点、隐藏真值SHA及独立RMSE。此表汇总正式评价器已保存的距离，不是本次重新计算表面距离。', '',
             'ASSD/HD95使用双向顶点采样到对侧三角形表面的距离；不是精确面积积分，也不是对应顶点误差。四腔总体指标由各腔距离样本合并计算，不能用四腔HD95的均值替代。单位均为mm，使用原物理坐标，不做GT配准。', '',
             '| 病例 | 参考顶点RMSE：初始→Joint | 参考ASSD：初始→Joint | 参考HD95：初始→Joint | 完整周期RMSE：初始→Joint | 相对运动RMSE：初始→Joint |',
             '|---|---:|---:|---:|---:|---:|']
    for record in records:
        pairs = [f"{record['metrics'][key]['before']:.6f}→{record['metrics'][key]['after']:.6f}" for key in METRICS]
        lines.append('| ' + record['patient'] + ' | ' + ' | '.join(pairs) + ' |')
    lines += ['', '病例中位数及配对方向如下；下降/不变/上升由保存值直接比较，不设事后改善阈值。', '',
              '| 指标 | 初始中位数 | Joint中位数 | 配对变化中位数 | 下降 / 不变 / 上升 |',
              '|---|---:|---:|---:|---:|']
    for key, value in aggregate.items():
        lines.append(f"| {key} | {value['before_median']:.6f} | {value['after_median']:.6f} | {value['paired_delta_median']:.6f} | {value['decreased']} / {value['unchanged']} / {value['increased']} |")
    lines += ['', '各腔参考表面误差的病例中位数：', '',
              '| 腔室 | 顶点RMSE：初始→Joint | ASSD：初始→Joint | HD95：初始→Joint |', '|---|---:|---:|---:|']
    for chamber, values in chamber_summary.items():
        pairs = [f"{values[key]['before_median']:.6f}→{values[key]['after_median']:.6f}" for key in ('reference_vertex_rmse_mm', 'ASSD_mm', 'HD95_mm')]
        lines.append('| ' + chamber + ' | ' + ' | '.join(pairs) + ' |')
    lines += ['', '这些指标检验合成TRAIN闭环的实际恢复误差；不能据此证明真实患者三维Shape/Pose准确性或未见病例泛化。Pose初始化冻结Shape和Motion，因此其表面/轨迹误差与初始值一致。此汇总不改变候选选择、门控、预算或安全阈值。', '',
              '逐例原始指标见 `synthetic_hidden_gt/patients/*/hidden_gt_evaluation.json`，本次身份检查、分腔汇总及CSV见同目录 `synthetic_surface_summary.json`、`synthetic_surface_cases.csv`。']
    (output / 'SYNTHETIC_SURFACE_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(summarized=len(records), required=required, aggregate=aggregate)))


if __name__ == '__main__':
    main()
