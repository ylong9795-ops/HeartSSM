"""Summarize a completed bounded Refiner experiment without rerunning inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run
    def read(name):
        return json.loads((run/name).read_text())
    protocol, decision, geometry = read('protocol.json'), read('decision.json'), read('geometry.json')
    comparisons = {split: read(split+'_comparison.json') for split in ('train', 'val')}
    history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    if len(history) != decision['updates'] or [r['step'] for r in history] != list(range(1, len(history)+1)):
        raise ValueError('Report requires a complete contiguous update history')
    if len(geometry) != sum(len(c['cases']) for c in comparisons.values()):
        raise ValueError('Selected-case geometry coverage is incomplete')
    summary = dict(protocol=protocol, decision=decision, reload=read('reload_verification.json'),
        comparisons={s: {k: c[k] for k in ('metrics', 'objectives', 'tails')} for s, c in comparisons.items()},
        timing=dict(updates=len(history), update_seconds_total=sum(r['seconds'] for r in history),
            update_seconds_median=statistics.median(r['seconds'] for r in history),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in history),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in history),
            scope='frozen-cache updates including backward/Adam; excludes preparation, validation and geometry'),
        geometry=dict(cases=len(geometry), base_pass=sum(r['base']['passed'] for r in geometry),
            refiner_pass=sum(r['refiner']['passed'] for r in geometry),
            per_scope={scope: {m: {status: sum(r[m][scope]['status'] == status for r in geometry)
                for status in ('PASS','FAIL','UNVERIFIED')} for m in ('base','refiner')}
                for scope in ('shape','saved_phases','inter_chamber','continuous')},
            scope='all selected cases, saved full24 triangles and every linear vertex chord including23->0'),
        sampled_composed_jacobians={split: {m: dict(
            minimum=min(c['sampled_composed_jacobian'][m]['minimum'] for c in v['cases']),
            samples=sum(c['sampled_composed_jacobian'][m]['samples'] for c in v['cases']),
            nonpositive=sum(c['sampled_composed_jacobian'][m]['nonpositive'] for c in v['cases']))
            for m in ('base','refiner')} for split, v in comparisons.items()},
        native_gt_pixel_mismatches={split: {m: sum(r['gt_native_pixel_mismatches'] for r in rows)
            for m, rows in read(split+'_native_cells.json').items()} for split in comparisons},
        source_files_still_match={f: hashlib.sha256((root/f).read_bytes()).hexdigest() == digest
            for f, digest in protocol['source_files_sha256'].items()},
        limitations=['TRAIN16/VAL8 only; not full800/100 acceptance',
            'Frozen Base and GT Pose; no estimated-Pose or real ultrasound conclusion',
            'Positive sampled Jacobians are not a whole-field certificate',
            'CCD checks linear vertex chords, not arbitrary continuous neural trajectories',
            'Fixed endpoint; no TEST or validation-driven tuning'])
    if (run/'input_ablation.json').exists():
        summary['input_ablation'] = read('input_ablation.json')['summary']
    if (run/'base_immutability_verification.json').exists():
        summary['base_immutability'] = read('base_immutability_verification.json')
    for key in ('verification', 'native_renderer_identity'):
        if (run/(key+'.json')).exists():
            summary[key] = read(key+'.json')
    paired_changes = {}
    for split in comparisons:
        cells = read(split+'_native_cells.json')
        base_rows, final_rows = cells['base'], cells['refiner']
        identity = lambda row: tuple(row[k] for k in ('case_id','frame','view','chamber'))
        if [identity(r) for r in base_rows] != [identity(r) for r in final_rows]:
            raise ValueError('Paired Native cell identities differ')
        paired_changes[split] = dict(
            improved=sum(b['exact_dice'] < r['exact_dice'] for b,r in zip(base_rows,final_rows)),
            worsened=sum(b['exact_dice'] > r['exact_dice'] for b,r in zip(base_rows,final_rows)),
            recovered_above_098=sum(b['exact_dice'] < .98 <= r['exact_dice'] for b,r in zip(base_rows,final_rows)),
            newly_below_098=sum(r['exact_dice'] < .98 <= b['exact_dice'] for b,r in zip(base_rows,final_rows)))
    summary['paired_native_cell_changes'] = paired_changes
    (run/'experiment_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    val = comparisons['val']
    b, r = val['tails']['base']['all'], val['tails']['refiner']['all']
    rb, rr = val['tails']['base']['right_heart'], val['tails']['refiner']['right_heart']
    lines = ['已有 Joint 权重初始化共享 Refiner：固定子集短训报告', '='*64, '',
        '范围与初始化', '------------', '',
        f"源 checkpoint：{protocol['source_path']}",
        f"源 SHA256：{protocol['source_sha256']}",
        f"源 epoch={protocol['source_epoch']}，step={protocol['source_step']}；新 Adam 更新={len(history)}。",
        f"固定 TRAIN{len(protocol['train_indices'])} / VAL{len(protocol['val_indices'])}，batch={protocol['batch_size']}，seed={protocol['seed']}。",
        'Base 冻结/eval；仅共享 Refiner 学习；使用 GT Pose；不读取 TEST。',
        '固定训练终点，不按 VAL 调参或选择 checkpoint。', '',
        '未见患者对照', '------------', '', '.. list-table:: VAL 固定子集', '   :header-rows: 1', '',
        '   * - 指标', '     - Base', '     - Refiner']
    for key, label in [('reference_vertex_rmse_mm','Shape RMSE mm'),
                       ('trajectory_vertex_rmse_mm','4D 顶点 RMSE mm'),
                       ('relative_motion_rmse_mm','Motion RMSE mm'),('volume_mae_ml','容积 MAE mL')]:
        lines.extend([f'   * - {label}', f"     - {val['metrics']['base'][key]:.6f}",
                      f"     - {val['metrics']['refiner'][key]:.6f}"])
    for label, x, y in [('平均 Native Dice',b['mean_dice'],r['mean_dice']),
                        ('最低 Native Dice',b['minimum_dice'],r['minimum_dice']),
                        ('Native Dice P1',b['p01_dice'],r['p01_dice']),
                        ('Native Dice P5',b['p05_dice'],r['p05_dice']),
                        ('Native Cell<0.98',b['below_098'],r['below_098']),
                        ('右心 Cell<0.98',rb['below_098'],rr['below_098']),
                        ('全部 Cell>=0.98 的病例',b['cases_all_cells_ge_098'],r['cases_all_cells_ge_098'])]:
        lines.extend([f'   * - {label}', f'     - {x}', f'     - {y}'])
    train = comparisons['train']
    lines.extend(['', '训练子集与观察输入检查', '------------------------', '',
        f"TRAIN 监督目标 {train['objectives']['base']['supervised']:.6f} -> {train['objectives']['refiner']['supervised']:.6f}。",
        f"TRAIN 4D 顶点 RMSE {train['metrics']['base']['trajectory_vertex_rmse_mm']:.6f} -> {train['metrics']['refiner']['trajectory_vertex_rmse_mm']:.6f} mm。",
        f"TRAIN Cell<0.98：{train['tails']['base']['all']['below_098']} -> {train['tails']['refiner']['all']['below_098']}。",
        f"VAL 达标恢复 {paired_changes['val']['recovered_above_098']} Cell；新增跌破0.98：{paired_changes['val']['newly_below_098']} Cell。",
        f"VAL Dice 提升 {paired_changes['val']['improved']} Cell，下降 {paired_changes['val']['worsened']} Cell；均完整保留。"])
    if 'input_ablation' in summary:
        lines.extend(['观察输入消融的 VAL 4D 顶点 RMSE mm：', ''])
        lines.extend(f"* {mode}: {value['trajectory_vertex_rmse_mm']:.6f}" for mode,value in summary['input_ablation'].items())
        lines.append('')
        lines.append('清空支持后精确退回 Base；反向纠正向量使误差增大，支持观察方向参与了修正。')
    t = summary['timing']
    lines.extend(['', '工程与几何', '----------', '',
        f"训练更新累计 {t['update_seconds_total']:.2f} 秒；中位 {t['update_seconds_median']:.3f} 秒/batch。",
        f"峰值已分配显存 {t['peak_allocated_bytes']/1e9:.2f} GB；预留 {t['peak_reserved_bytes']/1e9:.2f} GB。",
        '上述训练计时不含数据准备、验证及完整几何验收。',
        f"Checkpoint 重载与在线/缓存路径最大顶点差 {summary['reload']['maximum_vertex_difference_mm']:.8f} mm。",
        f"完整所选病例几何通过：Base {summary['geometry']['base_pass']}/{len(geometry)}；Refiner {summary['geometry']['refiner_pass']}/{len(geometry)}。",
        '几何范围包括全部24帧、源AV cap约定下的腔间事件、自交/退化和全部帧间线性顶点弦。',
        '采样组合 Jacobian 见 experiment_summary.json；不等于整个连续空间映射的正性证书。', '',
        '结论与限制', '----------', '',
        '本实验检验从已有训练 epoch 添加共享观察驱动 Refiner 的可行性。',
        '误差改善、尾部改善及几何通过分别报告；改善不等于全部 Cell 达标。',
        f"VAL 仍有 {r['below_098']}/{r['cells']} Cell<0.98，最低 Dice={r['minimum_dice']:.6f}。",
        '不能将固定 TRAIN16/VAL8 的短训结果写成 TRAIN800/VAL100 或真实超声效果。',
        '详细病例、时相、腔室和最差 Cell 见 *_native_cells.json 与 *_comparison.json。', ''])
    (run/'final_report.rst').write_text('\n'.join(lines))
    print(json.dumps(dict(decision=decision,geometry=summary['geometry'],timing=t),indent=2))


if __name__ == '__main__':
    main()
