"""Refresh the Chinese interim report from committed results, never optimization.

This report preserves missing/failed cases and does not certify goal completion.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tto.config import load_config,resolve
from tto.experiments import summarize
from tto.evaluator import dice_stats
from tto.io import read_json,sha256,write_json
from scripts.audit_real_tto_joint import load_cells


def native_groups(rows, trusted_threshold):
    subsets=[('监督集合','全部标注',rows),
             ('监督集合','可用监督',[row for row in rows if row['used_for_optimization']]),
             ('监督集合','高可信',[row for row in rows if row['confidence']>=trusted_threshold])]
    for dimension,field,labels in (
        ('视图','view',('A2C','A3C','A4C')),
        ('腔室','chamber',('LV','LA','RV','RA')),
        ('标签来源','label_source',sorted({row['label_source'] for row in rows}))):
        members=[(dimension,label,[row for row in rows if row[field]==label]) for label in labels]
        if sum(len(group) for _,_,group in members)!=len(rows):
            raise ValueError('Native grouping omits observations: '+field)
        subsets.extend(members)
    records=[]
    for dimension,label,members in subsets:
        stats=dice_stats(members);count=stats['cells']
        stats.update(fraction_ge095=stats['ge095']/count if count else None,
                     fraction_ge098=stats['ge098']/count if count else None)
        records.append(dict(dimension=dimension,label=label,stats=stats))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/real_tto_joint_v1.yaml')
    parser.add_argument('--output')
    args=parser.parse_args();config=load_config(args.config)
    root=resolve(args.output or config['output']['root'])
    comparison=summarize(root,config)
    path=root/'REAL_TTO_JOINT_REPORT.md'
    arm_table=path.read_text().split('\n',1)[1]
    budget=config['joint']['updates']
    arm_table=arm_table.replace(
        '所有实验臂使用同一合成先验、同一数据和相同 Joint 目标评估次数；Alternate 每次只更新一个参数组，其实际更新次数单独记录。',
        f'A组不优化；B/C/D/E/F/G各{budget}次Joint候选提议。D组Pose/Shape分别有{(budget+1)//2}/{budget//2}次提议；实际获准次数取决于固定安全规则，不能把提议预算视为获准更新次数。')
    arm_table=arm_table.replace(
        'ED-only 的梯度、候选选择和二维尾部准入仅使用参考观察；其余时相在提交端点后独立评价。',
        'G组共享完整周期Encoder和共同Pose初始化，仅Joint损失、候选选择及尾部准入为ED-only；其余时相在端点后评价。')
    lines=['# 全周期 Shape–Pose Joint TTO：当前实验报告','',
        f'更新于{time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())} UTC。本文是基于当前已提交产物的进展报告，不代表全部实验或最终需求审计完成。','',
        '合成16例闭环、真实C01/C02 Smoke与完成阶段恢复的具体证据分别见[合成报告](independent_audit/SYNTHETIC_RECOVERY_REPORT.md)和[Smoke报告](independent_audit/CLINICAL_SMOKE_REPORT.md)。合成Pose恢复与Shape保持不等于Shape精度提高或未见患者泛化。','',
        '| 主实验患者 | 状态 | Cell | Mean：Pose初始化→Joint | Minimum：Pose初始化→Joint | Joint P01 / P05 | ≥0.98 Cell | 可信Cell | Native插值支持槽/24 | Joint获准/尝试 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    all_cells=[];patient_records=[]
    for patient in config['experiment']['cohort']:
        folder=root/'patients'/patient
        committed=(folder/'export/completed.json').exists() and (folder/'states/export.pth').exists()
        if not committed:
            failure=folder/'failure.json'
            status='失败' if failure.exists() else '待完成'
            lines.append(f'| {patient} | {status} | — | — | — | — | — | — | — | — |')
            record=dict(patient=patient,status=status)
            if failure.exists():record['failure']=read_json(failure)
            patient_records.append(record);continue
        report=read_json(folder/'evaluation/metrics.json')
        before=report['comparisons']['pose_only_initialization']['all_annotated']
        after=report['comparisons']['final']['all_annotated']
        trusted=report['comparisons']['final']['trusted']['cells']
        cells=load_cells(folder/'evaluation/final_cells.csv');all_cells.extend(cells)
        support=sorted({phase for row in cells if row['used_for_optimization']
                        for phase in ([row['phase'],row['phase_upper']] if row['phase_alpha']>0 else [row['phase']])})
        budget_text='—';budget=None;audit_path=root/'independent_audit/joint'/patient/'audit.json'
        if audit_path.exists():
            audit=read_json(audit_path)
            if audit['state_sha256']!=sha256(folder/'states/export.pth'):
                raise ValueError('Saved audit belongs to a different endpoint: '+patient)
            budget=audit.get('optimization_budget')
            if budget is None and 'parameter_trajectories' in audit:
                attempts=audit['parameter_trajectories']['updates']
                budget=dict(attempted_updates=attempts,accepted_updates=attempts-audit['rollbacks_verified'],
                            scope='Counts derived from earlier parameter-trajectory and rollback audit')
            if budget is not None:budget_text=f"{budget['accepted_updates']}/{budget['attempted_updates']}"
        lines.append(f"| {patient} | {report['status']} | {after['cells']} | {before['mean']:.6f}→{after['mean']:.6f} | {before['minimum']:.6f}→{after['minimum']:.6f} | {after['P01']:.6f} / {after['P05']:.6f} | {after['ge098']}/{after['cells']} | {trusted} | {len(support)}/24 | {budget_text} |")
        patient_records.append(dict(patient=patient,status=report['status'],cells=after['cells'],
            optimization_budget=budget,
            native_time_interpolation_support=support,
            unsupported_model_slots=sorted(set(range(24))-set(support))))
    aggregate=dice_stats(all_cells)
    eligible=dice_stats([row for row in all_cells if row['used_for_optimization']])
    if eligible['cells']!=comparison['arms']['joint']['cells']:
        raise ValueError('CSV eligible-cell count differs from formal summary')
    reported=comparison['arms']['joint']['mean_dice']
    if reported is not None and abs(eligible['mean']-reported)>1e-12:
        raise ValueError('CSV eligible mean differs from formal summary')
    lines+=['',f"当前已导出病例的全标注Cell为{aggregate['cells']}，Mean={aggregate['mean']}，Minimum={aggregate['minimum']}，P01={aggregate['P01']}，P05={aggregate['P05']}，≥0.98计数={aggregate['ge098']}。未完成/失败病例没有填入虚构Dice；患者分母始终保留完整队列。",'',
        '插值支持槽数表示实际原生时刻的可用二维监督覆盖，不等于逐槽独立真值或逐槽梯度证明。模型仍重建并审核全部24帧；缺少直接监督的槽不能通过复制标签假装有标注。高可信、可用低权重与全部标注分别报告；共同相位置信因子在归一化均值中抵消，不能声称它降低整体二维监督相对于先验的强度。','']
    groups=native_groups(all_cells,config['supervision']['trusted_confidence_threshold'])
    lines+=['以下分组仅汇总当前已提交主实验端点，按Cell等权计算硬Dice，未用Loss置信度重新加权。视图、腔室和来源行分别划分同一全部标注集合；高可信空集合保留缺失指标，不填零。','',
            '| 分组 | Cell | Mean | Minimum | P01 | P05 | ≥0.95 | ≥0.98 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    number=lambda value:'—' if value is None else f'{value:.6f}'
    for group in groups:
        stats=group['stats'];count=stats['cells']
        rates=[f"{stats[key]}/{count} ({stats['fraction_'+key]:.2%})" if count else '—' for key in ('ge095','ge098')]
        values=[number(stats[key]) for key in ('mean','minimum','P01','P05')]
        lines.append('| '+group['dimension']+'：'+group['label']+' | '+str(count)+' | '+' | '.join(values+rates)+' |')
    lines.append('')
    for record in patient_records:
        if 'failure' in record:
            name=record['patient']
            lines += [f"- [{name}失败记录](patients/{name}/failure.json)：{record['failure'].get('error','详见记录')}。",'']
    diagnostic_path=root/'independent_audit/pose_initialization_diagnostics.json'
    if diagnostic_path.exists():
        diagnostic=read_json(diagnostic_path)['aggregate']
        lines += [f"Pose初始化事后诊断已检查{diagnostic['checked_patients']}例保存端点，其中{diagnostic['patients_near_translation_limit']}例有视图达到平移限值的99%，{diagnostic['patients_near_rotation_limit']}例接近旋转边界，{diagnostic['patients_with_mean_ranking_tradeoff']}例尾部排序与最高Mean候选存在权衡。这不是完整TTO完成计数，也不证明真实Pose在搜索范围之外；原候选及配置保持不变。详见[Pose初始化诊断](independent_audit/POSE_INITIALIZATION_DIAGNOSTICS.md)。",'']
    angle_path=root/'independent_audit/pose_angle_reporting_correction.json'
    if angle_path.exists():
        correction=read_json(angle_path)
        lines += [f"Pose角度数值修正已核对{correction['checked_committed_exports']}个已提交端点，检查通过为{correction['passed']}。辅助报告使用float64最近SO(3)旋转和atan2，相同保存矩阵角度严格为0；仅替代早期审计的角度字段，不修改Pose状态、优化或验收阈值。详见[Pose角度修正](independent_audit/pose_angle_reporting_correction.json)。",'']
    queue=root/'independent_audit/coverage_rescue_queue.json'
    if queue.exists():
        queued=read_json(queue)
        lines+=['采样补充实验状态：`'+queued['status']+'`，输入覆盖规则选出的患者：'+', '.join(queued['affected_patients'])+'。原协议采样失败继续保留；补充协议须独立执行16例门控、Smoke/恢复和匹配七臂，不计作原协议成功。见[输入覆盖审计](independent_audit/native_stride_coverage.json)和[补充预声明](independent_audit/coverage_rescue_predeclaration.json)。','']
    surface_path=root/'independent_audit/synthetic_surface_summary.json'
    if surface_path.exists():
        surface=read_json(surface_path)
        if surface['aggregate']:
            assd=surface['aggregate']['reference_ASSD_mm'];hd95=surface['aggregate']['reference_HD95_mm']
            lines += [f"合成隐藏GT的{surface['summarized']}/{surface['required']}例已保存表面误差汇总：参考ASSD病例中位数{assd['before_median']:.6f}→{assd['after_median']:.6f} mm，HD95中位数{hd95['before_median']:.6f}→{hd95['after_median']:.6f} mm；分别有{assd['increased']}例和{hd95['increased']}例误差上升。此项核对端点/隐藏真值SHA与独立RMSE，汇总原正式评价器的顶点到三角形表面距离，没有重新计算表面距离；结果不支持稳定的Shape精度改善。详见[合成表面与全周期误差](independent_audit/SYNTHETIC_SURFACE_REPORT.md)。",'']
    geometry_source_path=root/'independent_audit/geometry_backend_source_identity.json'
    if geometry_source_path.exists():
        identity=read_json(geometry_source_path)
        lines += [f"源几何身份补充检查已覆盖{identity['checked_exports']}个已提交端点（含合成与当前真实结果），与保存记录的一致性为{identity['current_committed_sources_match']}。检查范围为原始模块和固定支持文件的SHA及归档副本，不是新的几何验收，也不证明未列出的传递模块或二进制库身份。详见[源几何身份记录](independent_audit/geometry_backend_source_identity.json)。",'']
    lines+=['以下实验臂汇总按可用且已提交产物计算，配对不足时不推断优劣。','',arm_table,'',
        '真实数据无3D Mesh/Pose GT，不能由Native Dice推断真实三维精度或绝对Pose准确性。几何结论仅限已审核的参考/保存Mesh、跨腔和连续线性轨迹；完整空间FFD场证书仍为UNVERIFIED。Motion相对先验保持不证明初始生理运动正确。独立重渲染、参数回滚及其验证范围见各病例独立审计文件。']
    temporary=path.with_suffix('.tmp');temporary.write_text('\n'.join(lines)+'\n');temporary.replace(path)
    result=dict(updated=time.time(),reporter_sha256=sha256(Path(__file__)),patients=patient_records,
        all_annotated=aggregate,eligible=eligible,native_groups=groups,
        cohort_denominator=len(config['experiment']['cohort']),
        goal_completion_claim=False,scope='Interim report from committed exports and saved failure records; no optimization or new geometry/GT accuracy certificate')
    write_json(root/'independent_audit/interim_report_summary.json',result)
    print(dict(cohort_denominator=result['cohort_denominator'],exported_cells=aggregate['cells'],
               exported_patients=sum(record['status'] not in ('失败','待完成') for record in patient_records),
               mean_dice=aggregate['mean'],goal_completion_claim=False,report=str(path)))


if __name__=='__main__':main()
