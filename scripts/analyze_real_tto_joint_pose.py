"""Describe saved Pose initialization limits/ranking without changing candidates."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.audit_real_tto_joint import pose_changes
from tto.config import load_config,resolve
from tto.io import load_state,read_json,write_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/real_tto_joint_v1.yaml')
    parser.add_argument('--output')
    args=parser.parse_args();config=load_config(args.config)
    root=resolve(args.output or config['output']['root'])
    protocol=read_json(root/'protocol.json')
    if protocol['scientific_config']['pose']!=config['pose']:
        raise ValueError('Pose settings differ from saved experiment protocol')
    limits=config['pose'];records=[]
    lines=['# Pose初始化的保存候选与搜索边界诊断','',
        '仅检查已完成Pose初始化阶段；不更新参数、不重渲染或重选候选，不使用真实3D Mesh/Pose GT。尚无初始化端点的患者保留缺失状态。','',
        '| 患者 | 选中起点 | ≥0.98 Cell | 选中 Minimum / Mean | 最高Mean起点：Minimum / Mean | 接近边界视图数：平移 / 旋转 |',
        '|---|---:|---:|---:|---:|---:|']
    for patient in config['experiment']['cohort']:
        folder=root/'patients'/patient;path=folder/'states/pose_init.pth'
        if not path.exists():
            records.append(dict(patient=patient,status='FAILED_BEFORE_POSE_INIT' if (folder/'failure.json').exists() else 'PENDING'))
            continue
        state=load_state(path)['state'];result=state['pose_init_result']
        candidates=result.get('candidates')
        if not candidates:
            records.append(dict(patient=patient,status='NO_POSE_SEARCH',state_sha256=sha256(path)))
            continue
        chosen=next(c for c in candidates if c['start']==result['start'])
        best_mean=max(candidates,key=lambda c:c['score'][-1])
        change=pose_changes(state['pose_initial'],chosen['pose'])
        near_t=[x>=.99*limits['translation_limit_mm'] for x in change['translation_mm']]
        near_r=[x>=.99*limits['rotation_limit_degrees'] for x in change['rotation_degrees']]
        score=chosen['score'];alternative=best_mean['score']
        record=dict(patient=patient,status='CHECKED',state_sha256=sha256(path),
            selected_start=chosen['start'],selected_score=score,
            selected_is_predeclared_lexicographic_max=tuple(score)==max(tuple(c['score']) for c in candidates),
            highest_mean_start=best_mean['start'],highest_mean_score=alternative,
            mean_advantage_of_highest_mean_candidate=alternative[-1]-score[-1],
            minimum_change_of_highest_mean_candidate=alternative[1]-score[1],
            pose_change=change,near_translation_limit=near_t,near_rotation_limit=near_r,
            all_candidate_098_counts=[c['score'][0] for c in candidates],
            candidates=[dict(start=c['start'],score=c['score']) for c in candidates])
        records.append(record)
        lines.append(f"| {patient} | {chosen['start']} | {score[0]} | {score[1]:.6f} / {score[-1]:.6f} | {best_mean['start']}：{alternative[1]:.6f} / {alternative[-1]:.6f} | {sum(near_t)} / {sum(near_r)} |")
    checked=[r for r in records if r['status']=='CHECKED']
    aggregate=dict(checked_patients=len(checked),cohort_denominator=len(records),
        selected_ranking_verified=all(r['selected_is_predeclared_lexicographic_max'] for r in checked) if checked else None,
        patients_near_translation_limit=sum(any(r['near_translation_limit']) for r in checked),
        patients_near_rotation_limit=sum(any(r['near_rotation_limit']) for r in checked),
        patients_with_mean_ranking_tradeoff=sum(r['mean_advantage_of_highest_mean_candidate']>1e-12 for r in checked),
        patients_all_eight_candidate_098_counts_zero=sum(all(n==0 for n in r['all_candidate_098_counts']) for r in checked))
    lines+=['',f"当前已检查{len(checked)}/{len(records)}例Pose初始化。{aggregate['patients_near_translation_limit']}例至少一个视图接近平移边界，{aggregate['patients_near_rotation_limit']}例接近旋转边界；这里的“接近”仅指达到配置限值的99%，不是生理合理性或真实Pose误差的判断。",'',
        f"其中{aggregate['patients_all_eight_candidate_098_counts_zero']}例的全部保存起点均无Cell达到0.98；在这些现有初始化候选中替换硬排序也没有达标Cell可选。不能据此推断更换初始化后再运行Joint的结果。",'',
        '分数来自流水线保存的原生硬Dice；本检查未独立重渲染这些初始化候选。预声明排序为≥0.98数量、Minimum、P01、P05、Mean。在全部候选≥0.98计数均为0时，后续尾部指标决定排序。二维Loss按标签置信度加权，硬排序则在可用Cell集合中比较数量与尾部；不能把选中候选直接称为最高Mean候选。','',
        '最高Mean候选仅用于描述冻结Shape的Pose初始化阶段权衡，其后运行Joint的结果未知，不构成新的对照臂。搜索触及边界不证明真实Pose在边界之外，也不能由本表单独归因为域偏移或Shape容量不足。原Shape–Pose预算和候选选择保持不变。']
    result=dict(updated=time.time(),analyzer_sha256=sha256(Path(__file__)),
        pose_angle_helper_sha256=sha256(ROOT/'scripts/audit_real_tto_joint.py'),patients=records,aggregate=aggregate,
        scope='Post-initialization diagnostics only; no GT accuracy, candidate replacement, new stopping criterion or experiment protocol change')
    write_json(root/'independent_audit/pose_initialization_diagnostics.json',result)
    (root/'independent_audit/POSE_INITIALIZATION_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print(aggregate)


if __name__=='__main__':main()
