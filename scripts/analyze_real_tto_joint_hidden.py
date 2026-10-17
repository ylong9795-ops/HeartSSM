"""Post-endpoint hidden-GT attribution; no optimization or candidate selection.

Only completed synthetic endpoints are eligible. This script separates the
common Pose initialization from Joint and tests the unchanged native renderer
against hidden source geometry. It never provides truth to the TTO optimizer.
"""
import argparse
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from tto.config import load_config,resolve
from tto.io import load_state,read_json,sha256,write_json
from tto.evaluation import RealTTOEvaluator
from tto.synthetic_oracle import select_oracle_cases,hidden_patient
from train_neural_ffd import load_checkpoint
from utils.neural_ffd_losses import reconstruction_metrics


def pose_errors(pose,truth):
    predicted_R=pose['R'].numpy();predicted_t=pose['center_mm'].numpy()
    rotation=predicted_R.transpose(0,2,1)@truth['R'][0].numpy()
    cosine=np.clip((np.trace(rotation,axis1=-2,axis2=-1)-1)/2,-1,1)
    return dict(rotation_degrees=np.degrees(np.arccos(cosine)).tolist(),
                translation_mm=np.linalg.norm(predicted_t-truth['center_mm'][0].numpy(),axis=-1).tolist())


def plot_motion(record,folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(9,6))
    for axis,chamber in zip(axes.flat,('LV','LA','RV','RA')):
        before=record['stages']['initial']['gt_metrics_by_chamber'][chamber]
        final=record['stages']['joint']['gt_metrics_by_chamber'][chamber]
        curve=lambda values:np.asarray(values,dtype=float).reshape(24)
        axis.plot(np.arange(24)/24,curve(before['gt_volume_curve_ml']),'k--',label='Hidden source GT')
        axis.plot(np.arange(24)/24,curve(before['predicted_volume_curve_ml']),label='Initial prediction')
        axis.plot(np.arange(24)/24,curve(final['predicted_volume_curve_ml']),label='Joint endpoint')
        axis.set(title=chamber,xlabel='Cycle phase',ylabel='Volume (mL)');axis.legend(fontsize=8)
    figure.tight_layout();figure.savefig(folder/'hidden_gt_volume_comparison.png',dpi=140)
    figure.savefig(folder/'hidden_gt_volume_comparison.pdf');plt.close(figure)


def summarize_stages(records):
    """Describe committed cases only; these statistics never admit a candidate."""
    stages={}
    for name in ('initial','pose_initialization','joint'):
        values=[r['stages'][name] for r in records]
        cells=sum(v['native']['cells'] for v in values)
        result=dict(patients=len(values),cells=cells,
            native_ge098=sum(v['native']['ge098'] for v in values),
            patients_all_cells_ge098=sum(v['native']['all_cells_ge098'] for v in values))
        result['native_fraction_ge098']=result['native_ge098']/cells if cells else None
        result['native_cell_weighted_mean']=(sum(v['native']['mean']*v['native']['cells'] for v in values)/cells
                                             if cells else None)
        result['native_worst_cell']=min((v['native']['minimum'] for v in values),default=None)
        for metric in ('reference_rmse_mm','trajectory_rmse_mm','relative_motion_rmse_mm'):
            result['patient_median_'+metric]=float(np.median([v[metric] for v in values])) if values else None
        for metric in ('rotation_degrees','translation_mm'):
            per_patient=[float(np.mean(v['pose_error'][metric])) for v in values]
            result['patient_median_mean_view_'+metric]=float(np.median(per_patient)) if values else None
        stages[name]=result
    paired={}
    for metric in ('reference_rmse_mm','trajectory_rmse_mm','relative_motion_rmse_mm',
                   'native_mean','native_minimum','native_P01','native_P05','native_ge098'):
        delta=[r['joint_minus_pose_initialization'][metric] for r in records]
        paired[metric]=dict(patients=len(delta),negative=sum(d<0 for d in delta),
            zero=sum(d==0 for d in delta),positive=sum(d>0 for d in delta),
            median_delta=float(np.median(delta)) if delta else None)
    return dict(stages=stages,joint_minus_pose_initialization=paired,
        pose_initialization_selected_unchanged=sum(all(r['endpoint_matches_pose_initialization'].values()) for r in records),
        geometry_pass=sum(r['geometry_pass'] for r in records),
        motion_preserved=sum(r['motion_preserved'] for r in records),
        accepted_joint_updates=sum(r['accepted_joint_updates'] for r in records),
        attempted_joint_updates=sum(r['total_joint_updates'] for r in records),
        scope='Committed cases only; no imputation for unfinished cases; no full-cohort gate decision')


def audit_formal_gate(root,records,config):
    """Recompute the declared gate from independent, committed endpoint metrics."""
    path=root/'synthetic_hidden_gt/gate.json'
    if not path.exists():return dict(status='PENDING',verification_passed=None,recomputed_gate_passed=None)
    official=read_json(path);protocol=read_json(root/'protocol.json');criteria=config['synthetic']['gate']
    admitted=[r for r in records if r['status']=='COMPLETE']
    def median(stage,metric,pose=False):
        values=[float(np.mean(r['stages'][stage]['pose_error'][metric])) if pose else r['stages'][stage][metric] for r in admitted]
        return float(np.median(values)) if values else None
    medians=dict(rotation_before=median('initial','rotation_degrees',True),rotation_after=median('joint','rotation_degrees',True),
        translation_before=median('initial','translation_mm',True),translation_after=median('joint','translation_mm',True),
        shape_before=median('initial','reference_rmse_mm'),shape_after=median('joint','reference_rmse_mm'))
    required=config['synthetic']['cases']
    checks=dict(all_cases_evaluated=len(records)==required and required>=criteria['minimum_cases'],
        admitted_fraction=len(admitted)/required>=criteria['minimum_admitted_fraction'],
        shared_pose_supported=bool(records) and all(r['source_pose_shared_across_cycle'] for r in records),
        pose_rotation_recovery=medians['rotation_after'] is not None and medians['rotation_after']<=criteria['median_rotation_degrees'] and medians['rotation_after']<=medians['rotation_before'],
        pose_translation_recovery=medians['translation_after'] is not None and medians['translation_after']<=criteria['median_translation_mm'] and medians['translation_after']<=medians['translation_before'],
        shape_preserved=medians['shape_after'] is not None and medians['shape_after']<=criteria['median_shape_rmse_mm'] and medians['shape_after']<=medians['shape_before']+criteria['shape_rmse_growth_mm'],
        geometry_and_motion=bool(admitted) and all(r['geometry_pass'] and r['motion_preserved'] for r in admitted))
    agreement={}
    for key,value in medians.items():
        reported=official['medians'][key]
        atol=1e-3 if key.startswith('rotation') else 1e-4 if key.startswith('translation') else 1e-5
        agreement[key]=dict(recomputed=value,reported=reported,atol=atol,
            agrees=(value is None and reported is None) or
                (value is not None and reported is not None and math.isclose(value,reported,abs_tol=atol,rel_tol=0.)))
    identities=dict(criteria_unchanged=criteria==official['criteria']==protocol['scientific_config']['synthetic']['gate'],
        source=official['source_sha256']==protocol['source_sha256'],validation=official['validation_sha256']==protocol['validation_sha256'],
        config=official['protocol_sha256']==protocol['config_sha256'],seed=official['seed']==protocol['seed'],
        counts=official['total']==required and official['evaluated']==len(records) and official['admitted']==len(admitted))
    recomputed=all(checks.values())
    verified=(all(identities.values()) and all(v['agrees'] for v in agreement.values()) and
              checks==official['checks'] and recomputed==official['passed'])
    return dict(status='CHECKED',verification_passed=verified,recomputed_gate_passed=recomputed,
        reported_gate_passed=official['passed'],criteria=criteria,checks=checks,medians=medians,median_agreement=agreement,
        identity_checks=identities,gate_sha256=sha256(path),
        endpoint_identities=[dict(patient=r['patient'],endpoint_sha256=r['endpoint_sha256'],hidden_truth_sha256=r['hidden_truth_sha256']) for r in records],
        scope='Same declared criteria recomputed from independent post-endpoint metrics; numerical agreement tolerances do not change admission thresholds. Shape preservation does not establish Shape improvement or the >=0.98 Native Dice target.')


def write_recovery_report(root,records,summary):
    """Write reviewable Chinese evidence without claiming unfinished gates pass."""
    lines=['# 合成隐藏 GT 恢复：独立端点审计','',
        f"已提交并分析 {len(records)}/{summary['required']} 例。以下仅覆盖这些已完成病例，完整门控以正式 `synthetic_hidden_gt/gate.json` 为准。",'',
        'GT 在端点提交后才用于本报告，不用于优化、候选选择、超参数或停止预算。样本来自 Synthetic TRAIN，不能据此宣称未见患者泛化或真实患者三维精度。','',
        '| 病例 | Shape RMSE 初始→Joint（mm） | Pose 初始化→Joint ≥0.98 Cell | Joint 最低 Dice | Joint 旋转误差（三视图，°） | 获准/尝试更新 | 最终端点 | 几何/Motion |',
        '|---|---|---|---|---|---|---|---|']
    for r in records:
        initial=r['stages']['initial'];pose=r['stages']['pose_initialization'];joint=r['stages']['joint']
        angles='/'.join(f'{v:.3f}' for v in joint['pose_error']['rotation_degrees'])
        selected='Pose 初始化未变' if all(r['endpoint_matches_pose_initialization'].values()) else 'Joint 候选'
        lines.append(f"| {r['patient']} | {initial['reference_rmse_mm']:.6f}→{joint['reference_rmse_mm']:.6f} | "
            f"{pose['native']['ge098']}→{joint['native']['ge098']} / {joint['native']['cells']} | "
            f"{joint['native']['minimum']:.6f} | {angles} | {r['accepted_joint_updates']}/{r['total_joint_updates']} | {selected} | "
            f"{r['geometry_pass']}/{r['motion_preserved']} |")
    aggregate=summary['completed_case_aggregate'];before=aggregate['stages']['pose_initialization'];after=aggregate['stages']['joint']
    gate=summary['formal_gate_audit']
    if gate['status']=='CHECKED':
        lines+=['','## 预声明门控复核','',
            f"正式门控：{'通过' if gate['reported_gate_passed'] else '未通过'}；独立复核一致性：{'通过' if gate['verification_passed'] else '未通过'}。"
            '病例数、Pose/Shape 中位数、固定阈值、源码/配置/随机种子身份及几何/Motion 状态均逐项复核，详见 `synthetic_gate_recomputed.json`。','',
            '其中 Shape 条件是保持在预声明合理误差范围内，不等于 Shape 精度改善，也不等于全部 Cell 达到 Dice≥0.98。','']
    lines+=['','## 阶段归因','']
    if records:
        lines += [f"Pose 初始化后到 Joint 后，≥0.98 Cell 为 {before['native_ge098']}→{after['native_ge098']} / {after['cells']}；"
                  f"Joint 达标比例为 {after['native_fraction_ge098']:.2%}，全部 Cell 达标病例为 {after['patients_all_cells_ge098']}/{len(records)}。",'',
            f"参考 Shape RMSE 的患者中位数从 {before['patient_median_reference_rmse_mm']:.6f} 变为 {after['patient_median_reference_rmse_mm']:.6f} mm。"
            f"相对于 Pose 初始化，{aggregate['joint_minus_pose_initialization']['reference_rmse_mm']['positive']} 例误差增加，"
            f"{aggregate['joint_minus_pose_initialization']['reference_rmse_mm']['negative']} 例减少。二维 Dice 改善不能直接解释为三维 Shape 精度改善。",'']
        lines += [f"最终有 {aggregate['pose_initialization_selected_unchanged']}/{len(records)} 例保留了与 Pose 初始化后逐位相同的 Shape latent、三视图 Pose 和完整轨迹。"
                  '候选获准表示满足安全与覆盖门槛；最终输出仍按预声明排序保留最优合法状态，因此获准次数不等于最终端点改善次数。','']
    lines+=['Pose 初始化、Joint 两阶段的完整误差、逐腔 GT 指标和容积曲线保存在各例 `three_stage_hidden_gt.json` 与 PNG/PDF。',
        '汇总 JSON 的旋转/平移统计先求每病例三视图均值，再取病例中位数；Native 平均值按 Cell 数加权。未汇总的 P01/P05 仍以逐病例原始指标为准。','',
        'Motion 通过仅表示相对初始预测的固定信任域检查通过，不能证明初始预测运动正确。表中几何结果覆盖保存 Mesh 和既定连续轨迹检查；完整空间 FFD 场证书仍未验证。',
        '本报告不改变任何准入门槛。真实 C01/C02 Smoke、17 例队列和匹配消融需分别完成并验收。','']
    (root/'independent_audit/SYNTHETIC_RECOVERY_REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/real_tto_joint_v1.yaml')
    parser.add_argument('--output',default='outputs/real_tto_joint_v1_20261016')
    args=parser.parse_args();torch.set_num_threads(2)
    config=load_config(resolve(args.config));root=resolve(args.output)
    model,_=load_checkpoint(resolve(config['prior']['checkpoint']),'cpu')
    model.eval().requires_grad_(False)
    evaluator=RealTTOEvaluator(model,config['supervision'])
    calibration=read_json(resolve(config['prior']['calibration']))
    dataset,indices,_=select_oracle_cases(config,calibration)
    records=[]
    for ordinal,index in enumerate(indices):
        name=f'SYN_{ordinal:03d}';path=root/'synthetic_hidden_gt/patients'/name
        endpoint=path/'states/export.pth';committed=path/'export/completed.json'
        gt_path=root/'synthetic_hidden_gt/hidden_truth'/(name+'.pth')
        if not endpoint.exists() or not committed.exists():continue
        saved=load_state(endpoint);state=saved['state'];initial=state['initial'];result=state['result']
        truth=load_state(gt_path)
        patient,_=hidden_patient(dataset[index],name)
        if patient.identity!=saved['identity']['input_sha256']:raise ValueError('Hidden input identity changed')
        if truth['sample_sha256']!=patient.metadata['contract']['sample_sha256']:raise ValueError('Hidden truth identity changed')
        stages={}
        for stage,vertices,pose in (
            ('initial',initial['vertices_4d_initial'][0],state['pose_initial']),
            ('pose_initialization',initial['vertices_4d_initial'][0],state['pose']),
            ('joint',result['trajectory'],result['pose'])):
            rows=evaluator.annotated_cells(vertices.numpy(),{k:v.numpy() for k,v in pose.items()},patient,stage)
            relative=vertices-vertices[:1];gt=truth['vertices_gt'];relative_gt=gt-gt[:1]
            stages[stage]=dict(pose_error=pose_errors(pose,truth),
                reference_rmse_mm=float((vertices[0]-gt[0]).square().sum(-1).mean().sqrt()),
                trajectory_rmse_mm=float((vertices-gt).square().sum(-1).mean().sqrt()),
                relative_motion_rmse_mm=float((relative-relative_gt).square().sum(-1).mean().sqrt()),
                native=evaluator.summarize(rows)['all_annotated'],
                gt_metrics_by_chamber=reconstruction_metrics(vertices[None],gt[None],model.chamber_ranges,model.faces))
        shared=(torch.allclose(truth['R'],truth['R'][:1].expand_as(truth['R']),atol=1e-6,rtol=0) and
                torch.allclose(truth['center_mm'],truth['center_mm'][:1].expand_as(truth['center_mm']),atol=1e-5,rtol=0))
        diagnostic={}
        if shared:
            gt_pose=dict(R=truth['R'][0].numpy(),center_mm=truth['center_mm'][0].numpy())
            for label,vertices in [('hidden_GT_mesh_and_GT_pose',truth['vertices_gt']),
                                   ('initial_prediction_at_GT_pose',initial['vertices_4d_initial'][0])]:
                rows=evaluator.annotated_cells(vertices.numpy(),gt_pose,patient,'post_endpoint_diagnostic')
                diagnostic[label]=evaluator.summarize(rows)['all_annotated']
        else:diagnostic['status']='UNSUPPORTED_GT_POSE_NOT_SHARED'
        delta={key:stages['joint'][key]-stages['pose_initialization'][key]
               for key in ('reference_rmse_mm','trajectory_rmse_mm','relative_motion_rmse_mm')}
        for key in ('mean','minimum','P01','P05','ge098'):
            delta['native_'+key]=stages['joint']['native'][key]-stages['pose_initialization']['native'][key]
        reasons={}
        for update in result['history']:
            for reason in update['rejected_by']:reasons[reason]=reasons.get(reason,0)+1
        record=dict(patient=name,endpoint_sha256=sha256(endpoint),hidden_truth_sha256=sha256(gt_path),
            status=result['status'],geometry_pass=result['geometry']['passed'],motion_preserved=result['motion_report']['passed'],
            case_id=truth['case_id'],stages=stages,joint_minus_pose_initialization=delta,
            diagnostic_only=diagnostic,rejection_counts=reasons,
            accepted_joint_updates=sum(u['accepted'] for u in result['history']),
            total_joint_updates=len(result['history']),
            source_pose_shared_across_cycle=bool(shared),
            endpoint_matches_pose_initialization=dict(
                z_shape=torch.equal(result['z_shape'],initial['z_shape_initial'][0]),
                pose=all(torch.equal(result['pose'][key],state['pose'][key]) for key in ('R','center_mm')),
                trajectory=torch.equal(result['trajectory'],initial['vertices_4d_initial'][0])),
            scope='Independent post-endpoint GT evaluation only; no optimizer, candidate selection or hyperparameter change',
            interpretation='TRAIN unknown-Pose diagnostic; GT-pose measurements do not determine the best attainable decoder latent fit')
        folder=root/'independent_audit/synthetic'/name;folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'three_stage_hidden_gt.json',record);plot_motion(record,folder);records.append(record)
    summary=dict(updated=time.time(),analyzer_sha256=sha256(Path(__file__)),completed=len(records),required=config['synthetic']['cases'],
                 completed_case_aggregate=summarize_stages(records),
                 formal_gate_audit=audit_formal_gate(root,records,config),
                 patients=[dict(patient=r['patient'],joint_minus_pose_initialization=r['joint_minus_pose_initialization'],
                     endpoint_matches_pose_initialization=r['endpoint_matches_pose_initialization'],
                    diagnostic_only=r['diagnostic_only'],rejection_counts=r['rejection_counts']) for r in records],
                 scope='Post-endpoint stage attribution. No result is available for unfinished cases; no experimental protocol is changed.')
    write_json(root/'independent_audit/hidden_gt_stage_attribution.json',summary)
    write_json(root/'independent_audit/synthetic_gate_recomputed.json',dict(
        analyzer_sha256=summary['analyzer_sha256'],updated=summary['updated'],**summary['formal_gate_audit']))
    write_recovery_report(root,records,summary)
    print({'completed_post_endpoint_analyses':len(records),'required':config['synthetic']['cases']})
    if summary['formal_gate_audit']['verification_passed'] is False:raise AssertionError('Independent formal gate recomputation disagrees')


if __name__=='__main__':main()
