"""Sequential, matched-budget main/cohort/ablation execution on one GPU at a time.

python -m tto.experiments --config configs/real_tto_joint_v1.yaml --resume
"""
import argparse
from copy import deepcopy
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from .config import load_config,resolve
from .io import read_json,write_json,load_state,sha256


def smoke_gate(root,patients=('C01','C02')):
    checks={};evidence={}
    for patient in patients:
        path=root/'patients'/patient
        state_path=path/'states'/'export.pth'
        if not state_path.exists():checks[patient]=False;continue
        state=load_state(state_path)['state'];result=state['result'];initial=state['initial']
        history=result['history'];phases=sorted({p for row in history for p in row['supervised_phases']})
        gradients=bool(history) and all(any(h['gradients'][key]>0 for h in history) for key in ('shape','rotation','translation'))
        fields=initial['control_grids_initial']['fine_motion'][0]
        progress=load_state(root/'checkpoints'/patient/'joint.pth')['progress']
        import torch
        frozen=torch.equal(fields,progress['frozen_fields']) and torch.equal(initial['z_motion_initial'][0],progress['z_motion_initial'])
        checks[patient]=result['status']=='COMPLETE' and result['geometry']['passed'] and result['motion_report']['passed'] and gradients and frozen and any(p!=0 for p in phases)
        evidence[patient]=dict(status=result['status'],supervised_phases=phases,gradients=gradients,
            fields_and_latent_frozen=frozen,shape_4d=list(result['trajectory'].shape))
    measurement=read_json(root/'gpu_selection.json')['attempts'][-1]['measurement']
    checks['actual_nonreference_gradients']=all(measurement.get(key,0)>0 for key in
        ('nonreference_shape_gradient_norm','nonreference_translation_gradient_norm','nonreference_rotation_gradient_norm'))
    report=dict(passed=all(checks.values()),checks=checks,evidence=evidence,
        source='formal CLI saved clinical states; actual measured nonreference slice-only backward')
    write_json(root/'smoke_gate.json',report);return report


def summarize(root,config):
    root=Path(root);cohort=config['experiment']['cohort'];arms=config['experiment']['arms']
    rows=[];arm_reports={}
    for arm in arms:
        path=root if arm=='joint' else root/'ablations'/arm
        values=[];records=[]
        for patient in cohort:
            metric=path/'patients'/patient/'evaluation'/'metrics.json'
            completion=path/'patients'/patient/'export'/'completed.json'
            failure=path/'patients'/patient/'failure.json'
            if metric.exists() and completion.exists():
                report=read_json(metric);stats=report['comparisons']['final']['optimization_observations']
                row=dict(arm=arm,patient=patient,status=report['status'],stats=stats,
                         geometry_pass=report['geometry']['passed'],motion_pass=report['motion_relative_to_frozen_prior']['passed'])
                values.append(stats)
            elif failure.exists():row=dict(arm=arm,patient=patient,status='FAILED',failure=read_json(failure))
            else:row=dict(arm=arm,patient=patient,status='PENDING')
            rows.append(row);records.append(row)
        cells=sum(v['cells'] for v in values)
        arm_reports[arm]=dict(cohort_denominator=len(cohort),processed=sum(r['status']!='PENDING' for r in records),
            exported=len(values),geometry_pass=sum(r.get('geometry_pass',False) for r in records),
            motion_pass=sum(r.get('motion_pass',False) for r in records),cells=cells,
            mean_dice=sum(v['mean']*v['cells'] for v in values)/cells if cells else None,
            minimum=min((v['minimum'] for v in values),default=None),
            fraction_ge098=sum(v['ge098'] for v in values)/cells if cells else None,
            patients_all_cells_ge098=sum(v['all_cells_ge098'] for v in values),
            patients_all_cells_ge098_fraction=sum(v['all_cells_ge098'] for v in values)/len(cohort))
    paired={}
    lookup={(r['arm'],r['patient']):r for r in rows}
    for comparator in ('pose_only','shape_only','alternate','original_motion','ed_only'):
        pairs=[]
        for patient in cohort:
            main,other=lookup.get(('joint',patient),{}),lookup.get((comparator,patient),{})
            if 'stats' not in main or 'stats' not in other:continue
            if main['stats']['cells']!=other['stats']['cells']:raise ValueError('Ablation evaluated cell counts differ')
            pairs.append(dict(patient=patient,mean_dice_delta=main['stats']['mean']-other['stats']['mean'],
                minimum_delta=main['stats']['minimum']-other['stats']['minimum'],P05_delta=main['stats']['P05']-other['stats']['P05'],
                ge098_count_delta=main['stats']['ge098']-other['stats']['ge098']))
        paired[comparator]=dict(pairs=pairs,n=len(pairs),
            mean_dice_delta=float(np.mean([p['mean_dice_delta'] for p in pairs])) if pairs else None)
    result=dict(arms=arm_reports,patients=rows,paired_joint_minus_comparator=paired,
                complete=all(a['processed']==len(cohort) for a in arm_reports.values()),
                interpretation='All failures retained in patient denominator; available-cell averages have explicit cell denominator. Real 3D accuracy/Pose error unavailable.')
    write_json(root/'metrics'/'ablation_comparison.json',result)
    report=['# 全周期 Shape–Pose Joint TTO 科研报告','',
            '状态：'+('全部实验臂已处理队列。' if result['complete'] else '实验仍在执行，以下是已保存结果；未完成部分不视为通过。'),'',
            '| 实验臂 | 已处理/队列 | 有导出结果 | Mean Dice | 最低 Dice | ≥0.98 Cell 比例 | 全 Cell ≥0.98 患者/队列 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    def number(value):return '待计算' if value is None else f'{value:.6f}'
    for arm,value in arm_reports.items():
        report.append(f"| {arm} | {value['processed']}/{len(cohort)} | {value['exported']} | {number(value['mean_dice'])} | {number(value['minimum'])} | {number(value['fraction_ge098'])} | {value['patients_all_cells_ge098']}/{len(cohort)} |")
    report+=['','E/joint 为主实验，使用原 Dice+BCE+Boundary、全周期原生帧观察和 Frozen-Field Motion-Lock。',
        '所有实验臂使用同一合成先验、同一数据和相同 Joint 目标评估次数；Alternate 每次只更新一个参数组，其实际更新次数单独记录。',
        'ED-only 的梯度、候选选择和二维尾部准入仅使用参考观察；其余时相在提交端点后独立评价。',
        '真实数据没有三维 Mesh/Pose GT，二维改善不能单独证明真实三维 Shape 或绝对 Pose 更准确。',
        '可信标签、自动标签和相位不确定性分别记录；原始帧时刻评价与旧 24 槽最近邻评价单独报告。','',
        '各患者失败 Cell、Motion、几何和未验证项目详见 patients/*/evaluation/metrics.json。',
        '所有臂的逐患者差值见 metrics/ablation_comparison.json；配对不足时不作优劣结论。']
    (root/'REAL_TTO_JOINT_REPORT.md').write_text('\n'.join(report)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/real_tto_joint_v1.yaml')
    parser.add_argument('--output');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);config=load_config(args.config)
    if args.output:config['output']['root']=args.output
    root=resolve(config['output']['root']);root.mkdir(parents=True,exist_ok=True)
    order=['joint',*[a for a in config['experiment']['arms'] if a!='joint']]
    for arm in order:
        current=deepcopy(config);current['experiment']['arm']=arm
        output=root if arm=='joint' else root/'ablations'/arm
        current['output']['root']=str(output)
        if arm!='joint':
            gate=root/'synthetic_hidden_gt'/'gate.json'
            if not gate.exists() or not read_json(gate)['passed']:
                summarize(root,config);raise RuntimeError('Main synthetic recovery gate failed; ablations not admitted')
            current['synthetic']['validation_gate']=str(gate)
            current['experiment']['initialization_source']=str(root)
        path=root/'run_configs'/(arm+'.yaml');path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(current,sort_keys=False))
        log=root/'logs'/(arm+'.log');log.parent.mkdir(exist_ok=True)
        def execute(stage,patients=None,batch=None):
            command=[sys.executable,'-m','tto.run','--config',str(path),'--stage',stage,'--device','auto']
            if args.resume or stage=='all':command.append('--resume')
            if patients:command.extend(['--patients',*patients])
            if batch:command.extend(['--batch-size',str(batch)])
            with log.open('ab') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            return result,command
        if arm=='joint':
            gate=root/'synthetic_hidden_gt'/'gate.json'
            if not gate.exists() or not read_json(gate)['passed']:
                result,command=execute('synthetic_oracle')
                if result.returncode or not gate.exists() or not read_json(gate)['passed']:
                    summarize(root,config);return result.returncode or 1
            if not (root/'smoke_gate.json').exists() or not read_json(root/'smoke_gate.json')['passed']:
                result,command=execute('all',['C01','C02'],2)
                if result.returncode or not smoke_gate(root)['passed']:
                    summarize(root,config);return result.returncode or 1
                # A real completed-stage resume must leave all scientific states byte-identical.
                before={str(p):sha256(p) for patient in ('C01','C02') for p in (root/'patients'/patient/'states').glob('*.pth')}
                result,command=execute('all',['C01','C02'],2)
                unchanged=all(sha256(p)==digest for p,digest in before.items())
                write_json(root/'smoke_stage_resume.json',dict(passed=unchanged and result.returncode==0,states=before))
                if not unchanged or result.returncode:return 1
        result,command=execute('all')
        write_json(root/'experiment_status.json',dict(last_arm=arm,returncode=result.returncode,command=command))
        summarize(root,config)
        if result.returncode==75:return 75
        if result.returncode and not (output/'summary.json').exists():return result.returncode
    result=summarize(root,config)
    return 0 if result['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
