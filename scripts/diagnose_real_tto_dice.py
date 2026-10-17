"""Visual, post-hoc diagnosis of committed real TTO Native Dice failures.

No optimization state is modified. Target-dependent 2D translations/reflections
are diagnostic counterfactuals, never TTO results or recovered probe poses.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tto.config import load_config, resolve
from tto.dataset import RealTTODataset
from tto.evaluator import native_exact
from tto.io import read_json, sha256, write_json
from tto.losses import observation_cells

VIEWS = ('A2C', 'A3C', 'A4C')
CHAMBERS = ('LV', 'LA', 'RV', 'RA')
GREEN = '#53f482'
MAGENTA = '#ff5bd7'
STAGES = ('initial', 'pose_only_initialization', 'final')


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return str(row['native_frame']), row['view'], row['chamber']


def dice(first, second):
    denominator = int(first.sum()) + int(second.sum())
    return 2 * int((first & second).sum()) / denominator if denominator else 1.


def centroid(mask, spacing):
    yy, xx = np.nonzero(mask)
    return np.array([xx.mean() * spacing[1], yy.mean() * spacing[0]]) if len(xx) else np.full(2, np.nan)


def shifted(mask, dy, dx):
    matrix = np.array([[1., 0., dx], [0., 1., dy]])
    return cv2.warpAffine(mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)


def orient(mask, name):
    if name == 'horizontal_flip':
        return np.fliplr(mask)
    if name == 'vertical_flip':
        return np.flipud(mask)
    if name == 'rotate_180':
        return np.flipud(np.fliplr(mask))
    return mask


def translation_surface(target, prediction, spacing, maximum_mm=30.):
    """Exact pixel-grid Dice at all allowed shifts, including FOV clipping."""
    h, w = target.shape
    ylimit = min(h - 1, int(np.floor(maximum_mm / spacing[0])))
    xlimit = min(w - 1, int(np.floor(maximum_mm / spacing[1])))
    crop = (slice(h - 1 - ylimit, h + ylimit), slice(w - 1 - xlimit, w + xlimit))
    intersection = np.rint(fftconvolve(target.astype(float), prediction[::-1, ::-1].astype(float), 'full')[crop])
    predicted_area = np.rint(fftconvolve(np.ones_like(target, dtype=float), prediction[::-1, ::-1].astype(float), 'full')[crop])
    denominator = int(target.sum()) + predicted_area
    values = np.divide(2 * intersection, denominator, out=np.ones_like(intersection), where=denominator != 0)
    return values, ylimit, xlimit


def overlay(ax, image, target, prediction=None, error=False):
    if image.ndim == 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    ax.imshow(image, cmap='gray')
    if error:
        rgba = np.zeros((*target.shape, 4))
        rgba[target & ~prediction] = [0.15, 0.85, 1., .7]
        rgba[prediction & ~target] = [1., .15, .65, .7]
        rgba[target & prediction] = [1., 1., 1., .6]
        ax.imshow(rgba)
    else:
        if target.any() and not target.all():
            ax.contour(target, [.5], colors=[GREEN], linewidths=1.3)
        if prediction is not None and prediction.any() and not prediction.all():
            ax.contour(prediction, [.5], colors=[MAGENTA], linewidths=1.3)
    ax.set_xlim(0, target.shape[1] - 1)
    ax.set_ylim(target.shape[0] - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def save_figure(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=170, bbox_inches='tight', facecolor='white')
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


class NativeCases:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self.loaded, self.rendered = {}, {}

    def case(self, name):
        if name not in self.loaded:
            patient = RealTTODataset(resolve(self.config['data']['root']), [name], config=self.config['data'])[0]
            folder = self.root / 'patients' / name
            completion = read_json(folder / 'export/completed.json')
            if patient.identity != completion['identity']['input_sha256']:
                raise ValueError(f'Current native input differs from committed input: {name}')
            names = ['vertices_4d_initial', 'vertices_4d_frozen_field', 'probe_R_initial', 'probe_t_initial_mm',
                     'probe_R_joint_initial', 'probe_t_joint_initial_mm', 'probe_R_final', 'probe_t_final_mm',
                     'chamber_ranges', *('faces_' + chamber for chamber in CHAMBERS)]
            with np.load(folder / 'export/reconstruction.npz', allow_pickle=False) as z:
                data = {field: z[field].copy() for field in names}
            rows = observation_cells([patient], self.config['supervision'])
            indexed = {(str(row['native_frame']), VIEWS[row['view']], CHAMBERS[row['chamber']]): row for row in rows}
            self.loaded[name] = patient, data, indexed
        return self.loaded[name]

    def render(self, name, cell_key, stage):
        cache = name, cell_key, stage
        if cache not in self.rendered:
            patient, data, indexed = self.case(name)
            row = indexed[cell_key]
            vi, ci = row['view'], row['chamber']
            view = patient.views[vi]
            target = view['masks'][row['native_index'], ci].astype(bool)
            trajectory = data['vertices_4d_frozen_field' if stage == 'final' else 'vertices_4d_initial']
            vertices = (1 - row['mesh_alpha']) * trajectory[row['phase']] + row['mesh_alpha'] * trajectory[row['mesh_upper']]
            suffix = {'initial': 'initial', 'pose_only_initialization': 'joint_initial', 'final': 'final'}[stage]
            a, b = map(int, data['chamber_ranges'][ci])
            prediction = native_exact(vertices[a:b], data['faces_' + CHAMBERS[ci]],
                                      data['probe_R_' + suffix][vi], data['probe_t_' + suffix + '_mm'][vi],
                                      view['spacing'], target.shape).astype(bool)
            csv_rows = read_rows(self.root / 'patients' / name / 'evaluation' / (stage + '_cells.csv'))
            saved = next(value for value in csv_rows if key(value) == cell_key)
            actual = dice(target, prediction)
            if abs(actual - float(saved['exact_hard_dice'])) > 1e-12:
                raise ValueError(f'Independent native rendering disagrees: {cache}')
            self.rendered[cache] = target, prediction, np.asarray(view['spacing']), saved
        return self.rendered[cache]

    def image(self, name, cell_key):
        patient, _, indexed = self.case(name)
        row = indexed[cell_key]
        for slot, mapping in enumerate(patient.mapping):
            if mapping[row['view']]['nearest_index'] == row['native_index']:
                return patient.images[row['view']][slot]
        import pydicom
        contract = patient.metadata['contract']
        pixels = pydicom.dcmread(contract['dicom']).pixel_array
        region = contract['views'][row['view']]['region']
        h, w = patient.views[row['view']]['masks'].shape[-2:]
        return pixels[row['native_index'], region['y0']:region['y0'] + h, region['x0']:region['x0'] + w]


def cohort_figures(root, output):
    names = sorted(p.parents[1].name for p in root.glob('patients/*/export/completed.json')
                   if (p.parents[1] / 'states/export.pth').exists())
    rows, means, stage_summary = {}, {}, {}
    for stage in STAGES:
        all_rows = []
        means[stage] = []
        for name in names:
            values = read_rows(root / 'patients' / name / 'evaluation' / (stage + '_cells.csv'))
            all_rows.extend(values)
            rows[name, stage] = values
            means[stage].append(float(np.mean([float(x['exact_hard_dice']) for x in values])))
        values = np.array([float(x['exact_hard_dice']) for x in all_rows])
        stage_summary[stage] = dict(cells=len(values), mean=float(values.mean()), minimum=float(values.min()),
                                    ge098=int((values >= .98).sum()))
    for name in names:
        if any({key(x) for x in rows[name, stage]} != {key(x) for x in rows[name, 'final']} for stage in STAGES):
            raise ValueError('Stage cell identities differ')
    final = [row for name in names for row in rows[name, 'final']]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    x = np.arange(len(names))
    for shift, stage, color, label in zip([-.25, 0, .25], STAGES, ['#a7abb5', '#e6b85c', '#527fbd'],
                                         ['Initial prior + initial pose', 'After pose search', 'After Joint TTO']):
        axes[0, 0].bar(x + shift, means[stage], .24, color=color, label=label)
    axes[0, 0].set(xticks=x, xticklabels=names, ylim=(0, 1.02), ylabel='Mean Native Dice', title='Most improvement comes from pose search')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend(fontsize=8, loc='upper left')
    matrix = np.full((3, 4), np.nan)
    for vi, view in enumerate(VIEWS):
        for ci, chamber in enumerate(CHAMBERS):
            values = [float(row['exact_hard_dice']) for row in final if row['view'] == view and row['chamber'] == chamber]
            if values:
                matrix[vi, ci] = np.mean(values)
    im = axes[0, 1].imshow(np.ma.masked_invalid(matrix), cmap='RdYlGn', vmin=.4, vmax=1., aspect='auto')
    for vi in range(3):
        for ci in range(4):
            axes[0, 1].text(ci, vi, 'not labelled' if np.isnan(matrix[vi, ci]) else f'{matrix[vi, ci]:.3f}', ha='center', va='center')
    axes[0, 1].set(xticks=range(4), xticklabels=CHAMBERS, yticks=range(3), yticklabels=VIEWS, title='Final Dice: A4C is the weakest view')
    fig.colorbar(im, ax=axes[0, 1], label='Mean Native Dice', shrink=.8)
    sources = sorted({row['label_source'] for row in final})
    source_summary = {}
    for source, color in zip(sources, ['#6985aa', '#d39b54', '#aa69a1']):
        values = [float(row['exact_hard_dice']) for row in final if row['label_source'] == source]
        source_summary[source] = dict(cells=len(values), mean=float(np.mean(values)), minimum=min(values))
        axes[1, 0].hist(values, bins=np.linspace(0, 1, 26), weights=np.full(len(values), 1 / len(values)),
                        histtype='step', linewidth=2, label=f'{source}\nn={len(values)}, mean={np.mean(values):.3f}', color=color)
    axes[1, 0].axvline(.98, linestyle='--', color='red', label='0.98 target')
    axes[1, 0].set(xlabel='Final Native Dice', ylabel='Fraction within each label source', title='Manual labels also have large residual errors')
    axes[1, 0].legend(fontsize=8)
    before = np.array(means['pose_only_initialization'])
    after = np.array(means['final'])
    axes[1, 1].bar(x, after - before, color='#527fbd')
    axes[1, 1].set(xticks=x, xticklabels=names, ylabel='Joint minus pose-search mean Dice', title='Joint-stage gain is small under the current protocol')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(0, color='black', linewidth=.8)
    fig.suptitle(f'Committed main results only: {len(names)}/17 exported, {len(final)} native observations; no real 3D/Pose GT', fontsize=14)
    save_figure(fig, output, '01_cohort_dice_breakdown')
    return names, rows, dict(exported_patients=names, cohort_denominator=17, stages=stage_summary,
                             by_label_source=source_summary,
                             final_by_view_chamber=[[None if np.isnan(value) else float(value) for value in line] for line in matrix])


def examples_figure(cases, output):
    examples = [('C09', ('28', 'A4C', 'LV')), ('C09', ('22', 'A4C', 'RV')),
                ('C14', ('60', 'A4C', 'RA')), ('C01', ('42', 'A3C', 'LA'))]
    fig, axes = plt.subplots(len(examples), 4, figsize=(14, 13), constrained_layout=True)
    details = []
    for ri, (name, cell_key) in enumerate(examples):
        image = cases.image(name, cell_key)
        entry = dict(patient=name, native_frame=cell_key[0], view=cell_key[1], chamber=cell_key[2], stages={})
        for ci, stage in enumerate(STAGES):
            target, prediction, spacing, saved = cases.render(name, cell_key, stage)
            gap = float(np.linalg.norm(centroid(target, spacing) - centroid(prediction, spacing)))
            area_ratio = float(prediction.sum() / target.sum()) if target.any() else None
            overlay(axes[ri, ci], image, target, prediction)
            axes[ri, ci].set_title(f'{name} {cell_key[1]} {cell_key[2]} / frame {cell_key[0]}\n'
                                   f'Dice={dice(target,prediction):.3f}; centre gap={gap:.1f} mm\n'
                                   f'prediction/label area={area_ratio:.2f}', fontsize=9)
            entry['stages'][stage] = dict(dice=dice(target, prediction), centroid_gap_mm=gap, area_ratio=area_ratio)
        overlay(axes[ri, 3], image, target, prediction, error=True)
        axes[ri, 3].set_title(f'Final pixel error\nLabel: {saved["label_source"]}', fontsize=9)
        entry['label_source'] = saved['label_source']
        details.append(entry)
    fig.suptitle('Original native images and timestamps: initial / pose search / Joint / residual\n'
                 'Green: annotation; magenta: mesh slice. Error: cyan=missed label, pink=extra prediction, white=overlap.', fontsize=12)
    for ax, title in zip(axes[0], ['Initial prior + initial pose', 'Pose search only', 'Final Joint TTO', 'Final error map']):
        ax.text(.5, 1.32, title, transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    save_figure(fig, output, '02_native_failure_examples')
    return details


def shared_view_diagnostic(cases, output):
    name, view = 'C09', 'A4C'
    patient, _, indexed = cases.case(name)
    chosen = sorted((key for key in indexed if key[1] == view), key=lambda k: (int(k[0]), k[2]))
    rendered = [cases.render(name, cell_key, 'final') for cell_key in chosen]
    spacing = rendered[0][2]
    orientation_results = {}
    for orientation in ('identity', 'horizontal_flip', 'vertical_flip', 'rotate_180'):
        surfaces = [translation_surface(target, orient(prediction, orientation), spacing)
                    for target, prediction, _, _ in rendered]
        surface = np.mean([value[0] for value in surfaces], axis=0)
        iy, ix = np.unravel_index(surface.argmax(), surface.shape)
        dy, dx = int(iy - surfaces[0][1]), int(ix - surfaces[0][2])
        actual = [dice(target, shifted(orient(prediction, orientation), dy, dx)) for target, prediction, _, _ in rendered]
        if abs(float(np.mean(actual)) - float(surface[iy, ix])) > 1e-12:
            raise ValueError('FFT diagnostic scores disagree with explicit pixel translations')
        orientation_results[orientation] = dict(mean_dice=float(np.mean(actual)), shift_yx_pixels=[dy, dx],
                                                 shift_xy_mm=[float(dx * spacing[1]), float(dy * spacing[0])])
    best = max(orientation_results, key=lambda value: orientation_results[value]['mean_dice'])
    baseline = float(np.mean([dice(t, p) for t, p, _, _ in rendered]))
    frame = '28'
    fig, axes = plt.subplots(4, 4, figsize=(13, 13), constrained_layout=True)
    frame_records = []
    for ri, chamber in enumerate(CHAMBERS):
        cell_key = frame, view, chamber
        target, prediction, _, saved = cases.render(name, cell_key, 'final')
        image = cases.image(name, cell_key)
        surface, yl, xl = translation_surface(target, prediction, spacing)
        iy, ix = np.unravel_index(surface.argmax(), surface.shape)
        independent_shift = [int(iy - yl), int(ix - xl)]
        independent = shifted(prediction, *independent_shift)
        translations = orientation_results['identity']['shift_yx_pixels']
        common = shifted(prediction, *translations)
        transformed = shifted(orient(prediction, best), *orientation_results[best]['shift_yx_pixels'])
        for ci, mask in enumerate((prediction, common, transformed, independent)):
            overlay(axes[ri, ci], image, target, mask)
            axes[ri, ci].set_title(f'{chamber}: Dice={dice(target,mask):.3f}', fontsize=10)
        gap = centroid(target, spacing) - centroid(prediction, spacing)
        frame_records.append(dict(chamber=chamber, original_dice=dice(target,prediction),
                                  shared_translation_dice=dice(target,common),
                                  best_shared_orientation_dice=dice(target,transformed),
                                  independent_cell_translation_dice=dice(target,independent),
                                  centroid_target_minus_prediction_xy_mm=gap.tolist(),
                                  independent_shift_yx_pixels=independent_shift))
    titles = ['Actual Joint result', 'One shared 2D translation\n(all chambers + all times)',
              f'Best shared orientation + shift\n{best}', 'Separate shift for each Cell\nDIAGNOSTIC ONLY']
    for ax, title in zip(axes[0], titles):
        ax.text(.5, 1.17, title, transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')
    fig.suptitle('C09 A4C: diagnose placement versus contour mismatch (manual frame 28 shown)\n'
                 f'Whole-view {len(chosen)}-Cell mean: actual={baseline:.3f}; shared shift={orientation_results["identity"]["mean_dice"]:.3f}; '
                 f'best shared orientation+shift={orientation_results[best]["mean_dice"]:.3f}\n'
                 'Target-dependent 2D counterfactuals; NOT optimized 3D meshes, recovered poses, or replacement TTO results.', fontsize=11)
    save_figure(fig, output, '03_C09_shared_alignment_diagnostic')
    return dict(patient=name, view=view, cells=len(chosen), actual_mean_dice=baseline,
                shift_search_per_axis_mm=30., orientations=orientation_results, best_orientation=best,
                manual_frame=frame, per_chamber_manual_frame=frame_records,
                scope='One common 2D transformation fitted across every available chamber and original time. Per-Cell translations are separate counterfactual diagnostics. No changes to Mesh, Pose, results or scores in the TTO experiment.')


def temporal_figure(rows, output):
    groups = [('C09','A4C','LV'), ('C09','A4C','RV'), ('C14','A4C','RA'), ('C01','A3C','LA')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    details = []
    for ax, (name, view, chamber) in zip(axes.flat, groups):
        cells = [row for row in rows[name,'final'] if row['view']==view and row['chamber']==chamber]
        cells.sort(key=lambda row: int(row['phase']) + float(row['phase_alpha']))
        x = np.array([(int(row['phase']) + float(row['phase_alpha']))/24 for row in cells])
        target = np.array([float(row['target_area_mm2']) for row in cells])
        prediction = np.array([float(row['prediction_area_mm2']) for row in cells])
        manual = np.array([row['label_source']=='manual_ed_es' for row in cells])
        ax.plot(x,target,'o-',color=GREEN,markeredgecolor='#137534',label='Annotation area')
        ax.plot(x,prediction,'o-',color=MAGENTA,label='Final slice area')
        ax.scatter(x[manual],target[manual],marker='*',s=170,color='#176b36',zorder=4,label='Manual ED/ES')
        correlation=float(np.corrcoef(target,prediction)[0,1]) if target.std()>0 and prediction.std()>0 else None
        ax.set(title=f'{name} {view} {chamber}: area correlation={correlation:.2f}',
               xlabel='Model-cycle coordinate at original observation time (ED anchor estimated)',ylabel='Native section area (mm²)')
        ax.legend(fontsize=8)
        details.append(dict(patient=name,view=view,chamber=chamber,area_correlation=correlation,
                            target_minmax_mm2=[float(target.min()),float(target.max())],
                            prediction_minmax_mm2=[float(prediction.min()),float(prediction.max())],
                            caveat='Section area depends on both 3D motion and estimated section Pose; this does not uniquely identify a Motion error.'))
    fig.suptitle('Temporal residuals: one frozen prior and one shared view Pose must explain the whole cycle',fontsize=13)
    save_figure(fig, output, '04_temporal_area_residuals')
    return details


def optimizer_facts(root, names, config):
    facts=[]
    for name in names:
        folder=root/'patients'/name
        metric=read_json(folder/'evaluation/metrics.json')
        audit=root/'independent_audit/joint'/name
        meta_path=audit/'joint_parameter_trajectory.json'
        if not meta_path.exists():
            continue
        meta=read_json(meta_path)
        if meta['state_sha256']!=sha256(folder/'states/export.pth'):
            raise ValueError('Parameter-trajectory audit refers to a different committed endpoint')
        history=read_rows(audit/'joint_parameter_trajectory.csv')
        updates=[row for row in history if int(row['update'])>0]
        with np.load(folder/'export/reconstruction.npz',allow_pickle=False) as data:
            delta=np.linalg.norm(data['vertices_reference_final']-data['vertices_reference_initial'],axis=-1)
            pose_delta=np.linalg.norm(data['probe_t_final_mm']-data['probe_t_joint_initial_mm'],axis=-1)
        accepted=sum(row['accepted']=='True' for row in updates)
        facts.append(dict(patient=name,proposals=len(updates),admitted=accepted,rejected=len(updates)-accepted,
                          selected_update=meta['selected_update'],reference_maximum_change_mm=float(delta.max()),
                          joint_pose_translation_mm=pose_delta.tolist(),
                          shape_rms_train_std=metric['shape_latent_standardized_drift'],
                          parameter_history_sha256=sha256(audit/'joint_parameter_trajectory.csv'),
                          export_npz_sha256=sha256(folder/'export/reconstruction.npz')))
    return dict(configured_proposals=config['joint']['updates'],shape_lr=config['joint']['shape_lr'],
                pose_lr=config['joint']['pose_lr'],pose_parameter_units='translation: mm; rotation vector: degrees',
                units_source_sha256=sha256(ROOT/'tto/pose_optimizer.py'),patients=facts,
                scope='Actual committed changes, parameter-history admissions and selected endpoints; no claim that Adam learning rate times steps is a hard displacement bound.')


def write_report(result, output):
    cohort=result['cohort'];stages=cohort['stages'];alignment=result['shared_alignment']
    c09=next(row for row in result['optimization']['patients'] if row['patient']=='C09')
    pose=result['pose_initialization_aggregate']
    baseline_selected=[row['patient'] for row in result['optimization']['patients'] if row['selected_update']==0]
    text=[
        '# 真实 TTO 低 Dice 可视化诊断', '',
        f'本次快照覆盖新 Joint 主实验已提交的 {len(cohort["exported_patients"])} 例、{stages["final"]["cells"]} 个原生时间点 Cell；完整队列分母仍为17。'
        f'达到0.98的Cell为 {stages["final"]["ge098"]}/{stages["final"]["cells"]}。C03采样失败与尚未提交的患者未混入已提交指标。', '',
        '**主要发现：当前存在明显的切面位置偏差，以及单一整体平移不能消除的四腔相对轮廓/大小失配；Joint实际改动幅度很小。**', '',
        f'全体等Cell权重 Mean Dice：初始化 {stages["initial"]["mean"]:.4f} → Pose搜索 {stages["pose_only_initialization"]["mean"]:.4f} → Joint {stages["final"]["mean"]:.4f}。'
        '主要收益来自Pose初始化，Joint尚未充分消除剩余误差。', '',
        '## 直接看到的误差', '',
        '- C09/A4C/LV，人工标注第28帧：Dice=0.192，切面与标注中心相差26.49mm，预测面积/标注面积=1.11。面积接近却几乎错开，位置/切面捕获是明确问题。',
        '- C09/A4C/RV，第22帧：Dice=0.270，中心偏差14.83mm，预测面积为标注的2.92倍。除偏位外还有明显轮廓/面积失配。',
        '- C14/A4C/RA，第60帧：Dice=0.303，中心偏差16.60mm，预测面积为标注的2.32倍。',
        '- C01/A3C/LA，第42帧：Dice=0.632，中心偏差仅4.62mm，但面积仍为标注的1.57倍；位置改善后仍有轮廓残差。', '',
        '![真实原生图像、标注与切面](02_native_failure_examples.png)', '',
        '绿色为原始标注，紫色为三维Mesh切面。右列误差图：青色为漏掉的标注，粉色为多出的预测，白色为重叠。原生图像以灰度显示，避免原设备彩色叠加与诊断轮廓混淆。', '',
        '## 为什么不能只说“Pose整体平移一下就好”', '',
        f'C09/A4C全部68个Cell共用一次二维平移，均值从 {alignment["actual_mean_dice"]:.3f} 提高到 {alignment["orientations"]["identity"]["mean_dice"]:.3f}。'
        '人工第28帧的LV从0.192提高到0.506、LA从0.515提高到0.825，但RV从0.495下降到0.228。'
        '不同心腔所需的修正方向/幅度不一致。即使允许每个Cell独立平移，LV也仅到0.710，仍留下轮廓失配。', '',
        '这支持“存在四腔相对投影不匹配”，但不能仅凭二维图把责任唯一归到Shape：出平面Pose角度和Shape都能改变切面轮廓。'
        '同一视图/整个周期共用的水平翻转、垂直翻转和180度旋转均未超过不翻转方案；本病例不支持仅靠简单镜像解决。', '',
        '**这里的二维对齐使用标注做事后诊断，不是新的TTO结果，不是恢复出的三维Pose，也未写回实验。**', '',
        '![共享对齐的四腔权衡](03_C09_shared_alignment_diagnostic.png)', '',
        '## 为什么Joint没有把误差继续拉下来', '',
        f'- 当前Joint预算为{result["optimization"]["configured_proposals"]}步，Pose学习率为{result["optimization"]["pose_lr"]}，Pose参数单位是毫米和度。'
        f'C09的A4C在Joint中实际只再移动 {c09["joint_pose_translation_mm"][2]:.2f}mm，参考Mesh最大变化 {c09["reference_maximum_change_mm"]:.2f}mm，而LV剩余二维中心偏差为26.49mm。实际修正量与残差规模明显不匹配；这不是学习率乘步数的理论上界。',
        f'- C09的{c09["proposals"]}次候选中，{c09["rejected"]}次被拒绝；既有独立审核记录原因为运动保护，最终学习率因子为0.001953125。保护机制确实减少了可用更新，但这本身不能证明保护阈值应放宽。',
        f'- 在已审核的Pose初始化中，{pose["checked_patients"]}例有{pose["patients_near_translation_limit"]}例至少一个视图接近15mm平移边界、{pose["patients_near_rotation_limit"]}例接近20度旋转边界。捕获范围/初始Pose值得优先排查；撞边界不等于真实Pose误差已知。',
        '- 当前按尾部Dice优先选合法候选，部分病例最后仍选Pose初始化端点：'+', '.join(baseline_selected)+'。这些病例执行了Joint候选，不能把“已执行20步”解读成最终Shape已改善。', '',
        '## 时序与标签的证据边界', '',
        'C09/A4C/RV的预测切面面积与标注面积相关系数仅0.069；C14/A4C/RA为−0.096，且预测面积整体偏大。'
        '这显示全周期切面变化尚未解释好。切面面积同时依赖Pose、Shape和运动，不能用这些相关系数单独证明Motion Decoder错误。', '',
        f'人工ED/ES标签的均值为{cohort["by_label_source"]["manual_ed_es"]["mean"]:.3f}，自动标签为{cohort["by_label_source"]["patient_adapted_model"]["mean"]:.3f}，且最差Cell为人工标注。因此不能把主要误差直接归结为自动标签。', '',
        '![全周期面积残差](04_temporal_area_residuals.png)', '',
        '![全队列分解](01_cohort_dice_breakdown.png)', '',
        '## 建议的排查顺序', '',
        '1. 优先复核A4C物理坐标与初始Pose捕获，再分别标定Pose平移/旋转学习率和实际可产生的毫米/角度修正量。',
        '2. 用固定Pose的Shape可达性诊断及固定Shape的Pose诊断，区分出平面角度错误与四腔相对形态不可达；不能只增大整体平移。',
        '3. 在独立版本的受控实验里比较有效Joint更新、运动保护拒绝和尾部候选选择的影响，同时保持几何安全口径。',
        '4. 随后核查ED锚点、跨视图同步及先验运动幅度。', '',
        '本次只新增可视化与事后诊断文件，没有修改正在运行的主协议、模型、Pose、Mesh、损失、阈值或任何正式评分。真实数据无Mesh/Pose GT，上述证据不能证明唯一三维病因。', '',
        f'独立重渲染{result["independent_native_cells_rerendered"]}个“Cell×阶段”组合，全部在原始时间插值Mesh，复算Dice与保存值之差≤1e−12。'
        '生成代码、CSV和提交端点SHA见 diagnosis.json 与 source/。', '',
        '复现命令：', '',
        '```bash',
        'MPLCONFIGDIR=/tmp/heartssm-dice-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/yilong/.conda/envs/4dheart/bin/python scripts/diagnose_real_tto_dice.py --input outputs/real_tto_joint_v1_20261016 --output outputs/real_tto_dice_diagnosis_20261017',
        '```', '']
    markdown='\n'.join(text)
    (output/'DICE_DIAGNOSIS_REPORT.md').write_text(markdown)
    import html
    parts=[]
    for line in text:
        if line.startswith('!['):
            image_name=line.split('](',1)[1][:-1]
            data=base64.b64encode((output/image_name).read_bytes()).decode()
            parts.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(image_name)}"><figcaption>{html.escape(image_name)} · PNG/PDF原图保存在同目录</figcaption></figure>')
        elif line.startswith('## '):
            parts.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('# '):
            parts.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line:
            parts.append('<p>'+html.escape(line).replace('**','')+'</p>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实TTO低Dice诊断</title><style>body{font:17px/1.8 system-ui,sans-serif;color:#203047;background:#f4f6f9;margin:0}main{max-width:1240px;margin:auto;padding:30px;background:white}h1,h2{color:#142d50}figure{margin:28px 0}img{width:100%;height:auto;cursor:zoom-in}img.expanded{width:1900px;max-width:none;cursor:zoom-out}figcaption{font-size:13px;color:#617080}p{overflow-wrap:anywhere}</style><main>'+''.join(parts)+'</main><script>document.querySelectorAll("img").forEach(x=>x.onclick=()=>x.classList.toggle("expanded"))</script></html>'
    (output/'index.html').write_text(page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/real_tto_joint_v1.yaml')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    root, output=args.input.resolve(),args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    print('COHORT_PLOTS_BEGIN',flush=True)
    names,rows,summary=cohort_figures(root,output)
    cases=NativeCases(root,config)
    print('NATIVE_EXAMPLES_BEGIN',flush=True)
    examples=examples_figure(cases,output)
    print('SHARED_VIEW_DIAGNOSTIC_BEGIN',flush=True)
    alignment=shared_view_diagnostic(cases,output)
    temporal=temporal_figure(rows,output)
    pose=read_json(root/'independent_audit/pose_initialization_diagnostics.json')
    identity=dict(script_sha256=sha256(Path(__file__)), config_sha256=sha256(Path(args.config)),
                  protocol_sha256=sha256(root/'protocol.json'), input_root=str(root),
                  export_state_sha256={name:sha256(root/'patients'/name/'states/export.pth') for name in names},
                  csv_sha256={f'{name}/{stage}':sha256(root/'patients'/name/'evaluation'/(stage+'_cells.csv')) for name in names for stage in STAGES})
    result=dict(updated=time.time(),cohort=summary,examples=examples,shared_alignment=alignment,
                temporal_area_diagnostics=temporal,pose_initialization_aggregate=pose['aggregate'],
                optimization=optimizer_facts(root,names,config),
                independent_native_cells_rerendered=len(cases.rendered),native_agreement_tolerance=1e-12,
                provenance=identity,science_modified=False,
                limitations=['No real 3D Mesh or Probe Pose GT; unique causal attribution unavailable.',
                             '2D alignment uses observed annotations after TTO; it is not a valid ablation or recovered 3D result.',
                             'Pose/Shape reachability, camera orientation and temporal ambiguity need controlled experiments to separate.',
                             'Only committed main exports at snapshot time are included; C03 and pending cases stay in the cohort denominator.'])
    write_json(output/'diagnosis.json',result)
    write_report(result,output)
    source=output/'source'/Path(__file__).name
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    print('DIAGNOSIS_COMPLETE',json.dumps(dict(cohort=summary,alignment=alignment,examples=examples,temporal=temporal),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
