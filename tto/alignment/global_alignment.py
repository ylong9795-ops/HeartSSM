"""S1: frozen canonical meshes, analytic centres, broad native plane search.

No gradient optimiser, mesh/latent updates or free global SE(3) parameters.
Every proposed plane uses the repository's original native renderer.
"""
from copy import deepcopy
from dataclasses import replace
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from data.mask_transform import CHAMBERS, VIEWS
from ..evaluator import native_exact
from ..state import Stage
from ..supervision.observations import observation_cells
from .centres import balanced_correction, centroid, huber_location, measure


def canonical_pose(global_transform, pose):
    g = {k: np.asarray(v.detach().cpu() if isinstance(v, torch.Tensor) else v, float)
         for k, v in global_transform.items()}
    return dict(R=np.einsum('ij,vjk->vik', g['R'].T, np.asarray(pose['R'])),
        center_mm=(np.asarray(pose['center_mm'])-g['center_mm']) @ g['R'])


def shift_centre(pose, correction_xy_mm):
    return dict(R=pose['R'].copy(), center_mm=pose['center_mm']+pose['R'][:, :2] @ correction_xy_mm)


def score_cells(rows):
    """Equal chamber means; temporal weighting uses source/phase confidence."""
    results = {}
    for label, eligible in (('trusted_macro_dice', lambda r: r['fit_weight'] > 0),
                            ('all_macro_dice', lambda r: True)):
        scores = []
        for c in sorted({r['chamber'] for r in rows}):
            selected = [r for r in rows if r['chamber'] == c and eligible(r)]
            if selected:
                weights = [r['fit_weight'] if label.startswith('trusted') else 1. for r in selected]
                scores.append(float(np.average([0. if r.get('renderer_error') else r['exact_hard_dice']
                                                for r in selected], weights=weights)))
        results[label] = float(np.mean(scores)) if scores else None
    results['empty_prediction_cells'] = sum(r['prediction_area_mm2'] == 0 for r in rows)
    results['renderer_errors'] = sum(bool(r.get('renderer_error')) for r in rows)
    return results


class NativePlaneSearch:
    def __init__(self, model, patient, trajectory, policy, config):
        self.model, self.patient, self.trajectory = model, patient, trajectory
        self.config = config
        self.faces = {c: f.detach().cpu().numpy() for c, f in model.faces.items()}
        self.ranges = model.chamber_ranges.tolist()
        self.cells = observation_cells([patient], policy)
        self.fallback_chambers = []
        for vi in range(3):
            for c in range(4):
                group = [r for r in self.cells if r['view'] == vi and r['chamber'] == c]
                trusted = any(r['source_confidence'] >= config['source_min_weight'] and r['used'] for r in group)
                if group and not trusted: self.fallback_chambers.append([vi, c])
                for row in group:
                    eligible = row['used'] and (not trusted or row['source_confidence'] >= config['source_min_weight'])
                    row['fit_weight'] = row['source_confidence']*row['phase_confidence'] if eligible else 0.
        self.render_count = 0

    def render_rows(self, pose, cells):
        rows = []
        for cell in cells:
            vi, ci, phase = cell['view'], cell['chamber'], cell['phase']
            view = self.patient.views[vi]
            target = view['masks'][cell['native_index'], ci]
            a, b = self.ranges[ci]
            vertices = ((1-cell['mesh_alpha'])*self.trajectory[phase, a:b] +
                        cell['mesh_alpha']*self.trajectory[cell['mesh_upper'], a:b])
            error = None
            try:
                pred = native_exact(vertices, self.faces[CHAMBERS[ci]], pose['R'], pose['center_mm'],
                                    view['spacing'], target.shape)
            except Exception as exception:
                pred = np.zeros_like(target); error = repr(exception)
            self.render_count += 1
            row = dict(cell, patient=self.patient.patient_id, view_name=VIEWS[vi], chamber_name=CHAMBERS[ci],
                renderer_error=error, **measure(pred, target, view['spacing']))
            if error:
                row.update(centroid_error_xy_mm=None, centroid_error_norm_mm=None, centroid_status='RENDERER_ERROR',
                           exact_hard_dice=0.)
            rows.append(row)
        return rows

    def capture_centre(self, rotation, cells, depth_coordinate):
        """Projected 3D vertex means vs observed centres; an independent seed.

        Never substituted for a predicted binary centroid in diagnostics.
        """
        values, per_chamber = {}, []
        for cell in cells:
            vi, ci, phase = cell['view'], cell['chamber'], cell['phase']
            view = self.patient.views[vi]
            mask = view['masks'][cell['native_index'], ci]
            mu = centroid(mask)
            if mu is None: continue
            a, b = self.ranges[ci]
            vertices = ((1-cell['mesh_alpha'])*self.trajectory[phase, a:b] +
                        cell['mesh_alpha']*self.trajectory[cell['mesh_upper'], a:b])
            h, w = mask.shape
            offset = vertices.mean(0) @ rotation[:, :2] - (mu-[w/2, h/2])*view['spacing'][::-1]
            values.setdefault(ci, []).append((offset, cell['fit_weight']))
        for group in values.values():
            per_chamber.append(huber_location([v[0] for v in group], [v[1] for v in group], self.config['huber_delta_mm']))
        if not per_chamber: return None
        xy = huber_location(per_chamber, np.ones(len(per_chamber)), self.config['huber_delta_mm'])
        return rotation[:, :2] @ xy + rotation[:, 2]*depth_coordinate

    @staticmethod
    def diverse(candidates, count):
        selected = []
        for candidate in sorted(candidates, key=lambda x: (-x['score'], x['order'])):
            distinct = True
            for other in selected:
                angle = np.rad2deg(np.arccos(np.clip((np.trace(candidate['pose']['R'].T @ other['pose']['R'])-1)/2, -1, 1)))
                if angle < 2. and np.linalg.norm(candidate['pose']['center_mm']-other['pose']['center_mm']) < 2.:
                    distinct = False; break
            if distinct: selected.append(candidate)
            if len(selected) == count: break
        return selected

    def search_view(self, vi, initial_pose):
        config = self.config
        started, records, candidates, seen = time.monotonic(), [], [], set()
        all_cells = [r for r in self.cells if r['view'] == vi]
        baseline_rows = self.render_rows(initial_pose, all_cells)
        fit = [r for r in all_cells if r['fit_weight'] > 0]
        if not fit:
            return dict(view=VIEWS[vi], status='NO_USABLE_OBSERVATIONS', candidates=[],
                baseline_cells=baseline_rows, centre_cells=[], search_records=[], selected_cells=[])
        coarse = []
        for ci in sorted({r['chamber'] for r in fit}):
            group = sorted([r for r in fit if r['chamber'] == ci], key=lambda r: (-r['fit_weight'], r['phase']))
            limit = config['coarse_cells_per_chamber']
            if len(group) > limit:
                weight = max(r['fit_weight'] for r in group)
                best = sorted([r for r in group if r['fit_weight'] == weight], key=lambda r: r['phase'])
                if len(best) >= limit:
                    group = [best[i] for i in np.linspace(0, len(best)-1, limit).round().astype(int)]
                else: group = group[:limit]
            coarse.extend(group)
        baseline_correction = balanced_correction(baseline_rows, config['huber_delta_mm'])
        corrected = (shift_centre(initial_pose, baseline_correction['correction_xy_mm'])
                     if baseline_correction['correction_xy_mm'] is not None else deepcopy(initial_pose))
        centre_rows = self.render_rows(corrected, all_cells)
        budget_hit = False

        def propose(pose, stage, *, recenter=True):
            nonlocal budget_hit
            key = tuple(np.round(np.r_[pose['R'].ravel(), pose['center_mm']], 6))
            if key in seen: return
            if len(records) >= config['max_evaluations_per_view'] or time.monotonic()-started > config['max_seconds_per_view']:
                budget_hit = True; return
            seen.add(key)
            rows = self.render_rows(pose, coarse)
            adjustment = balanced_correction(rows, config['huber_delta_mm'])
            if recenter and adjustment['correction_xy_mm'] is not None:
                shifted = shift_centre(pose, adjustment['correction_xy_mm'])
                shifted_rows = self.render_rows(shifted, coarse)
                if score_cells(shifted_rows)['trusted_macro_dice'] >= score_cells(rows)['trusted_macro_dice']:
                    pose, rows = shifted, shifted_rows
            score = score_cells(rows)
            value = dict(pose=pose, score=score['trusted_macro_dice'], order=len(records), stage=stage,
                         coarse_metrics=score, centre_correction=adjustment)
            records.append(value); candidates.append(value)

        propose(deepcopy(initial_pose), 'initial', recenter=False)
        propose(corrected, 'centre_only', recenter=False)
        tilt, roll = config['coarse_tilt_deg'], config['coarse_roll_deg']
        angles = [(x, y, 0.) for x in (-tilt, 0., tilt) for y in (-tilt, 0., tilt)]
        angles += [(0., 0., -roll), (0., 0., roll)]
        phases = sorted({r['phase'] for r in coarse})
        chamber_vertices = np.concatenate([self.trajectory[t, self.ranges[c][0]:self.ranges[c][1]]
            for t in phases for c in sorted({r['chamber'] for r in coarse})])
        depth_offsets = np.arange(-config['coarse_depth_span_mm'], config['coarse_depth_span_mm']+1e-6,
                                  config['coarse_depth_step_mm'])
        for degrees in angles:
            if budget_hit: break
            rotation = initial_pose['R'] @ Rotation.from_rotvec(np.deg2rad(degrees)).as_matrix()
            origin_depth = float(corrected['center_mm'] @ rotation[:, 2])
            depths = np.unique(np.r_[origin_depth+depth_offsets,
                np.quantile(chamber_vertices @ rotation[:, 2], config['capture_quantiles'])])
            for depth in depths:
                if budget_hit: break
                base_center = corrected['center_mm'] + rotation[:, 2]*(depth-origin_depth)
                propose(dict(R=rotation, center_mm=base_center), 'coarse_depth_tilt')
                capture = self.capture_centre(rotation, coarse, depth)
                if capture is not None:
                    propose(dict(R=rotation, center_mm=capture), 'independent_mesh_capture')
        for step in config['refinement_steps']:
            if budget_hit: break
            beam = self.diverse(candidates, config['beam_width'])
            for parent in beam:
                for axis in range(4):
                    for sign in (-1., 1.):
                        pose = deepcopy(parent['pose'])
                        if axis == 0:
                            pose['center_mm'] += pose['R'][:, 2]*sign*step['depth_mm']
                        else:
                            rotation = np.zeros(3); rotation[axis-1] = sign*step['angle_deg']
                            pose['R'] = pose['R'] @ Rotation.from_rotvec(np.deg2rad(rotation)).as_matrix()
                        propose(pose, 'fine_depth_tilt_roll')
        # Re-rank a larger beam on all evidence, not only coarse sampled cells.
        finalists = self.diverse(candidates, max(config['retain_candidates']*2, config['beam_width']))
        for original in candidates[:2]:
            if all(original['order'] != c['order'] for c in finalists): finalists.append(original)
        full = []
        for candidate in finalists:
            rows = self.render_rows(candidate['pose'], all_cells)
            correction = balanced_correction(rows, config['huber_delta_mm'])
            if correction['correction_xy_mm'] is not None:
                pose = shift_centre(candidate['pose'], correction['correction_xy_mm'])
                revised = self.render_rows(pose, all_cells)
                if score_cells(revised)['trusted_macro_dice'] >= score_cells(rows)['trusted_macro_dice']:
                    candidate = dict(candidate, pose=pose); rows = revised
            metrics = score_cells(rows)
            full.append(dict(candidate, score=metrics['trusted_macro_dice'], metrics=metrics, cells=rows))
        retained = self.diverse(full, config['retain_candidates'])
        return dict(view=VIEWS[vi], status='BUDGET_EXHAUSTED' if budget_hit else 'SEARCH_COMPLETE',
            baseline_cells=baseline_rows, centre_cells=centre_rows, centre_estimate=baseline_correction,
            baseline_metrics=score_cells(baseline_rows), centre_metrics=score_cells(centre_rows),
            candidates=retained, selected_cells=retained[0]['cells'] if retained else [],
            search_records=records, search_evidence=[dict(phase=r['phase'], chamber=CHAMBERS[r['chamber']],
                source=r['source'], weight=r['fit_weight']) for r in coarse],
            evaluations=len(records), elapsed_seconds=time.monotonic()-started,
            pose_status='PROVISIONAL_NATIVE_2D_CANDIDATES; absolute 3D pose unverified')


def align_patient(model, patient, state, *, config, policy, initial_pose):
    state.validate()
    if patient.metadata.get('input_mode') != 'aligned24':
        raise ValueError('S1 requires identity-mapped aligned24 observations')
    trajectory = state.trajectory.detach().cpu().numpy()
    source_pose = ({k: v.detach().cpu().numpy() for k, v in state.observation_pose.items()}
                   if state.observation_pose is not None else initial_pose)
    seed = (canonical_pose(state.global_transform, source_pose) if state.observation_pose is not None
            else {k: np.asarray(v, float) for k, v in source_pose.items()})
    search = NativePlaneSearch(model, patient, trajectory, policy, config)
    results = []
    for vi in range(3):
        result = search.search_view(vi, {k: v[vi].copy() for k, v in seed.items()})
        results.append(result)
        print(f"S1 {patient.patient_id}/{VIEWS[vi]}: {result['status']}; candidates={len(result['candidates'])}", flush=True)
    selected = {k: torch.as_tensor(np.stack([r['candidates'][0]['pose'][k] if r['candidates'] else seed[k][vi]
        for vi, r in enumerate(results)]), dtype=torch.float64, device=state.trajectory.device)
        for k in ('R', 'center_mm')}
    diagnostics = deepcopy(state.diagnostics)
    diagnostics.update(pose_gauge='canonical_identity_global', alignment_status='POSE_CANDIDATES_UNVERIFIED',
        network_updates=0, latent_updates=0, mesh_updates=0, gradient_optimization=False,
        s1=dict(config=config, phase_semantics_status=state.diagnostics.get('s0_status', 'UNVERIFIED'),
            scope='static plane registration only; no patient network/latent/control fine tuning',
            search_complete=all(r['status'] in ('SEARCH_COMPLETE', 'NO_USABLE_OBSERVATIONS') for r in results),
            weak_source_fallback_chambers=search.fallback_chambers, render_calls=search.render_count,
            per_view=[{k: v for k, v in r.items() if k not in ('baseline_cells', 'centre_cells', 'search_records', 'selected_cells')}
                      for r in results]))
    for view in diagnostics['s1']['per_view']:
        view['candidates'] = [{k: v for k, v in c.items() if k != 'cells'} for c in view['candidates']]
    registered = replace(state, global_transform=dict(R=torch.eye(3, device=state.trajectory.device),
        center_mm=torch.zeros(3, device=state.trajectory.device)), observation_pose=selected,
        stage=Stage.GLOBAL_ALIGNED, diagnostics=diagnostics)
    registered.validate()
    return registered, results
