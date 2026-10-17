"""Independent Exact Native rendering and source-stratified cell reporting."""
import csv
from pathlib import Path

import numpy as np
import trimesh

from data.mask_transform import CHAMBERS, VIEWS
from utils.slice_evaluation import section_mask, source_renderer_identity
from .io import write_json


def native_exact(vertices, faces, rotation, center_mm, spacing_yx, shape):
    """Adapt legacy square fillPoly to native rectangles without moving pixel0."""
    h, w = map(int, shape)
    dy, dx = map(float, spacing_yx)
    side = max(h, w)
    rotation, center_mm = np.asarray(rotation, float), np.asarray(center_mm, float)
    pixel0 = center_mm - rotation @ np.array([w*dx/2, h*dy/2, 0.])
    padded_center = pixel0 + rotation @ np.array([side*dx/2, side*dy/2, 0.])
    # Historical section_mask catches section errors. Expose them before calling it.
    trimesh.Trimesh(vertices, faces, process=False).section(plane_origin=padded_center, plane_normal=rotation[:, 2])
    return section_mask(vertices, faces, rotation, padded_center,
        np.array([side*dx/2, side*dy/2, 1.]), image_size=side)[:h, :w]


def dice_stats(rows):
    x = np.array([r['exact_hard_dice'] for r in rows], float)
    if not len(x):
        return dict(cells=0, mean=None, minimum=None, P01=None, P05=None, ge095=0, ge098=0)
    return dict(cells=len(x), mean=float(x.mean()), minimum=float(x.min()),
        P01=float(np.quantile(x, .01)), P05=float(np.quantile(x, .05)),
        ge095=int((x >= .95).sum()), ge098=int((x >= .98).sum()))



class ExactNativeEvaluator:
    def __init__(self, model):
        self.model = model
        self.faces = {c: f.cpu().numpy() for c, f in model.faces.items()}

    def cells(self, trajectory, pose, patient, stage, phases=None):
        trajectory = np.asarray(trajectory)
        rows = []
        for phase in (range(24) if phases is None else phases):
            native = patient.native(phase)
            for vi, view in enumerate(VIEWS):
                for ci, chamber in enumerate(CHAMBERS):
                    if not native['annotated'][vi, ci]:
                        continue
                    target = native['masks'][vi][ci].astype(bool)
                    a, b = self.model.chamber_ranges[ci].tolist()
                    error = None
                    try:
                        prediction = native_exact(trajectory[phase, a:b], self.faces[chamber], pose['R'][vi],
                            pose['center_mm'][vi], native['spacing'][vi], target.shape).astype(bool)
                        denominator = int(prediction.sum()+target.sum())
                        dice = 2*int((prediction & target).sum())/denominator if denominator else 1.
                    except Exception as exception:
                        error = repr(exception);prediction = np.zeros_like(target);dice = 0.
                    mapping = patient.mapping[phase][vi]
                    rows.append(dict(patient=patient.patient_id, stage=stage, phase=phase,
                        view=view, chamber=chamber, exact_hard_dice=float(dice),
                        native_frame=mapping['nearest_native_frame'], time_error_ms=mapping['nearest_time_error_ms'],
                        label_source=native['sources'][vi][ci], source_weight=float(native['weights'][vi, ci]),
                        prediction_area_mm2=float(prediction.sum()*np.prod(native['spacing'][vi])),
                        target_area_mm2=float(target.sum()*np.prod(native['spacing'][vi])), renderer_error=error))
        return rows

    def evaluate(self, trajectory, pose, patient, *, stage, motion=None, geometry=None, output=None):
        rows = self.cells(trajectory, pose, patient, stage)
        report = dict(patient=patient.patient_id, stage=stage, all_observations=dice_stats(rows),
            by_source={s: dice_stats([r for r in rows if r['label_source'] == s])
                       for s in sorted({r['label_source'] for r in rows})},
            by_view={v: dice_stats([r for r in rows if r['view'] == v]) for v in VIEWS},
            by_chamber={c: dice_stats([r for r in rows if r['chamber'] == c]) for c in CHAMBERS},
            motion_diagnostics=motion, geometry=geometry, renderer=source_renderer_identity(),
            acquisition_status=patient.metadata['contract'].get('acquisition_status', 'UNVERIFIED'),
            interpretation='Independent native 2D agreement; no real 3D/Pose ground truth')
        if output is not None:
            output = Path(output);output.mkdir(parents=True, exist_ok=True)
            with (output/'cells.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['patient'])
                writer.writeheader();writer.writerows(rows)
            write_json(output/'metrics.json', report)
        return report, rows
