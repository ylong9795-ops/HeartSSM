"""Actual NeuralFFDNet surface audits using pinned repository geometry kernels.

No GeometryHeartFFDNet control certificate is applied to this older network.
Unavailable backends are UNVERIFIED and never admit a candidate.
"""
import numpy as np
import trimesh

from data.mask_transform import CHAMBERS
from utils.geometry_pair_policy import SourceAVCapPolicy, SETTINGS, source_modules


class GeometryGuard:
    def __init__(self, model):
        self.model = model
        self.faces = {c: f.detach().cpu().numpy() for c, f in model.faces.items()}
        self.ranges = {c: slice(*r) for c, r in zip(CHAMBERS, model.chamber_ranges.tolist())}
        self.backend_error = None
        try:
            self.policy = SourceAVCapPolicy(model, SETTINGS)
            _, self.certificate = source_modules()
            self.pmp = self.certificate.backends()[-1]
        except Exception as error:
            self.backend_error = repr(error)

    def discrete(self, vertices):
        vertices = np.asarray(vertices, np.float64)
        if vertices.ndim == 2:
            vertices = vertices[None]
        if self.backend_error:
            return dict(status='UNVERIFIED', passed=False, error=self.backend_error, frames=len(vertices))
        records, cross, errors = [], [], []
        for phase, frame in enumerate(vertices):
            for c, region in self.ranges.items():
                try:
                    points = frame[region]
                    if not np.isfinite(points).all():
                        raise ValueError('Nonfinite surface vertices')
                    mesh = trimesh.Trimesh(points, self.faces[c], process=False)
                    pairs = []
                    self.pmp.self_intersections(self.certificate.polyhedron(points, self.faces[c]), pairs)
                    area = float((mesh.area_faces*2).min())
                    passed = not pairs and area >= 1e-8 and mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
                    records.append(dict(phase=phase, chamber=c, passed=bool(passed), self_intersections=len(pairs),
                        min_twice_area_mm2=area, watertight=bool(mesh.is_watertight),
                        winding_consistent=bool(mesh.is_winding_consistent), signed_volume_ml=float(mesh.volume/1000)))
                except Exception as error:
                    errors.append(dict(phase=phase, chamber=c, error=repr(error)))
            try:
                cross.append(dict(phase=phase, passed=True, details=self.policy.audit_frames(frame[None], workers=1)))
            except Exception as error:
                cross.append(dict(phase=phase, passed=False, error=repr(error), details=getattr(error, 'report', None)))
        verified = not errors
        passed = verified and all(r['passed'] for r in records) and all(r['passed'] for r in cross)
        return dict(status=('PASS' if passed else 'FAIL') if verified else 'UNVERIFIED', passed=bool(passed),
            frames=len(vertices), mesh_records=records, cross_chamber=cross, errors=errors,
            scope='supplied discrete surfaces; fixed source AV cap support; no continuous claim')

    def continuous(self, trajectory):
        if self.backend_error:
            return dict(status='UNVERIFIED', passed=False, error=self.backend_error)
        try:
            value = np.asarray(trajectory, np.float64)
            if value.shape != (24, len(self.model.template_vertices_mm), 3):
                raise ValueError('Full24 fixed-topology trajectory required')
            report = self.policy.certify(value, continuous=True)
            return dict(status='PASS' if report['passed'] else 'FAIL', passed=bool(report['passed']),
                details=report, scope='all24 piecewise-linear vertex chords including23->0; not a continuous neural decoder')
        except Exception as error:
            return dict(status='UNVERIFIED', passed=False, error=repr(error))

    def audit(self, trajectory, *, continuous=True):
        value = np.asarray(trajectory)
        shape = self.discrete(value[0])
        frames = self.discrete(value)
        temporal = self.continuous(value) if continuous else dict(status='UNVERIFIED', passed=False, reason='not requested')
        cross = frames.get('cross_chamber', [])
        cross_verified = len(cross) == len(value) and not self.backend_error
        cross_pass = cross_verified and all(x['passed'] for x in cross)
        return dict(passed=shape['passed'] and frames['passed'] and temporal['passed'],
            shape=shape, saved_phases=frames, inter_chamber=dict(
                status=('PASS' if cross_pass else 'FAIL') if cross_verified else 'UNVERIFIED',
                passed=bool(cross_pass), records=cross), continuous=temporal,
            spatial_ffd_certificate=dict(status='UNVERIFIED', reason='No compatible whole-field certificate for original NeuralFFDNet'),
            topology='unchanged source chamber faces and vertex identities')
