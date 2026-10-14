"""Pinned source AV cap support for the two source valve interfaces.

Reuse source topology/predicate functions; never infer exemptions from a
patient, prediction, training outcome, or a new SSM decoder.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from data.mask_transform import CHAMBERS

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'synthetic_bundle_20261009'
SUPPORT=BUNDLE/'configs/geometry/source_support.json'
SUPPORT_SHA256='faa136bcfcacbc0db5be95b01875654077f84c2f31b728e2d59cf388ba7f78e4'
SOURCE_SHA256='4c68caa46eb96ee83f604011b2ace79c2e30297220f7e0cb2ae00b0bd7b03b87'
ALLOWED_PAIRS=(('LV','LA'),('RV','RA'))
SETTINGS=dict(policy='fixed-source-av-cap-support-v1',source_support_sha256=SUPPORT_SHA256,
    allowed_pairs=[list(p) for p in ALLOWED_PAIRS],boundary_tolerance_mm=1e-4,
    maximum_remote_face_pairs=0,maximum_noncap_penetrating_vertices=0,maximum_other_pair_contacts=0)


def source_modules():
    # Both paths are needed by the original geometry package. No source files
    # are copied or installed into the existing conda environment.
    for path in (BUNDLE,BUNDLE/'src'):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    from heart4d.geometry import inter_chamber_policy,certificate
    return inter_chamber_policy,certificate


class SourceAVCapPolicy:
    def __init__(self,model,settings):
        if settings!=SETTINGS:raise ValueError('Fixed source AV geometry settings differ')
        self.settings=dict(settings)
        raw=SUPPORT.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=SUPPORT_SHA256:raise ValueError('Frozen source support hash differs')
        bundle=json.loads(raw)
        if bundle['source_identity']!=SOURCE_SHA256:raise ValueError('Source asset identity differs')
        policy,_=source_modules()
        self.ranges={c:slice(*r) for c,r in zip(CHAMBERS,model.chamber_ranges.cpu().tolist())}
        self.faces={c:model.faces[c].cpu().numpy() for c in CHAMBERS}
        rows=policy.validate_bundle(bundle,self.faces,self.ranges)
        self.rows={row.pair:row for row in rows if row.pair in ALLOWED_PAIRS}
        self.caps={c:np.empty(0,dtype=np.int64) for c in CHAMBERS}
        for pair,row in self.rows.items():
            self.caps[pair[0]]=np.asarray(row.allowed_face_support_a,dtype=np.int64)
            self.caps[pair[1]]=np.asarray(row.allowed_face_support_b,dtype=np.int64)
        source_files=('src/heart4d/geometry/inter_chamber_policy.py','src/heart4d/geometry/certificate.py',
                      'src/heart4d/geometry/topology.py','src/heart4d/geometry/policy.py')
        self.identity=dict(settings=self.settings,source_asset_sha256=SOURCE_SHA256,
            source_policy_hash=bundle['policy_hash'],topology_hash=bundle['topology_hash'],
            support_sha256=SUPPORT_SHA256,
            source_implementation_sha256={p:hashlib.sha256((BUNDLE/p).read_bytes()).hexdigest() for p in source_files},
            cap_face_counts={c:len(ids) for c,ids in self.caps.items()})

    def audit_frames(self,vertices,*,workers=4):
        """Actual surfaces at every supplied frame; only pinned cap IDs exempt.

        This is a discrete-frame audit. Continuous native CCD is separate;
        neither the safety contract nor its support is inferred from results.
        """
        from itertools import combinations
        from utils.geometry_v2_checks import chamber_pair_audit, GeometrySafetyError
        value=np.asarray(vertices,dtype=np.float64)
        if value.ndim!=3 or value.shape[-1]!=3 or not np.isfinite(value).all():
            raise ValueError('Finite [T,N,3] source/prediction required')
        if len(value)>1 and workers>1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts=list(pool.map(lambda frame:self.audit_frames(frame[None],workers=1),value))
            records=[dict(record,phase=phase) for phase,part in enumerate(parts) for record in part['records']]
            return dict(status='pass',frames_checked=len(value),pairs_checked=len(records),
                records=records,policy_identity=self.identity,scope='actual discrete-frame surfaces')
        records=[]
        for phase,frame in enumerate(value):
            for a,b in combinations(CHAMBERS,2):
                pair=(a,b)
                row=chamber_pair_audit(frame[self.ranges[a]],self.faces[a],
                    frame[self.ranges[b]],self.faces[b],
                    boundary_tolerance_mm=self.settings['boundary_tolerance_mm'])
                hits=np.asarray(row.get('contact_face_pairs',[]),dtype=np.int64).reshape(-1,2)
                allowed=(np.isin(hits[:,0],self.caps[a]) & np.isin(hits[:,1],self.caps[b])
                         if pair in ALLOWED_PAIRS else np.zeros(len(hits),dtype=bool))
                illegal=0;gross=False
                for chamber,direction in ((a,'a_inside_b'),(b,'b_inside_a')):
                    enclosed=np.asarray(row['directions'].get(direction,{}).get('indices',[]),dtype=np.int64)
                    cap_vertices=(np.unique(self.faces[chamber][self.caps[chamber]])
                                  if pair in ALLOWED_PAIRS else np.empty(0,dtype=np.int64))
                    illegal+=int((~np.isin(enclosed,cap_vertices)).sum())
                    gross|=len(enclosed)==len(frame[self.ranges[chamber]])
                record=dict(phase=phase,pair=list(pair),surface_contact_pairs=len(hits),
                    expected_cap_contacts=int(allowed.sum()),remote_face_pairs=int((~allowed).sum()),
                    noncap_penetrating_vertices=illegal,gross_containment=gross,
                    maximum_enclosed_boundary_depth_mm=row['maximum_penetration_mm'])
                records.append(record)
                if record['remote_face_pairs'] or illegal or gross:
                    raise GeometrySafetyError('Unsupported cross-chamber interaction',record)
        return dict(status='pass',frames_checked=len(value),pairs_checked=len(records),
            records=records,policy_identity=self.identity,scope='actual discrete-frame surfaces')

    def certify(self,vertices,*,continuous=False):
        """All supplied frames, and optionally every chord including 23->0.

        The old certificate demotes AV failures to diagnostics. This adapter
        explicitly PROMOTES unsupported AV failures back to rejection; allowed
        cap face contacts are the only exception. Source VV/atrial exceptions
        are absent here; only the two immutable source AV interfaces apply.
        """
        _,certificate=source_modules()
        value=np.asarray(vertices,dtype=np.float64)
        if continuous:
            if value.shape[0]!=24:raise ValueError('Complete24-phase trajectory required')
            value=np.concatenate((value,value[:1]),axis=0)  # audit only; network still has24 independent frames
        report=certificate.certify_geometry(value,self.faces,self.ranges,cap_face_ids=self.caps,
            require_continuous=continuous,check_trajectory=continuous,include_av_diagnostics=True)
        promoted={key:number for key,number in report.get('diagnostic_failures',{}).items()
                  if 'cross_chamber' in key}
        report['failures'].update(promoted)
        report['passed']=bool(report['passed'] and not promoted)
        # Additional explicit containment/depth diagnostics use the verified
        # native VTK operator already present in v2. No depth limit is fitted
        # to outcomes; the blocking threshold is zero outside pinned caps.
        from utils.geometry_v2_checks import chamber_pair_audit
        containment=[]
        from itertools import combinations
        for phase,frame in enumerate(value):
            for pair in combinations(CHAMBERS,2):
                a,b=pair;ra,rb=self.ranges[a],self.ranges[b]
                row=chamber_pair_audit(frame[ra],self.faces[a],frame[rb],self.faces[b],
                    boundary_tolerance_mm=self.settings['boundary_tolerance_mm'])
                illegal=0;gross=False
                for chamber,direction in ((a,'a_inside_b'),(b,'b_inside_a')):
                    direction_report=row['directions'].get(direction,{})
                    enclosed=np.asarray(direction_report.get('indices',[]),dtype=np.int64)
                    allowed_vertices=(np.unique(self.faces[chamber][self.caps[chamber]])
                                      if pair in ALLOWED_PAIRS else np.empty(0,dtype=np.int64))
                    illegal+=int((~np.isin(enclosed,allowed_vertices)).sum())
                    gross|=len(enclosed)==len(frame[self.ranges[chamber]])
                if illegal or gross:
                    report['failures'][f'geometry:noncap_or_gross_containment:{a}-{b}:{phase}']=max(illegal,1)
                containment.append(dict(phase=phase,pair=list(pair),noncap_penetrating_vertices=illegal,
                    gross_containment=gross,maximum_enclosed_boundary_depth_mm=row['maximum_penetration_mm'],
                    penetrating_vertices=row['penetrating_vertices']))
        report['passed']=bool(report['passed'] and not report['failures'])
        report.update(av_cap_policy_identity=self.identity,noncap_containment=containment,
            allowance_scope='Only immutable source LV/LA and RV/RA cap Face-IDs',
            depth_scope='Enclosed-surface distance is diagnostic; zero noncap penetration is the threshold')
        return report


def bind_geometry_policy(model,settings):
    model._source_av_cap_policy=SourceAVCapPolicy(model,settings)
    return model._source_av_cap_policy.identity


def validate_source_geometry(model):
    from utils.geometry_joint_checks import validate_template
    validate_template(model)
    return model._source_av_cap_policy.audit_frames(model.template_vertices_mm.detach().cpu().numpy()[None])


def validate_patient_geometry(model,output):
    vertices=output['vertices_4d'].detach().cpu().numpy()
    if vertices.ndim!=4 or vertices.shape[1]!=24:
        raise ValueError('Complete [B,24,N,3] geometry required')
    return dict(status='pass',patients=[model._source_av_cap_policy.audit_frames(case) for case in vertices])


def validate_output_geometry(model,output):
    """Frozen physical certificate plus native CGAL actual triangle checks."""
    from utils.geometry_joint_checks import validate_prediction
    from utils.geometry_v2_checks import GeometrySafetyError
    report=validate_prediction(model,output,sample_jacobian=True,check_intersections=False)
    _,certificate=source_modules()
    *_,pmp=certificate.backends()
    vertices=output['vertices_4d'].detach().cpu().numpy()
    checks=0
    for patient,cycle in enumerate(vertices):
        for phase,frame in enumerate(cycle):
            for chamber,region in model._source_av_cap_policy.ranges.items():
                pairs=[]
                pmp.self_intersections(certificate.polyhedron(frame[region],model._source_av_cap_policy.faces[chamber]),pairs)
                if pairs:
                    raise GeometrySafetyError('Native CGAL actual mesh self-intersection',
                        dict(patient=patient,phase=phase,chamber=chamber,pairs=len(pairs)))
                checks+=1
    report.update(actual_mesh_self_intersecting_pairs=0,actual_chamber_phase_checks=checks,
                  actual_intersection_backend='pinned source CGAL',inter_chamber=validate_patient_geometry(model,output))
    return report
