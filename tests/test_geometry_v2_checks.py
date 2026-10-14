"""Independent cross-chamber and section-topology rejection tests."""
import unittest
import numpy as np
import trimesh
from utils.geometry_v2_checks import (chamber_pair_audit,section_topology,four_chamber_audit,
                                     validate_four_chamber_prediction,GeometrySafetyError)
from data.mask_transform import CHAMBERS


class GeometrySafetyTests(unittest.TestCase):
    def test_separate_overlapping_and_contained_closed_surfaces(self):
        sphere=trimesh.creation.icosphere(subdivisions=1,radius=1.)
        far=chamber_pair_audit(sphere.vertices,sphere.faces,sphere.vertices+[4,0,0],sphere.faces)
        self.assertEqual(far['penetrating_vertices'],0)
        overlap=chamber_pair_audit(sphere.vertices,sphere.faces,sphere.vertices+[.5,0,0],sphere.faces)
        self.assertGreater(overlap['surface_contact_pairs'],0)
        self.assertGreater(overlap['penetrating_vertices'],0)
        nested=chamber_pair_audit(sphere.vertices*.5,sphere.faces,sphere.vertices*2,sphere.faces)
        self.assertEqual(nested['surface_contact_pairs'],0)
        self.assertGreater(nested['penetrating_vertices'],0)

    def test_closed_and_legally_empty_plane_sections(self):
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=1.)
        closed=section_topology(sphere.vertices,sphere.faces,np.eye(3),np.zeros(3))
        self.assertEqual(closed['closed_components'],1)
        self.assertEqual(closed['open_or_branch_vertices'],0)
        empty=section_topology(sphere.vertices,sphere.faces,np.eye(3),np.array([0,0,3]))
        self.assertTrue(empty['empty'])

    def test_surface_touch_is_not_silently_certified_as_disjoint(self):
        sphere=trimesh.creation.icosphere(subdivisions=1,radius=1.)
        vertices=np.concatenate([sphere.vertices+center for center in
            ((0,0,0),(2,0,0),(10,0,0),(15,0,0))])
        n=len(sphere.vertices)
        report=four_chamber_audit(vertices,[(i*n,(i+1)*n) for i in range(4)],
                                  {c:sphere.faces for c in CHAMBERS})
        self.assertGreater(report['surface_contact_pairs'],0)
        self.assertFalse(report['strict_disjoint_cavity_safe'])

    def test_every_patient_and_last_phase_actual_mesh_are_checked(self):
        import torch
        from test_geometry_heartffdnet import fixture_model
        model=fixture_model()
        trajectory=model.template_vertices_mm[None,None].expand(2,24,-1,-1).clone()
        report=validate_four_chamber_prediction(model,{'vertices_4d':trajectory})
        self.assertEqual(report['phases_checked'],48)
        # Only the second patient's last phase is invalid; no [0] shortcut.
        trajectory[1,23,42:84,1]-=6
        with self.assertRaises(GeometrySafetyError) as caught:
            validate_four_chamber_prediction(model,{'vertices_4d':trajectory})
        self.assertEqual(caught.exception.report['patient'],1)
        self.assertEqual(caught.exception.report['phase'],23)

    def test_cap_policy_rejects_remote_contacts_and_containment(self):
        from utils.geometry_pair_policy import SourceAVCapPolicy,SETTINGS
        from test_geometry_heartffdnet import fixture_model
        model=fixture_model()
        policy=SourceAVCapPolicy.__new__(SourceAVCapPolicy)
        policy.settings=SETTINGS;policy.identity={'scope':'test'}
        policy.ranges={c:slice(*r) for c,r in zip(CHAMBERS,model.chamber_ranges.tolist())}
        policy.faces={c:model.faces[c].numpy() for c in CHAMBERS}
        policy.caps={c:np.empty(0,dtype=np.int64) for c in CHAMBERS}
        v=model.template_vertices_mm.numpy().copy()
        self.assertEqual(policy.audit_frames(v[None])['status'],'pass')
        v[42:84,1]-=6
        with self.assertRaises(GeometrySafetyError):policy.audit_frames(v[None])
        # Even cap-supported vertices cannot license gross containment.
        policy.caps={c:np.arange(len(f)) for c,f in policy.faces.items()}
        center=v[:42].mean(0)
        v[42:84]=center+(v[:42]-center)*.5
        with self.assertRaises(GeometrySafetyError):policy.audit_frames(v[None])


if __name__=='__main__':unittest.main()
