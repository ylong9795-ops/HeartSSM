"""CPU dataset, native renderer and storage contract tests; no patient-specific fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from tto.config import load_config
from tto.dataset import RealTTODataset, collate_patients, DEFAULT_VALID
from tto.evaluator import native_exact
from tto.geometry.guard import GeometryGuard
from tto.io import ROOT, load_state, save_state
from data.mask_transform import CHAMBERS, VIEWS
from utils.slice_evaluation import section_mask


def fixture(root, patient, offset=0):
    folder = root/patient;folder.mkdir()
    views = []
    for vi, name in enumerate(VIEWS):
        h, w = 28+2*vi+offset, 36+2*vi+offset
        masks = np.zeros((4, 4, h, w), np.uint8)
        yy, xx = np.mgrid[:h, :w]
        for t in range(4):
            for ci in range(4):
                if DEFAULT_VALID[vi, ci]:
                    masks[t, ci] = (xx-w/2-ci*.5)**2+(yy-h/2)**2 < (4+t*.2)**2
        np.savez(folder/f'{name}.npz', masks=masks)
        np.save(folder/f'{name}.npy', np.zeros((4,h,w), np.uint8))
        views.append(dict(name=name, masks_npz=f'{name}.npz', images_npy=f'{name}.npy',
            spacing_yx_mm=[.7, 1.1], channels=list(CHAMBERS), times_s=[0., .25, .5, .75],
            annotated=np.broadcast_to(DEFAULT_VALID[vi], (4,4)).tolist(),
            sources=[['manual_ed_es' if c else 'not_applicable' for c in DEFAULT_VALID[vi]]]*4))
    manifest = dict(version='real-tto-input-v1', patient_id=patient, cycle_start_s=0.,
        cycle_duration_s=.5, anchor_time_s=0., ED_frame=0, ES_frame=1, views=views)
    (folder/'tto_input.json').write_text(json.dumps(manifest))


class RealTTOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture(self.root, 'new_patient');fixture(self.root, 'other_patient', 2)
        self.config = load_config(ROOT/'configs/tto.yaml')['data']
        self.config.update(model_size=32, canonical_fov_mm={v:60. for v in VIEWS})
        self.config['load_images'] = True

    def tearDown(self):
        self.temporary.cleanup()

    def test_heterogeneous_native_centres_sources_and_batch_isolation(self):
        dataset = RealTTODataset(self.root, ['new_patient','other_patient'], config=self.config)
        p, q = dataset[0], dataset[1]
        batch = collate_patients([p,q])
        self.assertEqual(tuple(batch['observations']['masks'].shape), (2,24,3,4,32,32))
        self.assertIs(batch['patients'][0], p)
        self.assertNotEqual(p.views[0]['masks'].shape, q.views[0]['masks'].shape)
        self.assertEqual(p.native()['sources'][0][0], 'manual_ed_es')
        self.assertEqual(float(p.native()['weights'][0,0]), 1.)
        self.assertEqual(p.native()['masks'][0].shape[-2:], (28,36))
        for vi, view in enumerate(p.views):
            transform = np.array(p.metadata['model_to_native'])[0, vi]
            centre = transform @ np.array([16.,16.,1.])
            h,w = view['masks'].shape[-2:]
            np.testing.assert_allclose(centre, [w/2,h/2,1.], atol=1e-10)
        self.assertEqual(len(p.images), 3)
        before = q.views[0]['masks'].copy();p.views[0]['masks'][0] = 0
        np.testing.assert_array_equal(before, q.views[0]['masks'])
        with self.assertRaises(ValueError):collate_patients([q,q])

    def test_foreign_patient_and_unknown_annotation_trust_rejected(self):
        path = self.root/'new_patient/tto_input.json';data = json.loads(path.read_text())
        data['patient_id'] = 'other_patient';path.write_text(json.dumps(data))
        dataset = RealTTODataset(self.root,['new_patient'],config=self.config)
        with self.assertRaises(ValueError):dataset[0]
        data['patient_id'] = 'new_patient';data['views'][0]['sources'][0][0] = 'unknown_auto_label'
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):dataset[0]

    def test_native_exact_square_parity_and_rectangular_physical_translation(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=12.)
        r = np.eye(3);spacing = [.7,1.1];center = np.array([.137,-.263,.619])
        for side in (64,128,224):
            actual = native_exact(mesh.vertices,mesh.faces,r,center,spacing,(side,side))
            expected = section_mask(mesh.vertices,mesh.faces,r,center,np.array([side*1.1/2,side*.7/2,1.]),image_size=side)
            np.testing.assert_array_equal(actual,expected)
        a = native_exact(mesh.vertices,mesh.faces,r,center,spacing,(70,94))
        b = native_exact(mesh.vertices+np.array([1.1,0,0]),mesh.faces,r,center,spacing,(70,94))
        c = native_exact(mesh.vertices+np.array([0,.7,0]),mesh.faces,r,center,spacing,(70,94))
        np.testing.assert_array_equal(a[:,:-1],b[:,1:]);np.testing.assert_array_equal(a[:-1],c[1:])


    def test_atomic_resume_rejects_foreign_input(self):
        p = self.root/'state.pth'
        identity = dict(patient='arbitrary', input_sha256='abc')
        save_state(p,dict(identity=identity,parameter=torch.arange(3.),optimizer={'step':1},
            native_time_error=np.float64(.125),geometry=np.array([True,False]),source=np.str_('manual_ed_es')))
        restored=load_state(p,identity)
        self.assertTrue(torch.equal(restored['parameter'],torch.arange(3.)))
        self.assertEqual(restored['native_time_error'],.125)
        self.assertEqual(restored['geometry'],[True,False])
        with self.assertRaises(ValueError):load_state(p,dict(patient='foreign',input_sha256='abc'))


    def test_geometry_unavailable_is_unverified(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        v = torch.tensor(mesh.vertices,dtype=torch.float32)
        model = SimpleNamespace(template_vertices_mm=torch.cat([v]*4),chamber_ranges=torch.tensor([[i*len(v),(i+1)*len(v)] for i in range(4)]),
            faces={c:torch.tensor(mesh.faces) for c in CHAMBERS})
        with patch('tto.geometry.guard.SourceAVCapPolicy',side_effect=ImportError('missing backend')):
            report = GeometryGuard(model).audit(np.tile(model.template_vertices_mm.numpy()[None],(24,1,1)))
        self.assertFalse(report['passed'])
        for key in ('shape','saved_phases','inter_chamber','continuous'):
            self.assertEqual(report[key]['status'],'UNVERIFIED')


if __name__ == '__main__':
    unittest.main()
