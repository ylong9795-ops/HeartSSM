"""No temporal remapping, named metadata, and physical pretrained input contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data.mask_transform import CHAMBERS, VIEWS
from tto.aligned24 import PHASE_SEMANTICS
from tto.config import load_config, validate_config
from tto.dataset import RealTTODataset, collate_patients
from tto.pipeline import TTOPipeline
from tto.supervision.observations import observation_cells


class Aligned24Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / 'case'; self.folder.mkdir()
        self.config = load_config('configs/tto.yaml')
        self.config['data']['root'] = str(self.root)
        self.channels = ['LV', 'RV', 'LA', 'RA']
        specs = []
        self.raw = {}
        for vi, name in enumerate(VIEWS):
            h, w = 38+vi*2, 54+vi*2
            masks = np.zeros((24, 4, h, w), np.uint8)
            for t in range(24):
                for c in range(4):
                    masks[t, c, 9+c*4:12+c*4, 10+t%9:13+t%9] = 1
            self.raw[name] = masks
            np.savez(self.folder / f'{name}.npz', masks=masks)
            sources = [['manual_ed_es', 'patient_adapted_model', 'manual_review', 'temporal_fallback']]*24
            annotated = np.ones((24, 4), bool)
            # Missing LA in frame 5 tests permutation of availability too.
            annotated[5, 2] = False
            specs.append(dict(name=name, channels=self.channels, masks_npz=f'{name}.npz',
                times_s=(np.cumsum(np.arange(1, 25))*.013+vi*.001).tolist(),
                spacing_yx_mm=[.7, 1.2], annotated=annotated.tolist(), sources=sources,
                frame_ids=list(range(100, 124))))
        self.contract = dict(version='real-tto-input-v1', patient_id='case', input_mode='aligned24',
            phases=(np.arange(24)/24).tolist(), views=specs[::-1],
            # Invalid RR metadata must NEVER be read in aligned24 mode.
            cycle_duration_s=-1, cycle_start_s=99999, anchor_time_s=-99999,
            phase_alignment=dict(status='UNVERIFIED', phase_grid_status='UNVERIFIED',
                evidence='source reference not established', phase_grid_evidence='nonuniform native sample times',
                synthetic_reference=PHASE_SEMANTICS))
        self.write()

    def write(self):
        (self.folder / 'tto_input.json').write_text(json.dumps(self.contract))

    def load(self):
        return RealTTODataset(self.root, ['case'], config=self.config['data'])[0]

    def test_identity_frames_named_channels_sources_availability_and_native_cells(self):
        with patch.object(RealTTODataset, '_repository_contract', side_effect=AssertionError('RR path')):
            p = self.load()
        batch = collate_patients([p])['observations']
        self.assertEqual(batch['masks'].shape, (1, 24, 3, 4, 224, 224))
        self.assertEqual(batch['sdf'].shape, batch['masks'].shape)
        self.assertEqual(batch['view_present'].shape, (1, 24, 3))
        self.assertEqual(batch['annotated'].shape, (1, 24, 3, 4))
        self.assertEqual(batch['phases'].shape, (1, 24))
        for vi, name in enumerate(VIEWS):
            np.testing.assert_array_equal(p.views[vi]['masks'], self.raw[name][:, [0, 2, 1, 3]])
            for t in range(24):
                self.assertEqual(p.mapping[t][vi]['lower_index'], t)
                self.assertEqual(p.mapping[t][vi]['upper_index'], t)
                self.assertEqual(p.mapping[t][vi]['upper_weight'], 0.)
            self.assertEqual(p.native(5)['sources'][vi], ['manual_ed_es', 'manual_review', 'patient_adapted_model', 'temporal_fallback'])
            np.testing.assert_allclose(p.native(0)['weights'][vi], [1., 1., .25, .1])
        self.assertFalse(bool(batch['annotated'][0, 5, 0, 1]))
        self.assertTrue(bool(batch['annotated'][0, 5, 0, 2]))
        rows = observation_cells([p], self.config['supervision'])
        self.assertEqual(len(rows), 24*3*4-3)
        self.assertTrue(all(r['native_index'] == r['phase'] and r['mesh_alpha'] == 0 for r in rows))
        self.assertFalse(TTOPipeline(self.config).preflight(p)['s0_input_audit']['ready_for_shape_tto'])

    def test_physical_centres_spacing_and_sdf_use_mm_once(self):
        p = self.load()
        for vi, view in enumerate(p.views):
            matrix = np.asarray(p.metadata['model_to_native'])[0, vi]
            h, w = view['masks'].shape[-2:]
            np.testing.assert_allclose(matrix @ [112, 112, 1], [w/2, h/2, 1], atol=1e-10)
            # One model-pixel step has exactly the declared physical spacing.
            mm = np.asarray(p.metadata['model_spacing_yx_mm'])[0, vi]
            np.testing.assert_allclose(matrix[0, 0]*1.2, mm[1])
            np.testing.assert_allclose(matrix[1, 1]*.7, mm[0])
        valid = p.encoder['annotated'][..., None, None].expand_as(p.encoder['sdf'])
        inside = (p.encoder['masks'] > 0) & valid
        outside = (p.encoder['masks'] == 0) & valid
        self.assertTrue((p.encoder['sdf'][inside] < 0).all())
        self.assertTrue((p.encoder['sdf'][outside] > 0).all())
        self.assertEqual(float(p.encoder['sdf'].abs().max()), 15.)
        self.assertTrue((p.encoder['sdf'][~valid] == 0).all())
        self.assertTrue((p.encoder['sdf'][inside].abs() > .9).any())

    def test_precomputed_encoder_consumed_and_wrong_units_values_or_centre_rejected(self):
        p = self.load()
        values = {k: v.numpy() for k, v in p.encoder.items() if isinstance(v, torch.Tensor)}
        values.update(views=np.array(VIEWS), channels=np.array(CHAMBERS), sdf_units=np.array('mm'),
            model_to_native=np.asarray(p.metadata['model_to_native']),
            pixel_spacing_yx_mm=np.asarray(p.metadata['model_spacing_yx_mm']))
        self.contract['encoder_npz'] = 'encoder.npz'; self.write()
        np.savez(self.folder / 'encoder.npz', **values)
        restored = self.load()
        torch.testing.assert_close(restored.encoder['sdf'], p.encoder['sdf'], rtol=0, atol=0)
        # Encoder archives may independently declare legacy chamber/view order.
        reordered = dict(values, views=np.array(VIEWS)[[2, 0, 1]], channels=np.array(self.channels))
        for key in ('masks', 'sdf', 'annotated'):
            reordered[key] = values[key][:, [2, 0, 1]][:, :, [0, 2, 1, 3]]
        for key in ('view_present', 'model_to_native', 'pixel_spacing_yx_mm'):
            reordered[key] = values[key][:, [2, 0, 1]]
        np.savez(self.folder / 'encoder.npz', **reordered)
        torch.testing.assert_close(self.load().encoder['sdf'], p.encoder['sdf'], rtol=0, atol=0)
        for key, bad in [('sdf_units', np.array('dimensionless')),
                         ('sdf', values['sdf']/15),
                         ('model_to_native', values['model_to_native']+1)]:
            np.savez(self.folder / 'encoder.npz', **dict(values, **{key: bad}))
            with self.assertRaisesRegex(ValueError, 'SDF|encoder'):
                self.load()

    def test_bad_phases_duplicates_and_raw_mode_are_rejected(self):
        original = deepcopy(self.contract)
        for key, bad in [('phases', np.linspace(0, 1, 24).tolist()),
                         ('input_mode', 'raw_rr')]:
            self.contract = deepcopy(original); self.contract[key] = bad; self.write()
            with self.assertRaises(ValueError): self.load()
        self.contract = deepcopy(original)
        self.contract['views'][0]['channels'] = ['LV', 'LA', 'RV', 'RV']; self.write()
        with self.assertRaises(ValueError): self.load()
        self.contract = deepcopy(original); self.write()
        self.config['data']['input_mode'] = 'raw_rr'
        with self.assertRaisesRegex(ValueError, 'never enter'):
            self.load()
        config = deepcopy(self.config); config['data'].pop('input_mode')
        with self.assertRaises(ValueError): validate_config(config)


if __name__ == '__main__':
    unittest.main()
