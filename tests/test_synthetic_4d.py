"""Small certified fixtures; no 10K dependency or geometry generation.

Identity/tamper and reference-vs-cycle assertions adapted from EchoGSF
tests/s1/unit/test_dataset.py (157aa7624cfda8e7319ec325af5d1088065f026d).
Run: python -m unittest discover -s tests -v
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mask_transform import MaskTransform, mask_to_boundary, mask_to_sdf, normalize_masks
from data.synthetic_4d import Synthetic4DDataset, VISIBLE, _coordinate_hash, _sha256


class Synthetic4DTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.asset = self.root / "asset.json"
        self.asset.write_text(json.dumps(dict(format="wholeheart-original-ssm-v1", scale_mm=10.,
            center_mm=[1, 2, 3], vertices=12, temporal_resampling="explicit_linear_closure_v1")))
        self.asset_hash = _sha256(self.asset)
        (self.root / "plan.json").write_text('{}')
        self.manifest = self.root / "train.json"
        self.rows = []
        self.payloads = []
        for index in range(2):
            vertices = np.arange(36, dtype=np.float32).reshape(1, 12, 3) / 20 + index
            vertices = np.repeat(vertices, 25, axis=0)
            vertices[:, :, 0] += np.sin(np.arange(25) * 2 * np.pi / 24).astype(np.float32)[:, None] / 10
            vertices[-1] = vertices[0]
            masks = np.zeros((25, 3, 4, 16, 20), np.uint8)
            for view, chamber in np.argwhere(VISIBLE):
                masks[:, view, chamber, 2 + chamber:5 + chamber, 3 + view:8 + view] = 1
            payload = dict(vertices=vertices, masks=masks, asset_sha256=np.array(self.asset_hash),
                probe_R=np.broadcast_to(np.eye(3, dtype=np.float32), (25, 3, 3, 3)).copy(),
                probe_t=np.zeros((25, 3, 3), np.float32), probe_s=np.ones((25, 3, 3), np.float32))
            self.payloads.append(payload)
            self.rows.append(dict(candidate_id=index, path=f"{index}.npz", qc_path=f"{index}.qc.json"))
            self.write_case(index)
        self.write_manifest()

    def write_manifest(self, **changes):
        manifest = dict(format="wholeheart-simple-training-v2", complete=True, accepted_count=2,
                        asset_sha256=self.asset_hash, asset_path="asset.json",
                        plan_sha256=_sha256(self.root / "plan.json"), samples=self.rows)
        manifest.update(changes)
        self.manifest.write_text(json.dumps(manifest))

    def write_case(self, index):
        path = self.root / f"{index}.npz"
        np.savez_compressed(path, **self.payloads[index])
        coordinate = _coordinate_hash(self.payloads[index]["vertices"])
        qc = self.root / f"{index}.qc.json"
        qc.write_text(json.dumps(dict(status="accept", metrics=dict(coordinate_sha256=coordinate))))
        self.rows[index].update(sha256=_sha256(path), qc_sha256=_sha256(qc), coordinate_sha256=coordinate)
        if self.manifest.exists():
            self.write_manifest()

    def dataset(self, **options):
        return Synthetic4DDataset(self.manifest, self.asset, **options)

    def test_case_loading(self):
        data = self.dataset()
        self.assertEqual(len(data), 2)
        sample = data[0]
        self.assertEqual(sample['masks'].shape, (24, 3, 4, 16, 20))
        self.assertEqual(sample['asset_sha256'], self.asset_hash)
        np.testing.assert_array_equal(sample['vertices_gt'], self.payloads[0]['vertices'][:-1].astype(float) * 10)
        for key in ('probe_R', 'probe_t', 'probe_s'):
            np.testing.assert_array_equal(sample['pose_gt'][key], self.payloads[0][key][:-1])
        self.assertNotIn('proposal_coefficients', sample)
        self.assertEqual(sample['masks'].dtype, torch.uint8)

    def test_lazy_loading(self):
        (self.root / '0.npz').unlink()
        dataset = self.dataset()
        with self.assertRaisesRegex(ValueError, 'Case 0'):
            dataset[0]

    def test_phase_handling(self):
        sample = self.dataset()[0]
        np.testing.assert_allclose(sample['phases'], np.arange(24) / 24)
        whole = self.dataset(expected_phases=25, remove_duplicate_endpoint=False)[0]
        self.assertEqual(len(whole['masks']), 25)
        self.assertTrue(torch.equal(whole['vertices_gt'][0], whole['vertices_gt'][-1]))
        self.assertEqual(whole['phases'][-1].item(), 1)

    def test_reference_frame(self):
        cycle = self.dataset()[1]
        reference = self.dataset(reference_frame=6)[1]
        for key in ('masks', 'vertices_gt', 'native_masks', 'phases'):
            self.assertTrue(torch.equal(reference[key][0], cycle[key][6]))
        self.assertEqual(reference['case_id'], cycle['case_id'])

    def test_chamber_mapping(self):
        original = self.dataset()[0]
        vi, ci = [2, 0, 1], [3, 1, 0, 2]
        payload = self.payloads[0]
        payload['masks'] = payload['masks'][:, vi][:, :, ci]
        for key in ('probe_R', 'probe_t', 'probe_s'):
            payload[key] = payload[key][:, vi]
        payload['view_names'] = np.array(['A4C', 'A2C', 'A3C'])
        payload['chamber_names'] = np.array(['RA', 'LA', 'LV', 'RV'])
        payload['channel_annotated'] = VISIBLE[vi][:, ci]
        self.write_case(0)
        self.assertTrue(torch.equal(self.dataset()[0]['masks'], original['masks']))
        del payload['channel_annotated']
        self.write_case(0)
        self.assertTrue(torch.equal(self.dataset()[0]['masks'], original['masks']))

    def test_missing_annotations(self):
        payload = self.payloads[0]
        payload['channel_annotated'] = VISIBLE.copy()
        payload['channel_annotated'][0, 0] = False
        payload['masks'][:, 0, 1] = 0  # annotated empty, distinct from missing
        self.write_case(0)
        sample = self.dataset(mask_transform=MaskTransform())[0]
        self.assertFalse(sample['annotated'][0, 0])
        self.assertTrue(sample['annotated'][0, 1])
        self.assertFalse(sample['masks'][:, 0, 0].any())
        self.assertFalse(sample['sdf'][:, 0, 0].any())
        self.assertTrue((sample['sdf'][:, 0, 1] == 15).all())

    def test_per_frame_validity(self):
        payload = self.payloads[0]
        payload['view_present'] = np.ones((25, 3), bool)
        payload['channel_annotated'] = np.repeat(VISIBLE[None], 25, axis=0)
        payload['view_present'][7, 1] = False
        payload['channel_annotated'][7, 1] = False
        self.write_case(0)
        sample = self.dataset(mask_transform=MaskTransform())[0]
        self.assertEqual(sample['view_present'].shape, (24, 3))
        self.assertEqual(sample['annotated'].shape, (24, 3, 4))
        self.assertFalse(sample['masks'][7, 1].any())
        self.assertFalse(sample['sdf'][7, 1].any())
        self.assertTrue(sample['masks'][6, 1].any())

    def test_mask_normalization(self):
        masks = self.payloads[0]['masks'][:1]
        result = normalize_masks(masks, (2, 1), target_size=(32, 40),
                                 canonical_fov_mm={v: (64, 40) for v in ('A2C', 'A3C', 'A4C')})
        self.assertEqual(result['masks'].shape, (1, 3, 4, 32, 40))
        np.testing.assert_allclose(result['pixel_spacing'][0, 0], (2, 1))
        matrix = result['model_to_native'][0, 0]
        np.testing.assert_allclose(matrix @ [20, 16, 1], [10, 8, 1])
        yy, xx = np.nonzero(result['masks'][0, 0, 0])
        self.assertAlmostEqual(xx.mean() - 20, 5 - 10)
        self.assertAlmostEqual(yy.mean() - 16, 3 - 8)
        self.assertEqual(int(result['masks'][0, 0, 0].sum()), int(masks[0, 0, 0].sum()))

    def test_identity_transform(self):
        masks = self.payloads[0]['masks'][:1]
        result = normalize_masks(masks, (2, 1))
        np.testing.assert_array_equal(result['masks'], masks)
        np.testing.assert_array_equal(result['model_to_native'][0, 0], np.eye(3))

    def test_resize_nearest_and_physical_scale(self):
        masks = np.zeros((1, 3, 4, 10, 10), np.uint8)
        masks[0, 0, 0, 3:6, 3:6] = 1
        result = normalize_masks(masks, (1, 1), target_size=20)
        self.assertEqual(set(np.unique(result['masks'])), {0, 1})
        np.testing.assert_array_equal(result['pixel_spacing'][0, 0], [.5, .5])
        self.assertEqual(int(result['masks'].sum()), 36)

    def test_clipping_rejected(self):
        masks = self.payloads[0]['masks'][:1]
        with self.assertRaisesRegex(ValueError, 'clips'):
            normalize_masks(masks, (1, 1), canonical_fov_mm={v: 2 for v in ('A2C', 'A3C', 'A4C')})

    def test_sdf_sign(self):
        sdf = mask_to_sdf(self.payloads[0]['masks'][:1], (1, 1))
        self.assertLess(sdf[0, 0, 0, 3, 4], 0)
        self.assertGreater(sdf[0, 0, 0, 0, 0], 0)

    def test_sdf_spacing(self):
        masks = np.zeros((1, 3, 4, 7, 7), np.uint8)
        masks[0, 0, 0, 3, 3] = 1
        sdf = mask_to_sdf(masks, (2, 3), clip_mm=None)
        self.assertEqual(sdf[0, 0, 0, 2, 3], 2)
        self.assertEqual(sdf[0, 0, 0, 3, 2], 3)
        self.assertAlmostEqual(float(sdf[0, 0, 0, 2, 2]), np.sqrt(13), places=5)

    def test_sdf_clipping_normalization_and_full(self):
        masks = np.ones((1, 3, 4, 7, 7), np.uint8)
        masks[0, 0, 1] = 0
        sdf = mask_to_sdf(masks, (1, 1), clip_mm=5, normalize=True)
        self.assertTrue((sdf[0, 0, 0] == -1).all())
        self.assertTrue((sdf[0, 0, 1] == 1).all())
        with self.assertRaises(ValueError):
            mask_to_sdf(masks, (1, 1), clip_mm=None, normalize=True)

    def test_boundary(self):
        masks = self.payloads[0]['masks'][:1]
        boundary = mask_to_boundary(masks, annotated=VISIBLE)
        self.assertEqual(boundary[0, 0, 0, 3, 5], 0)
        self.assertEqual(boundary[0, 0, 0, 2, 5], 1)
        self.assertFalse(boundary[0, 0, 2].any())

    def test_batch_loading(self):
        dataset = self.dataset(mask_transform=MaskTransform())
        for batch_size, workers, context in [(1, 0, None), (2, 0, None), (2, 2, 'fork'), (2, 2, 'spawn')]:
            batch = next(iter(DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                        multiprocessing_context=context, timeout=30 if workers else 0)))
            self.assertEqual(batch['masks'].shape, (batch_size, 24, 3, 4, 16, 20))
            self.assertEqual(len(set(batch['case_id'])), batch_size)
            for i in range(batch_size):
                self.assertTrue(torch.equal(batch['vertices_gt'][i], dataset[i]['vertices_gt']))
                self.assertTrue(torch.equal(batch['pose_gt']['probe_R'][i], dataset[i]['pose_gt']['probe_R']))

    def test_mesh_identity(self):
        self.payloads[0]['vertices'] = self.payloads[0]['vertices'][:, :-1]
        self.write_case(0)
        with self.assertRaisesRegex(ValueError, 'vertex count'):
            self.dataset()[0]

    def test_fixed_template(self):
        template = torch.tensor(self.payloads[0]['vertices'][0])
        ranges = torch.tensor([[0, 3], [3, 6], [6, 9], [9, 12]])
        faces = {c: torch.tensor([[0, 1, 2]]) for c in ('LV', 'LA', 'RV', 'RA')}
        global_faces = torch.cat([faces[c] + 3 * i for i, c in enumerate(faces)])
        digest = hashlib.sha256()
        for value in (template, global_faces, ranges):
            digest.update(value.numpy().tobytes())
        path = self.root / 'template.pth'
        state = dict(template_vertices=template, vertex_ranges=ranges,
                     **{'faces.' + c: f for c, f in faces.items()})
        specification = dict(asset_sha256=self.asset_hash, norm_to_mm=10., template_sha256=digest.hexdigest())
        torch.save(dict(model=state, specification=specification), path)
        dataset = self.dataset(template=path, template_file_sha256=_sha256(path))
        self.assertEqual(dataset[0]['template_sha256'], digest.hexdigest())
        state['faces.LV'][0, 1] = 2
        torch.save(dict(model=state, specification=specification), path)
        with self.assertRaisesRegex(ValueError, 'topology identity'):
            self.dataset(template=path)

    def test_invalid_sample(self):
        (self.root / '0.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Modified source file'):
            self.dataset()[0]

    def test_invalid_closure_all_arrays(self):
        for key in ('masks', 'vertices', 'probe_R', 'probe_t', 'probe_s'):
            with self.subTest(key=key):
                original = self.payloads[0][key].copy()
                self.payloads[0][key][-1].flat[0] += 1
                self.write_case(0)
                with self.assertRaises(ValueError):
                    self.dataset()[0]
                self.payloads[0][key] = original

    def test_invalid_availability(self):
        self.payloads[0]['view_present'] = np.array([0, 1, 1])
        self.write_case(0)
        with self.assertRaisesRegex(ValueError, 'absent view'):
            self.dataset()[0]

    def test_qc_and_coordinate_tamper(self):
        qc = self.root / '0.qc.json'
        qc.write_text(json.dumps(dict(status='reject', metrics=dict(coordinate_sha256=self.rows[0]['coordinate_sha256']))))
        self.rows[0]['qc_sha256'] = _sha256(qc)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Unaccepted'):
            self.dataset()[0]
        self.write_case(0)
        self.rows[0]['coordinate_sha256'] = 'bad'
        qc.write_text(json.dumps(dict(status='accept', metrics=dict(coordinate_sha256='bad'))))
        self.rows[0]['qc_sha256'] = _sha256(qc)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'coordinate hash'):
            self.dataset()[0]

    def test_invalid_plan_and_manifest(self):
        (self.root / 'plan.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'generation plan'):
            self.dataset()
        self.write_manifest(complete=False)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.dataset()

    def test_duplicate_membership_and_ed_repair(self):
        self.write_manifest(samples=[self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.dataset()
        self.write_manifest(format='wholeheart-ed-observation-repair-v1')
        with self.assertRaisesRegex(ValueError, 'ED-only'):
            self.dataset()

    def test_absolute_path_relocation(self):
        for row in self.rows:
            row['path'] = '/archive/' + row['path']
            row['qc_path'] = '/archive/' + row['qc_path']
        self.write_manifest()
        sample = self.dataset(path_map={'/archive': self.root})[0]
        self.assertEqual(sample['candidate_id'], 0)

    def test_sharded_manifest(self):
        path = self.root / 'sharded.json'
        path.write_text(json.dumps(dict(format='wholeheart-sharded-training-v1', complete=True,
            samples=[dict(path='.', manifest_sha256=_sha256(self.manifest))])))
        self.manifest.rename(self.root / 'manifest.json')
        data = Synthetic4DDataset(path, self.asset)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[1]['candidate_id'], 1)

    def test_invalid_parameters(self):
        for options in [dict(expected_phases=23), dict(reference_frame=-1), dict(reference_frame=True)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.dataset(**options)
        with self.assertRaises(ValueError):
            mask_to_sdf(self.payloads[0]['masks'][:1], (0, 1))

    def test_nonbinary_nonfinite_and_unknown_order(self):
        self.payloads[0]['masks'][0, 0, 0, 0, 0] = 2
        self.write_case(0)
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.dataset()[0]
        self.payloads[0]['masks'][0, 0, 0, 0, 0] = 0
        self.payloads[0]['vertices'][3, 0, 0] = np.nan
        self.write_case(0)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.dataset()[0]
        self.payloads[0]['vertices'][3, 0, 0] = 0
        self.payloads[0]['view_names'] = np.array(['A2C', 'SAX', 'A4C'])
        self.write_case(0)
        with self.assertRaisesRegex(ValueError, 'Unknown view'):
            self.dataset()[0]

    def test_original_case_identity(self):
        self.rows[0]['case_id'] = 'original-case-zero'
        self.write_manifest()
        self.assertEqual(self.dataset()[0]['case_id'], 'original-case-zero')
        self.rows[1]['case_id'] = 'original-case-zero'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'original case IDs'):
            self.dataset()

    def test_verify_false_keeps_qc_and_closure_checks(self):
        self.payloads[0]['probe_t'][-1, 0, 0] = 1
        np.savez_compressed(self.root / '0.npz', **self.payloads[0])
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            self.dataset(verify=False)[0]

    def test_binary_255_storage(self):
        self.payloads[0]['masks'] *= 255
        self.write_case(0)
        sample = self.dataset(mask_transform=MaskTransform())[0]
        self.assertEqual(set(torch.unique(sample['masks']).tolist()), {0, 1})
        self.assertTrue(torch.isfinite(sample['sdf']).all())


if __name__ == '__main__':
    unittest.main()
