"""Original-label cache correctness, parallel equality and training IO order."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,default_collate

from data.mask_transform import CHAMBERS,VIEWS,normalize_masks
from train_neural_ffd import evaluate,exact_metrics,training_loader,render_source_slice
from utils.validation_runtime import ValidationRuntime


class FixtureDataset(Dataset):
    manifest_sha256 = 'fixture'
    cache_version = 0

    def __init__(self):
        import trimesh
        from synthetic_bundle_20261009.data_generation.core import section_mask
        mesh = trimesh.creation.icosphere(subdivisions=1,radius=.6)
        vertices = np.concatenate([mesh.vertices+center for center in
                                  ((-1.,-1.,0.),(-1.,1.,0.),(1.,-1.,0.),(1.,1.,0.))])
        vertices = np.stack((vertices,vertices+[.031,0.,0.]))
        n = len(mesh.vertices)
        self.ranges = torch.tensor([[i*n,(i+1)*n] for i in range(4)])
        self.faces = {c:torch.tensor(mesh.faces.copy()) for c in CHAMBERS}
        R = np.broadcast_to(np.eye(3),(2,3,3,3)).copy()
        t = np.broadcast_to([.07,-.13,.137],(2,3,3)).copy()
        s = np.broadcast_to([4.2,3.7,1.],(2,3,3)).copy()
        annotated = np.array([[1,1,0,0],[1,1,0,0],[1,1,1,1]],bool)
        masks = np.zeros((2,3,4,40,40),np.uint8)
        for frame in range(2):
            for view,chamber in np.argwhere(annotated):
                left,right = self.ranges[chamber]
                masks[frame,view,chamber] = section_mask(vertices[frame,left:right],mesh.faces,
                    R[frame,view],t[frame,view],s[frame,view],40)
        spacing = 2*s[...,[1,0]]*10/40
        transformed = normalize_masks(masks,spacing,target_size=32,
            canonical_fov_mm=dict(zip(VIEWS,[48.,50.,52.])),
            view_present=np.ones(3,bool),annotated=annotated)
        self.sample = {key:torch.tensor(value) for key,value in transformed.items()}
        self.sample.update(native_masks=torch.tensor(masks),vertices_gt=torch.tensor(vertices*10),
            sdf=torch.zeros_like(self.sample['masks'],dtype=torch.float32),
            native_pixel_spacing=torch.tensor(spacing),
            pose_gt={k:torch.tensor(v) for k,v in dict(probe_R=R,probe_t=t,probe_s=s).items()},
            view_present=torch.ones(3,dtype=torch.bool),annotated=torch.tensor(annotated),
            phases=torch.arange(2,dtype=torch.float64)/2,phases_gt=torch.arange(2,dtype=torch.float64)/2)
        self.reads = 0

    def __len__(self):
        return 2

    def __getitem__(self,index):
        self.reads += 1
        return dict(self.sample,case_id=f'case-{index}')


class FixtureModel(nn.Module):
    def __init__(self,dataset):
        super().__init__()
        self.register_buffer('chamber_ranges',dataset.ranges.clone())
        self.register_buffer('vertices',dataset.sample['vertices_gt'].clone())
        for c in CHAMBERS:
            self.register_buffer('faces_'+c,dataset.faces[c].clone())
        self.bias = nn.Parameter(torch.tensor(.2,dtype=torch.float64))
        self.norm_to_mm = 10.

    @property
    def faces(self):
        return {c:getattr(self,'faces_'+c) for c in CHAMBERS}

    def forward(self,batch):
        p = self.vertices[None].clone()
        p[...,0] += self.bias
        return dict(vertices_4d=p,vertices_ref=p[:,0])


class ValidationRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data = FixtureDataset()
        self.model = FixtureModel(self.data)

    def test_renderer_matches_recorded_generation_symbol(self):
        from utils.slice_evaluation import source_renderer_identity
        identity = source_renderer_identity()
        self.assertEqual(identity['generation_core_sha256'],
            '4bd081a41efeed061959df89752e9afa79e7bb82c39833af128d7ad1ce4182b5')
        self.assertEqual(identity['section_mask_sha256'],
            '3ae51014042ec4abb82f138e80c0dc3ed326d53050fc22a454ed4847ca6003e1')
        batch = default_collate([self.data[0]])
        # The later topology-preserving renderer must never replace the
        # original generator for prediction or GT consistency checks.
        with patch('synthetic_bundle_20261009.data_generation.core.section_mask',
                   side_effect=AssertionError('Wrong generation version')):
            report = exact_metrics(self.model,self.model(batch)['vertices_4d'].detach(),batch)
        self.assertEqual(report['summary']['gt_native_pixel_mismatches'],0)

    def test_cached_gt_halves_render_calls_without_changing_metrics(self):
        batch = default_collate([self.data[0]])
        prediction = self.model(batch)['vertices_4d'].detach()
        cache = {}
        with patch('train_neural_ffd.render_source_slice',wraps=render_source_slice) as render:
            first = exact_metrics(self.model,prediction,batch,cache)
            self.assertEqual(render.call_count,32)
            render.reset_mock()
            second = exact_metrics(self.model,prediction,batch,cache)
            self.assertEqual(render.call_count,16)
        self.assertEqual(first,second)
        self.assertEqual(first,exact_metrics(self.model,prediction,batch))

    def test_changed_label_or_pose_invalidates_reference_cache(self):
        batch = default_collate([self.data[0]])
        prediction = self.model(batch)['vertices_4d'].detach()
        for field in ('native_masks','pose_gt'):
            cache = {}
            exact_metrics(self.model,prediction,batch,cache)
            changed = dict(batch)
            if field=='native_masks':
                changed[field] = batch[field].clone()
                changed[field][0,0,0,0,0,0] ^= 1
            else:
                changed[field] = {key:value.clone() for key,value in batch[field].items()}
                changed[field]['probe_t'][0,0,0,2] += .5
            with self.assertRaisesRegex(ValueError,'GT renderer/label mismatch'):
                exact_metrics(self.model,prediction,changed,cache)

    def test_parallel_reports_equal_serial_and_new_predictions_are_not_cached(self):
        serial = evaluate(self.model,self.data,'cpu',cases=2,exact=True)
        with ValidationRuntime(workers=2,cache_cases=2) as runtime:
            cold = evaluate(self.model,self.data,'cpu',cases=2,exact=True,runtime=runtime)
            reads = self.data.reads
            warm = evaluate(self.model,self.data,'cpu',cases=2,exact=True,runtime=runtime)
            self.assertEqual(self.data.reads,reads)
            self.assertEqual(cold['cases'],serial['cases'])
            self.assertEqual(warm['cases'],serial['cases'])
            self.assertEqual(cold['performance']['gt_cells_verified'],32)
            self.assertEqual(warm['performance']['gt_cells_reused'],32)
            self.assertEqual(warm['performance']['input_cache_hits'],2)
            with torch.no_grad():
                self.model.bias.add_(1.)
            updated = evaluate(self.model,self.data,'cpu',cases=2,exact=True,runtime=runtime)
            new_serial = evaluate(self.model,self.data,'cpu',cases=2,exact=True)
            self.assertEqual(updated['cases'],new_serial['cases'])
            self.assertNotEqual(updated['cases'],warm['cases'])

    def test_dataset_change_invalidates_cache_and_lru_is_bounded(self):
        with ValidationRuntime(workers=0,cache_cases=1) as runtime:
            evaluate(self.model,self.data,'cpu',cases=2,exact=True,runtime=runtime)
            self.assertLessEqual(len(runtime.inputs),1)
            self.data.cache_version += 1
            self.data.sample['native_masks'][0,0,0,0,0] ^= 1
            with self.assertRaisesRegex(ValueError,'GT renderer/label mismatch'):
                evaluate(self.model,self.data,'cpu',cases=2,exact=True,runtime=runtime)

    def test_spawn_reader_order_and_training_phase_contract_match_serial(self):
        for stage in ('shape','joint'):
            generator = torch.Generator().manual_seed(7)
            serial = list(training_loader(self.data,[1,0],batch_size=1,workers=0,prefetch=2,
                          device='cpu',generator=generator,stage=stage))
            generator = torch.Generator().manual_seed(7)
            parallel = list(training_loader(self.data,[1,0],batch_size=1,workers=2,prefetch=2,
                            device='cpu',generator=generator,stage=stage))
            self.assertEqual([x['case_id'] for x in parallel],[x['case_id'] for x in serial])
            for a,b in zip(serial,parallel):
                self.assertTrue(torch.equal(a['masks'],b['masks']))
                self.assertTrue(torch.equal(a['sdf'],b['sdf']))
                self.assertTrue(torch.equal(a['vertices_gt'],b['vertices_gt']))
                self.assertNotIn('native_masks',b)
                self.assertEqual('pose_gt' in b,stage=='joint')
                self.assertTrue(torch.equal(b['phases'],b['phases_gt']))


if __name__=='__main__':
    unittest.main()
