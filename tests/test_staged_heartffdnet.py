"""G1 reference/loss/all-parameter gradient and future freeze contracts."""
import copy
import unittest
from unittest.mock import patch
import torch
from tests.test_geometry_heartffdnet import fixture_model, observations
from data.mask_transform import CHAMBERS
from net.geometry_heartffdnet import loss_output_adapter
from net.staged_heartffdnet import StagedHeartFFDNet, reference_loss
from utils.neural_ffd_losses import NeuralFFDLoss


class StagedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)

    def test_reference_loss_and_all_gradients_equal_g1(self):
        g1 = fixture_model().eval()
        stage = copy.deepcopy(g1)
        stage.__class__ = StagedHeartFFDNet
        x = observations()
        truth = g1.template_vertices_mm[None,None].expand(1,24,-1,-1)+.3
        criterion = NeuralFFDLoss(g1.chamber_ranges,g1.faces)
        a = g1(x)
        la = criterion(loss_output_adapter(a),truth,
                       control_spacings_mm=g1.control_spacings_mm,stage='shape')
        la['total'].backward()
        with patch.object(stage,'_deform',wraps=stage._deform) as deform:
            b = stage.reference_output(x)
        self.assertEqual(deform.call_count,3)
        torch.testing.assert_close(a['vertices_ref'],b['vertices_ref'],rtol=0,atol=0)
        lb = reference_loss(criterion,b,truth,stage.control_spacings_mm)
        torch.testing.assert_close(la['total'],lb['total'],rtol=0,atol=0)
        for name in ('shape','edge','laplacian','normal','control'):
            torch.testing.assert_close(la['terms'][name],lb['terms'][name],rtol=0,atol=0)
        lb['total'].backward()
        for (name,p),(other,q) in zip(g1.named_parameters(),stage.named_parameters()):
            self.assertEqual(name,other)
            self.assertIsNotNone(p.grad,name)
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-8,msg=name)
        self.assertEqual(float(stage.heads.deform3.conv.weight.grad[3:].abs().max()),0.)
        poisoned = truth.clone();poisoned[:,1:] = float('nan')
        reference_loss(criterion,b,poisoned,stage.control_spacings_mm)

    def test_frozen_reference_and_reserved_motion(self):
        model = fixture_model();model.__class__ = StagedHeartFFDNet
        model.freeze_reference().train()
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        x = observations()
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            v = model(x)
            torch.testing.assert_close(v,model.predict_reference(x),rtol=0,atol=0)
        for name,value in before.items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        with self.assertRaises(NotImplementedError):model.predict_motion(x,v)



# Input-pipeline tests use workspace-local scratch; never default /tmp.
from pathlib import Path
import hashlib
import json
import tempfile
from types import SimpleNamespace
from utils.staged_data import (CachedStageSamples, EpochOrder, make_loader,
                               configure_runtime_temp, workspace_path)


class CacheFixture:
    def __init__(self,root,count=5):
        self.verify=True;self.manifest_sha256='manifest';self.asset_sha256='asset'
        self.template_sha256='template';self.norm_to_mm=1.;self.expected_phases=24
        self.reference_frame=None;self.remove_duplicate_endpoint=True;self.calls=0
        self.mask_transform=SimpleNamespace(options={'target_size':224},use_sdf=True,sdf_clip_mm=15.,sdf_normalize=False)
        self.sources=[]
        for i in range(count):
            p=root/f'source{i}';q=root/f'qc{i}';p.write_bytes(str(i).encode());q.write_bytes(b'qc')
            self.sources.append(dict(path=str(p),qc_path=str(q),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),qc_sha256=hashlib.sha256(b'qc').hexdigest()))
    def __getitem__(self,index):
        self.calls+=1
        return dict(case_id=str(index),sdf=torch.arange(24,dtype=torch.float32)+index,
                    sample_sha256=self.sources[index]['sha256'],template_sha256=self.template_sha256)


class StagedDataTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]/'outputs/goal04s/acceleration_checks'
        self.root.mkdir(parents=True,exist_ok=True)
        configure_runtime_temp(self.root/'runtime_tmp')
        self.tmp=tempfile.TemporaryDirectory(dir=self.root)
        self.path=Path(self.tmp.name);self.ds=CacheFixture(self.path)
    def tearDown(self):self.tmp.cleanup()

    def test_lossless_hit_no_reprocessing_and_config_invalidation(self):
        cache=CachedStageSamples(self.ds,range(5),self.path/'cache')
        first=cache[0];second=cache[0]
        self.assertEqual(self.ds.calls,1)
        torch.testing.assert_close(first['sdf'],second['sdf'],rtol=0,atol=0)
        other=CachedStageSamples(self.ds,range(5),self.path/'cache')
        other[0];self.assertEqual(self.ds.calls,1) # persistent across instances
        self.ds.mask_transform.sdf_clip_mm=10.
        changed=CachedStageSamples(self.ds,range(5),self.path/'cache')
        self.assertNotEqual(cache.namespace,changed.namespace)
        changed[0];self.assertEqual(self.ds.calls,2)

    def test_modified_source_and_corrupt_cache_rejected(self):
        cache=CachedStageSamples(self.ds,range(5),self.path/'cache');cache[0]
        row,path=cache.entry(0)
        with path.open('ab') as f:f.write(b'corruption')
        with self.assertRaisesRegex(ValueError,'Corrupt'):cache[0]
        Path(row['path']).write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError,'Modified source'):cache[0]

    def test_prefetch_order_resume_and_rng(self):
        cache=CachedStageSamples(self.ds,range(5),self.path/'cache')
        sampler=EpochOrder(5,2026)
        loader=make_loader(cache,workers=2,prefetch=1,sampler=sampler,pin_memory=False)
        state=torch.get_rng_state().clone()
        for epoch,offset in ((0,0),(1,2),(2,0)):
            sampler.set_position(epoch,offset)
            ids=[int(batch['case_id'][0]) for batch in loader]
            expected=torch.randperm(5,generator=torch.Generator().manual_seed(2026+epoch)).tolist()[offset:]
            self.assertEqual(ids,expected)
        torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
        del loader

    def test_workspace_only_paths(self):
        with self.assertRaisesRegex(ValueError,'workspace'):workspace_path('/tmp/forbidden-stage-s-cache')

class StagedRuntimeTests(unittest.TestCase):
    def test_loader_upgrade_rejects_network_source_change(self):
        from train_staged_heartffd import check_resume_code,code_identity,LEGACY_LOADER_TRAINER_SHA256
        old=code_identity();old.pop('utils/staged_data.py')
        old['train_staged_heartffd.py']=LEGACY_LOADER_TRAINER_SHA256
        saved={'runtime':{'code_identity':old}}
        with self.assertRaisesRegex(ValueError,'Source identity'):check_resume_code(saved)
        check_resume_code(saved,True)
        old['net/staged_heartffdnet.py']='tampered'
        with self.assertRaisesRegex(ValueError,'Source identity'):check_resume_code(saved,True)

    def test_cached_trainer_upgrade_keeps_cache_implementation_pinned(self):
        from train_staged_heartffd import check_resume_code,code_identity,CACHED_TRAINER_SHA256
        old=code_identity();old['train_staged_heartffd.py']=CACHED_TRAINER_SHA256
        saved={'runtime':{'code_identity':old}}
        check_resume_code(saved,True)
        old['utils/staged_data.py']='modified-cache'
        with self.assertRaisesRegex(ValueError,'Source identity'):check_resume_code(saved,True)

    def test_execution_policy_preserved_without_state_dict_changes(self):
        from train_staged_heartffd import configure_execution
        model=fixture_model();before={k:v.clone() for k,v in model.state_dict().items()}
        policy=configure_execution(model,'off')
        self.assertFalse(model.encoder.checkpoint_cnn)
        configure_execution(model,'on');self.assertTrue(model.encoder.checkpoint_cnn)
        configure_execution(model,saved={'runtime':{'execution':policy}})
        self.assertFalse(model.encoder.checkpoint_cnn)
        for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)

    def test_expired_evaluation_is_marked_incomplete(self):
        import time
        from train_staged_heartffd import evaluate
        result=evaluate(torch.nn.Identity(),[{'value':torch.ones(1)}],'cpu',deadline=time.monotonic()-1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['cases'],[])





class RefinementTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def fixture(self):
        from tests.test_neural_ffd import BatchedSliceTests
        helper=BatchedSliceTests();helper.setUp()
        p,x,ranges,faces=helper.fixture()
        # Deliberately different native sampling; supervision remains T=1.
        def one(value):
            if isinstance(value,dict):return {k:one(v) for k,v in value.items()}
            return value[:1,:1] if isinstance(value,torch.Tensor) else value
        p=p[:1,:1];x=one(x)
        from utils.slice_supervision import exact_section_mask
        from data.mask_transform import mask_to_sdf
        native=torch.zeros(1,1,3,4,48,48,dtype=torch.double)
        spacing=x['pixel_spacing']*32/48
        for v in range(3):
            for c,(a,b) in enumerate(ranges):
                native[0,0,v,c]=torch.tensor(exact_section_mask(p[0,0,a:b].numpy(),faces[CHAMBERS[c]].numpy(),
                    x['pose_gt']['probe_R'][0,0,v].numpy(),x['pose_gt']['probe_t'][0,0,v].numpy()*2.5,
                    spacing[0,0,v].numpy(),(48,48)),dtype=torch.double)
        sdf=mask_to_sdf(native[0].numpy(),spacing[0].numpy(),view_present=x['view_present'][0].numpy(),annotated=x['annotated'][0].numpy())
        x.update(native_masks=native,native_sdf=torch.tensor(sdf)[None],native_pixel_spacing=spacing,native_sdf_units='mm')
        return p+torch.tensor([.19,-.11,.03]),x,ranges,faces

    def test_native_grid_ignores_model_targets_and_preserves_gradients(self):
        from utils.slice_supervision import MeshSliceSupervisor
        from data.mask_transform import CHAMBERS
        p,x,ranges,faces=self.fixture();x=copy.deepcopy(x)
        explicit=dict(x,masks=x['native_masks'],sdf=x['native_sdf'],pixel_spacing=x['native_pixel_spacing'])
        reference=None
        for backend in ('serial','batched'):
            value=p.clone().requires_grad_()
            slicer=MeshSliceSupervisor(ranges,faces,2.5,pixel_stride=2,execution_backend=backend,supervision_grid='native_grid')
            poisoned=dict(x,masks=torch.full_like(x['masks'],float('nan')),sdf=torch.full_like(x['sdf'],float('nan')),
                          pixel_spacing=x['pixel_spacing']*100)
            y=slicer(value,poisoned);grad=torch.autograd.grad(y['total'],value)[0]
            expected=MeshSliceSupervisor(ranges,faces,2.5,pixel_stride=2,execution_backend=backend)(value,explicit)
            torch.testing.assert_close(y['total'],expected['total'],rtol=1e-10,atol=1e-10)
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)
            if reference is None:reference=(y['total'].detach(),grad)
            else:
                torch.testing.assert_close(y['total'],reference[0],rtol=1e-10,atol=1e-10)
                torch.testing.assert_close(grad,reference[1],rtol=1e-9,atol=1e-9)
        missing=dict(x);missing.pop('native_sdf')
        with self.assertRaisesRegex(ValueError,'native_grid requires'):slicer(p,missing)
        bad=dict(x,native_sdf_units='normalized')
        with self.assertRaisesRegex(ValueError,'physical mm'):slicer(p,bad)

    def test_chamber_balancing_is_mean_of_chamber_means(self):
        from utils.slice_supervision import MeshSliceSupervisor
        p,x,ranges,faces=self.fixture()
        # Retain the actual2/2/4 visibility, giving6 left vs2 right cells.
        visible=torch.tensor([[1,1,0,0],[1,1,0,0],[1,1,1,1]],dtype=torch.bool)[None,None]
        x['annotated']=visible;x['view_present']=visible.any(-1)
        expected=[]
        for c in range(4):
            single=copy.deepcopy(x);single['annotated']=visible.clone();single['annotated'][...,:]=False
            single['annotated'][...,c]=visible[...,c];single['view_present']=single['annotated'].any(-1)
            expected.append(MeshSliceSupervisor(ranges,faces,2.5,supervision_grid='native_grid',pixel_stride=2)(p,single)['total'])
        for backend in ('serial','batched'):
            result=MeshSliceSupervisor(ranges,faces,2.5,supervision_grid='native_grid',pixel_stride=2,
                                      reduction='chamber_mean',execution_backend=backend)(p,x)
            torch.testing.assert_close(result['total'],torch.stack(expected).mean(),rtol=1e-10,atol=1e-10)

    def test_native_sdf_cache_reuses_existing_physical_sdf_operator(self):
        from utils.reference_refinement import NativeReferenceSamples
        import utils.reference_refinement as module
        from data.mask_transform import mask_to_sdf
        root=Path(__file__).resolve().parents[1]/'outputs/goal04s/refinement_checks';root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as path:
            mask=torch.zeros(24,3,4,32,32,dtype=torch.uint8);mask[...,8:24,8:24]=1
            sample=dict(native_masks=mask,native_pixel_spacing=torch.full((24,3,2),.7),
                view_present=torch.ones(3,dtype=torch.bool),annotated=torch.ones(3,4,dtype=torch.bool),
                sample_sha256='source',qc_sha256='qc',template_sha256='template',norm_to_mm=1.)
            samples=NativeReferenceSamples([sample],path)
            with patch.object(module,'mask_to_sdf',wraps=mask_to_sdf) as op:
                a=samples[0];b=samples[0];self.assertEqual(op.call_count,1)
            self.assertEqual(tuple(a['native_sdf'].shape),(1,3,4,32,32))
            torch.testing.assert_close(a['native_sdf'],b['native_sdf'],rtol=0,atol=0)

    def test_completed_evaluation_cache_requires_same_state_and_protocol(self):
        from utils.reference_refinement import cached_full_report
        root=Path(__file__).resolve().parents[1]/'outputs/goal04s/refinement_checks'
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path=Path(directory)/'report.json'
            payload=dict(complete=True,model_state_sha256='state',baseline_report_sha256='baseline',code_identity={'source':'hash'})
            path.write_text(json.dumps(payload))
            self.assertIsNotNone(cached_full_report(path,'state','baseline',{'source':'hash'}))
            self.assertIsNone(cached_full_report(path,'changed','baseline',{'source':'hash'}))
            self.assertIsNone(cached_full_report(path,'state','other-baseline',{'source':'hash'}))
            payload['complete']=False;path.write_text(json.dumps(payload))
            self.assertIsNone(cached_full_report(path,'state','baseline',{'source':'hash'}))

    def test_guards_reject3d_unknown_and_tail_degradation(self):
        from utils.reference_refinement import refinement_guard
        import yaml
        root=Path(__file__).resolve().parents[1]
        baseline=json.loads((root/'outputs/goal04s/train800_cosine40_complete_memoryfast/best_full_val/val_evaluation.json').read_text())
        protocol=yaml.safe_load((root/'configs/staged_heartffd_refine.yaml').read_text())
        self.assertTrue(refinement_guard(baseline,baseline,protocol)['eligible'])
        self.assertFalse(refinement_guard(baseline,baseline,protocol)['hard_target_met'])
        for key in ('surface_assd_mm','reference_vertex_rmse_mm'):
            bad=copy.deepcopy(baseline);bad['summary'][key]+=.1
            guard=refinement_guard(bad,baseline,protocol)
            self.assertFalse(guard['eligible']);self.assertIn(key,guard['failures'])
        bad=copy.deepcopy(baseline)
        for case in bad['cases']:
            for c in CHAMBERS:
                region=case['metrics'][c]['regions']['unknown']
                if region['count']:region['rmse_mm']*=2
        self.assertIn('unknown_rmse_mm',refinement_guard(bad,baseline,protocol)['failures'])
        bad=copy.deepcopy(baseline);bad['cases'][0]['exact']['cells'][0]['exact_dice']=.1
        self.assertIn('minimum_dice',refinement_guard(bad,baseline,protocol)['failures'])
        bad=copy.deepcopy(baseline);bad['cases'][0]['exact']['cells'].pop()
        self.assertFalse(refinement_guard(bad,baseline,protocol)['eligible'])


class RefinementAccelerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2026)

    def test_large_chunks_preserve_native_loss_and_vertex_gradients(self):
        from utils.slice_supervision import MeshSliceSupervisor
        p,x,ranges,faces=RefinementTests().fixture()
        for stride in (4,2):
            for reduction in ('cell_mean','chamber_mean'):
                reference=None
                for chunk in (256,1024,4096):
                    v=p.clone().requires_grad_()
                    loss=MeshSliceSupervisor(ranges,faces,2.5,supervision_grid='native_grid',
                        execution_backend='batched',pixel_stride=stride,chunk_size=chunk,reduction=reduction)(v,x)['total']
                    grad=torch.autograd.grad(loss,v)[0]
                    if reference is None:reference=(loss.detach(),grad)
                    else:
                        torch.testing.assert_close(loss,reference[0],rtol=1e-10,atol=1e-10)
                        torch.testing.assert_close(grad,reference[1],rtol=1e-9,atol=1e-10)

    def test_group_gradient_report_matches_and_rejects_bad_gradients(self):
        from train_geometry_heartffdnet import finite_gradient_report
        from utils.reference_refinement import reference_gradient_report
        modules=[torch.nn.Linear(3,2) for _ in range(6)]
        model=SimpleNamespace(encoder=modules[0],lifting=modules[1],unet=modules[2],
            heads=SimpleNamespace(deform3=modules[3],deform1=modules[4],deform2=modules[5]))
        for module in modules:
            for p in module.parameters():p.grad=torch.randn_like(p)
        self.assertEqual(reference_gradient_report(model),finite_gradient_report(model))
        first=modules[0];saved=[p.grad.clone() for p in first.parameters()]
        for bad in ('nan','inf','zero','missing'):
            for p,g in zip(first.parameters(),saved):
                p.grad=None if bad=='missing' else torch.zeros_like(g) if bad=='zero' else g.clone()
            if bad in ('nan','inf'):next(first.parameters()).grad[0,0]=float(bad)
            with self.assertRaises(ValueError):reference_gradient_report(model)
            with self.assertRaises(ValueError):finite_gradient_report(model)
        for p,g in zip(first.parameters(),saved):p.grad=g
        next(first.parameters()).grad=None # A partially unused parameter is allowed, as before.
        self.assertEqual(reference_gradient_report(model),finite_gradient_report(model))

    def test_reference_jacobian_matches_legacy_zero_padding_and_rejects_motion(self):
        import utils.geometry_joint_checks as checks
        model=fixture_model().eval();model.__class__=StagedHeartFFDNet
        checks.validate_template(model)
        with torch.no_grad():out=model.reference_output(observations())
        fine=out['control_grids']['fine']
        padded=dict(out,control_grids=dict(out['control_grids'],fine=torch.cat(
            (fine,torch.zeros_like(fine).expand(-1,23,-1,-1,-1,-1)),1)))
        with patch.object(checks,'bspline_ffd_physical',wraps=checks.bspline_ffd_physical) as op:
            legacy=checks.validate_prediction(model,padded,sample_jacobian=True)
            self.assertEqual(op.call_count,26*7)
        with patch.object(checks,'bspline_ffd_physical',wraps=checks.bspline_ffd_physical) as op:
            current=checks.validate_prediction(model,out,sample_jacobian=True,reference_only=True,check_intersections=True)
            self.assertEqual(op.call_count,3*7)
        self.assertEqual(current.pop('actual_mesh_self_intersecting_pairs'),0)
        self.assertEqual(current,legacy)
        with self.assertRaisesRegex(ValueError,'T=1'):
            checks.validate_prediction(model,padded,sample_jacobian=True,reference_only=True)
        with self.assertRaisesRegex(ValueError,'T=1'):
            checks.validate_prediction(model,dict(out,vertices_4d=out['vertices_4d'].expand(-1,24,-1,-1)),reference_only=True)

    def exact_fixture(self):
        from utils.geometry_joint_checks import validate_template
        from utils.slice_evaluation import render_source_slice
        model=fixture_model().eval();model.__class__=StagedHeartFFDNet;validate_template(model)
        x=observations();x['annotated'].zero_();x['annotated'][:,:,0,0]=True
        x['pose_gt']['probe_t'][:,:,:,2]=.137
        x['vertices_gt']=model.template_vertices_mm[None,None].expand(1,24,-1,-1).clone()
        mask=render_source_slice(model.template_vertices_mm[:42].numpy(),model.faces['LV'].numpy(),
            x['pose_gt']['probe_R'][0,0,0].numpy(),x['pose_gt']['probe_t'][0,0,0].numpy(),
            x['pose_gt']['probe_s'][0,0,0].numpy(),model.norm_to_mm,(32,32))
        x['native_masks']=torch.zeros_like(x['masks'],dtype=torch.uint8)
        x['native_masks'][0,:,0,0]=torch.as_tensor(mask)
        x['case_id']=['cache-case-1']
        return model,x

    def test_epoch_evaluation_reuses_gt_but_recomputes_predictions(self):
        import train_neural_ffd as legacy
        from train_staged_heartffd import evaluate
        from torch.utils.data import DataLoader
        model,x=self.exact_fixture();loader=DataLoader([x],collate_fn=lambda samples:samples[0])
        cache={}
        with patch.object(legacy,'render_source_slice',wraps=legacy.render_source_slice) as render:
            first=evaluate(model,loader,'cpu',native_exact=True,reference_cache=cache)
            self.assertEqual(render.call_count,2)
            second=evaluate(model,loader,'cpu',native_exact=True,reference_cache=cache)
            self.assertEqual(render.call_count,3) # Prediction is rendered again; GT is not.
        self.assertEqual(first,second)
        with patch.object(legacy,'render_source_slice',wraps=legacy.render_source_slice) as render:
            x['case_id']=['cache-case-2']
            evaluate(model,loader,'cpu',native_exact=True,reference_cache=cache)
            self.assertEqual(render.call_count,2)
        self.assertEqual(len(cache),2)

    def test_gt_cache_invalidates_changed_pose_labels_topology_and_renderer(self):
        import train_neural_ffd as legacy
        from train_staged_heartffd import phase0
        model,x=self.exact_fixture();batch=phase0(x);truth=batch['vertices_gt'];cache={}
        baseline=legacy.exact_metrics(model,truth,batch,cache)
        with patch.object(legacy,'render_source_slice',wraps=legacy.render_source_slice) as render:
            self.assertEqual(legacy.exact_metrics(model,truth,batch,cache),baseline)
            self.assertEqual(render.call_count,1)
        # Each change invalidates cached GT and must rerender/check it.
        for key in ('pose','labels','truth'):
            changed=copy.deepcopy(batch)
            if key=='pose':changed['pose_gt']['probe_t'][0,0,0,0]+=2
            elif key=='labels':changed['native_masks'][0,0,0,0,0,0]=1
            else:changed['vertices_gt'][...,0]+=2
            with self.assertRaisesRegex(ValueError,'GT renderer/label mismatch'):
                legacy.exact_metrics(model,truth,changed,copy.deepcopy(cache))
        changed=copy.deepcopy(batch);changed['native_pixel_spacing']*=2
        with patch.object(legacy,'render_source_slice',wraps=legacy.render_source_slice) as render:
            legacy.exact_metrics(model,truth,changed,copy.deepcopy(cache))
            self.assertEqual(render.call_count,2)
        old=model.faces['LV'].clone();getattr(model,'faces_LV').copy_(old.flip(0))
        with patch.object(legacy,'render_source_slice',wraps=legacy.render_source_slice) as render:
            legacy.exact_metrics(model,truth,batch,copy.deepcopy(cache));self.assertEqual(render.call_count,2)
        getattr(model,'faces_LV').copy_(old)
        identity=dict(legacy.source_renderer_identity(),test_revision='changed')
        with patch.object(legacy,'source_renderer_identity',return_value=identity),patch.object(
                legacy,'render_source_slice',wraps=legacy.render_source_slice) as render:
            legacy.exact_metrics(model,truth,batch,copy.deepcopy(cache));self.assertEqual(render.call_count,2)


if __name__ == '__main__':unittest.main()
