"""Independent numerical and end-to-end contracts for Neural HeartFFDNet."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import BSpline

from utils.ffd_physical import bspline_ffd_physical, control_geometry
from net.multiview_encoder import MultiViewTemporalEncoder,EncoderDiagnostics
from net.neural_ffdnet import NeuralFFDNet
from utils.slice_supervision import (plane_segments,signed_segment_distance,section_soft_mask,
                                     exact_section_mask,MissingSectionError,MeshSliceSupervisor,
                                     batched_plane_segments,batched_signed_segment_distance)
from utils.neural_ffd_losses import NeuralFFDLoss


class NeuralTestCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        # Do not leave an active multi-thread OpenMP pool for the legacy
        # Dataset's explicit fork-worker test later in unittest discovery.
        torch.set_num_threads(1)


def small_model(frames=4,regional_motion=False,spatial_fusion='none',dynamic_spatial='none'):
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=1,radius=2.)
    vertices = torch.cat([torch.tensor(mesh.vertices,dtype=torch.float32)+torch.tensor(center)
                          for center in ((-3.,-3.,0.),(-3.,3.,0.),(3.,-3.,0.),(3.,3.,0.))])
    count = len(mesh.vertices)
    ranges = torch.tensor([[i*count,(i+1)*count] for i in range(4)])
    faces = {c:torch.tensor(mesh.faces.copy()) for c in ('LV','LA','RV','RA')}
    regional = {}
    if regional_motion:
        centers = torch.tensor([[-3.,-3.,0.],[-3.,3.,0.],[3.,-3.,0.],[3.,3.,0.]])
        regional = dict(regional_motion=True,regional_motion_lower_mm=(centers-6).tolist(),
                        regional_motion_upper_mm=(centers+6).tolist(),regional_motion_hidden=8)
    return NeuralFFDNet(vertices,ranges,faces,template_sha256='fixture-template',asset_sha256='fixture-asset',norm_to_mm=1.,
        encoder_config=dict(shape_dim=12,motion_dim=4,width=16,heads=2,temporal_layers=1,
                            spatial_sides=(4,2),query_slots=2,cnn_chunk_size=4),
        decoder_config=dict(frames=frames,channels=4,seed_shape=(4,4,4),global_shape_dim=4,
                            lower_mm=(-10.,-10.,-10.),upper_mm=(10.,10.,10.),
                            spatial_fusion=spatial_fusion,dynamic_spatial=dynamic_spatial,
                            dynamic_spatial_width=8,dynamic_spatial_heads=2,dynamic_spatial_query_chunk=64,**regional))


class DynamicSpatialTests(NeuralTestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(603)

    def test_once_computed_cycle_tokens_and_gt_pose_independence(self):
        net=small_model(frames=24,dynamic_spatial='query_difference_v1').eval()
        x=observations(2,24,32)
        counts=[]
        handle=net.encoder.stem.register_forward_pre_hook(lambda m,a:counts.append(len(a[0])))
        with torch.no_grad():encoded=net.encode(x)
        handle.remove()
        self.assertEqual(sum(counts),2*24*3)
        self.assertEqual(encoded['dynamic_spatial']['tokens'].shape,(2,24,60,16))
        self.assertEqual(encoded['dynamic_spatial']['valid'].shape,(2,24,60))
        self.assertEqual(encoded['dynamic_spatial']['channel_valid'].shape,(2,24,3,4))
        poisoned=dict(x,pose_gt='must not be read',vertices_gt=torch.full((1,),float('nan')))
        with torch.no_grad():other=net.encode(poisoned)
        for key in ('z_shape','z_motion'):self.assertTrue(torch.equal(encoded[key],other[key]))
        self.assertTrue(torch.equal(encoded['dynamic_spatial']['tokens'],other['dynamic_spatial']['tokens']))

    def test_strict_inheritance_zero_equivalence_and_freeze_scopes(self):
        from train_neural_ffd import save_checkpoint,load_checkpoint,initialize_from_checkpoint,configure_trainable
        old=small_model().eval();new=small_model(dynamic_spatial='query_difference_v1').eval()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pth'
            save_checkpoint(path,old,None,config={},identity={'val':'fixed'})
            metadata=initialize_from_checkpoint(new,path,{'val':'fixed'})
            self.assertTrue(metadata['new_keys'])
            self.assertTrue(all(k.startswith('dynamic_conditioner.') for k in metadata['new_keys']))
            for key,value in old.state_dict().items():self.assertTrue(torch.equal(value,new.state_dict()[key]),key)
            x=observations(2,4,32)
            with torch.no_grad():a,b=old(x),new(x)
            self.assertTrue(torch.equal(a['vertices_4d'],b['vertices_4d']))
            self.assertTrue(torch.equal(a['vertices_ref'],b['vertices_ref']))
            for key in ('coarse','medium','fine_motion','regions'):
                self.assertTrue(torch.equal(a['control_grids'][key],b['control_grids'][key]))
            criterion=NeuralFFDLoss(new.chamber_ranges,new.faces)
            target=a['vertices_4d']+.2
            for output in (a,b):
                losses=criterion(output,target,slice_result={'total':output['vertices_4d'].sum()*0+.12},
                    control_spacings_mm=new.control_spacings_mm,regional_spacing_mm=new.regional_spacing_mm)
                if output is a:reference=losses
                else:
                    self.assertEqual(set(losses['terms']),set(reference['terms']))
                    self.assertEqual(len(losses['terms']),9)
                    for key,value in losses['terms'].items():self.assertTrue(torch.equal(value,reference['terms'][key]),key)
            save_checkpoint(path,new,None,config={},identity={'val':'fixed'})
            restored,_=load_checkpoint(path)
            with torch.no_grad():torch.testing.assert_close(restored.eval()(x)['vertices_4d'],b['vertices_4d'],rtol=0,atol=0)
        configure_trainable(new,{'freeze':'dynamic_only'})
        self.assertTrue(all(p.requires_grad==name.startswith('dynamic_conditioner.') for name,p in new.named_parameters()))
        configure_trainable(old,{'freeze':'motion_head_only'})
        self.assertFalse(old.deform3.reference.weight.requires_grad)
        self.assertTrue(old.deform3.motion.weight.requires_grad)
        new.decoder_config['lower_mm']=[-11.,-10.,-10.]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pth';save_checkpoint(path,old,None,config={},identity={'val':'fixed'})
            with self.assertRaisesRegex(ValueError,'configuration'):initialize_from_checkpoint(new,path,{'val':'fixed'})

    def test_query_reads_are_reused_and_adjacent_difference_has_correct_index(self):
        from unittest.mock import patch
        net=small_model(frames=24,dynamic_spatial='query_difference_v1').eval()
        with torch.no_grad():encoded=net.encode(observations(1,24,32))
        shape,motion=encoded['z_shape'],encoded['z_motion']
        spatial=encoded['dynamic_spatial'];module=net.dynamic_conditioner
        torch.nn.init.normal_(module.output.weight,std=.1)
        with torch.no_grad(),patch.object(module,'_read',wraps=module._read) as reads,patch.object(
                net.deform3,'motion_field',wraps=net.deform3.motion_field) as calls:
            out=net.decode_4d(shape,motion,dynamic_spatial=spatial,return_details=True)
            self.assertEqual(sum(call.args[1].shape[1] for call in reads.call_args_list),24)
            self.assertEqual([call.args[-1] for call in calls.call_args_list],list(range(23)))
            for phase,call in enumerate(calls.call_args_list):
                field=calls._mock_wraps(*call.args)
                torch.testing.assert_close(field,out['control_grids']['fine_motion'][:,phase+1])
                torch.testing.assert_close(net._deform(out['vertices_4d'][:,phase],field,2),out['vertices_4d'][:,phase+1])
            changed={key:value.clone() for key,value in spatial.items()}
            changed['tokens'][:,12,:4] += .01*torch.randn_like(changed['tokens'][:,12,:4])
            later=net.decode_4d(shape,motion,dynamic_spatial=changed,return_details=True)
            self.assertTrue(torch.equal(out['vertices_4d'][:,:12],later['vertices_4d'][:,:12]))
            self.assertFalse(torch.equal(out['control_grids']['fine_motion'][:,12],later['control_grids']['fine_motion'][:,12]))
            self.assertTrue(torch.equal(out['vertices_ref'],later['vertices_ref']))

    def test_missing_and_changed_annotation_never_create_placeholder_motion(self):
        net=small_model(dynamic_spatial='query_difference_v1').eval()
        torch.nn.init.normal_(net.dynamic_conditioner.output.weight,std=.1)
        with torch.no_grad():encoded=net.encode(observations(2,4,32))
        spatial=encoded['dynamic_spatial']
        channels=torch.zeros_like(spatial['channel_valid'])
        channels[:,0,0,0]=True;channels[:,1,0,1]=True
        valid=channels.any(-1)[...,net.dynamic_conditioner.token_views]
        changed=dict(tokens=spatial['tokens'].clone(),valid=valid,channel_valid=channels)
        changed['tokens'][~valid]=float('nan')
        with torch.no_grad():
            with_branch=net.decode_4d(encoded['z_shape'],encoded['z_motion'],dynamic_spatial=changed)
            without=net.decode_4d(encoded['z_shape'],encoded['z_motion'])
        self.assertTrue(torch.equal(with_branch,without))
        self.assertTrue(torch.isfinite(with_branch).all())
        # A missing view between otherwise legal pairs is handled with common
        # masks, not a subtraction of the unobserved read from its neighbor.
        channels=spatial['channel_valid'].clone();channels[:,2,0]=False
        changed=dict(tokens=spatial['tokens'],channel_valid=channels,
            valid=channels.any(-1)[...,net.dynamic_conditioner.token_views])
        with torch.no_grad():out=net.decode_4d(encoded['z_shape'],encoded['z_motion'],dynamic_spatial=changed)
        self.assertTrue(torch.isfinite(out).all())

    def test_branch_leaves_zero_then_has_finite_gradients_and_local_response(self):
        from train_neural_ffd import configure_trainable
        net=small_model(dynamic_spatial='query_difference_v1').eval()
        with torch.no_grad():encoded=net.encode(observations(1,4,32))
        shape,motion=encoded['z_shape'].detach(),encoded['z_motion'].detach()
        spatial={key:value.detach() for key,value in encoded['dynamic_spatial'].items()}
        configure_trainable(net,{'freeze':'dynamic_only'})
        optimizer=torch.optim.Adam(net.dynamic_conditioner.parameters(),lr=.02)
        with torch.no_grad():base=net.decode_4d(shape,motion)
        target=base.clone();target[:,1:,:,0]+=torch.tensor([.03,.06,.04])[None,:,None]
        for step in range(3):
            optimizer.zero_grad();out=net.decode_4d(shape,motion,dynamic_spatial=spatial)
            loss=(out-target).square().mean();loss.backward()
            self.assertGreater(float(net.dynamic_conditioner.output.weight.grad.norm()),0.)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
            if step>0:self.assertGreater(float(net.dynamic_conditioner.attention.in_proj_weight.grad.norm()),0.)
            optimizer.step()
        leaf=spatial['tokens'].clone().requires_grad_(True);spatial=dict(spatial,tokens=leaf)
        out=net.decode_4d(shape,motion,dynamic_spatial=spatial,return_details=True)
        grad=torch.autograd.grad((out['vertices_4d']-target).square().mean(),leaf)[0]
        self.assertGreater(float(grad.norm()),0.);self.assertTrue(torch.isfinite(grad).all())
        changed=dict(spatial,tokens=leaf.detach().clone())
        changed['tokens'][:,2,:4] += .01*torch.randn_like(changed['tokens'][:,2,:4])
        with torch.no_grad():other=net.decode_4d(shape,motion,dynamic_spatial=changed,return_details=True)
        delta=other['control_grids']['fine_motion'][:,2]-out['control_grids']['fine_motion'][:,2].detach()
        self.assertGreater(float(delta.square().mean().sqrt()),1e-10)
        self.assertGreater(float((delta-delta.mean((1,2,3),keepdim=True)).square().mean().sqrt()),1e-10)
        self.assertTrue(torch.equal(other['vertices_4d'][:,:2],out['vertices_4d'][:,:2]))
        self.assertGreater(float((other['vertices_4d'][:,2:]-out['vertices_4d'][:,2:].detach()).norm(dim=-1).max()),1e-9)

    def test_batch_parity_and_zero_change_neutrality(self):
        net=small_model(dynamic_spatial='query_difference_v1').eval()
        torch.nn.init.normal_(net.dynamic_conditioner.output.weight,std=.1)
        with torch.no_grad():
            encoded=net.encode(observations(2,4,32))
            together=net.decode_4d(**encoded)
            for patient in range(2):
                single=dict(z_shape=encoded['z_shape'][patient:patient+1],z_motion=encoded['z_motion'][patient:patient+1],
                    dynamic_spatial={key:value[patient:patient+1] for key,value in encoded['dynamic_spatial'].items()})
                torch.testing.assert_close(together[patient:patient+1],net.decode_4d(**single),atol=2e-6,rtol=1e-6)
            features=net._reference(encoded['z_shape'])[1]
            zero=features.new_zeros(2,8,*net.grid_shapes[1])
            conditioned,stats=net.dynamic_conditioner.increment(features,zero)
            self.assertTrue(torch.equal(features,conditioned))
            self.assertEqual(float(stats.sum()),0.)


class SpatialFusionTests(NeuralTestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(104)

    def test_encoder_reuses_cnn_and_only_reference_tokens(self):
        net = small_model(spatial_fusion='reference_cross_attention_v1').eval()
        x = observations(2,4,32)
        seen = []
        handle = net.encoder.stem.register_forward_pre_hook(lambda m,a:seen.append(a[0].shape[0]))
        with torch.no_grad():
            encoded = net.encode(x)
        handle.remove()
        self.assertEqual(sum(seen),2*4*3)
        self.assertEqual(len(seen),6)  # chunk size 4, no extra reference pass
        spatial = encoded['reference_spatial']
        self.assertEqual(spatial['tokens'].shape,(2,3*(4**2+2**2),16))
        self.assertEqual(spatial['valid'].shape,(2,60))
        changed = dict(x,sdf=x['sdf'].clone())
        changed['sdf'][:,1:] += 4
        with torch.no_grad():
            later = net.encode(changed)
            legacy = net.encoder(x)
        self.assertTrue(torch.equal(spatial['tokens'],later['reference_spatial']['tokens']))
        for key in ('z_shape','z_motion'):
            self.assertTrue(torch.equal(encoded[key],legacy[key]))

    def test_zero_initialization_checkpoint_and_motion_semantics(self):
        from train_neural_ffd import save_checkpoint,load_checkpoint
        from unittest.mock import patch
        old = small_model(frames=24).eval()
        torch.manual_seed(104)
        new = small_model(frames=24,spatial_fusion='reference_cross_attention_v1').eval()
        for key,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,new.state_dict()[key]),key)
        x = observations(1,24,32)
        with torch.no_grad():
            a,b = old(x),new(x)
        for key in ('vertices_ref','vertices_4d','z_shape','z_motion'):
            self.assertTrue(torch.equal(a[key],b[key]),key)
        for key in ('coarse','medium','fine_motion','regions'):
            self.assertTrue(torch.equal(a['control_grids'][key],b['control_grids'][key]),key)
        self.assertEqual(old.geometry_sha256,new.geometry_sha256)
        self.assertEqual(old.grid_shapes,new.grid_shapes)
        # Verify the original recurrence, phase index and per-phase field on
        # nonzero fused features too, without introducing a new Motion head.
        torch.nn.init.normal_(new.spatial_fusion.output.weight,std=.02)
        with torch.no_grad(), patch.object(new.deform3,'motion_field',wraps=new.deform3.motion_field) as calls:
            out = new(x)
            self.assertEqual([call.args[-1] for call in calls.call_args_list],list(range(23)))
            for phase,call in enumerate(calls.call_args_list):
                field = calls._mock_wraps(*call.args)
                torch.testing.assert_close(field,out['control_grids']['fine_motion'][:,phase+1])
                torch.testing.assert_close(new._deform(out['vertices_4d'][:,phase],field,2),
                                           out['vertices_4d'][:,phase+1])
            changed_motion = new.decode_4d(out['z_shape'],out['z_motion']+2,
                reference_spatial=out['reference_spatial'],return_details=True)
            self.assertTrue(torch.equal(out['vertices_ref'],changed_motion['vertices_ref']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fusion.pth'
            save_checkpoint(path,new,None,config={},identity={})
            restored,_ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(restored.eval()(x)['vertices_4d'],out['vertices_4d'],rtol=0,atol=0)

    def test_training_escape_gradients_and_local_fine_response(self):
        net = small_model(spatial_fusion='reference_cross_attention_v1').eval()
        x = observations(1,4,32)
        latent = net.encode(x)
        shape = latent['z_shape'].detach()
        spatial = {k:v.detach() for k,v in latent['reference_spatial'].items()}
        target = net.template_vertices_mm[None]+torch.tensor([.7,-.2,.4])
        optimizer = torch.optim.Adam(net.spatial_fusion.parameters(),lr=.01)
        loss = (net.decode_shape(shape,spatial)-target).square().mean()
        loss.backward()
        self.assertGreater(float(net.spatial_fusion.output.weight.grad.norm()),0.)
        optimizer.step()
        self.assertGreater(float(net.spatial_fusion.output.weight.norm()),0.)
        # Detaching every latent isolates the new route to the original CNN.
        latent = net.encode(x)
        reference = net._reference(shape,latent['reference_spatial'])
        for value in (reference[0],reference[4]):
            gradients = torch.autograd.grad(value.square().mean(),
                (latent['reference_spatial']['tokens'],net.encoder.stem[0].weight),retain_graph=True)
            for grad in gradients:
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.norm()),0.)
            self.assertGreater(float(gradients[0][:,:48].norm()),0.)
            self.assertGreater(float(gradients[0][:,48:].norm()),0.)
        # Two opposite local token perturbations preserve global token mean.
        # Compare actual Fine controls, not just an attention/gradient norm.
        changed = dict(spatial,tokens=spatial['tokens'].clone())
        perturb = torch.linspace(-3.,3.,16)
        changed['tokens'][:,0] += perturb
        changed['tokens'][:,5] -= perturb
        torch.testing.assert_close(changed['tokens'].mean(1),spatial['tokens'].mean(1))
        with torch.no_grad():
            base = net._reference(shape,spatial)
            moved = net._reference(shape,changed)
        delta = moved[4]-base[4]
        self.assertTrue(torch.isfinite(delta).all())
        self.assertGreater(float(delta.norm()),1e-7)
        self.assertGreater(float(delta.flatten(1,3).std(1).norm()),1e-8)
        self.assertGreater(float((moved[0]-base[0]).norm()),1e-7)
        self.assertTrue(torch.equal(base[2],moved[2]))  # Coarse remains latent-only
        # Query position must matter, even with spatially constant 3D content.
        medium = torch.ones(1,4,8,8,8)
        d = net.spatial_fusion(medium,changed)-net.spatial_fusion(medium,spatial)
        self.assertGreater(float(d.flatten(2).std(-1).norm()),1e-8)

    def test_missing_reference_and_masked_tokens_remain_neutral(self):
        net = small_model(spatial_fusion='reference_cross_attention_v1').eval()
        torch.nn.init.normal_(net.spatial_fusion.output.weight,std=.1)
        spatial = dict(tokens=torch.randn(2,60,16),valid=torch.ones(2,60,dtype=torch.bool))
        spatial['valid'][0] = False
        spatial['valid'][1,3:10] = False
        medium = torch.randn(2,4,8,8,8)
        a = net.spatial_fusion(medium,spatial)
        spatial['tokens'][~spatial['valid']] = float('nan')
        b = net.spatial_fusion(medium,spatial)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(a[0],medium[0]))


class DecoderTests(NeuralTestCase):
    def test_regional_motion_support_units_and_domain_failures(self):
        net = small_model(regional_motion=True)
        v = net.template_vertices_mm[None]
        controls = torch.zeros(1,4,5,5,5,3)
        controls[:,2,...,1] = 1.25
        changed,rms = net.apply_motion_regions(v,controls)
        torch.testing.assert_close(changed[:,84:126,1]-v[:,84:126,1],torch.full((1,42),1.25))
        self.assertTrue(torch.equal(changed[:,:84],v[:,:84]))
        self.assertTrue(torch.equal(changed[:,126:],v[:,126:]))
        torch.testing.assert_close(rms,torch.tensor([[0.,0.,1.25,0.]]))
        with self.assertRaisesRegex(ValueError,'support domain'):
            net.apply_motion_regions(v+100,controls)
        controls[0,0,0,0,0,0] = float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            net.apply_motion_regions(v,controls)

    def test_regional_motion_24_frames_initial_equivalence_and_latent_gradients(self):
        net = small_model(frames=24,regional_motion=True)
        z,m = torch.randn(1,12,requires_grad=True),torch.randn(1,4,requires_grad=True)
        output = net.decode_4d(z,m,return_details=True)
        self.assertEqual(output['vertices_4d'].shape,(1,24,168,3))
        self.assertEqual(output['control_grids']['regional_motion'].shape,(1,24,4,5,5,5,3))
        self.assertTrue(torch.equal(output['vertices_ref'],net.decode_shape(z)))
        net.regional_motion = False
        initial = net.decode_4d(z,m)
        net.regional_motion = True
        self.assertTrue(torch.equal(initial,output['vertices_4d']))
        # One ordinary update opens the inner/latent path from a zero output.
        optimizer = torch.optim.Adam(net.regional_motion_heads.parameters(),lr=.001)
        desired = output['vertices_4d'].detach().clone()
        desired[:,1:,84:126,0] += .2
        loss = (output['vertices_4d']-desired).square().mean()
        loss.backward()
        self.assertGreater(float(net.regional_motion_heads[2][-1].weight.grad.norm()),0.)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        updated = net.decode_4d(z,m,return_details=True)
        region_controls = updated['control_grids']['regional_motion'][:,1:]
        gradients = torch.autograd.grad(region_controls.square().sum(),
            (m,z,net.regional_motion_heads[2][0].weight),retain_graph=True)
        for grad in gradients:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.norm()),0.)
        self.assertTrue(torch.equal(updated['vertices_ref'],output['vertices_ref']))
        self.assertTrue(torch.isfinite(updated['vertices_4d']).all())
        for c in net.faces:
            self.assertTrue(torch.equal(net.faces[c],self.net.faces[c]))
        criterion = NeuralFFDLoss(net.chamber_ranges,net.faces,weights={'slice':0.})
        losses = criterion(updated,desired,control_spacings_mm=net.control_spacings_mm,
            regional_spacing_mm=net.regional_spacing_mm,regional_motion_spacings_mm=net.regional_motion_spacings_mm)
        self.assertTrue(torch.isfinite(losses['total']))
        control_gradient = torch.autograd.grad(losses['terms']['control'],net.regional_motion_heads[2][-1].weight)[0]
        self.assertTrue(torch.isfinite(control_gradient).all())
        self.assertGreater(float(control_gradient.norm()),0.)

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        self.net = small_model()

    def test_reference_independence_batch_and_full_forward(self):
        output = self.net(observations())
        self.assertEqual(output['vertices_4d'].shape,(2,4,168,3))
        self.assertEqual(output['control_grids']['fine_motion'].shape,(2,4,16,16,16,3))
        z,m = output['z_shape'],output['z_motion']
        other = self.net.decode_4d(z,m+3)
        self.assertTrue(torch.equal(other[:,0],output['vertices_ref']))
        self.assertFalse(torch.equal(other[:,1:],output['vertices_4d'][:,1:]))
        separate = self.net.decode_4d(z[:1],m[:1])
        self.assertTrue(torch.allclose(separate,output['vertices_4d'][:1],atol=1e-6))
        self.assertTrue(torch.equal(self.net.decode_shape(z),output['vertices_ref']))
        with self.assertRaises(ValueError):
            self.net.encode(dict(observations(),template_sha256=['wrong','wrong']))
        with self.assertRaises(ValueError):
            self.net.encode(dict(observations(),norm_to_mm=torch.tensor([10.,10.])))

    def test_identity_and_region_support(self):
        z = torch.randn(2,12)
        m = torch.randn(2,4)
        regions = torch.zeros(2,4,4,4,4,3)
        regions[:,1,...,0] = 1.
        v = self.net.template_vertices_mm[None].expand(2,-1,-1)
        changed = self.net.apply_regions(v,regions)
        delta = changed-v
        self.assertTrue(torch.allclose(delta[:,42:84,0],torch.ones(2,42),atol=1e-6))
        self.assertTrue(torch.equal(delta[:,:42],torch.zeros_like(delta[:,:42])))
        self.assertTrue(torch.equal(delta[:,84:],torch.zeros_like(delta[:,84:])))
        for layer in (self.net.deform1,self.net.deform2,self.net.deform3.reference,self.net.deform3.motion,*self.net.regional_heads):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        self.assertTrue(torch.equal(self.net.decode_4d(z,m),v[:,None].expand(-1,4,-1,-1)))

    def test_frozen_decoder_latent_optimization(self):
        self.net.eval().requires_grad_(False)
        target_z,target_m = torch.randn(1,12),torch.randn(1,4)
        with torch.no_grad():
            target = self.net.decode_4d(target_z,target_m)
        z,m = torch.zeros(1,12,requires_grad=True),torch.zeros(1,4,requires_grad=True)
        optimizer = torch.optim.Adam((z,m),lr=.15)
        initial = (self.net.decode_4d(z,m)-target).square().mean().item()
        for _ in range(8):
            optimizer.zero_grad()
            loss = (self.net.decode_4d(z,m)-target).square().mean()
            loss.backward()
            self.assertGreater(z.grad.norm().item(),0)
            self.assertGreater(m.grad.norm().item(),0)
            optimizer.step()
        final = (self.net.decode_4d(z,m)-target).square().mean().item()
        self.assertLess(final,initial)
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))


class SliceTests(NeuralTestCase):
    def setUp(self):
        torch.set_num_threads(2)
        import trimesh
        sphere = trimesh.creation.icosphere(subdivisions=2,radius=4.)
        self.vertices = torch.tensor(sphere.vertices.copy(),dtype=torch.double)
        self.faces = torch.tensor(sphere.faces.copy())
        self.R = torch.eye(3,dtype=torch.double)
        self.t = torch.tensor([0.,0.,1.37],dtype=torch.double)

    def test_true_section_independent_exact_and_gradient(self):
        v = self.vertices.clone().requires_grad_()
        result = section_soft_mask(v,self.faces,self.R,self.t,[.25,.25],(64,64),sigma_mm=.15)
        hard = exact_section_mask(v.detach().numpy(),self.faces.numpy(),self.R.numpy(),self.t.numpy(),[.25,.25],(64,64))
        soft = result['soft_mask'].detach().numpy()>.5
        dice = 2*(soft*hard).sum()/(soft.sum()+hard.sum())
        self.assertGreater(dice,.94)  # integer fillPoly has a one-pixel boundary offset
        target = torch.tensor(hard,dtype=torch.double)
        shifted = v+torch.tensor([.31,-.17,0.],dtype=torch.double)
        p = section_soft_mask(shifted,self.faces,self.R,self.t,[.25,.25],(64,64),sigma_mm=.3)['soft_mask']
        loss = 1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1)
        loss.backward()
        self.assertGreater(v.grad.norm().item(),0)
        self.assertTrue(torch.isfinite(v.grad).all())
        # This is a section rather than projection: moving the plane outside
        # the sphere removes its curve even though its XY projection persists.
        with self.assertRaises(MissingSectionError):
            section_soft_mask(v,self.faces,self.R,self.t+torch.tensor([0.,0.,10.]),[.25,.25],(64,64),require_intersection=True)

    def test_intersection_distance_gradcheck(self):
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=0,radius=4.)
        v = torch.tensor(mesh.vertices.copy(),dtype=torch.double,requires_grad=True)
        faces = torch.tensor(mesh.faces.copy())
        points = torch.tensor([[.37,.81],[2.73,1.13],[-3.27,-1.41]],dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda vertices:signed_segment_distance(points,plane_segments(vertices,faces,self.R,self.t)),(v,),atol=1e-4))

    def test_four_d_joint_supervision_reaches_shape_motion_and_encoder(self):
        from data.mask_transform import mask_to_sdf
        net = small_model()
        b,t,h = 1,4,32
        gt = net.template_vertices_mm[None,None].repeat(b,t,1,1)
        gt[:,1:,:,0] += torch.tensor([.2,.4,.2])[None,:,None]
        x = observations(b,t,h)
        x['pose_gt'] = dict(probe_R=torch.eye(3)[None,None,None].repeat(b,t,3,1,1),
                            probe_t=torch.tensor([0.,0.,.23])[None,None,None].repeat(b,t,3,1))
        x['pixel_spacing'] = torch.full((b,t,3,2),.625)
        mask = np.zeros((t,3,4,h,h),np.uint8)
        for frame in range(t):
            for view,chamber in x['annotated'][0].nonzero().tolist():
                c = ('LV','LA','RV','RA')[chamber]
                left,right = net.chamber_ranges[chamber]
                mask[frame,view,chamber] = exact_section_mask(gt[0,frame,left:right].numpy(),net.faces[c].numpy(),
                    np.eye(3),[0.,0.,.23],[.625,.625],(h,h))
        x['vertices_gt'] = gt
        x['masks'] = torch.tensor(mask)[None]
        x['sdf'] = torch.tensor(mask_to_sdf(mask,[.625,.625],annotated=x['annotated'][0].numpy()))[None]
        out = net(x)
        supervisor = MeshSliceSupervisor(net.chamber_ranges,net.faces,1.,sigma_mm=.4,pixel_stride=1,
            target_mode='paired_soft_geometry_v1')
        slices = supervisor(out['vertices_4d'],x)
        self.assertEqual(slices['diagnostics']['valid_cells'],t*8)
        criterion = NeuralFFDLoss(net.chamber_ranges,net.faces)
        losses = criterion(out,gt,slice_result=slices,control_spacings_mm=net.control_spacings_mm,
                           regional_spacing_mm=net.regional_spacing_mm)
        # Check supervision routes independently, not just a total requires_grad.
        for term,parameter in (('shape',net.deform3.reference.weight),('motion',net.deform3.motion.weight),
                               ('slice',net.deform3.motion.weight),('slice',net.encoder.stem[0].weight)):
            grad = torch.autograd.grad(losses['terms'][term],parameter,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(),0)
        losses['total'].backward()
        self.assertGreater(net.seed.weight.grad.norm().item(),0)
        self.assertGreater(net.regional_heads[0].weight.grad.norm().item(),0)
        # Missing observed sections and missed foreground samples must be
        # exposed, rather than returning an independent zero loss.
        far = dict(x,pose_gt={key:value.clone() for key,value in x['pose_gt'].items()})
        far['pose_gt']['probe_t'][...,2] = 100.
        with self.assertRaises(MissingSectionError):
            supervisor(out['vertices_4d'],far)
        tiny = dict(x,masks=torch.zeros_like(x['masks']))
        tiny['masks'][0,0,0,0,1,1] = 1
        strided = MeshSliceSupervisor(net.chamber_ranges,net.faces,1.,pixel_stride=4)
        with self.assertRaisesRegex(ValueError,'pixel_stride misses'):
            strided(out['vertices_4d'],tiny)


class CheckpointTests(NeuralTestCase):
    def test_training_config_rejects_unimplemented_options(self):
        import copy
        import yaml
        from train_neural_ffd import load_config
        path = Path(__file__).resolve().parents[1]/'configs/neural_ffd_1k_direct.yaml'
        config,_ = load_config(path)
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)/'config.yaml'
            for section,key in (('training','geometry_learning_rate'),
                                ('training','geometry_learning_rate_multiplier'),(None,'pose_framework')):
                wrong = copy.deepcopy(config)
                if section is None:
                    wrong[key] = {}
                else:
                    wrong[section][key] = 1e-3
                candidate.write_text(yaml.safe_dump(wrong))
                with self.assertRaisesRegex(ValueError,'Unknown'):
                    load_config(candidate)

    def test_regional_motion_initialization_and_fixed_buffer_integrity(self):
        from train_neural_ffd import save_checkpoint,initialize_from_checkpoint,load_checkpoint
        torch.set_num_threads(2)
        old,current = small_model(),small_model(regional_motion=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.pth'
            save_checkpoint(path,old,None,config={},identity={'split':'fixed'})
            result = initialize_from_checkpoint(current,path,{'split':'fixed'})
            self.assertTrue(all(key.startswith('regional_motion') for key in result['new_keys']))
            for key,value in old.state_dict().items():
                self.assertTrue(torch.equal(value,current.state_dict()[key]),key)
            save_checkpoint(path,current,None,config={},identity={'split':'fixed'})
            restored,saved = load_checkpoint(path)
            self.assertTrue(restored.regional_motion)
            saved['model']['regional_motion_origins_mm'][0,0] += 1
            torch.save(saved,path)
            with self.assertRaisesRegex(ValueError,'fixed control/topology buffer'):
                load_checkpoint(path)

    def test_audited_initialization_preserves_all_old_weights(self):
        from train_neural_ffd import save_checkpoint,initialize_from_checkpoint
        torch.set_num_threads(2)
        old = small_model()
        current = NeuralFFDNet(old.template_vertices_mm,old.chamber_ranges,old.faces,
            template_sha256=old.template_sha256,asset_sha256=old.asset_sha256,norm_to_mm=old.norm_to_mm,
            encoder_config=old.encoder_config,decoder_config=old.decoder_config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.pth'
            save_checkpoint(path,old,None,config={},identity={'split':'fixed'},step=800)
            result = initialize_from_checkpoint(current,path,{'split':'fixed'})
            self.assertEqual(result['new_keys'],[])
            self.assertEqual(result['source_step'],800)
            for key,value in old.state_dict().items():
                self.assertTrue(torch.equal(value,current.state_dict()[key]),key)
            with self.assertRaisesRegex(ValueError,'identical source data'):
                initialize_from_checkpoint(current,path,{'split':'wrong'})
            current.encoder_config['shape_dim'] = 999
            with self.assertRaisesRegex(ValueError,'configuration'):
                initialize_from_checkpoint(current,path,{'split':'fixed'})

    def test_long_schedule_rates_and_independent_checkpoint_selection(self):
        from train_neural_ffd import learning_rate_for_epoch,stage_for_epoch,best_checkpoint_candidates,format_epoch_summary
        config = dict(training=dict(epochs=180,shape_epochs=30,motion_epochs=30,
                                    learning_rate=1e-4,joint_min_learning_rate=1e-5))
        self.assertEqual([stage_for_epoch(i,config) for i in (0,29,30,59,60,179)],
                         ['shape','shape','motion','motion','joint','joint'])
        for i in (0,29,30,59,60):
            self.assertEqual(learning_rate_for_epoch(i,config,stage_for_epoch(i,config)),1e-4)
        self.assertAlmostEqual(learning_rate_for_epoch(179,config,'joint'),1e-5)
        rates = [learning_rate_for_epoch(i,config,'joint') for i in range(60,180)]
        self.assertTrue(all(a>=b for a,b in zip(rates,rates[1:])))
        legacy = dict(training=dict(epochs=30,shape_epochs=3,motion_epochs=5,learning_rate=1e-4))
        self.assertEqual(learning_rate_for_epoch(29,legacy,'joint'),1e-4)
        summary = dict(complete=True,stage='joint',epoch=80,train_mean_loss=1.,validation=dict(
            reference_vertex_rmse_mm=3.,relative_motion_rmse_mm=4.,mean_exact_dice=.99))
        candidates = best_checkpoint_candidates(summary)
        self.assertEqual(candidates['best_joint']['score'],5.)
        self.assertEqual(candidates['best_dice']['score'],-.99)
        # Better projection overlap must not conceal worsening reference error.
        worse = dict(summary,validation=dict(summary['validation'],reference_vertex_rmse_mm=6.,mean_exact_dice=1.))
        self.assertGreater(best_checkpoint_candidates(worse)['best_joint']['score'],5.)
        self.assertLess(best_checkpoint_candidates(worse)['best_dice']['score'],-.99)
        self.assertEqual(best_checkpoint_candidates(dict(summary,complete=False)),{})
        self.assertEqual(best_checkpoint_candidates(dict(summary,stage='shape'))['best_shape']['score'],3.)
        self.assertEqual(best_checkpoint_candidates(dict(summary,stage='motion'))['best_motion']['score'],5.)
        self.assertEqual(format_epoch_summary(summary,180),
            'Epoch 80/180 [joint] | Train loss=1.0000 | Val Shape RMSE=3.000 mm | Motion RMSE=4.000 mm | Dice=0.9900')

    def test_reference_dice_summary_separates_cycle_cells(self):
        from train_neural_ffd import validation_summary
        metrics = {c:dict(reference_vertex_rmse_mm=1.,trajectory_vertex_rmse_mm=2.,
                         relative_motion_rmse_mm=3.,volume_mae_ml=4.) for c in ('LV','LA','RV','RA')}
        report = dict(case_count=1,cases=[dict(metrics=metrics,exact=dict(
            summary=dict(cells=2,mean_exact_dice=.8),
            cells=[dict(frame=0,exact_dice=.9),dict(frame=1,exact_dice=.7)]))])
        summary = validation_summary(report)
        self.assertAlmostEqual(summary['reference_exact_dice'],.9)
        self.assertAlmostEqual(summary['mean_exact_dice'],.8)
        report['cases'][0].pop('exact')
        self.assertIsNone(validation_summary(report)['reference_exact_dice'])

    def test_optimizer_restore_geometry_identity_and_stages(self):
        from train_neural_ffd import save_checkpoint,load_checkpoint,stage_for_epoch,make_optimizer
        torch.set_num_threads(2)
        net = small_model()
        optimizer = make_optimizer(net,dict(learning_rate=.001))
        z,m = torch.randn(1,12),torch.randn(1,4)
        optimizer.zero_grad()
        loss = net.decode_4d(z,m).square().mean()
        loss.backward();optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pth'
            save_checkpoint(path,net,optimizer,config={'fixture':True},identity={'fixture':'abc'},epoch=2,sample_offset=1,step=3)
            restored,saved = load_checkpoint(path)
            restored_optimizer = make_optimizer(restored,dict(learning_rate=.001))
            restored_optimizer.load_state_dict(saved['optimizer'])
            self.assertEqual((saved['epoch'],saved['sample_offset'],saved['step']),(2,1,3))
            self.assertTrue(torch.equal(net.decode_4d(z,m),restored.decode_4d(z,m)))
            for current,current_optimizer in ((net,optimizer),(restored,restored_optimizer)):
                current_optimizer.zero_grad()
                current.decode_4d(z,m).square().mean().backward()
                current_optimizer.step()
            for key,value in net.state_dict().items():
                self.assertTrue(torch.equal(value,restored.state_dict()[key]),key)
            saved['model']['control_origins_mm'][0,0] += 1
            torch.save(saved,path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
        config = dict(training=dict(shape_epochs=2,motion_epochs=3))
        self.assertEqual([stage_for_epoch(i,config) for i in (0,2,5)],['shape','motion','joint'])


def observations(batch=2, frames=4, side=32):
    masks = (torch.rand(batch, frames, 3, 4, side, side) > .6).float()
    return dict(masks=masks, sdf=(.5-masks)*8, phases=torch.arange(frames)[None].expand(batch,-1)/frames,
                view_present=torch.ones(batch,3,dtype=torch.bool),
                annotated=torch.tensor([[1,1,0,0],[1,1,0,0],[1,1,1,1]],dtype=torch.bool)[None].expand(batch,-1,-1))


class EncoderTests(NeuralTestCase):
    def test_integrated_diagnostics_preserve_latents_and_release_hooks(self):
        from types import SimpleNamespace
        torch.set_num_threads(2)
        net = MultiViewTemporalEncoder(shape_dim=12,motion_dim=4,width=16,heads=2,
            temporal_layers=1,spatial_sides=(4,2),query_slots=2,feature_transfer='direct_residual_v1').eval()
        x = observations()
        x['view_present'] = x['view_present'][:,None].repeat(1,4,1)
        x['annotated'] = x['annotated'][:,None].repeat(1,4,1,1)
        x['view_present'][0,0] = False
        x['annotated'][0,0] = False
        with torch.no_grad():
            baseline = net(x)
            with EncoderDiagnostics(SimpleNamespace(encoder=net)) as monitor:
                measured = net(x)
                summary = monitor.summary()
        for key in baseline:
            self.assertTrue(torch.equal(baseline[key],measured[key]))
        self.assertEqual(summary['valid_frames'],7)
        self.assertEqual(summary['observed_reference_cases'],1)
        self.assertEqual(summary['cases'],2)
        self.assertGreater(summary['rms']['shape_direct'],0.)
        self.assertTrue(all(np.isfinite(value) for value in summary['rms'].values()))
        for module in net.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        x['phases'] = x['phases'].roll(1,1)
        with self.assertRaises(ValueError), EncoderDiagnostics(SimpleNamespace(encoder=net)):
            net(x)
        for module in net.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_latents_spatial_scales_and_gt_independence(self):
        torch.set_num_threads(2)
        for mode in ('legacy','direct_residual_v1'):
            with self.subTest(feature_transfer=mode):
                net = MultiViewTemporalEncoder(shape_dim=12, motion_dim=4, width=16, heads=2,
                                              temporal_layers=1, spatial_sides=(4,2), query_slots=2, checkpoint_cnn=True,
                                              feature_transfer=mode)
                x = observations()
                latents = net(x)
                self.assertEqual(latents['z_shape'].shape, (2,12))
                self.assertEqual(latents['z_motion'].shape, (2,4))
                maps = net.spatial_features(torch.randn(2,12,32,32))
                self.assertEqual([tuple(v.shape) for v in maps], [(2,16,4,4),(2,16,2,2)])
                (latents['z_shape'].square().sum()+latents['z_motion'].square().sum()).backward()
                self.assertGreater(net.stem[0].weight.grad.norm().item(), 0)
                net.eval()
                with torch.no_grad():
                    first = net(x)
                    x['vertices_gt'], x['pose_gt'], x['proposal_coefficients'] = torch.tensor(float('nan')), 'forbidden', 'forbidden'
                    x['pixel_spacing'], x['norm_to_mm'] = 'forbidden', 'forbidden'
                    invalid = ~x['annotated'][:,None,:,:,None,None].expand_as(x['masks'])
                    x['masks'] = x['masks'].masked_fill(invalid, float('nan'))
                    x['sdf'] = x['sdf'].masked_fill(invalid, 999.)
                    second = net(x)
                for key in first:
                    self.assertTrue(torch.equal(first[key], second[key]))

    def test_missing_view_frame_and_bad_phase(self):
        torch.set_num_threads(2)
        net = MultiViewTemporalEncoder(width=16, heads=2, temporal_layers=1, spatial_sides=(4,2), query_slots=2).eval()
        x = observations(batch=1)
        x['view_present'] = x['view_present'][:,None].repeat(1,4,1)
        x['annotated'] = x['annotated'][:,None].repeat(1,4,1,1)
        x['view_present'][:,1] = False
        x['annotated'][:,1] = False
        with torch.no_grad():
            self.assertTrue(torch.isfinite(net(x)['z_shape']).all())
        x['phases'] = x['phases'].roll(1,1)
        with self.assertRaises(ValueError):
            net(x)


class PhysicalFFDTests(NeuralTestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_scipy_tensor_product_and_closed_boundaries(self):
        torch.manual_seed(1)
        shape = (5, 6, 7)
        lo = torch.tensor([-7., -4., -9.], dtype=torch.double)
        hi = torch.tensor([3., 8., 1.], dtype=torch.double)
        o, h = control_geometry(lo, hi, shape)
        v = torch.stack((lo, hi, lo + .371 * (hi - lo)))[None].repeat(2, 1, 1)
        c = torch.randn(2, *shape, 3, dtype=torch.double)
        out = bspline_ffd_physical(v, c, o, h).numpy()
        q = ((v - o) / h).numpy()
        bases = [BSpline(np.arange(g + 4) - 2, np.eye(g), 3)(q[..., axis])
                 for axis, g in enumerate(shape)]
        expected = np.einsum('bnx,bny,bnz,bxyzd->bnd', *bases, c.numpy())
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_batch_translation_affine_and_mm_equivariance(self):
        lo, hi = torch.tensor([-10., -20., -30.]), torch.tensor([10., 20., 30.])
        o, h = control_geometry(lo, hi, (6, 7, 8))
        v = torch.tensor([[[0., 2., 5.]], [[1., -2., -5.]]])
        axes = [o[i] + torch.arange(g) * h[i] for i, g in enumerate((6, 7, 8))]
        nodes = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
        c = torch.stack((.1 * nodes + 2, -.2 * nodes - 3))
        expected = torch.stack((.1 * v[0] + 2, -.2 * v[1] - 3))
        self.assertTrue(torch.allclose(bspline_ffd_physical(v, c, o, h), expected, atol=2e-6))
        self.assertTrue(torch.allclose(bspline_ffd_physical(v*10, c*10, o*10, h*10), expected*10, atol=2e-5))
        self.assertTrue(torch.equal(bspline_ffd_physical(v, torch.zeros_like(c), o, h), torch.zeros_like(v)))

    def test_vertex_and_control_gradients(self):
        o = torch.zeros(3, dtype=torch.double)
        h = torch.tensor([2., 3., 4.], dtype=torch.double)
        v = torch.tensor([[[3.4, 5.1, 6.8]], [[4.4, 5.4, 7.2]]], dtype=torch.double, requires_grad=True)
        c = torch.randn(2, 5, 5, 5, 3, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: bspline_ffd_physical(a, b, o, h), (v, c)))

    def test_invalid_domain_batch_and_nonfinite(self):
        v, c = torch.ones(2, 1, 3)*2, torch.zeros(2, 5, 5, 5, 3)
        for bad_v, bad_c in ((v - 4, c), (v + 4, c), (v, c[:1]), (v*float('nan'), c)):
            with self.assertRaises(ValueError):
                bspline_ffd_physical(bad_v, bad_c, torch.zeros(3), torch.ones(3))


class BatchedSliceTests(unittest.TestCase):
    def setUp(self):
        import trimesh
        torch.set_num_threads(2)
        mesh = trimesh.creation.icosphere(subdivisions=1,radius=2.)
        self.vertices = torch.tensor(mesh.vertices.copy(),dtype=torch.double)
        self.faces = torch.tensor(mesh.faces.copy())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(1)

    def test_packed_distances_and_vertex_gradients_equal_independent_sections(self):
        v = self.vertices[None].repeat(3,1,1).requires_grad_()
        r = torch.eye(3,dtype=torch.double)[None].repeat(3,1,1)
        tr = torch.tensor([[0.,0.,.31],[.17,-.23,.72],[0.,0.,10.]],dtype=torch.double)
        points = torch.tensor([[.13,.29],[1.73,.53],[-1.17,-2.31]],dtype=torch.double)[None].repeat(3,1,1)
        segments,present = batched_plane_segments(v,self.faces,r,tr)
        self.assertFalse(present[2].any())
        for chunk in (1,8):
            batched = batched_signed_segment_distance(points,segments,present,chunk)[:2]
            serial = torch.stack([signed_segment_distance(points[i],plane_segments(v[i],self.faces,r[i],tr[i]),chunk)
                                  for i in range(2)])
            torch.testing.assert_close(batched,serial,atol=1e-12,rtol=1e-12)
            bg = torch.autograd.grad(batched.square().sum(),v,retain_graph=True)[0]
            sg = torch.autograd.grad(serial.square().sum(),v,retain_graph=True)[0]
            torch.testing.assert_close(bg,sg,atol=1e-11,rtol=1e-11)

    def fixture(self):
        b,t,h = 2,3,32
        ranges = torch.tensor([[i*len(self.vertices),(i+1)*len(self.vertices)] for i in range(4)])
        centers = torch.tensor([[-3.,-3.,0.],[-3.,3.,0.],[3.,-3.,0.],[3.,3.,0.]],dtype=torch.double)
        mesh = (self.vertices[None]+centers[:,None]).reshape(-1,3)
        trajectory = mesh[None,None].repeat(b,t,1,1)
        trajectory[1,...,0] += .23
        trajectory[:,1:,:,1] += .17
        r = torch.eye(3,dtype=torch.double)[None,None,None].repeat(b,t,3,1,1)
        angle = .13
        r[:,:,1] = torch.tensor([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],
                               [0.,np.sin(angle),np.cos(angle)]],dtype=torch.double)
        tr = torch.zeros(b,t,3,3,dtype=torch.double)
        tr[...,2] = .31
        tr[1,2,2,2] = 20.  # valid empty/empty sections
        spacing = torch.full((b,t,3,2),.625,dtype=torch.double)
        spacing[:,:,1,0] = .7  # anisotropic physical pixel grid
        valid = torch.ones(b,t,3,4,dtype=torch.bool)
        valid[:,1,0] = False  # fully absent frame/view
        valid[0,0,1,2:] = False  # chamber-specific missing observations
        masks = torch.zeros(b,t,3,4,h,h,dtype=torch.double)
        faces = dict(zip(('LV','LA','RV','RA'),[self.faces]*4))
        for patient,frame,view,chamber in valid.nonzero().tolist():
            left,right = ranges[chamber]
            masks[patient,frame,view,chamber] = torch.tensor(exact_section_mask(
                trajectory[patient,frame,left:right].numpy(),self.faces.numpy(),
                r[patient,frame,view].numpy(),tr[patient,frame,view].numpy(),
                spacing[patient,frame,view].numpy(),(h,h)),dtype=torch.double)
        observations = dict(masks=masks,sdf=torch.full_like(masks,2.7),annotated=valid,
            view_present=valid.any(-1),phases=torch.arange(t,dtype=torch.double)[None].repeat(b,1)/t,
            pose_gt=dict(probe_R=r,probe_t=tr/2.5),pixel_spacing=spacing)
        predicted = trajectory+torch.tensor([.11,-.07,.02],dtype=torch.double)
        return predicted,observations,ranges,faces

    def test_supervisor_losses_gradients_and_diagnostics_equal_with_empty_and_missing_cells(self):
        predicted,obs,ranges,faces = self.fixture()
        serial = MeshSliceSupervisor(ranges,faces,2.5,sigma_mm=.4,pixel_stride=2,chunk_size=31)
        v = predicted.clone().requires_grad_()
        original = serial(v,obs)
        grad = torch.autograd.grad(original['total'],v)[0]
        self.assertGreater(grad.norm().item(),0.)
        self.assertEqual(original['diagnostics']['matched_empty_sections'],4)
        for chunk in (1,5,128):
            fast = MeshSliceSupervisor(ranges,faces,2.5,sigma_mm=.4,pixel_stride=2,chunk_size=31,
                                       execution_backend='batched',cell_chunk_size=chunk)
            vb = predicted.clone().requires_grad_()
            result = fast(vb,obs)
            self.assertEqual(result['diagnostics'],original['diagnostics'])
            for key in ('total','dice','bce','boundary_mm'):
                torch.testing.assert_close(result[key],original[key],atol=1e-11,rtol=1e-11)
            bg = torch.autograd.grad(result['total'],vb)[0]
            torch.testing.assert_close(bg,grad,atol=1e-11,rtol=1e-10)

    def test_batched_backend_preserves_invalid_supervision_errors(self):
        predicted,obs,ranges,faces = self.fixture()
        fast = MeshSliceSupervisor(ranges,faces,2.5,pixel_stride=2,execution_backend='batched')
        obs['masks'][1,2,2,0,4,4] = 1
        with self.assertRaises(MissingSectionError):
            fast(predicted,obs)
        obs['masks'].zero_()
        obs['masks'][0,0,0,0,1,1] = 1
        with self.assertRaisesRegex(ValueError,'pixel_stride misses'):
            fast(predicted,obs)
        obs['masks'].zero_()
        obs['pose_gt']['probe_R'][0,0,0,0,0] = 2
        with self.assertRaisesRegex(ValueError,'orthonormal'):
            fast(predicted,obs)

    def test_paired_soft_targets_stationary_at_gt_and_serial_batched_parity(self):
        predicted,obs,ranges,faces = self.fixture()
        obs['vertices_gt'] = (predicted-torch.tensor([.11,-.07,.02],dtype=torch.double)).detach()
        results,gradients = [],[]
        for backend in ('serial','batched'):
            slicer = MeshSliceSupervisor(ranges,faces,2.5,sigma_mm=.4,pixel_stride=2,chunk_size=31,
                execution_backend=backend,target_mode='paired_soft_geometry_v1')
            exact_gt = obs['vertices_gt'].clone().requires_grad_()
            result = slicer(exact_gt,obs)
            grad = torch.autograd.grad(result['total'],exact_gt)[0]
            self.assertLess(abs(float(result['total'])),1e-12)
            self.assertLess(float(grad.norm()),1e-12)
            value = predicted.clone().requires_grad_()
            perturbed = slicer(value,obs)
            gradient = torch.autograd.grad(perturbed['total'],value)[0]
            self.assertGreater(float(perturbed['total']),0.)
            self.assertGreater(float(gradient.norm()),0.)
            results.append(perturbed)
            gradients.append(gradient)
        for key in ('total','dice','bce','boundary_mm'):
            torch.testing.assert_close(results[0][key],results[1][key],atol=1e-11,rtol=1e-10)
        torch.testing.assert_close(gradients[0],gradients[1],atol=1e-11,rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
