"""V2 contracts: physical evidence, time identity, validity, gradient and resume."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from data.mask_transform import mask_to_sdf
from net.geometry_heartffdnet_v2 import GeometryHeartFFDNetV2
from net.geometry_observation_lifting import GeometryObservationLifting
from net.geometry_view_fusion import SupportAwareViewFusion
from net.local_temporal_motion import temporal_conditions,resample_physical_features
from train_geometry_heartffd_v2 import save_checkpoint,load_checkpoint,gradient_report
from utils.geometry_joint_checks import validate_template,validate_prediction
from test_geometry_heartffdnet import fixture_model,observations


def model(variant='A4'):
    source=fixture_model()
    return GeometryHeartFFDNetV2(source.template_vertices_mm,source.chamber_ranges,source.faces,
        template_sha256=source.template_sha256,asset_sha256=source.asset_sha256,norm_to_mm=source.norm_to_mm,
        encoder_config=source.encoder_config,lifting_config=source.lifting_config,decoder_config=source.decoder_config,
        motion_config=dict(variant=variant,temporal_width=4,fine_width=4,phase_width=4))


def measured(batch=1,side=32):
    x=observations(batch,side)
    masks=torch.zeros_like(x['masks'])
    for t in range(24):
        offset=int(round(2*np.sin(2*np.pi*t/24)))
        masks[:,t,:,:,side//4:3*side//4,side//4+offset:3*side//4+offset]=1
    x['masks']=masks
    sdf=[]
    for b in range(batch):
        sdf.append(mask_to_sdf(masks[b].numpy(),x['pixel_spacing'][b].numpy(),
            view_present=x['view_present'][b].numpy(),annotated=x['annotated'][b].numpy()))
    x['sdf']=torch.tensor(np.stack(sdf))
    return x


class V2Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(2026);torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(1)

    def test_batch2_once_cnn_and_joint_head(self):
        net=model().eval();x=measured(2);counts=[]
        handle=net.encoder.stem.register_forward_pre_hook(lambda m,a:counts.append(len(a[0])))
        with torch.no_grad(),patch.object(net.heads.deform3,'forward',wraps=net.heads.deform3.forward) as head:
            output=net(x,return_features=True)
        handle.remove()
        self.assertEqual(sum(counts),144);self.assertEqual(head.call_count,1)
        self.assertEqual(output['vertices_4d'].shape,(2,24,168,3))
        self.assertEqual(output['control_grids']['fine'].shape,(2,24,24,24,32,3))
        self.assertEqual(output['features']['motion'].shape,(2,24,4,24,24,32))
        self.assertEqual(net.heads.deform3.readout.conv.groups,24)
        self.assertTrue(torch.allclose(output['view_fusion_weights'].sum(2)[~output['support_regions']['unknown']],torch.ones_like(output['view_fusion_weights'].sum(2)[~output['support_regions']['unknown']])))
        one={k:(v[:1] if isinstance(v,torch.Tensor) else {a:b[:1] for a,b in v.items()} if isinstance(v,dict) else v) for k,v in x.items()}
        with torch.no_grad():single=net(one)
        torch.testing.assert_close(single['vertices_4d'],output['vertices_4d'][:1],rtol=1e-5,atol=1e-6)
        validate_template(net);validate_prediction(net,output,sample_jacobian=True)

    def test_exact_zero_control_has_finite_identity_gradient(self):
        from net.geometry_heartffdnet_v2 import bound_control_field
        net=model()
        raw=torch.zeros(2,24,4,4,4,3,requires_grad=True)
        bounded=bound_control_field(net.heads,raw,2.,torch.ones(3))
        bounded.sum().backward()
        torch.testing.assert_close(raw.grad,torch.ones_like(raw),atol=0,rtol=0)
        self.assertTrue((bounded==0).all())

    def test_uniform_translation_control_has_finite_tanh_gradient(self):
        from net.geometry_heartffdnet_v2 import bound_control_field
        net=model()
        raw=torch.ones(2,24,4,4,4,3,requires_grad=True)
        bounded=bound_control_field(net.heads,raw,2.,torch.ones(3))
        bounded.sum().backward()
        expected=1-torch.tanh(raw.detach()/2).square()
        torch.testing.assert_close(raw.grad,expected,atol=1e-6,rtol=1e-6)

    def test_bound_matches_frozen_operator_in_active_and_inactive_regions(self):
        from net.geometry_heartffdnet_v2 import bound_control_field
        net=model()
        for scale in (.001,4.):
            raw=(torch.randn(2,3,4,4,4,3)*scale).requires_grad_()
            copy=raw.detach().clone().requires_grad_()
            new=bound_control_field(net.heads,raw,2.,torch.ones(3))
            old=net.heads._bound(copy,2.,torch.ones(3))
            torch.testing.assert_close(new,old,atol=0,rtol=0)
            new.square().sum().backward();old.square().sum().backward()
            torch.testing.assert_close(raw.grad,copy.grad,atol=1e-6,rtol=1e-5)

    def test_high_resolution_sdf_physical_sampling_and_support(self):
        x=measured(side=224)
        side=torch.arange(224,dtype=torch.float32)
        # Analytic signed in-plane coordinate field: x_phys=(pixel_x-112)*20/224.
        x['sdf']=((side-112)*20/224)[None,None,None,None,None].expand(1,24,3,4,224,224).clone()
        maps=dict(scale14=torch.zeros(1,24,3,1,14,14),scale7=torch.zeros(1,24,3,1,7,7),validity=x['annotated'])
        projections=torch.nn.ModuleList([torch.nn.Conv2d(1,1,1) for _ in range(2)])
        query=torch.tensor([[[[2.,0.,0.],[2.,0.,5.],[2.,0.,30.]]]])
        lift=GeometryObservationLifting(sigma_mm=8.)
        y=lift(maps,x,query,1.,projections)
        torch.testing.assert_close(y['sampled_slice_sdf_mm'][0,0,0,:,0,0,:2],torch.tensor([[2.,2.]]).expand(4,-1),atol=1e-5,rtol=1e-5)
        self.assertEqual(y['direct_support'][0,0,0,0,0].tolist(),[True,False,False])
        self.assertEqual(y['propagated_support'][0,0,0,0,0].tolist(),[False,True,False])
        self.assertFalse(y['view_support'][...,2].any())

    def test_fusion_preserves_view_conflicts_and_missing_chambers(self):
        x=measured();net=model().eval()
        with torch.no_grad():encoded=net.encode_conditions(x)
        lifted=encoded['fine']['lifted'];fuse=net.view_fusion
        # Two supported views with distinct feature values at one voxel.
        h=lifted['view_features'].clone();h[:,:,0]=1;h[:,:,1]=-1;h[:,:,2]=5
        support=torch.ones_like(lifted['view_support']);support[:,:,2]=False
        data=dict(lifted,view_features=h,view_support=support,
            confidence=support.float(),feature_validity=support[:,:,:,None].expand_as(h))
        y=fuse(data)
        self.assertTrue((y['weights'][:,:,2]==0).all())
        self.assertGreater(float(y['conflict'].mean()),.1)
        self.assertFalse(torch.equal(y['features'][:,:,0],y['features'][:,:,12]))
        # An unannotated channel cannot become a conflicting background value.
        fv=data['feature_validity'].clone();fv[:,:,1,-4:]=False
        data['feature_validity']=fv
        missing=fuse(data)
        self.assertTrue((missing['conflict'][:,:,-4:]==0).all())
        x['annotated'][...,0,2:]=False
        x['annotated'][...,1,:]=False;x['view_present'][...,1]=False
        poisoned=copy.deepcopy(x)
        poisoned['sdf'][...,0,2:,:,:]=float('nan');poisoned['masks'][...,0,2:,:,:]=float('nan')
        poisoned['sdf'][...,1,:,:,:]=float('nan');poisoned['masks'][...,1,:,:,:]=float('nan')
        with torch.no_grad():a,b=net(x),net(poisoned)
        torch.testing.assert_close(a['vertices_4d'],b['vertices_4d'],rtol=0,atol=0)
        self.assertTrue((b['view_fusion_weights'][:,:,1]==0).all())

    def test_masked_differences_circular_previous_and_phase_id(self):
        h=torch.arange(24,dtype=torch.float32).reshape(1,24,1,1,1,1).expand(1,24,1,2,2,2)
        valid=torch.ones_like(h,dtype=torch.bool);valid[:,5]=False
        phases=torch.arange(24)[None]/24
        y=temporal_conditions(h,valid,phases)
        self.assertEqual(float(y['previous_delta'][0,0,0,0,0,0]),-23.)
        self.assertTrue((y['previous_delta'][:,5:7]==0).all())
        self.assertTrue((y['reference_delta'][:,5]==0).all())
        self.assertFalse(torch.equal(y['conditions'][:,1],y['conditions'][:,2]))
        masks=dict(reference=torch.zeros_like(valid),previous=torch.zeros_like(valid))
        z=temporal_conditions(h,valid,phases,masks)
        self.assertTrue((z['reference_delta']==0).all());self.assertTrue((z['previous_delta']==0).all())

    def test_physical_medium_to_fine_interpolation_axes(self):
        x,y,z=torch.meshgrid(torch.arange(4.),torch.arange(5.),torch.arange(6.),indexing='ij')
        features=(x+2*y+3*z)[None,None,None].expand(2,24,1,-1,-1,-1)
        points=torch.tensor([[[[1.5,2.,3.],[2.,1.5,2.5]]]])
        result=resample_physical_features(features,points,torch.zeros(3),torch.ones(3))
        torch.testing.assert_close(result[0,0,0,0,0],torch.tensor([14.5,12.5]))

    def test_shape_motion_actual_gradients_and_gt_is_not_read(self):
        net=model();x=measured();x['vertices_gt']=torch.full((1,),float('nan'))
        for kind in ('shape','motion'):
            net.zero_grad(set_to_none=True);out=net(x);p=out['vertices_4d']
            loss=(p[:,0]-net.template_vertices_mm-.2).square().mean() if kind=='shape' else ((p[:,-1]-p[:,0])-.2).square().mean()
            loss.backward();grads=gradient_report(net)
            self.assertTrue(all(g>0 for g in grads.values()))
            if kind=='motion':
                weights=net.heads.deform3.readout.conv.weight.grad.reshape(24,-1).norm(dim=1)
                self.assertTrue((weights[1:]>0).all())

    def test_independent_version_and_strict_next_update_resume(self):
        net=model();x=measured();opt=torch.optim.Adam(net.parameters(),lr=.001)
        config=dict(training=dict(shape_updates=1,motion_updates=1,joint_updates=1,batch_size=1))
        def update(m,o):
            o.zero_grad(set_to_none=True);p=m(x)['vertices_4d'];p.sub(.2).square().mean().backward();o.step()
        update(net,opt)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'v2.pth';save_checkpoint(path,net,opt,config=config,identity={'train':'frozen'},step=1,runtime={})
            restored,s=load_checkpoint(path,config=config,identity={'train':'frozen'})
            other=torch.optim.Adam(restored.parameters(),lr=.001);other.load_state_dict(s['optimizer_state'])
            from train_neural_ffd import restore_rng
            restore_rng({'rng':s['rng_state']},'cpu');update(net,opt)
            restore_rng({'rng':s['rng_state']},'cpu');update(restored,other)
            for key,value in net.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'data identity'):load_checkpoint(path,identity={'train':'different'})
            s['architecture']='geometry-ma-joint-ffd-v1';s['model_spec']['motion']=None
            torch.save(s,path)
            with self.assertRaises((ValueError,RuntimeError)):load_checkpoint(path)

    def test_shared_initialization_and_component_switches(self):
        from train_geometry_heartffd_v2 import build_model
        for variant in ('A1','A2','A3','A4'):
            torch.manual_seed(2026);net=model(variant)
            with torch.no_grad():encoded=net.encode_conditions(measured())
            self.assertEqual(net.use_sdf,variant!='A1')
            self.assertEqual(net.use_fusion,variant in ('A3','A4'))
            self.assertEqual(net.use_fine_skip,variant=='A4')
            if variant!='A4':self.assertTrue((encoded['fine_skip']==0).all())


if __name__=='__main__':unittest.main()
