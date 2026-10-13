"""Independent physical lifting, joint architecture, gradient and reload contracts."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from data.mask_transform import CHAMBERS, normalize_masks
from net.geometry_heartffdnet import GeometryHeartFFDNet, loss_output_adapter
from net.geometry_lifting import project_physical_points, GeometryFeatureLifting
from utils.geometry_joint_checks import validate_template, validate_prediction, triangle_self_intersections
from utils.neural_ffd_losses import NeuralFFDLoss
from utils.slice_evaluation import section_mask
from train_geometry_heartffdnet import save_checkpoint, load_checkpoint, training_collate, finite_gradient_report


def observations(batch=1, side=32):
    masks = torch.randint(0,2,(batch,24,3,4,side,side)).float()
    return dict(masks=masks,sdf=masks*2-1,
        view_present=torch.ones(batch,24,3,dtype=torch.bool),
        annotated=torch.ones(batch,24,3,4,dtype=torch.bool),
        phases=torch.arange(24)[None].expand(batch,-1)/24,
        pose_gt=dict(probe_R=torch.eye(3).expand(batch,24,3,3,3).clone(),
            probe_t=torch.zeros(batch,24,3,3),probe_s=torch.full((batch,24,3,3),10.)),
        model_to_native=torch.eye(3).expand(batch,24,3,3,3).clone(),
        pixel_spacing=torch.full((batch,24,3,2),20/side),
        native_pixel_spacing=torch.full((batch,24,3,2),20/side),sdf_units='mm')


def fixture_model():
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=1,radius=1.)
    vertices = torch.cat([torch.tensor(sphere.vertices,dtype=torch.float32)+torch.tensor(center)
        for center in ((-3.,-3.,0.),(-3.,3.,0.),(3.,-3.,0.),(3.,3.,0.))])
    n = len(sphere.vertices)
    return GeometryHeartFFDNet(vertices,torch.tensor([[i*n,(i+1)*n] for i in range(4)]),
        {c:torch.tensor(sphere.faces.copy()) for c in CHAMBERS},
        template_sha256='fixture',asset_sha256='fixture',norm_to_mm=1.,
        encoder_config=dict(width=16,spatial_sides=(4,2),cnn_chunk_size=12,checkpoint_cnn=False),
        lifting_config=dict(projected_width=2,sigma_mm=2.),
        decoder_config=dict(channels=8,lower_mm=[-10.,-10.,-10.],upper_mm=[10.,10.,10.]))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2026)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(1)

    def test_projection_renderer_rotation_case_and_canonical_fov(self):
        x = observations(2)
        r = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        x['pose_gt']['probe_R'][1] = r
        x['pose_gt']['probe_t'][1,...,0] = 1.
        p = torch.tensor([[0.,0.,0.],[2.,3.,4.]])
        result = project_physical_points(p,x,1.,(32,32))
        torch.testing.assert_close(result['native_pixels'][0,0,0],torch.tensor([[16.,16.],[19.2,20.8]]))
        torch.testing.assert_close(result['native_pixels'][1,0,0],torch.tensor([[16.,17.6],[20.8,14.4]]))
        torch.testing.assert_close(result['distance_mm'][0,0,0],torch.tensor([0.,4.]))
        # Real MaskTransform affine: fixed 40mm model FOV, native 20mm FOV.
        transformed = normalize_masks(np.zeros((24,3,4,32,32)),(20/32,20/32),target_size=32,
            canonical_fov_mm=dict(A2C=40,A3C=40,A4C=40))
        for key in ('model_to_native','pixel_spacing','native_pixel_spacing'):
            x[key] = torch.tensor(transformed[key])[None].expand(2,*transformed[key].shape).float()
        result = project_physical_points(p,x,1.,(32,32))
        torch.testing.assert_close(result['model_pixels'][0,0,0,1],torch.tensor([17.6,18.4]))
        # Compare the original renderer algebra directly (including source epsilon).
        q = (p[1].numpy()-np.array([1,0,0]))@r.numpy()
        expected = (q[:2]/(10+1e-8)+1)*16
        np.testing.assert_allclose(result['native_pixels'][1,0,0,1].numpy(),expected,atol=1e-5)

    def test_linear_feature_sampling_and_unknown_flags(self):
        x = observations(side=224)
        lift = GeometryFeatureLifting(input_width=1,projected_width=1,sigma_mm=1.)
        with torch.no_grad():
            for proj in lift.projections:proj.weight.fill_(1);proj.bias.zero_()
        maps = {}
        for key,side in (('scale14',14),('scale7',7)):
            # Feature pixel center mapped back to model integer pixel index.
            stride, offset = (16,4) if side == 14 else (32,8)
            coordinate = torch.arange(side,dtype=torch.float32)*stride+offset
            maps[key] = coordinate[None,None,None,None,None,:].expand(1,24,3,1,side,side).clone().requires_grad_()
        valid = x['annotated'].clone();maps['validity'] = valid
        query = torch.tensor([[[[0.,0.,0.],[2.,0.,0.],[0.,0.,9.],[30.,0.,0.]]]])
        out = lift(maps,x,query,1.)
        torch.testing.assert_close(out['features'][0,0,0,0,0,:2],torch.tensor([112.,134.4]),atol=1e-5,rtol=1e-5)
        self.assertEqual(out['support'][0,0,0,0].tolist(),[True,True,False,False])
        self.assertTrue((out['features'][0,0,:2,0,0,2:] == 0).all())
        out['features'].sum().backward()
        self.assertGreater(float(maps['scale14'].grad.abs().sum()),0)
        valid[...,0,:] = False
        maps['scale14'].data[:,:,0] = 1e6;maps['scale7'].data[:,:,0] = 1e6
        clean = lift(maps,x,query,1.)
        torch.testing.assert_close(clean['features'][0,0,0,0,0,:2],torch.tensor([112.,134.4]),atol=1e-5,rtol=1e-5)
        valid.zero_();unknown = lift(maps,x,query,1.)
        self.assertFalse(unknown['support'].any())
        self.assertTrue((unknown['features'][:,:,:3] == 0).all())

    def test_batch_joint_shapes_no_cross_case_or_gt_encoder_input(self):
        model = fixture_model().eval();x = observations(2)
        with torch.no_grad(),patch.object(model.heads.deform3,'forward',wraps=model.heads.deform3.forward) as joint:
            y = model(x)
        self.assertEqual(joint.call_count,1)
        self.assertEqual(model.heads.deform3.conv.out_channels,72)
        self.assertEqual(y['vertices_4d'].shape,(2,24,168,3))
        for key,shape in (('coarse',(2,6,6,8,3)),('medium',(2,12,12,16,3)),('fine',(2,24,24,24,32,3))):
            self.assertEqual(tuple(y['control_grids'][key].shape),shape)
        one = {k:(v[:1] if isinstance(v,torch.Tensor) else {a:b[:1] for a,b in v.items()} if isinstance(v,dict) else v) for k,v in x.items()}
        one['vertices_gt'] = torch.full((1,),float('nan'))
        with torch.no_grad():z = model(one)
        torch.testing.assert_close(z['vertices_4d'],y['vertices_4d'][:1],rtol=1e-5,atol=1e-6)
        self.assertFalse(any('motion_field' in name or 'shape_head' in name or 'motion_head' in name for name,_ in model.named_modules()))
        validate_template(model);validate_prediction(model,y,sample_jacobian=True)
        self.assertFalse(torch.equal(y['vertices_4d'][:,0],y['vertices_4d'][:,-1]))

    def test_missing_chambers_poisoned_input_cannot_leak(self):
        model = fixture_model().eval();x = observations()
        x['annotated'][...,0,2:] = False
        x['annotated'][...,1,:] = False;x['view_present'][...,1] = False
        a = copy.deepcopy(x);a['masks'][...,0,2:,:,:] = float('nan');a['sdf'][...,0,2:,:,:] = float('nan')
        a['masks'][...,1,:,:,:] = float('nan');a['sdf'][...,1,:,:,:] = float('nan')
        with torch.no_grad():y,z = model(x),model(a)
        torch.testing.assert_close(y['vertices_4d'],z['vertices_4d'],rtol=0,atol=0)

    def test_actual_parameter_shape_and_phase23_motion_gradients(self):
        model = fixture_model();x = observations()
        for kind in ('shape','motion'):
            model.zero_grad(set_to_none=True)
            y = model(x);p = y['vertices_4d']
            loss = (p[:,0]-(model.template_vertices_mm[None]+.3)).square().mean() if kind == 'shape' else ((p[:,-1]-p[:,0])-.2).square().mean()
            loss.backward();report = finite_gradient_report(model)
            self.assertTrue(all(value > 0 for value in report.values()))
            input_gradient = model.unet.conv0.weight.grad.reshape(model.unet.conv0.out_channels,24,model.lifting.output_width,-1)
            self.assertTrue((input_gradient.square().sum((0,2,3)) > 0).all())
            if kind == 'motion':
                phase_gradient = model.heads.deform3.conv.weight.grad.reshape(24,-1).norm(dim=1)
                self.assertTrue((phase_gradient[1:] > 0).all())
        # A non-reference observation changes the joint prediction.
        model.eval();changed = copy.deepcopy(x);changed['sdf'][:,17] *= -3
        with torch.no_grad():a,b = model(x),model(changed)
        self.assertGreater(float((a['control_grids']['fine'][:,17]-b['control_grids']['fine'][:,17]).abs().max()),1e-10)

    def test_checkpoint_geometry_and_architecture_rejection(self):
        model = fixture_model().eval();x = observations()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'joint.pth'
            save_checkpoint(path,model,None,config={'fixed':1},identity={'train':'fixed'},training_stage='motion')
            other,saved = load_checkpoint(path,config={'fixed':1},identity={'train':'fixed'})
            with torch.no_grad():torch.testing.assert_close(model(x)['vertices_4d'],other.eval()(x)['vertices_4d'],rtol=0,atol=0)
            self.assertEqual(saved['training_stage'],'motion')
            with self.assertRaisesRegex(ValueError,'data identity'):load_checkpoint(path,identity={'train':'changed'})
            with self.assertRaisesRegex(ValueError,'configuration'):load_checkpoint(path,config={})
            saved['model_state']['control_spacings_mm'][0,0] += 1;torch.save(saved,path)
            with self.assertRaisesRegex(ValueError,'buffer'):load_checkpoint(path)
            saved['architecture'] = 'neural-heartffd-incremental-v1';torch.save(saved,path)
            with self.assertRaisesRegex(ValueError,'architecture'):load_checkpoint(path)

    def test_recurrence_constant_fields_and_loss_adapter(self):
        model = fixture_model();x = observations()
        # Keep network prediction one-shot, independently verify physical application.
        fields = dict(coarse=torch.zeros(1,6,6,8,3),medium=torch.zeros(1,12,12,16,3),fine=torch.zeros(1,24,24,24,32,3))
        fields['coarse'][...,0]=.1;fields['medium'][...,1]=.2
        fields['fine'][:,0,...,2]=.3;fields['fine'][:,1:,...,0]=.01
        with patch.object(model.heads,'forward',return_value=fields):y = model(x)
        torch.testing.assert_close(y['vertices_ref'],model.template_vertices_mm[None]+torch.tensor([.1,.2,.3]))
        torch.testing.assert_close(y['vertices_4d'][:,-1],y['vertices_ref']+torch.tensor([.23,0,0]),atol=3e-6,rtol=1e-5)
        criterion = NeuralFFDLoss(model.chamber_ranges,model.faces)
        losses = criterion(loss_output_adapter(y),y['vertices_4d'].detach()+.2,
            slice_result={'total':y['vertices_4d'].sum()*0+.1},control_spacings_mm=model.control_spacings_mm)
        self.assertEqual(len(losses['terms']),9)

    def test_existing_ffdnet_modules_are_reused_by_identity(self):
        from net.ffdnet import FFDNet
        original = []
        def factory(*args,**kwargs):
            source = FFDNet(*args,**kwargs);original.append(source);return source
        with patch('net.geometry_heartffdnet.FFDNet',side_effect=factory):
            model = fixture_model()
        source = original[0]
        for key in ('conv2','conv3','conv4','conv5','deconv2','deconv3','deconv4','deconv5'):
            self.assertIs(getattr(model.unet,key),getattr(source,key))
        for key in ('deform1','deform2','deform3'):
            self.assertIs(getattr(model.heads,key),getattr(source,key))
        self.assertEqual(model.unet.conv2.conv.stride,(1,1,1))
        self.assertEqual(FFDNet(n_frame=24,dim_hid=8).conv2.conv.stride,(2,2,2))

    def test_invalid_phase_pose_calibration_and_support_are_errors(self):
        model = fixture_model().eval();x = observations()
        x['phases'][0,-1] = 1.
        with self.assertRaisesRegex(ValueError,'Phases'):model(x)
        x = observations();x['pose_gt']['probe_R'][0,3,1,0,0] = 2.
        with self.assertRaisesRegex(ValueError,'poses'):model(x)
        x = observations();x['pixel_spacing'] *= 2
        with self.assertRaisesRegex(ValueError,'spacing mismatch'):model(x)
        v = model.template_vertices_mm[None].clone();v[0,0,0] = 12
        with self.assertRaisesRegex(ValueError,'support domain'):
            model._deform(v,torch.zeros(1,6,6,8,3),0)

    def test_optimizer_rng_resume_reproduces_next_update(self):
        model = fixture_model();x = observations();optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
        def update(net,opt):
            opt.zero_grad(set_to_none=True)
            y = net(x)['vertices_4d']
            y.sub(.2).square().mean().backward();opt.step()
        update(model,optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'resume.pth'
            save_checkpoint(path,model,optimizer,config={},identity={},step=1)
            restored,saved = load_checkpoint(path)
            other = torch.optim.Adam(restored.parameters(),lr=1e-3)
            other.load_state_dict(saved['optimizer_state'])
            from train_neural_ffd import restore_rng
            restore_rng({'rng':saved['rng_state']},'cpu');update(model,optimizer)
            restore_rng({'rng':saved['rng_state']},'cpu');update(restored,other)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)

    def test_triangle_intersection_detector(self):
        vertices = np.array([[0,0,0],[2,0,0],[0,2,0],[.5,.5,-1],[.5,.5,1],[1.5,.5,0.]])
        faces = np.array([[0,1,2],[3,4,5]])
        self.assertEqual(triangle_self_intersections(vertices,faces),1)
        vertices[3:] += 10
        self.assertEqual(triangle_self_intersections(vertices,faces),0)
        vertices[3:] = np.array([[.2,.2,0],[.8,.2,0],[.2,.8,0]])
        self.assertEqual(triangle_self_intersections(vertices,faces),1)


if __name__ == '__main__':unittest.main()
