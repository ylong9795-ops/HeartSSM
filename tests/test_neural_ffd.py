"""Independent numerical and end-to-end contracts for Neural HeartFFDNet."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import BSpline

from utils.ffd_physical import bspline_ffd_physical, control_geometry
from net.multiview_encoder import MultiViewTemporalEncoder
from net.neural_ffdnet import NeuralFFDNet
from utils.slice_supervision import (plane_segments,signed_segment_distance,section_soft_mask,
                                     exact_section_mask,MissingSectionError,MeshSliceSupervisor)
from utils.neural_ffd_losses import NeuralFFDLoss


class NeuralTestCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        # Do not leave an active multi-thread OpenMP pool for the legacy
        # Dataset's explicit fork-worker test later in unittest discovery.
        torch.set_num_threads(1)


def small_model(frames=4):
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=1,radius=2.)
    vertices = torch.cat([torch.tensor(mesh.vertices,dtype=torch.float32)+torch.tensor(center)
                          for center in ((-3.,-3.,0.),(-3.,3.,0.),(3.,-3.,0.),(3.,3.,0.))])
    count = len(mesh.vertices)
    ranges = torch.tensor([[i*count,(i+1)*count] for i in range(4)])
    faces = {c:torch.tensor(mesh.faces.copy()) for c in ('LV','LA','RV','RA')}
    return NeuralFFDNet(vertices,ranges,faces,template_sha256='fixture-template',asset_sha256='fixture-asset',norm_to_mm=1.,
        encoder_config=dict(shape_dim=12,motion_dim=4,width=16,heads=2,temporal_layers=1,
                            spatial_sides=(4,2),query_slots=2,cnn_chunk_size=4),
        decoder_config=dict(frames=frames,channels=4,seed_shape=(4,4,4),global_shape_dim=4,
                            lower_mm=(-10.,-10.,-10.),upper_mm=(10.,10.,10.)))


class DecoderTests(NeuralTestCase):
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
        x['masks'] = torch.tensor(mask)[None]
        x['sdf'] = torch.tensor(mask_to_sdf(mask,[.625,.625],annotated=x['annotated'][0].numpy()))[None]
        out = net(x)
        supervisor = MeshSliceSupervisor(net.chamber_ranges,net.faces,1.,sigma_mm=.4,pixel_stride=1)
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
    def test_optimizer_restore_geometry_identity_and_stages(self):
        from train_neural_ffd import save_checkpoint,load_checkpoint,stage_for_epoch
        torch.set_num_threads(2)
        net = small_model()
        optimizer = torch.optim.Adam(net.parameters(),lr=.001)
        z,m = torch.randn(1,12),torch.randn(1,4)
        optimizer.zero_grad()
        loss = net.decode_4d(z,m).square().mean()
        loss.backward();optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pth'
            save_checkpoint(path,net,optimizer,config={'fixture':True},identity={'fixture':'abc'},epoch=2,sample_offset=1,step=3)
            restored,saved = load_checkpoint(path)
            restored_optimizer = torch.optim.Adam(restored.parameters(),lr=.001)
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
    def test_latents_spatial_scales_and_gt_independence(self):
        torch.set_num_threads(2)
        net = MultiViewTemporalEncoder(shape_dim=12, motion_dim=4, width=16, heads=2,
                                      temporal_layers=1, spatial_sides=(4,2), query_slots=2, checkpoint_cnn=True)
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


if __name__ == '__main__':
    unittest.main()
