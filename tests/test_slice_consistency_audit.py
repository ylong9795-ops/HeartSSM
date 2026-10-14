"""Perfect exact Dice is not zero binary soft loss on the same true mesh."""
import unittest
import numpy as np
import torch
from test_geometry_heartffdnet import fixture_model,observations
from data.mask_transform import mask_to_sdf
from utils.slice_evaluation import section_mask
from utils.slice_supervision import MeshSliceSupervisor
from utils.slice_consistency_audit import paired_slice_cells


class SliceConsistencyTests(unittest.TestCase):
    def test_same_gt_mesh_has_perfect_native_dice_and_positive_soft_loss(self):
        model=fixture_model();x=observations();x['annotated'].zero_();x['annotated'][:,:,0,0]=True
        x['pose_gt']['probe_t'][:,:,0,2]=.137
        vertices=model.template_vertices_mm[None,None].expand(1,24,-1,-1).clone()
        mask=section_mask(vertices[0,0,:42].numpy(),model.faces['LV'].numpy(),np.eye(3),
            np.array([0,0,.137],dtype=np.float32),np.full(3,10.,dtype=np.float32),32)
        x['masks'].zero_();x['masks'][0,:,0,0]=torch.as_tensor(mask)
        x['sdf']=torch.as_tensor(mask_to_sdf(x['masks'][0].numpy(),x['pixel_spacing'][0].numpy(),
            view_present=x['view_present'][0].numpy(),annotated=x['annotated'][0].numpy()))[None]
        x.update(native_masks=x['masks'].byte(),vertices_gt=vertices)
        slicer=MeshSliceSupervisor(model.chamber_ranges,model.faces,1.,pixel_stride=1,sigma_mm=.3)
        report=paired_slice_cells(model,vertices,x,slicer)
        self.assertEqual(report['summary']['gt_native_pixel_mismatches'],0)
        self.assertEqual(report['summary']['mean_exact_native_dice'],1.)
        self.assertGreater(report['summary']['mean_soft_total'],1e-4)


if __name__=='__main__':unittest.main()
