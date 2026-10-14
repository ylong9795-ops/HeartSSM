"""Full-cycle metrics preserve independent endpoints, chambers and patients."""
import unittest
import torch
from test_geometry_heartffdnet import fixture_model,observations
from utils.geometry_motion_metrics import observed_vertex_regions,detailed_motion_metrics


class MotionMetricTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model=fixture_model()
        self.phases=torch.arange(24)[None]/24

    def test_volume_extrema_are_per_chamber_and_not_source_reference(self):
        template=self.model.template_vertices_mm
        phases=self.phases.squeeze(0)
        scale=1+.1*torch.sin(2*torch.pi*phases)
        trajectory=template[None,None]*scale[None,:,None,None]
        report=detailed_motion_metrics(trajectory,trajectory,self.model.chamber_ranges,
            self.model.faces,self.phases)
        lv=report['cases'][0]['chambers']['LV']
        self.assertEqual(lv['relative_motion_rmse_mm'],0)
        self.assertEqual(lv['volume']['maximum_phase'],.25)
        self.assertEqual(lv['volume']['minimum_phase'],.75)
        self.assertAlmostEqual(lv['volume']['ef_extremum_proxy'],1-(.9/1.1)**3,places=5)
        self.assertIsNone(report['cases'][0]['chambers']['LA']['volume']['ef_extremum_proxy'])

    def test_closing_step_and_static_amplitude_are_not_hidden(self):
        truth=self.model.template_vertices_mm[None,None].expand(2,24,-1,-1).clone()
        prediction=truth.clone();prediction[1,23,:,0]+=.3
        report=detailed_motion_metrics(prediction,truth,self.model.chamber_ranges,
            self.model.faces,self.phases.expand(2,-1))
        a,b=[case['chambers']['LV'] for case in report['cases']]
        self.assertEqual(a['closing_step_error_rms_mm'],0)
        self.assertIsNone(a['normalized_jerk_per_cycle3'])
        self.assertTrue(a['volume']['flat_curve'])
        self.assertIsNone(a['volume']['maximum_phase'])
        self.assertAlmostEqual(b['closing_step_error_rms_mm'],.3,places=6)
        self.assertGreater(b['jerk_rms_mm_per_cycle3'],0)
        self.assertGreater(b['motion_amplitude_rms_mm'],0)
        self.assertIsNone(b['motion_amplitude_ratio'])

    def test_chamber_annotation_and_per_patient_fov_define_regions(self):
        truth=self.model.template_vertices_mm[None,None].expand(2,24,-1,-1).clone()
        x=observations(2)
        x['annotated'].zero_();x['annotated'][:,:,0,0]=True
        x['pose_gt']['probe_t'][1,...,0]=100
        regions=observed_vertex_regions(truth,x,self.model.chamber_ranges,1.)
        self.assertTrue(regions['direct'][0,:,:42].any())
        self.assertTrue(regions['unknown'][0,:,42:].all())
        self.assertTrue(regions['unknown'][1].all())
        report=detailed_motion_metrics(truth,truth,self.model.chamber_ranges,self.model.faces,
            self.phases.expand(2,-1),regions=regions)
        missing=report['cases'][1]['chambers']['LV']['regions']['direct']
        self.assertEqual(missing['vertex_phase_count'],0)
        self.assertIsNone(missing['relative_motion_rmse_mm'])


if __name__=='__main__':unittest.main()
