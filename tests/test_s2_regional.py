"""S2 observability, common gauge, original decoder and motion contracts."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from tto.alignment.chamber_alignment import relative_translation_hints
from tto.alignment.regional_decoder import FixedGlobalRegionalDecoder
from tto.config import load_config
from tto.io import ROOT,load_state,read_json
from tto.optimization.regional_shape import observation_acceptance
from tto.prior import FrozenNeuralFFDPrior
from tto.state import TTOState


def projected_rows(displacements,shared=None):
    rotations=np.stack([np.eye(3),Rotation.from_euler('y',55,degrees=True).as_matrix(),
                        Rotation.from_euler('z',15,degrees=True).as_matrix()])
    shared=np.zeros(3) if shared is None else np.asarray(shared)
    rows=[]
    for ci in range(4):
        for vi in (range(3) if ci<2 else (2,)):
            e=-rotations[vi,:,:2].T@(displacements[ci]+shared)
            for phase in (0,12):
                rows.append(dict(chamber=ci,view=vi,phase=phase,fit_weight=1.,renderer_error=None,
                                 centroid_error_xy_mm=e.copy()))
    return rows,dict(R=rotations,center_mm=np.zeros((3,3)))


class RelativeHintTests(unittest.TestCase):
    def test_sign_multiview_and_single_view_nullspace_with_zero_common_component(self):
        displacement=np.array([[2.,1.,2.],[-1.,2.,-2.],[1.,-1.,0.],[-2.,-2.,0.]])
        rows,pose=projected_rows(displacement)
        result=relative_translation_hints(rows,pose,ridge=1e-10,max_hint_mm=100.)
        np.testing.assert_allclose(result['displacements_mm'],displacement,atol=1e-6)
        np.testing.assert_allclose(result['displacements_mm'].sum(0),0,atol=1e-10)
        for ci in (2,3):
            self.assertEqual(result['chambers'][ci]['rank'],2)
            self.assertAlmostEqual(float(pose['R'][2,:,2]@result['displacements_mm'][ci]),0.,places=10)

    def test_pure_shared_shift_does_not_create_relative_chamber_changes(self):
        rows,pose=projected_rows(np.zeros((4,3)),shared=[5.,-3.,2.])
        result=relative_translation_hints(rows,pose,ridge=1e-6)
        np.testing.assert_allclose(result['displacements_mm'],0,atol=1e-7)
        np.testing.assert_allclose(result['removed_shared_component_mm'],[5.,-3.,2.],atol=1e-7)

    def test_candidate_gate_checks_every_chamber_not_only_lv(self):
        baseline=dict(macro_trusted_dice=.6,chambers={c:dict(trusted_dice=.6) for c in ('LV','LA','RV','RA')})
        candidate=deepcopy(baseline);candidate.update(macro_trusted_dice=.65,renderer_errors=0)
        candidate['chambers']['LV']['trusted_dice']=.9;candidate['chambers']['RV']['trusted_dice']=.4
        options=load_config('configs/tto.yaml')['alignment']['regional']
        ok,reasons=observation_acceptance(baseline,candidate,options)
        self.assertFalse(ok);self.assertIn('RV trusted native Dice regression',reasons)


@unittest.skipUnless((ROOT/'outputs/neural_ffd_1k_long_20261010_225918/best_joint.pth').exists(),'Original checkpoint unavailable')
class OriginalRegionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.prior=FrozenNeuralFFDPrior(load_config('configs/tto.yaml')['prior']['checkpoint'])
        cls.initial=load_state(ROOT/'outputs/s1_native_alignment_20261017/cohort/patients/C09/aligned.pth')
        stats=read_json(ROOT/'configs/real_tto_prior_calibration.json')
        cls.decoder=FixedGlobalRegionalDecoder(cls.prior,cls.initial['z_shape'],cls.initial['conditions'],stats['shape_std'][32:])

    def test_exact_original_decoder_and_global_motion_freeze(self):
        raw=torch.linspace(-.2,.2,32).reshape(4,8).requires_grad_()
        ref,z=self.decoder.decode(raw)
        direct=self.prior.decode_shape(z,self.initial['conditions'])
        torch.testing.assert_close(ref,direct,atol=0,rtol=0)
        torch.testing.assert_close(z[:,:32],self.initial['z_shape'][:,:32],atol=0,rtol=0)
        ref.square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all());self.assertGreater(float(raw.grad.norm()),0.)
        delta=ref.detach()[0]-self.initial['vertices_reference']
        common=torch.stack([delta[a:b].mean(0) for a,b in self.prior.model.chamber_ranges.tolist()]).mean(0)
        self.assertLess(float(common.norm()),1e-4)
        self.prior.assert_frozen()

    def test_frozen_motion_code_allows_original_decoder_motion_change(self):
        raw=torch.linspace(-.3,.3,32).reshape(4,8)
        z=self.decoder.assert_original(raw)
        with torch.no_grad():
            result=self.prior.decode_original(z,self.initial['z_motion'],self.initial['conditions'])
        new=result['vertices_4d'][0];old=self.initial['trajectory']
        error=((new-new[:1])-(old-old[:1])).abs().max()
        self.assertGreater(float(error),1e-6)
        self.assertEqual(result['vertices_4d'].shape,(1,24,10631,3))
        self.prior.assert_frozen()

    def test_saved_state_enforces_frozen_global_motion_pose_and_no_offsets(self):
        payload=deepcopy(self.initial)
        payload['diagnostics']['s2_candidate']=dict(partition_lock_version='regional-only-v1',
            frozen_parameters=dict(global32=payload['z_shape'][:,:32].clone(),motion32=payload['z_motion'].clone(),
                pose={k:v.clone() for k,v in payload['observation_pose'].items()}),
            temporary_offsets_used_in_final_mesh=False)
        state=TTOState.from_payload(payload)
        for key in ('z_shape','z_motion'):
            broken=deepcopy(payload);broken[key][0,0]+=1
            with self.assertRaisesRegex(ValueError,'Global32|Motion32'):
                TTOState.from_payload(broken)
        broken=deepcopy(payload);broken['observation_pose']['center_mm'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'same shared S1'):
            TTOState.from_payload(broken)
        broken=deepcopy(payload);broken['diagnostics']['s2_candidate']['temporary_offsets_used_in_final_mesh']=True
        with self.assertRaisesRegex(ValueError,'initialization hints only'):
            TTOState.from_payload(broken)


if __name__=='__main__':unittest.main()
