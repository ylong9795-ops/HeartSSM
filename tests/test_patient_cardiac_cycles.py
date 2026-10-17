"""LV area-defined cycles, unchanged anchors, and original-frame coverage."""
import unittest
import json
import numpy as np

from scripts.locate_patient_cardiac_cycles import locate_cycles


class LVAreaCycleTests(unittest.TestCase):
    def curves(self, period=20, n=144):
        area = 1000+300*np.cos(np.arange(n)*2*np.pi/period)
        return {'1': area.copy(), '2': area*2+100, '3': area*.6+300}

    def locate(self, areas=None, ed=40, es=50, window=range(30, 78)):
        return locate_cycles(self.curves() if areas is None else areas, ed, es, list(window), 40.)

    def test_peak_valley_peak_is_located_from_area_without_ecg(self):
        result = self.locate()
        cycle = result['detection']['selected']
        self.assertEqual((cycle['first_peak'], cycle['area_minimum'], cycle['next_peak']), (40, 50, 60))
        self.assertEqual(result['status'], 'accepted_lv_area_native24')
        self.assertEqual(cycle['supporting_views'], ['1', '2', '3'])
        self.assertEqual(result['existing48']['complete_lv_cycle_count'], 1)

    def test_cycle_evidence_roundtrips_as_finite_json(self):
        result = self.locate()
        saved = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(saved['native24'], result['native24'])

    def test_native24_covers_both_area_peaks_and_fixed_manual_frames(self):
        result = self.locate(ed=38)
        crop = result['native24']
        self.assertEqual(result['manual_phase_indices'], {'ED': 38, 'ES': 50})
        self.assertEqual(len(crop['source_frame_indices']), 24)
        self.assertTrue(np.all(np.diff(crop['source_frame_indices']) == 1))
        self.assertLessEqual(crop['first_source_frame'], 38)
        self.assertGreaterEqual(crop['last_source_frame'], 60)
        self.assertEqual(crop['manual_phase_indices']['ED']+crop['first_source_frame'], 38)

    def test_24_intervals_need_25_original_frames_including_both_peaks(self):
        result = self.locate(self.curves(period=24), ed=48, es=60, window=range(34, 82))
        self.assertEqual(result['minimum_frames_for_cycle_and_manual'], 25)
        self.assertEqual(result['status'], 'rejected_native24_too_short')
        self.assertIsNone(result['native24'])

    def test_short_neighboring_beat_cannot_replace_the_annotated_cycle(self):
        x = np.arange(130); area = np.full(130, 1300.)
        peaks = [0, 10, 45, 80, 92, 104, 116, 129]
        for start, end in zip(peaks[:-1], peaks[1:]):
            points = (x >= start) & (x <= end)
            area[points] = 1000+300*np.cos((x[points]-start)*2*np.pi/(end-start))
        result = self.locate({'1': area, '2': area*2, '3': area*.8}, ed=45, es=63, window=range(30, 78))
        self.assertIsNone(result['native24'])
        self.assertGreater(result['minimum_frames_for_cycle_and_manual'], 24)
        self.assertAlmostEqual(result['detection']['selected']['first_peak'], 45, delta=1)

    def test_isolated_manual_area_spike_does_not_create_a_new_cycle(self):
        areas = self.curves(); areas['1'][40] = 100000.
        before = {k: v.copy() for k, v in areas.items()}
        result = self.locate(areas)
        self.assertEqual(result['detection']['selected']['next_peak'], 60)
        self.assertEqual(result['status'], 'accepted_lv_area_native24')
        for key in areas:
            np.testing.assert_array_equal(areas[key], before[key])

    def test_flat_third_view_does_not_overrule_two_consistent_views(self):
        areas = self.curves(); areas['3'][:] = 1000
        result = self.locate(areas)
        self.assertEqual(result['status'], 'accepted_lv_area_native24')
        self.assertEqual(result['detection']['selected']['supporting_views'], ['1', '2'])

    def test_flat_or_single_view_variation_is_unverified(self):
        areas = self.curves(); areas['2'][:] = 1000; areas['3'][:] = 1000
        result = self.locate(areas)
        self.assertEqual(result['status'], 'review_required_lv_area')
        self.assertIsNone(result['native24'])

    def test_es_far_from_area_minimum_is_not_silently_relabelled(self):
        result = self.locate(es=42)
        self.assertEqual(result['status'], 'review_required_lv_area')
        self.assertEqual(result['manual_phase_indices']['ES'], 42)
        self.assertIsNone(result['native24'])

    def test_incomplete_first_cycle_is_not_invented_at_source_boundary(self):
        result = self.locate(ed=0, es=10, window=range(0, 48))
        self.assertEqual(result['status'], 'review_required_lv_area')
        self.assertIsNone(result['detection']['selected'])

    def test_saved_crop_satisfies_independent_exhaustive_window_search(self):
        for ed in range(37, 44):
            result = self.locate(ed=ed)
            bounds = result['coverage_envelope']; crop = result['native24']
            valid = [start for start in range(144-23) if start <= min(ed, 50, bounds['first_peak'])
                     and max(ed, 50, bounds['next_peak']) <= start+23]
            self.assertTrue(valid)
            self.assertIn(crop['first_source_frame'], valid)

    def test_malformed_sources_and_nonconsecutive_windows_fail(self):
        with self.assertRaises(ValueError):
            self.locate(areas={'1': np.ones(144)})
        areas = self.curves(); areas['1'][4] = np.nan
        with self.assertRaises(ValueError):
            self.locate(areas)
        with self.assertRaises(ValueError):
            self.locate(window=list(range(30, 77))+[79])
        with self.assertRaises(ValueError):
            self.locate(ed=4)


if __name__ == '__main__':
    unittest.main()
