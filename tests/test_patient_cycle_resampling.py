"""Fixed manual anchors and temporal fidelity in native cycle downsampling."""
import json
import unittest
import numpy as np

from scripts.resample_patient_cycles_24frames import anchored_downsample, make_plan, mapping_rows


class CycleResamplingTests(unittest.TestCase):
    def test_long_cycle_has_24_unique_ordered_native_frames_with_all_anchors(self):
        plan = anchored_downsample(56, 87, [56, 57, 69, 70, 87])
        indices = plan['source_frame_indices']
        self.assertEqual(len(indices), 24)
        self.assertEqual((indices[0], indices[-1]), (56, 87))
        self.assertTrue((np.diff(indices) > 0).all())
        self.assertTrue({56, 57, 69, 70, 87} <= set(indices))

    def test_temporal_mapping_reconstructs_native_source_with_one_hot_weights(self):
        plan = anchored_downsample(56, 87, [57, 69, 70])
        plan['first_source_frame'] = 56
        for row in mapping_rows(plan, 35.744569):
            source = row['target_lower_source_frame']*row['nearest_lower_weight']+row['target_upper_source_frame']*row['nearest_upper_weight']
            self.assertEqual(source, row['source_frame'])
            self.assertEqual(row['nearest_lower_weight']+row['nearest_upper_weight'], 1)
            self.assertLessEqual(abs(row['nearest_time_error_ms']), 35.744569/2+1e-9)

    def test_anchor_sampling_does_not_fabricate_uniform_timestamps(self):
        plan = anchored_downsample(56, 87, [57, 69, 70])
        self.assertGreater(len(set(np.diff(plan['source_frame_indices']).tolist())), 1)
        self.assertGreater(len(set(np.round(np.diff(plan['target_source_frame_positions']), 6).tolist())), 1)

    def test_all_native_frames_are_preserved_if_source_has_exactly_24(self):
        plan = anchored_downsample(10, 33, [11, 22])
        self.assertEqual(plan['source_frame_indices'], list(range(10, 34)))

    def test_preserves_random_anchor_sets_without_duplicates(self):
        rng = np.random.default_rng(2026)
        for length in range(24, 90):
            anchors = rng.choice(np.arange(10, 10+length), size=min(12, length), replace=False).tolist()
            plan = anchored_downsample(10, 10+length-1, anchors)
            self.assertEqual(len(set(plan['source_frame_indices'])), 24)
            self.assertTrue(set(anchors) <= set(plan['source_frame_indices']))
            json.dumps(plan, allow_nan=False)

    def test_manual_ed_before_area_peak_is_retained_in_full_coverage(self):
        coverage = dict(status='rejected_native24_too_short', sensitivity=[dict(reliable=True)],
                        coverage_envelope=dict(first_peak=79, next_peak=105),
                        detection=dict(selected=dict(first_peak=80, area_minimum=89, next_peak=105)))
        phases = {'ED': 78, 'ES': 89}
        plan = make_plan(coverage, phases)
        self.assertEqual((plan['first_source_frame'], plan['last_source_frame']), (78, 105))
        self.assertEqual(plan['source_interval_frame_count'], 28)
        for phase, frame in phases.items():
            self.assertEqual(plan['source_frame_indices'][plan['phase_indices'][phase]], frame)
        self.assertTrue({78, 79, 80, 89, 105} <= set(plan['source_frame_indices']))

    def test_unverified_cycle_is_not_rescued_by_resampling(self):
        with self.assertRaises(ValueError):
            make_plan(dict(status='review_required_lv_area'), {'ED': 1, 'ES': 5})

    def test_invalid_anchor_count_or_source_span_fails(self):
        for args in [(0, 10, [1, 5]), (10, 50, [9]), (10, 50, list(range(10, 35)))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                anchored_downsample(*args)


if __name__ == '__main__':
    unittest.main()
