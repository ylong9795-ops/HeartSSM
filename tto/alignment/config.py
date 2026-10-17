"""Strict, shared S1 search policy; no patient-specific tuning."""
import math

KEYS = ('initial_pose_file', 'initial_pose_sha256', 'source_min_weight', 'huber_delta_mm',
    'coarse_depth_span_mm', 'coarse_depth_step_mm', 'coarse_tilt_deg', 'coarse_roll_deg',
    'capture_quantiles', 'refinement_steps', 'retain_candidates', 'beam_width',
    'coarse_cells_per_chamber', 'max_evaluations_per_view', 'max_seconds_per_view')


def validate_pose_config(config):
    if not isinstance(config, dict) or set(config) != set(KEYS):
        raise ValueError(f'alignment.pose requires exactly {KEYS}')
    if not isinstance(config['initial_pose_file'], str) or not config['initial_pose_file']:
        raise ValueError('Explicit TRAIN-only initial pose file required')
    checksum = config['initial_pose_sha256']
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Pinned initial pose SHA256 required')
    for key in ('source_min_weight', 'huber_delta_mm', 'coarse_depth_span_mm', 'coarse_depth_step_mm',
                'coarse_tilt_deg', 'coarse_roll_deg', 'max_seconds_per_view'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Positive finite alignment.pose.{key} required')
    if config['source_min_weight'] > 1 or config['coarse_depth_span_mm'] < 40:
        raise ValueError('S1 requires source weight <=1 and a depth capture span >=40 mm')
    if config['coarse_depth_step_mm'] > config['coarse_depth_span_mm']:
        raise ValueError('Depth step must resolve the declared capture range')
    for key in ('retain_candidates', 'beam_width', 'coarse_cells_per_chamber', 'max_evaluations_per_view'):
        if type(config[key]) is not int or config[key] < (2 if key != 'max_evaluations_per_view' else 1):
            raise ValueError(f'alignment.pose.{key} must retain multiple hypotheses/evidence')
    quantiles = config['capture_quantiles']
    if not isinstance(quantiles, list) or not quantiles or any(type(v) not in (float, int) or not 0 < v < 1 for v in quantiles):
        raise ValueError('Interior capture quantiles required')
    steps = config['refinement_steps']
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError('At least two coarse-to-fine levels required')
    for step in steps:
        if set(step) != {'depth_mm', 'angle_deg'} or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in step.values()):
            raise ValueError('Positive depth_mm/angle_deg refinement steps required')
    if any(a['depth_mm'] <= b['depth_mm'] or a['angle_deg'] <= b['angle_deg'] for a, b in zip(steps, steps[1:])):
        raise ValueError('Refinement steps must decrease')
