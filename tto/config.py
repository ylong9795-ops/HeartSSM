"""One strict scaffold configuration. Paths resolve from repository root."""
from copy import deepcopy
from pathlib import Path
import math

import yaml
from .io import ROOT

VERSION = 'tto-scaffold-v1'


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _keys(value, expected, section):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{section} requires exactly: {", ".join(expected)}')


def _number(value, name, *, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if (integer and not isinstance(value, int)) or value < 0 or (positive and value <= 0):
        raise ValueError(f'Invalid {name}: {value}')


def validate_config(config):
    if not isinstance(config, dict) or config.get('version') != VERSION:
        raise ValueError(f'Expected {VERSION}; legacy TTO protocols are retired')
    _keys(config, ('version', 'prior', 'data', 'supervision', 'loss', 'alignment', 'runtime', 'output'), 'config')
    prior = config['prior']
    _keys(prior, ('checkpoint', 'sha256', 'shape_dim', 'motion_dim'), 'prior')
    if prior['shape_dim'] != 64 or prior['motion_dim'] != 32:
        raise ValueError('Original Shape64 / Motion32 required')
    if not isinstance(prior['checkpoint'], str) or not prior['checkpoint']:
        raise ValueError('Explicit pretrained checkpoint required')
    checksum = prior['sha256']
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Pinned lowercase checkpoint SHA256 required')
    data = config['data']
    _keys(data, ('root', 'input_mode', 'frames', 'model_size', 'sdf_clip_mm', 'canonical_fov_mm', 'load_images', 'source_weights'), 'data')
    if data['input_mode'] not in ('aligned24', 'raw_rr'):
        raise ValueError('data.input_mode must explicitly select aligned24 or raw_rr')
    if data['input_mode'] == 'aligned24' and data['model_size'] != 224:
        raise ValueError('aligned24 requires the pretrained 224 x 224 grid')
    if data['frames'] != 24:
        raise ValueError('Exactly 24 model phases required')
    _number(data['model_size'], 'model_size', positive=True, integer=True)
    _number(data['sdf_clip_mm'], 'sdf_clip_mm', positive=True)
    _keys(data['canonical_fov_mm'], ('A2C', 'A3C', 'A4C'), 'canonical_fov_mm')
    for value in data['canonical_fov_mm'].values():
        _number(value, 'canonical_fov_mm', positive=True)
    if not isinstance(data['root'], str) or not data['root'] or not isinstance(data['load_images'], bool):
        raise ValueError('data.root and boolean load_images required')
    if not isinstance(data['source_weights'], dict) or not data['source_weights']:
        raise ValueError('Explicit annotation source trust weights required')
    for value in data['source_weights'].values():
        _number(value, 'source_weight')
        if value > 1:
            raise ValueError('Source weights must be in [0,1]')
    policy = config['supervision']
    _keys(policy, ('max_nearest_time_error_ms', 'unverified_phase_confidence', 'trusted_confidence_threshold'), 'supervision')
    for name, value in policy.items():
        _number(value, name)
        if name != 'max_nearest_time_error_ms' and value > 1:
            raise ValueError(f'{name} must be in [0,1]')
    loss = config['loss']
    _keys(loss, ('sigma_mm', 'pixel_stride', 'chunk_size', 'cell_chunk_size', 'dice_weight', 'bce_weight', 'boundary_weight', 'target_mode'), 'loss')
    if loss['target_mode'] != 'binary_native_v1':
        raise ValueError('Only native binary 2D supervision is supported')
    for name in ('sigma_mm', 'pixel_stride', 'chunk_size', 'cell_chunk_size'):
        _number(loss[name], name, positive=True, integer=name != 'sigma_mm')
    for name in ('dice_weight', 'bce_weight', 'boundary_weight'):
        _number(loss[name], name)
    if sum(loss[name] for name in ('dice_weight', 'bce_weight', 'boundary_weight')) <= 0:
        raise ValueError('At least one 2D loss component must be active')
    _keys(config['alignment'], ('chamber_mode', 'pose', 'regional'), 'alignment')
    from .alignment.config import validate_pose_config
    validate_pose_config(config['alignment']['pose'])
    from .alignment.s2_config import validate_s2_config
    validate_s2_config(config['alignment']['regional'])
    if config['alignment']['chamber_mode'] != 'initial_candidates_only':
        raise ValueError('Chamber translations may only generate initial candidates')
    _keys(config['runtime'], ('device', 'threads', 'seed'), 'runtime')
    if config['runtime']['device'] != 'cpu':
        raise ValueError('Scaffold initialization supports cpu; GPU workload admission is not wired yet')
    _number(config['runtime']['threads'], 'threads', positive=True, integer=True)
    _number(config['runtime']['seed'], 'seed', integer=True)
    _keys(config['output'], ('root',), 'output')
    if not isinstance(config['output']['root'], str) or not config['output']['root']:
        raise ValueError('Output root required')
    return deepcopy(config)


def load_config(path):
    return validate_config(yaml.safe_load(resolve(path).read_text()))
