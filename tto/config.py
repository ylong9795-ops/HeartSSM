"""Versioned configuration. Relative paths always resolve from repository root."""
from copy import deepcopy
from pathlib import Path
import yaml

from .io import ROOT


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    version = config.get('version')
    if version not in ('real-tto-v1', 'real-tto-joint-v1'):
        raise ValueError('Expected version: real-tto-v1 or real-tto-joint-v1')
    required = ('prior', 'data', 'loss', 'pose', 'motion_limits', 'joint', 'runtime', 'output')
    required += ('shape',) if version == 'real-tto-v1' else ('supervision', 'synthetic', 'experiment')
    for key in required:
        if key not in config:
            raise ValueError(f'Missing config section {key}')
    if config['prior'].get('shape_dim') != 64 or config['prior'].get('motion_dim') != 32:
        raise ValueError('This protocol requires original 64D Shape / 32D Motion')
    if config['data'].get('frames') != 24:
        raise ValueError('This protocol requires 24 phases')
    if config['runtime']['batch_size'] < 1:
        raise ValueError('Positive batch_size required')
    return deepcopy(config)
