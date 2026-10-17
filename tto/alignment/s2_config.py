"""Shared S2 protocol, checked before patient data/model work."""
import math

KEYS=('steps','priority_steps','learning_rate','seed_scales','pixel_stride','chunk_size','sigma_mm',
    'center_scale_mm','missing_center_penalty','weights','edge_prior_weight','normal_prior_weight',
    'av_prior_weight','surface_prior_weight','latent_prior_weight','huber_delta_mm','hint_ridge',
    'observable_singular_ratio','max_hint_mm','latent_fit_ridge',
    'native_dice_regression_tolerance','minimum_dice_gain','geometry_timeout_seconds','max_seconds_per_case')


def validate_s2_config(config):
    if not isinstance(config,dict) or set(config)!=set(KEYS):raise ValueError(f'alignment.regional requires exactly {KEYS}')
    for k in ('steps','priority_steps','pixel_stride','chunk_size'):
        if type(config[k]) is not int or config[k]<1:raise ValueError(f'Invalid S2 {k}')
    if config['priority_steps']>config['steps']:raise ValueError('Priority steps exceed total steps')
    for k in set(KEYS)-{'steps','priority_steps','pixel_stride','chunk_size','seed_scales','weights'}:
        value=config[k]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError(f'Invalid S2 {k}')
    for k in ('learning_rate','sigma_mm','center_scale_mm','missing_center_penalty','huber_delta_mm','hint_ridge',
              'latent_fit_ridge','max_hint_mm','geometry_timeout_seconds','max_seconds_per_case','observable_singular_ratio'):
        if config[k]<=0:raise ValueError(f'Positive S2 {k} required')
    if not 0<config['observable_singular_ratio']<1:raise ValueError('Invalid observability rank threshold')
    scales=config['seed_scales']
    if not isinstance(scales,list) or not scales or scales[0]!=0 or any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in scales):
        raise ValueError('S2 includes an unshifted baseline branch and explicit finite initialization scales')
    weights=config['weights']
    if set(weights)!={'center','boundary_mm','dice','anatomy'} or any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in weights.values()):
        raise ValueError('All four S2 objective components must be active')
