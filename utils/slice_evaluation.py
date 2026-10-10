"""Evaluation directly on original labels with their source renderer and pose.

This module is never used by the differentiable training objective. It calls
the original data-generation function, including its endpoint assembly,
normalization, s+1e-8 projection, int32 truncation, clipping and fillPoly.
"""
import hashlib
import inspect

import numpy as np


EVALUATION_POLICY = 'source-native-fillpoly-original-label-v2'


def source_renderer_identity():
    from synthetic_bundle_20261009.data_generation import core
    import cv2
    import trimesh
    return dict(policy=EVALUATION_POLICY, evaluation_grid='original_native', target='native_masks',
                libraries=dict(numpy=np.__version__,opencv=cv2.__version__,trimesh=trimesh.__version__),
                renderer='synthetic_bundle_20261009.data_generation.core.section_mask',
                section_assembly=core.SECTION_ASSEMBLY_VERSION,
                section_mask_sha256=hashlib.sha256(inspect.getsource(core.section_mask).encode()).hexdigest(),
                section_path_sha256=hashlib.sha256(inspect.getsource(core.section_path).encode()).hexdigest())


def render_source_slice(vertices_mm, faces, rotation, translation_normalized,
                        probe_scale_normalized, norm_to_mm, native_shape):
    """Render in the source grid with the recorded source R, t, s and size.

    Only convert physical vertices back to source coordinate units. There is
    no image resampling, pose adjustment or reference to network-input masks.
    """
    from synthetic_bundle_20261009.data_generation.core import section_mask

    vertices = np.asarray(vertices_mm, dtype=np.float64)
    rotation = np.asarray(rotation)
    translation = np.asarray(translation_normalized)
    scale = np.asarray(probe_scale_normalized)
    native_shape = tuple(native_shape)
    if (vertices.ndim != 2 or vertices.shape[-1] != 3 or not np.isfinite(vertices).all()
            or not np.isfinite(norm_to_mm) or norm_to_mm <= 0
            or rotation.shape != (3, 3) or translation.shape != (3,) or scale.shape != (3,)
            or not all(np.isfinite(x).all() for x in (rotation, translation, scale))
            or (scale <= 0).any() or len(native_shape) != 2
            or min(native_shape) < 1 or native_shape[0] != native_shape[1]):
        raise ValueError('Valid source square grid and original pose/scale required')
    return section_mask(vertices / norm_to_mm, np.asarray(faces), rotation,
                        translation, scale, image_size=native_shape[0])
