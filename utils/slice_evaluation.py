"""Evaluation directly on original labels with their source renderer and pose.

This module is never used by the differentiable training objective. The 1K
generation plan pins core.py to 4bd081a4..., whose section_mask is copied here
verbatim. The later bundled core uses topological endpoint assembly and can
disagree with the existing labels; it must not replace the recorded generator.
"""
import hashlib
import inspect

import numpy as np
import cv2
import trimesh


EVALUATION_POLICY = 'source-native-fillpoly-original-label-v3-generation-v1'
GENERATION_CORE_SHA256 = '4bd081a41efeed061959df89752e9afa79e7bb82c39833af128d7ad1ce4182b5'


def section_mask(vertices: np.ndarray, faces: np.ndarray, R: np.ndarray, t: np.ndarray,
                 s: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Rasterise one chamber section using the exact LV ``trimesh + fillPoly`` core.

    This is intentionally copied from ``LV_only/search_probe_real.py`` with
    only the Dice-specific ground-truth lines removed.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normal = R[:, 2]
    origin = t
    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
    except Exception:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    if section is None or len(section.vertices) < 3:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    vert3d = section.vertices.astype(np.float64)
    local = (vert3d - t[None, :]) @ R
    px = (local[:, 0] / (s[0] + 1e-8) + 1.0) * 0.5 * image_size
    py = (local[:, 1] / (s[1] + 1e-8) + 1.0) * 0.5 * image_size

    img = np.zeros((image_size, image_size), dtype=np.uint8)
    for entity in section.entities:
        if len(entity.points) < 3:
            continue
        pts = np.stack([px[entity.points], py[entity.points]], axis=-1).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, image_size - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, image_size - 1)
        cv2.fillPoly(img, [pts], 255)
    return (img > 127).astype(np.uint8)


def source_renderer_identity():
    return dict(policy=EVALUATION_POLICY, evaluation_grid='original_native', target='native_masks',
                libraries=dict(numpy=np.__version__,opencv=cv2.__version__,trimesh=trimesh.__version__),
                renderer='utils.slice_evaluation.section_mask',
                generation_core_sha256=GENERATION_CORE_SHA256,
                section_assembly='original-trimesh-mesh-section-merged-endpoints-v1',
                section_mask_sha256=hashlib.sha256(inspect.getsource(section_mask).encode()).hexdigest())


def render_source_slice(vertices_mm, faces, rotation, translation_normalized,
                        probe_scale_normalized, norm_to_mm, native_shape):
    """Render in the source grid with the recorded source R, t, s and size.

    Only convert physical vertices back to source coordinate units. There is
    no image resampling, pose adjustment or reference to network-input masks.
    """
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
