"""Physical mask grids and optional contours, independent of mesh/pose/network.

Adapted from EchoGSF s1_shape/observation.py at
157aa7624cfda8e7319ec325af5d1088065f026d: inverse nearest-neighbour warp,
physical origin at (W/2,H/2), and EDT(outside)-EDT(inside). Extended to time,
anisotropic spacing, explicit transforms and finite empty-channel distances.
"""
from collections.abc import Mapping

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

VIEWS = ("A2C", "A3C", "A4C")
CHAMBERS = ("LV", "LA", "RV", "RA")


def _validity(time, view_present=None, annotated=None):
    present = np.ones(3, bool) if view_present is None else np.asarray(view_present)
    channels = np.ones((3, 4), bool) if annotated is None else np.asarray(annotated)
    if present.shape not in ((3,), (time, 3)) or channels.shape not in ((3, 4), (time, 3, 4)):
        raise ValueError("Invalid static or per-frame availability shape")
    if not np.isin(present, [0, 1]).all() or not np.isin(channels, [0, 1]).all():
        raise ValueError("Availability must be binary")
    present = np.broadcast_to(present.astype(bool), (time, 3))
    channels = np.broadcast_to(channels.astype(bool), (time, 3, 4))
    if np.any(channels & ~present[..., None]):
        raise ValueError("Annotated channel in absent view")
    return present[..., None] & channels


def _masks_and_spacing(masks, pixel_spacing):
    masks = np.asarray(masks)
    if masks.ndim != 5 or masks.shape[1:3] != (3, 4) or min(masks.shape) < 1:
        raise ValueError("Masks must be [T,3,4,H,W]")
    if not ((masks == 0) | (masks == 1) | (masks == 255)).all():
        raise ValueError("Masks must be binary independent channels")
    spacing = np.asarray(pixel_spacing, dtype=np.float64)
    if spacing.shape not in ((2,), (3, 2), (masks.shape[0], 3, 2)):
        raise ValueError("Spacing must be y,x in mm: [2], [3,2] or [T,3,2]")
    spacing = np.broadcast_to(spacing, (masks.shape[0], 3, 2))
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Positive finite calibrated pixel spacing required")
    return masks > 0, spacing


def normalize_masks(masks, pixel_spacing, *, target_size=None, canonical_fov_mm=None,
                    view_present=None, annotated=None, allow_clipping=False):
    """Fixed per-view FOV, never chamber/case bounding-box normalization.

    FOV values are full widths in mm (square) or (height,width) pairs. None
    preserves each native physical FOV, for explicit native-grid experiments.
    model_to_native maps homogeneous (x,y,1) pixel indices, not y,x. All
    chambers share one warp per view/frame. No flips or pose alignment occur.
    """
    masks, spacing = _masks_and_spacing(masks, pixel_spacing)
    time, _, _, height, width = masks.shape
    valid = _validity(time, view_present, annotated)
    masks = masks & valid[..., None, None]
    if target_size is None:
        out_h, out_w = height, width
    elif isinstance(target_size, (int, np.integer)):
        out_h = out_w = int(target_size)
    else:
        out_h, out_w = map(int, target_size)
    if min(out_h, out_w) < 1:
        raise ValueError("Positive target image dimensions required")
    if canonical_fov_mm is None:
        destination = spacing * np.array([height / out_h, width / out_w])
    else:
        if not isinstance(canonical_fov_mm, Mapping) or set(canonical_fov_mm) != set(VIEWS):
            raise ValueError("Fixed FOV must specify A2C, A3C, A4C")
        fov = np.asarray([canonical_fov_mm[v] for v in VIEWS], dtype=np.float64)
        if fov.shape == (3,):
            fov = np.repeat(fov[:, None], 2, axis=1)
        if fov.shape != (3, 2) or not np.isfinite(fov).all() or np.any(fov <= 0):
            raise ValueError("Positive full physical FOV widths required")
        destination = np.broadcast_to(fov / [out_h, out_w], spacing.shape)
    transforms = np.broadcast_to(np.eye(3), (time, 3, 3, 3)).copy()
    if (out_h, out_w) == (height, width) and np.array_equal(destination, spacing):
        return dict(masks=masks.astype(np.uint8), model_to_native=transforms,
                    pixel_spacing=destination.copy(), native_pixel_spacing=spacing.copy(),
                    clipped_foreground=np.zeros((time, 3, 4), np.int64))
    output = np.zeros((time, 3, 4, out_h, out_w), np.uint8)
    clipped = np.zeros((time, 3, 4), np.int64)
    for frame in range(time):
        for view in range(3):
            ratio_y, ratio_x = destination[frame, view] / spacing[frame, view]
            inverse = np.array([[ratio_x, 0, width / 2 - ratio_x * out_w / 2],
                                [0, ratio_y, height / 2 - ratio_y * out_h / 2],
                                [0, 0, 1]], dtype=np.float64)
            transforms[frame, view] = inverse
            for chamber in np.flatnonzero(valid[frame, view]):
                native = masks[frame, view, chamber]
                y, x = np.nonzero(native)
                model_x = (x - inverse[0, 2]) / ratio_x
                model_y = (y - inverse[1, 2]) / ratio_y
                clipped[frame, view, chamber] = np.count_nonzero(
                    (model_x < -.5) | (model_x >= out_w - .5) |
                    (model_y < -.5) | (model_y >= out_h - .5))
                if (out_h, out_w) == (height, width) and np.allclose(inverse, np.eye(3), atol=1e-12, rtol=0):
                    output[frame, view, chamber] = native
                else:
                    output[frame, view, chamber] = cv2.warpAffine(
                        native.astype(np.uint8), inverse[:2], (out_w, out_h),
                        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if clipped.any() and not allow_clipping:
        raise ValueError(f"Fixed FOV clips {clipped.sum()} native foreground pixels")
    return dict(masks=output, model_to_native=transforms,
                pixel_spacing=destination.copy(), native_pixel_spacing=spacing.copy(),
                clipped_foreground=clipped)


def mask_to_sdf(masks, pixel_spacing, *, view_present=None, annotated=None,
                clip_mm=15.0, normalize=False):
    """Float32 2D EDT, negative inside and positive outside, in millimetres.

    Empty annotated channels have +clip_mm (or +FOV diagonal without clipping),
    full channels have its negative. Unannotated channels are zero placeholders
    and must be excluded using availability. normalize=True divides by clip_mm
    and changes output units to a dimensionless [-1,1] value.
    """
    masks, spacing = _masks_and_spacing(masks, pixel_spacing)
    valid = _validity(len(masks), view_present, annotated)
    if clip_mm is not None and (not np.isfinite(clip_mm) or clip_mm <= 0):
        raise ValueError("SDF clipping distance must be positive finite mm")
    if normalize and clip_mm is None:
        raise ValueError("SDF normalization requires clip_mm")
    result = np.zeros(masks.shape, np.float32)
    for frame, view, chamber in np.argwhere(valid):
        inside = masks[frame, view, chamber]
        sampling = spacing[frame, view]
        saturation = clip_mm if clip_mm is not None else np.linalg.norm(sampling * inside.shape)
        if not inside.any():
            sdf = np.full(inside.shape, saturation, dtype=np.float64)
        elif inside.all():
            sdf = np.full(inside.shape, -saturation, dtype=np.float64)
        else:
            sdf = distance_transform_edt(~inside, sampling=sampling) - distance_transform_edt(inside, sampling=sampling)
            if clip_mm is not None:
                np.clip(sdf, -clip_mm, clip_mm, out=sdf)
        if normalize:
            sdf /= clip_mm
        result[frame, view, chamber] = sdf
    return result


def mask_to_boundary(masks, *, view_present=None, annotated=None):
    """Inner 4-connected pixel boundary; missing channels produce no contour."""
    masks = np.asarray(masks)
    masks, _ = _masks_and_spacing(masks, (1, 1))
    valid = _validity(len(masks), view_present, annotated)
    result = np.zeros(masks.shape, np.uint8)
    for frame, view, chamber in np.argwhere(valid):
        inside = masks[frame, view, chamber]
        result[frame, view, chamber] = inside & ~binary_erosion(inside, border_value=0)
    return result


class MaskTransform:
    """Picklable callable for Dataset/DataLoader; accepts measured spacing only."""
    def __init__(self, *, target_size=None, canonical_fov_mm=None, use_sdf=True,
                 sdf_clip_mm=15.0, sdf_normalize=False, allow_clipping=False):
        self.options = dict(target_size=target_size, canonical_fov_mm=canonical_fov_mm,
                            allow_clipping=allow_clipping)
        self.use_sdf = use_sdf
        self.sdf_clip_mm = sdf_clip_mm
        self.sdf_normalize = sdf_normalize

    def __call__(self, masks, pixel_spacing, *, view_present, annotated):
        result = normalize_masks(masks, pixel_spacing, view_present=view_present,
                                 annotated=annotated, **self.options)
        if self.use_sdf:
            result["sdf"] = mask_to_sdf(result["masks"], result["pixel_spacing"],
                                       view_present=view_present, annotated=annotated,
                                       clip_mm=self.sdf_clip_mm, normalize=self.sdf_normalize)
        return result
