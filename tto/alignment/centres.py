"""Binary-cell measurements and chamber-balanced Huber centres; no gradients."""
import numpy as np


def centroid(mask):
    y, x = np.nonzero(mask)
    return np.array([x.mean(), y.mean()]) if len(x) else None


def measure(predicted, observed, spacing_yx):
    predicted, observed = np.asarray(predicted, bool), np.asarray(observed, bool)
    p, o = centroid(predicted), centroid(observed)
    pa, oa = int(predicted.sum()), int(observed.sum())
    spacing = np.asarray(spacing_yx, float)
    error = (p-o)*spacing[::-1] if p is not None and o is not None else None
    return dict(prediction_centroid_xy_px=p, observation_centroid_xy_px=o,
        centroid_error_xy_mm=error, centroid_error_norm_mm=float(np.linalg.norm(error)) if error is not None else None,
        prediction_area_mm2=float(pa*np.prod(spacing)), observation_area_mm2=float(oa*np.prod(spacing)),
        area_ratio=float(pa/oa) if oa else None,
        exact_hard_dice=float(2*(predicted & observed).sum()/(pa+oa)) if pa+oa else 1.,
        centroid_status='VALID' if error is not None else 'BOTH_EMPTY' if not pa and not oa else
            'PREDICTION_EMPTY' if not pa else 'OBSERVATION_EMPTY')


def huber_location(values, weights, delta_mm=3.):
    """Minimise sum w*rho(||x-d||) by deterministic IRLS."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 2 or values.shape[1] != 2 or weights.shape != (len(values),)
            or not len(values) or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights <= 0).any() or delta_mm <= 0):
        raise ValueError('Positive finite weighted 2D centre evidence required')
    d = np.median(values, axis=0)
    for _ in range(100):
        radius = np.linalg.norm(values-d, axis=1)
        effective = weights*np.minimum(1., delta_mm/np.maximum(radius, 1e-12))
        updated = np.average(values, axis=0, weights=effective)
        if np.linalg.norm(updated-d) < 1e-8: return updated
        d = updated
    return d


def balanced_correction(rows, delta_mm=3.):
    """Time aggregation per chamber, then equal-weight chamber aggregation."""
    chambers = []
    for c in sorted({r['chamber'] for r in rows}):
        valid = [r for r in rows if r['chamber'] == c and r.get('centroid_error_xy_mm') is not None
                 and r['fit_weight'] > 0 and not r.get('renderer_error')]
        if valid:
            d = huber_location([r['centroid_error_xy_mm'] for r in valid],
                               [r['fit_weight'] for r in valid], delta_mm)
            chambers.append(dict(chamber=c, cells=len(valid), correction_xy_mm=d))
    correction = (huber_location([r['correction_xy_mm'] for r in chambers], np.ones(len(chambers)), delta_mm)
                  if chambers else None)
    return dict(correction_xy_mm=correction, per_chamber=chambers,
        status='ESTIMATED' if chambers else 'NO_NONEMPTY_PREDICTED_CENTRES',
        weighting='source-weighted time Huber per chamber, then equal chamber Huber; no area weights')
