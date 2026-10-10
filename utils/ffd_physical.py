"""Batched cardinal cubic B-spline FFD in physical millimetres.

Basis: HeartSSM utils.ffd.bspline_basis. Control-to-mesh interpretation checked
against Kong et al., HeartFFDNet src/custom_layers.py and make_control_grid.py.
Unlike Kong's clamped-knot matrix, this evaluates a uniform cardinal lattice
at the CURRENT vertices (needed for HeartSSM's sequential deformations).
"""
import torch

from utils.ffd import bspline_basis


def control_geometry(lower_mm, upper_mm, grid_shape):
    """Return actual control-point origin/spacing for a closed supported box.

    Control point i is at origin + i*spacing. Four supporting nodes are
    floor(q)-1 ... floor(q)+2. One guard node on each side gives the complete
    domain q in [1, G-2]; at its upper endpoint use the left-hand cell u=1.
    """
    lower = torch.as_tensor(lower_mm)
    if not lower.is_floating_point():
        lower = lower.float()
    upper = torch.as_tensor(upper_mm, dtype=lower.dtype, device=lower.device)
    shape = torch.as_tensor(grid_shape, device=lower.device)
    if (lower.shape != (3,) or upper.shape != (3,) or shape.shape != (3,)
            or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
            or (upper <= lower).any() or (shape < 4).any()):
        raise ValueError('Finite increasing physical bounds and grid dimensions >=4 required')
    spacing = (upper - lower) / (shape - 3)
    return lower - spacing, spacing


def bspline_ffd_physical(vertices_mm, control_grid_mm, origin_mm, spacing_mm):
    """[B,N,3], [B,Gx,Gy,Gz,3], [3], [3] -> displacement [B,N,3].

    Invalid support is an error, never extrapolated or silently clamped.
    Gradients flow through both vertices and the per-patient control fields.
    The boundary cell choice is solely a representation of the closed upper
    endpoint, not an out-of-domain fallback.
    """
    v, c = vertices_mm, control_grid_mm
    if (v.ndim != 3 or v.shape[-1] != 3 or c.ndim != 5 or c.shape[-1] != 3
            or v.shape[0] != c.shape[0] or min(c.shape[1:4]) < 4
            or not v.is_floating_point() or v.dtype != c.dtype or v.device != c.device):
        raise ValueError('Matching floating vertices [B,N,3] and controls [B,Gx,Gy,Gz,3] required')
    origin = torch.as_tensor(origin_mm, device=v.device, dtype=v.dtype)
    spacing = torch.as_tensor(spacing_mm, device=v.device, dtype=v.dtype)
    if (origin.shape != (3,) or spacing.shape != (3,) or (spacing <= 0).any()
            or not all(torch.isfinite(x).all() for x in (v, c, origin, spacing))):
        raise ValueError('FFD requires finite values and positive physical spacing')
    shape = torch.tensor(c.shape[1:4], device=v.device)
    q = (v - origin) / spacing
    # Check the physical bounds first: FP rounding can move a boundary query
    # a few ulps. Only roundoff within this explicitly checked box is removed.
    lower, upper = origin + spacing, origin + (shape - 2) * spacing
    tol = 8 * torch.finfo(v.dtype).eps * torch.maximum(lower.abs(), upper.abs()).clamp_min(1)
    if ((v < lower - tol) | (v > upper + tol)).any():
        raise ValueError('Vertices outside cubic B-spline support domain (mm)')
    q = torch.minimum(torch.maximum(q, torch.ones_like(q)), (shape - 2).to(q))
    cell = torch.minimum(torch.floor(q).long(), shape - 3)
    u = q - cell.to(q)
    basis = bspline_basis(u)  # [B,N,3,4]
    shifts = torch.cartesian_prod(*[torch.arange(4, device=v.device)] * 3)
    indices = cell[..., None, :] - 1 + shifts
    flat_indices = (indices[..., 0] * c.shape[2] + indices[..., 1]) * c.shape[3] + indices[..., 2]
    points = torch.gather(c.reshape(c.shape[0], -1, 3), 1,
                          flat_indices.reshape(c.shape[0], -1, 1).expand(-1, -1, 3))
    points = points.reshape(v.shape[0], v.shape[1], 64, 3)
    weights = (basis[..., 0, shifts[:, 0]] * basis[..., 1, shifts[:, 1]]
               * basis[..., 2, shifts[:, 2]])
    return (weights[..., None] * points).sum(-2)
