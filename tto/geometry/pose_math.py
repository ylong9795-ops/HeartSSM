"""Rigid geometry only. Angles are degrees, translations and points are mm."""
import torch


def so3(degrees):
    x, y, z = torch.deg2rad(degrees).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    return torch.matrix_exp(skew.reshape(*degrees.shape[:-1], 3, 3))


def transform_points(points, transform):
    """Canonical-to-patient SE(3): column convention R*x + center_mm."""
    return points @ transform['R'].transpose(-1, -2) + transform['center_mm']


def compose_plane_pose(delta, base):
    """Right SO(3) composition for geometry/gradient checks; no search policy."""
    return dict(R=base['R'] @ so3(delta[..., 3:]),
                center_mm=base['center_mm'] + delta[..., :3])
