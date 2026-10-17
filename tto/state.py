"""Versioned patient state. No chamber offsets or frozen motion fields."""
from dataclasses import dataclass, field, fields
from enum import Enum

import torch

from .geometry.pose_math import transform_points

STATE_VERSION = 'tto-state-v1'


class Stage(str, Enum):
    INITIALIZED = 'initialized'
    GLOBAL_ALIGNED = 'global_alignment'
    CHAMBER_INITIALIZED = 'chamber_candidates'
    SHAPE = 'shape'
    MOTION = 'motion'
    JOINT = 'joint'


@dataclass
class TTOState:
    patient_id: str
    identity: dict
    z_shape: torch.Tensor  # [1,64]
    z_motion: torch.Tensor  # [1,32]
    conditions: dict
    vertices_reference: torch.Tensor  # [N,3], canonical physical mm
    trajectory: torch.Tensor  # [24,N,3], original decoder output
    global_transform: dict  # single shared canonical-to-patient SE(3)
    observation_pose: dict | None = None  # three planes in patient coordinates
    stage: Stage = Stage.INITIALIZED
    diagnostics: dict = field(default_factory=dict)

    def validate(self):
        if not self.patient_id or self.identity.get('patient_id') != self.patient_id:
            raise ValueError('State identity/patient mismatch')
        if self.z_shape.shape != (1, 64) or self.z_motion.shape != (1, 32):
            raise ValueError('Original Shape64 / Motion32 latents required')
        if self.vertices_reference.ndim != 2 or self.vertices_reference.shape[-1] != 3:
            raise ValueError('Reference must be [N,3]')
        if self.trajectory.shape != (24, *self.vertices_reference.shape):
            raise ValueError('A full 24-phase fixed-topology trajectory is required')
        for value in (self.z_shape, self.z_motion, self.vertices_reference, self.trajectory):
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite TTO state')
        if not torch.equal(self.vertices_reference, self.trajectory[0]):
            raise ValueError('Reference differs from original decoder phase zero')
        self._validate_pose(self.global_transform, ())
        if self.observation_pose is not None:
            self._validate_pose(self.observation_pose, (3,))
        if self.diagnostics.get('pose_gauge') == 'canonical_identity_global':
            if not torch.equal(self.global_transform['R'], torch.eye(3, device=self.global_transform['R'].device)) or not torch.equal(
                    self.global_transform['center_mm'], torch.zeros(3, device=self.global_transform['center_mm'].device)):
                raise ValueError('S1 canonical pose gauge requires identity global transform')
        candidate=self.diagnostics.get('s2_candidate',{})
        if candidate.get('partition_lock_version') == 'regional-only-v1':
            frozen=candidate['frozen_parameters']
            if not torch.equal(self.z_shape[:,:32],torch.as_tensor(frozen['global32']).to(self.z_shape)):
                raise ValueError('S2 Global32 must remain frozen')
            if not torch.equal(self.z_motion,torch.as_tensor(frozen['motion32']).to(self.z_motion)):
                raise ValueError('S2 Motion32 must remain frozen; regenerated motion may change')
            if self.observation_pose is None or any(not torch.equal(value,torch.as_tensor(frozen['pose'][key]).to(value))
                                                  for key,value in self.observation_pose.items()):
                raise ValueError('S2 must use the same shared S1 observation poses')
            if candidate.get('temporary_offsets_used_in_final_mesh') is not False:
                raise ValueError('S2 chamber offsets are initialization hints only')
        self.stage = Stage(self.stage)

    @staticmethod
    def _validate_pose(pose, prefix):
        if set(pose) != {'R', 'center_mm'}:
            raise ValueError('Rigid pose requires only R and center_mm')
        r, t = pose['R'], pose['center_mm']
        if r.shape != (*prefix, 3, 3) or t.shape != (*prefix, 3):
            raise ValueError('Unexpected rigid pose dimensions')
        if not torch.isfinite(r).all() or not torch.isfinite(t).all():
            raise ValueError('Nonfinite rigid pose')
        eye = torch.eye(3, device=r.device, dtype=r.dtype).expand_as(r)
        if not torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-5, rtol=1e-5) or not torch.allclose(
                torch.linalg.det(r), r.new_ones(prefix), atol=1e-5, rtol=1e-5):
            raise ValueError('Pose must be a proper SO(3) rotation')

    def patient_trajectory(self):
        return transform_points(self.trajectory, self.global_transform)

    def to_payload(self):
        self.validate()
        # Storage performs the detach/copy; dataclasses.asdict cannot deepcopy
        # non-leaf decoded tensors when an optimizer eventually supplies them.
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        payload.update(version=STATE_VERSION, stage=self.stage.value)
        return payload

    @classmethod
    def from_payload(cls, payload):
        value = dict(payload)
        if value.pop('version', None) != STATE_VERSION:
            raise ValueError('Unsupported TTO state; legacy state migration is not implicit')
        try:
            result = cls(**value)
        except TypeError as error:
            raise ValueError('Unexpected or missing fields in TTO state') from error
        result.validate()
        return result
