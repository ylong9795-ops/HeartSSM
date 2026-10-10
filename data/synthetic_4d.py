"""Lazy EchoGSF three-view observations and saved fixed-topology 4D truth.

Ported/adapted from shared/data/{source_data,synthetic}.py, generation/utils.py
and s1_shape/checkpoint.py, EchoGSF commit 157aa7624cfda8e7319ec325af5d1088065f026d.
No EchoGSF imports, generation, marching cubes, coefficient or network loading.
Coordinates are centered physical mm (vertices * scale_mm), as in its reader;
add asset center_mm only when explicitly exporting to original SSM world space.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .mask_transform import CHAMBERS, VIEWS, MaskTransform, _validity

# Generator visibility is annotation policy, never inferred from foreground.
VISIBLE = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], bool)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinate_hash(vertices):
    return hashlib.sha256(np.asarray(vertices, dtype="<f4").tobytes()).hexdigest()


def _read_bytes(path, expected, verify):
    data = Path(path).read_bytes()
    if verify and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Modified source file: " + str(path))
    return data


def _resolve(value, parent, path_map):
    """Explicit archive relocation only; never guess by filename or glob order."""
    path = Path(value)
    for old, new in sorted(path_map.items(), key=lambda pair: -len(str(pair[0]))):
        try:
            return Path(os.path.abspath(Path(new) / path.relative_to(old)))
        except ValueError:
            pass
    return Path(os.path.abspath(parent / path))


def _source_rows(path, asset_hash, path_map, verify, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Cyclic source manifest")
    seen.add(path)
    manifest = json.loads(path.read_text())
    source_digest = _sha256(path)
    if not manifest.get("complete") or manifest.get("retired", False):
        raise ValueError("Incomplete or retired source manifest")
    kind = manifest.get("format")
    if kind == "wholeheart-ed-observation-repair-v1":
        raise ValueError("ED-only repairs cannot certify full-cycle observations")
    if kind == "wholeheart-sharded-training-v1":
        rows = []
        for shard in manifest["samples"]:
            child = _resolve(shard["path"], path.parent, path_map) / "manifest.json"
            if verify and _sha256(child) != shard["manifest_sha256"]:
                raise ValueError("Modified source shard")
            rows.extend(_source_rows(child, asset_hash, path_map, verify, seen))
    elif kind in ("wholeheart-simple-training-v1", "wholeheart-simple-training-v2"):
        if manifest["asset_sha256"] != asset_hash:
            raise ValueError("Source asset identity mismatch")
        if verify and _sha256(path.parent / "plan.json") != manifest["plan_sha256"]:
            raise ValueError("Modified source generation plan")
        rows = []
        for index, source in enumerate(manifest["samples"]):
            row = dict(source)
            for field in ("sha256", "qc_sha256", "coordinate_sha256", "candidate_id"):
                if field not in row:
                    raise ValueError("Missing source identity: " + field)
            row.update(path=str(_resolve(row["path"], path.parent, path_map)),
                       qc_path=str(_resolve(row["qc_path"], path.parent, path_map)),
                       asset_sha256=asset_hash, source_manifest_sha256=source_digest,
                       source_index=index)
            rows.append(row)
        if manifest.get("accepted_count", len(rows)) != len(rows):
            raise ValueError("Manifest membership count mismatch")
    else:
        raise ValueError("Unsupported source manifest: " + str(kind))
    seen.remove(path)
    return rows


def _load_template(path, asset_hash, scale_mm, vertex_count, expected_sha256):
    """Extract shared geometry buffers only, using the existing topology digest."""
    identity = _sha256(path)
    if expected_sha256 is not None and identity != expected_sha256:
        raise ValueError("Modified template artifact")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state, specification = saved["model"], saved["specification"]
    vertices = state["template_vertices"].clone()
    ranges = state["vertex_ranges"].clone()
    if (specification["asset_sha256"] != asset_hash or
            specification["norm_to_mm"] != scale_mm or
            vertices.shape != (vertex_count, 3) or not torch.isfinite(vertices).all() or
            ranges.shape != (4, 2) or ranges.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Template asset/scale/vertex contract mismatch")
    bounds = ranges.tolist()
    if (bounds[0][0] != 0 or bounds[-1][1] != vertex_count or
            any(a >= b for a, b in bounds) or
            any(bounds[i][1] != bounds[i + 1][0] for i in range(3))):
        raise ValueError("Invalid contiguous chamber ranges")
    faces = {}
    for chamber, (left, right) in zip(CHAMBERS, bounds):
        value = state["faces." + chamber].clone()
        if (value.ndim != 2 or value.shape[1] != 3 or value.numel() == 0 or
                value.dtype not in (torch.int32, torch.int64) or
                value.min() < 0 or value.max() >= right - left):
            raise ValueError("Invalid fixed template faces")
        faces[chamber] = value
    global_faces = torch.cat([faces[c].long() + a for c, (a, b) in zip(CHAMBERS, bounds)])
    digest = hashlib.sha256()
    for value in (vertices, global_faces, ranges):
        digest.update(value.numpy().tobytes())
    if digest.hexdigest() != specification["template_sha256"]:
        raise ValueError("Template topology identity mismatch")
    return dict(vertices=vertices, faces=faces, chamber_ranges=ranges,
                template_sha256=digest.hexdigest(), artifact_sha256=identity)


class Synthetic4DDataset(Dataset):
    """One lazy case per item; default_collate works with fixed source shapes.

    Inputs: masks, optional sdf, phases (relative observed cycle time),
    view_present and annotated. GT/geometry fields are supervision/provenance,
    never encoder inputs. phases_gt retains synthetic reference phase separately.
    Static validity remains [3]/[3,4]; per-frame validity remains [T,3]/[T,3,4].
    reference_frame selects a single validated frame with T=1 for Shape training;
    frame zero is the source reference, not necessarily every chamber's volume ED.
    verify=False skips file/coordinate digests, never shape, closure or QC checks.
    """
    def __init__(self, manifest, asset, mask_transform=None, verify=True, *,
                 path_map=None, expected_phases=24, remove_duplicate_endpoint=True,
                 reference_frame=None, template=None, template_file_sha256=None):
        self.manifest = Path(manifest).resolve()
        self.manifest_sha256 = _sha256(self.manifest)
        self.asset_path = Path(asset).resolve()
        self.asset_sha256 = _sha256(self.asset_path)
        self.asset = json.loads(self.asset_path.read_text())
        if self.asset.get("format") != "wholeheart-original-ssm-v1":
            raise ValueError("Unsupported physical asset")
        if self.asset.get("temporal_resampling") != "explicit_linear_closure_v1":
            raise ValueError("Unknown source phase contract; explicit adaptation required")
        self.norm_to_mm = float(self.asset["scale_mm"])
        self.vertex_count = self.asset["vertices"]
        if (not np.isfinite(self.norm_to_mm) or self.norm_to_mm <= 0 or
                type(self.vertex_count) is not int or self.vertex_count < 1 or
                np.asarray(self.asset["center_mm"]).shape != (3,) or
                not np.isfinite(self.asset["center_mm"]).all()):
            raise ValueError("Invalid physical asset scale/count/center")
        self.verify = bool(verify)
        self.sources = _source_rows(self.manifest, self.asset_sha256, path_map or {}, self.verify)
        if not self.sources:
            raise ValueError("Empty source membership")
        for field in ("path", "sha256", "coordinate_sha256"):
            if len({row[field] for row in self.sources}) != len(self.sources):
                raise ValueError("Duplicate source membership: " + field)
        case_ids = [row["case_id"] for row in self.sources if "case_id" in row]
        if any(not isinstance(value, str) or not value for value in case_ids) or len(set(case_ids)) != len(case_ids):
            raise ValueError("Invalid or duplicate original case IDs")
        self.mask_transform = MaskTransform(use_sdf=False) if mask_transform is None else mask_transform
        self.remove_duplicate_endpoint = bool(remove_duplicate_endpoint)
        self.expected_phases = expected_phases
        if expected_phases != (24 if self.remove_duplicate_endpoint else 25):
            raise ValueError("Verified native asset stores 25 closed / 24 independent phases")
        if reference_frame is not None and (type(reference_frame) is not int or not 0 <= reference_frame < expected_phases):
            raise ValueError("Reference frame outside independent cycle")
        self.reference_frame = reference_frame
        self.shared_geometry = None if template is None else _load_template(
            template, self.asset_sha256, self.norm_to_mm, self.vertex_count, template_file_sha256)
        self.template_sha256 = "" if self.shared_geometry is None else self.shared_geometry["template_sha256"]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        source = self.sources[index]
        try:
            return self._read_case(source)
        except (ValueError, KeyError, OSError) as error:
            raise ValueError(f"Case {source['candidate_id']} ({source['path']}): {error}") from error

    def _read_case(self, source):
        qc = json.loads(_read_bytes(source["qc_path"], source["qc_sha256"], self.verify))
        if qc["status"] != "accept" or qc["metrics"]["coordinate_sha256"] != source["coordinate_sha256"]:
            raise ValueError("Unaccepted source geometry or QC coordinate mismatch")
        with np.load(io.BytesIO(_read_bytes(source["path"], source["sha256"], self.verify)), allow_pickle=False) as z:
            if str(z["asset_sha256"]) != self.asset_sha256:
                raise ValueError("Sample asset identity mismatch")
            masks, vertices = z["masks"], z["vertices"]
            poses = {key: z[key] for key in ("probe_R", "probe_t", "probe_s")}
            present = z["view_present"] if "view_present" in z else np.ones(3, bool)
            views = tuple(z["view_names"].tolist()) if "view_names" in z else VIEWS
            chambers = tuple(z["chamber_names"].tolist()) if "chamber_names" in z else CHAMBERS
            if len(views) != 3 or set(views) != set(VIEWS) or len(chambers) != 4 or set(chambers) != set(CHAMBERS):
                raise ValueError("Unknown view/chamber order")
            annotated = z["channel_annotated"] if "channel_annotated" in z else VISIBLE[
                [VIEWS.index(v) for v in views]][:, [CHAMBERS.index(c) for c in chambers]]
        if (masks.ndim != 5 or masks.shape[:3] != (25, 3, 4) or min(masks.shape[-2:]) < 1 or
                not ((masks == 0) | (masks == 1) | (masks == 255)).all()):
            raise ValueError("Native mask shape/binary contract mismatch")
        if (vertices.shape != (25, self.vertex_count, 3) or vertices.dtype != np.float32 or
                not np.isfinite(vertices).all()):
            raise ValueError("GT vertex count/dtype/finite contract mismatch")
        if self.verify and _coordinate_hash(vertices) != source["coordinate_sha256"]:
            raise ValueError("GT coordinate hash mismatch")
        shapes = dict(probe_R=(25, 3, 3, 3), probe_t=(25, 3, 3), probe_s=(25, 3, 3))
        if any(value.shape != shapes[key] or not np.isfinite(value).all() for key, value in poses.items()):
            raise ValueError("Pose/time/finite contract mismatch")
        rotations = poses["probe_R"]
        if (np.any(poses["probe_s"] <= 0) or np.any(np.linalg.det(rotations) <= 0) or
                not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-4, rtol=0)):
            raise ValueError("Rotation/physical FOV contract mismatch")
        _validity(25, present, annotated)
        for name, value in dict(masks=masks, vertices=vertices, **poses).items():
            if not np.array_equal(value[0], value[-1]):
                raise ValueError("Duplicate endpoint mismatch: " + name)
        for value, dimensions in ((present, 2), (annotated, 3)):
            if value.ndim == dimensions and not np.array_equal(value[0], value[-1]):
                raise ValueError("Availability endpoint mismatch")
        view_index = [views.index(v) for v in VIEWS]
        chamber_index = [chambers.index(c) for c in CHAMBERS]
        if view_index != [0, 1, 2]:
            masks = masks[:, view_index]
            poses = {key: value[:, view_index] for key, value in poses.items()}
        if chamber_index != [0, 1, 2, 3]:
            masks = masks[:, :, chamber_index]
        present = np.take(present, view_index, axis=-1).astype(bool)
        annotated = np.take(np.take(annotated, view_index, axis=-2), chamber_index, axis=-1).astype(bool)
        frames = np.arange(self.expected_phases)
        selection = slice(0, self.expected_phases)
        if self.reference_frame is not None:
            frames = frames[self.reference_frame:self.reference_frame + 1]
            selection = slice(self.reference_frame, self.reference_frame + 1)
        masks, vertices = masks[selection], vertices[selection]
        poses = {key: value[selection] for key, value in poses.items()}
        if present.ndim == 2:
            present = present[selection]
        if annotated.ndim == 3:
            annotated = annotated[selection]
        valid = _validity(len(frames), present, annotated)
        native = ((masks > 0) & valid[..., None, None]).astype(np.uint8)
        spacing = 2 * poses["probe_s"][..., [1, 0]].astype(np.float64) * self.norm_to_mm / masks.shape[-2:]
        transformed = self.mask_transform(native, spacing, view_present=present, annotated=annotated)
        result = {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in transformed.items()}
        physical = vertices.astype(np.float64) * self.norm_to_mm
        if not np.isfinite(physical).all():
            raise ValueError("Nonfinite physical GT")
        result.update(native_masks=torch.from_numpy(native), vertices_gt=torch.from_numpy(physical),
                      pose_gt={key: torch.from_numpy(value.copy()) for key, value in poses.items()},
                      phases=torch.from_numpy(frames.astype(np.float64) / 24),
                      phases_gt=torch.from_numpy(frames.astype(np.float64) / 24),
                      phase_semantics="relative_observed_cycle_time", frame_indices=torch.from_numpy(frames),
                      view_present=torch.from_numpy(present.copy()), annotated=torch.from_numpy(annotated.copy()),
                      channel_annotated=torch.from_numpy(annotated.copy()),
                      case_id=source.get("case_id", f"{self.asset_sha256}:{source['candidate_id']}:{source['coordinate_sha256']}"),
                      candidate_id=source["candidate_id"], sample_sha256=source["sha256"],
                      coordinate_sha256=source["coordinate_sha256"], asset_sha256=self.asset_sha256,
                      source_path=source["path"], qc_sha256=source["qc_sha256"],
                      template_sha256=self.template_sha256, manifest_sha256=self.manifest_sha256,
                      source_manifest_sha256=source["source_manifest_sha256"], source_index=source["source_index"],
                      norm_to_mm=self.norm_to_mm, center_mm=torch.tensor(self.asset["center_mm"], dtype=torch.float64),
                      sdf_units="dimensionless" if getattr(self.mask_transform, "sdf_normalize", False) else "mm")
        return result
