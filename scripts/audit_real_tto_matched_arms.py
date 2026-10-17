"""Audit committed ablation pairs by cell identity, initialization and budget.

Read-only with respect to scientific outputs. This does not rerender meshes,
recompute geometry, certify intermediate Adam rollback, or infer 3D accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tto.io import digest_json, load_state, read_json, sha256, write_json

ARMS = ("frozen_prior", "pose_only", "shape_only", "alternate", "original_motion", "ed_only")
VIEWS = ("A2C", "A3C", "A4C")
CHAMBERS = ("LV", "LA", "RV", "RA")
CELL_METADATA = ("patient", "native_time_s", "phase", "phase_upper", "phase_alpha",
                 "label_source", "source_weight", "target_area_mm2", "confidence",
                 "phase_confidence", "used_for_optimization", "supervision_reason")
INITIAL_ARRAYS = ("z_shape_initial", "z_motion_initial", "vertices_reference_initial",
                  "vertices_4d_initial", "frozen_fields", "probe_R_initial",
                  "probe_t_initial_mm", "faces_LV", "faces_LA", "faces_RV", "faces_RA",
                  "chamber_ranges", "phases")


def cells(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["native_frame"], row["view"], row["chamber"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate native observation identity in {path}")
    return result


def metadata_differences(first: dict, second: dict) -> list:
    errors = []
    if first.keys() != second.keys():
        errors.append("Native frame/view/chamber identity sets differ")
    for key in first.keys() & second.keys():
        changed = [field for field in CELL_METADATA if first[key][field] != second[key][field]]
        if changed:
            errors.append(f"Observation metadata differs at {key}: {changed}")
    return errors


def same_array(first, second) -> bool:
    return (first.dtype == second.dtype and first.shape == second.shape
            and first.tobytes() == second.tobytes())


def normalized_science(protocol: dict) -> dict:
    result = deepcopy(protocol["scientific_config"])
    result["experiment"].pop("arm", None)
    result["experiment"].pop("initialization_source", None)
    result["synthetic"].pop("validation_gate", None)
    return result


def manifest_keys(state: dict) -> set:
    return {(str(row["native_frame"]), VIEWS[row["view"]], CHAMBERS[row["chamber"]])
            for row in state["result"]["supervision_manifest"] if row["used"]}


def pair(main: Path, other: Path, arm: str, main_protocol: dict) -> dict:
    errors = []
    protocol = read_json(other.parents[1] / "protocol.json")
    for field in ("source_sha256", "validation_sha256", "seed"):
        if protocol[field] != main_protocol[field]:
            errors.append(f"Protocol {field} differs")
    if normalized_science(protocol) != normalized_science(main_protocol):
        errors.append("Scientific settings differ beyond the declared arm/import/gate fields")
    saved_main = load_state(main / "states/export.pth")
    saved_other = load_state(other / "states/export.pth")
    for folder, saved, own_protocol in ((main, saved_main, main_protocol), (other, saved_other, protocol)):
        if read_json(folder / "export/completed.json")["identity"] != saved["identity"]:
            errors.append(f"Completion identity differs: {folder}")
        if saved["identity"]["protocol_sha256"] != digest_json(own_protocol):
            errors.append(f"State protocol identity differs: {folder}")
    for field in ("patient", "input_sha256", "checkpoint_sha256", "calibration_sha256"):
        if saved_main["identity"][field] != saved_other["identity"][field]:
            errors.append(f"State {field} differs")
    state_main, state_other = saved_main["state"], saved_other["state"]
    if state_main["result"]["status"] != "COMPLETE" or state_other["result"]["status"] != "COMPLETE":
        errors.append("Pair includes an endpoint without COMPLETE status")
    checked_arrays = list(INITIAL_ARRAYS)
    if arm != "frozen_prior":
        checked_arrays += ["probe_R_joint_initial", "probe_t_joint_initial_mm"]
    with np.load(main / "export/reconstruction.npz", allow_pickle=False) as left:
        with np.load(other / "export/reconstruction.npz", allow_pickle=False) as right:
            for field in checked_arrays:
                if not same_array(left[field], right[field]):
                    errors.append(f"Initialization/topology array bytes differ: {field}")
            if arm == "frozen_prior":
                for field, initial in (("probe_R_joint_initial", "probe_R_initial"),
                                       ("probe_t_joint_initial_mm", "probe_t_initial_mm")):
                    if not same_array(right[field], right[initial]):
                        errors.append(f"Frozen Prior unexpectedly changed its initial Pose: {field}")
    main_rows, other_rows = cells(main / "evaluation/final_cells.csv"), cells(other / "evaluation/final_cells.csv")
    errors.extend(metadata_differences(main_rows, other_rows))
    main_manifest, other_manifest = read_json(main / "input_manifest.json"), read_json(other / "input_manifest.json")
    if main_manifest != other_manifest:
        errors.append("Input/native-time/physical-coordinate manifests differ")
    expected = manifest_keys(state_main)
    if arm == "ed_only":
        frames = {VIEWS[i]: str(mapping["nearest_native_frame"])
                  for i, mapping in enumerate(main_manifest["mapping"][0])}
        expected = {key for key in expected if key[0] == frames[key[1]]}
    actual = manifest_keys(state_other)
    if actual != expected:
        errors.append("Joint loss membership differs from the declared full-cycle or reference-observation arm")
    budget = main_protocol["scientific_config"]["joint"]["updates"]
    history = state_other["result"]["history"]
    if len(history) != (0 if arm == "frozen_prior" else budget):
        errors.append("Actual attempted-update budget differs")
    if len(state_main["result"]["history"]) != budget:
        errors.append("Main attempted-update budget differs")
    for index, row in enumerate(history):
        groups = (["pose"] if arm == "pose_only" or (arm == "alternate" and index % 2 == 0)
                  else ["shape"] if arm == "shape_only" or (arm == "alternate" and index % 2 == 1)
                  else ["shape", "pose"])
        if row["parameter_groups_updated"] != groups:
            errors.append(f"Declared parameter groups differ at proposal {index + 1}")
        if row["supervision_cells"] != len(actual):
            errors.append(f"Actual loss coverage differs at proposal {index + 1}")
    mode = "original_decoder" if arm == "original_motion" else "frozen_field"
    if state_other["result"]["motion_mode"] != mode:
        errors.append("Motion forward differs from the arm")
    return {"patient": main.name, "arm": arm, "status": "VERIFIED_MATCHED" if not errors else "MISMATCH",
            "passed": not errors, "errors": errors, "evaluation_cells": len(main_rows),
            "main_loss_cells": len(manifest_keys(state_main)), "comparator_loss_cells": len(actual),
            "attempted_updates": len(history),
            "parameter_group_proposals": dict(Counter(group for row in history for group in row["parameter_groups_updated"])),
            "identical_initial_arrays": checked_arrays,
            "main_export_sha256": sha256(main / "states/export.pth"),
            "comparator_export_sha256": sha256(other / "states/export.pth")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    protocol = read_json(root / "protocol.json")
    records = []
    for arm in ARMS:
        for patient in protocol["scientific_config"]["experiment"]["cohort"]:
            main_folder = root / "patients" / patient
            other_folder = root / "ablations" / arm / "patients" / patient
            both_committed = all((folder / "export/completed.json").exists()
                                 and (folder / "states/export.pth").exists()
                                 for folder in (main_folder, other_folder))
            if both_committed:
                try:
                    records.append(pair(main_folder, other_folder, arm, protocol))
                except Exception as error:
                    records.append(dict(patient=patient, arm=arm, status="AUDIT_ERROR", passed=False, error=repr(error)))
            else:
                failures = {label: read_json(folder / "failure.json") for label, folder in
                            (("main", main_folder), ("comparator", other_folder)) if (folder / "failure.json").exists()}
                terminal = all((folder / "failure.json").exists() or
                               ((folder / "export/completed.json").exists() and (folder / "states/export.pth").exists())
                               for folder in (main_folder, other_folder))
                status = "UNPAIRED_FAILURE" if failures and terminal else "UNPAIRED_PENDING" if failures else "PENDING"
                records.append(dict(patient=patient, arm=arm, status=status,
                                    passed=None, failures=failures))
    generator_sha = sha256(Path(__file__))
    report = dict(updated=time.time(), generator_sha256=generator_sha, pairs=records,
                  statuses=dict(Counter(row["status"] for row in records)),
                  complete=all(row["status"] in ("VERIFIED_MATCHED", "UNPAIRED_FAILURE") for row in records),
                  all_available_pairs_matched=all(row["passed"] is not False for row in records),
                  objective_complete=False,
                  scope="Actual committed pairs only. Exact cell identities/metadata and initial array bytes, protocol and attempted parameter-group budgets; no rerender, geometry recomputation, intermediate-Adam certification or real 3D/Pose accuracy claim. Unpaired failures stay in the cohort denominator.")
    audit = root / "independent_audit"
    write_json(audit / "matched_arm_identity_audit.json", report)
    archive = audit / "audit_code" / f"{generator_sha}.py"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: report[key] for key in ("statuses", "complete", "all_available_pairs_matched")}))
    raise SystemExit(1 if any(row["passed"] is False for row in records) else 0)


if __name__ == "__main__":
    main()
