"""Preserve every objective clause for subsequent evidence-based final review.

This is a coverage inventory, not a verifier of scientific completion. Source
text and artifact existence never promote a clause to a verified status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def inventory(source: Path) -> dict:
    raw = source.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    sections, context, structure = [], [], []
    current = None
    accounted = []
    for number, text in enumerate(lines, 1):
        stripped = text.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            current = {"id": f"S{len(sections) + 1:02d}", "title": text,
                       "source_line": number, "clauses": []}
            sections.append(current)
            structure.append({"source_line": number, "text": text})
        elif stripped.startswith("#") or re.fullmatch(r"[-*_]{3,}", stripped):
            structure.append({"source_line": number, "text": text})
        else:
            if re.match(r"\d+\.\s", stripped):
                kind = "numbered_item"
            elif stripped.startswith("- "):
                kind = "bullet"
            elif stripped.startswith("|"):
                kind = "table_line"
            else:
                kind = "text_or_formula"
            entry = {"id": f"{current['id'] if current else 'CONTEXT'}-L{number:04d}",
                     "source_line": number, "text": text, "kind": kind,
                     "status": "UNREVIEWED", "evidence": [],
                     "finding": "Pending clause-specific inspection; inventory coverage is not acceptance evidence."}
            (current["clauses"] if current else context).append(entry)
        accounted.append(number)
    expected = [i for i, line in enumerate(lines, 1) if line.strip()]
    if accounted != expected or len(set(accounted)) != len(accounted):
        raise ValueError("Objective content coverage is incomplete or duplicated")
    if len(sections) != 17:
        raise ValueError(f"Expected all 17 objective sections, found {len(sections)}")
    count = sum(len(section["clauses"]) for section in sections) + len(context)
    return {"updated": time.time(), "objective_file": str(source.resolve()),
            "objective_sha256": hashlib.sha256(raw).hexdigest(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "purpose": "Lossless 17-section clause inventory for the final evidence audit; no completion inference.",
            "objective_complete": False,
            "coverage": {"sections": len(sections), "clauses_and_context": count,
                         "nonempty_source_lines": len(expected),
                         "all_nonempty_lines_accounted_once": True,
                         "verified_clauses": 0, "unreviewed_clauses": count},
            "document_context": context, "structure": structure, "sections": sections,
            "review_rules": [
                "Inspect evidence for each clause at its actual scope; existence or a green summary is insufficient.",
                "Record incomplete, contradicted, missing and indirectly supported requirements explicitly.",
                "Keep factual scientific failure distinct from missing execution or missing evidence.",
                "No completion claim until every requirement, named artifact, command, test, gate and invariant is reviewed.",
                "This generated inventory intentionally marks all clauses UNREVIEWED; do not treat it as a completed audit."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.objective)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "objective_clause_inventory.json"
    if target.exists():
        previous = json.loads(target.read_text())
        reviewed = [item for section in previous["sections"] for item in section["clauses"]
                    if item["status"] != "UNREVIEWED"]
        if reviewed:
            raise ValueError("Refusing to overwrite reviewed clauses; preserve the completed review separately")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    report = ["# 原目标逐条验收库存（待审核）", "",
              "此文件只核对原文覆盖，所有条款仍待逐条检查，不能作为实验或目标已完成的证据。", "",
              f"原文 SHA256：`{result['objective_sha256']}`。",
              f"覆盖全部 17 章，{result['coverage']['clauses_and_context']} 个内容条目；"
              "原文所有非空行均恰好登记一次。", ""]
    for section in result["sections"]:
        report.extend([section["title"], ""])
        for item in section["clauses"]:
            report.extend([f"- [ ] `{item['id']}`（原文第 {item['source_line']} 行）", "",
                           "    " + item["text"], ""])
    (args.output / "OBJECTIVE_CLAUSE_INVENTORY.md").write_text("\n".join(report))
    archive = args.output / "audit_code" / f"{result['generator_sha256']}.py"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result["coverage"]))


if __name__ == "__main__":
    main()
