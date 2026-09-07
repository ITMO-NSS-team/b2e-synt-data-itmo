from __future__ import annotations

import json
from pathlib import Path

from sim.skill_eval.types import EvalCase


def load_cases(path: str | Path, case_ids: list[str] | None = None) -> list[EvalCase]:
    wanted = set(case_ids) if case_ids else None
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = str(raw["case_id"])
        if wanted is not None and case_id not in wanted:
            continue
        cases.append(EvalCase(
            case_id=case_id,
            category=str(raw["category"]),
            question=str(raw["question"]),
            runtime_actor_employee_id=str(raw["runtime_actor_employee_id"]),
            expected_skill=raw.get("expected_skill"),
            expected_skill_kind=raw.get("expected_skill_kind"),
            gold=raw.get("gold") or {},
            snapshot_id=raw.get("snapshot_id"),
            business_task=raw.get("business_task") or {},
            raw=raw,
        ))
    if wanted is not None:
        found = {case.case_id for case in cases}
        missing = wanted - found
        if missing:
            raise FileNotFoundError(
                f"cases not in {path}: {sorted(missing)}")
    return cases
