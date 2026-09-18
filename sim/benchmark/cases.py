"""Authorial case JSON to validated, ready-only suites.

The published contract is ``DEFAULT_SCHEMA_PATH``. Pass ``schema_path`` to use
another file; this module checks operational fields without a stand dependency.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from .contracts import response_schema, validate_response_contract
from .path_lib import DEFAULT_SCHEMA_PATH
_EVALUATION_OUTCOMES = frozenset({
    "answer", "access_control", "no_data", "missing_skill", "out_of_scope",
})


@dataclass(frozen=True, slots=True)
class _CaseSchema:
    fields: frozenset[str]
    categories: frozenset[str]
    case_id: re.Pattern[str]
    sha256: re.Pattern[str]


@lru_cache(maxsize=8)
def _case_schema(schema_path: str) -> _CaseSchema:
    contract = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return _CaseSchema(
        fields=frozenset(contract["required"]),
        categories=frozenset(contract["properties"]["category"]["enum"]),
        case_id=re.compile(contract["properties"]["case_id"]["pattern"]),
        sha256=re.compile(contract["properties"]["skill_registry_hash"]["pattern"]),
    )


def resolve_schema_path(schema_path: str | Path | None = None) -> Path:
    """Resolve the authorial case schema path.

    Args:
        schema_path: Optional schema file. Relative paths are resolved from CWD.
            ``None`` uses ``DEFAULT_SCHEMA_PATH``.

    Returns:
        Absolute path of the schema that ``validate_case`` will load.
    """
    path = Path(schema_path) if schema_path is not None else DEFAULT_SCHEMA_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_comparison(comparison: Any) -> None:
    fields = {
        "ordered", "row_key", "allow_extra_rows", "numeric_absolute_tolerance",
    }
    if not isinstance(comparison, dict) or set(comparison) != fields:
        raise ValueError("evaluation_contract.comparison has missing or unknown fields")
    for field in ("ordered", "allow_extra_rows"):
        if not isinstance(comparison[field], bool):
            raise ValueError(f"evaluation_contract.comparison.{field} must be boolean")
    keys = comparison["row_key"]
    if not isinstance(keys, list) or any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError("evaluation_contract.comparison.row_key must be an array of strings")
    if len(keys) != len(set(keys)):
        raise ValueError("evaluation_contract.comparison.row_key contains duplicates")
    tolerance = comparison["numeric_absolute_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ValueError(
            "evaluation_contract.comparison.numeric_absolute_tolerance must be >= 0"
        )


def _validate_evaluation_contract(contract: Any) -> None:
    if not isinstance(contract, dict):
        raise ValueError("evaluation_contract must be an object")
    if set(contract) != {"expected_outcome", "gold_result", "comparison"}:
        raise ValueError(
            "evaluation_contract must contain expected_outcome, gold_result and comparison"
        )
    if contract["expected_outcome"] not in _EVALUATION_OUTCOMES:
        raise ValueError("evaluation_contract.expected_outcome is unknown")
    _validate_comparison(contract["comparison"])
    if contract["expected_outcome"] == "answer":
        if contract["gold_result"] is None:
            raise ValueError("evaluation_contract.gold_result is required for outcome=answer")
        if contract["comparison"]["row_key"] and not isinstance(contract["gold_result"], list):
            raise ValueError("comparison.row_key is only valid for array gold_result")
    elif contract["gold_result"] is not None:
        raise ValueError("evaluation_contract.gold_result must be null for non-answer outcomes")


def validate_case(raw: Any, *, schema_path: str | Path | None = None) -> None:
    """Validate the current v3.0 authorial shape.

    Draft cases may omit actor and gold fields. Ready cases must pass
    ``require_ready`` as well.

    Args:
        raw: Parsed JSON object for one case.
        schema_path: Optional override for the published v3 schema.

    Raises:
        ValueError: If required fields, enums, or nested contracts are invalid.
    """
    schema = _case_schema(str(resolve_schema_path(schema_path)))
    if not isinstance(raw, dict):
        raise ValueError("case must be a JSON object")
    missing = schema.fields - raw.keys()
    extra = raw.keys() - schema.fields
    if missing or extra:
        raise ValueError(f"case fields: missing={sorted(missing)}, unknown={sorted(extra)}")
    if raw["schema_version"] != "3.0":
        raise ValueError("schema_version must be '3.0'")
    if not schema.case_id.fullmatch(_nonempty(raw["case_id"], "case_id")):
        raise ValueError("case_id contains unsafe characters")
    if not isinstance(raw["status"], str) or raw["status"] not in {"draft", "ready"}:
        raise ValueError("status must be draft or ready")
    if not isinstance(raw["category"], str) or raw["category"] not in schema.categories:
        raise ValueError(f"unknown category: {raw['category']!r}")
    for field in ("query", "source_type", "source_business_process", "snapshot_id"):
        _nonempty(raw[field], field)
    if not isinstance(raw["employee_role"], str) or raw["employee_role"] not in {"self", "manager", "hr"}:
        raise ValueError("employee_role must be self, manager or hr")
    if raw["employee_id"] is not None:
        _nonempty(raw["employee_id"], "employee_id")
    if not schema.sha256.fullmatch(_nonempty(raw["skill_registry_hash"], "skill_registry_hash")):
        raise ValueError("skill_registry_hash must be sha256:<64 lowercase hex digits>")
    skills = raw["expected_skills"]
    if not isinstance(skills, list) or any(not isinstance(s, str) or not s.strip() for s in skills):
        raise ValueError("expected_skills must be an array of non-empty strings")
    if len(set(skills)) != len(skills):
        raise ValueError("expected_skills contains duplicates")
    if raw["response_contract"] is not None:
        validate_response_contract(raw["response_contract"])
    if raw["evaluation_contract"] is not None:
        _validate_evaluation_contract(raw["evaluation_contract"])
    if raw["status"] == "ready":
        require_ready(raw)


def require_ready(raw: dict[str, Any]) -> None:
    """Refuse incomplete cases; never fill actor or gold by guessing.

    Args:
        raw: Already structurally validated case object.

    Raises:
        ValueError: If status is draft, required ready fields are missing, or
            gold violates the public response contract.
    """
    if raw["status"] != "ready":
        raise ValueError(f"{raw['case_id']}: draft case is not runnable")
    if (
        raw["employee_id"] is None
        or raw["response_contract"] is None
        or raw["evaluation_contract"] is None
    ):
        raise ValueError(
            f"{raw['case_id']}: ready requires employee_id, response_contract and evaluation_contract"
        )
    expected = raw["evaluation_contract"]["expected_outcome"]
    fixed = {
        "answerable": {"answer"},
        "access_control": {"access_control"},
        "no_data": {"no_data"},
        "out_of_scope": {"out_of_scope"},
        # A missing ready-made skill may still be solvable with base tools.
        "missing_skill": {"answer", "missing_skill"},
    }
    if expected not in fixed[raw["category"]]:
        raise ValueError(
            f"{raw['case_id']}: expected_outcome {expected!r} is incompatible "
            f"with category {raw['category']!r}"
        )
    expected_response = {
        "result": raw["evaluation_contract"]["gold_result"],
        "message": None,
    }
    errors = list(
        Draft202012Validator(response_schema(raw["response_contract"])).iter_errors(
            expected_response
        )
    )
    if errors:
        raise ValueError(
            f"{raw['case_id']}: evaluation_contract violates response_contract: "
            f"{errors[0].message}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Validated authorial case loaded from a JSON file.

    Attributes:
        source_path: Absolute path of the source case file.
        raw: Parsed case object after schema validation.
    """

    source_path: Path
    raw: dict[str, Any]

    @property
    def case_id(self) -> str:
        """Stable identifier from the authorial JSON."""
        return self.raw["case_id"]

    @property
    def status(self) -> str:
        """Authorial status: ``draft`` or ``ready``."""
        return self.raw["status"]


def load_case(path: str | Path, *, schema_path: str | Path | None = None) -> BenchmarkCase:
    """Load and validate one authorial JSON case.

    Args:
        path: Path to a ``.json`` case file.
        schema_path: Optional override for the published v3 schema.

    Returns:
        Validated case wrapper. Draft files are allowed.

    Raises:
        ValueError: If the file is not a valid v3 case.
    """
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    validate_case(raw, schema_path=schema_path)
    return BenchmarkCase(source_path=source, raw=raw)


def load_suite(
    cases_dir: str | Path,
    case_ids: Iterable[str] | None = None,
    *,
    on_draft: str = "skip",
    schema_path: str | Path | None = None,
) -> list[BenchmarkCase]:
    """Load ready cases from a directory in deterministic order.

    Args:
        cases_dir: Directory of authorial ``*.json`` files.
        case_ids: If set, only these ids are returned; unknown ids fail.
        on_draft: ``skip`` drops drafts, ``error`` refuses them.
        schema_path: Optional override for the published v3 schema.

    Returns:
        Ready cases sorted by ``case_id``. Source files are never rewritten.

    Raises:
        ValueError: If the directory, ids, drafts or case bodies are invalid.
    """
    if on_draft not in {"skip", "error"}:
        raise ValueError("on_draft must be skip or error")
    root = Path(cases_dir)
    if not root.is_dir():
        raise ValueError(f"cases directory does not exist: {root}")
    requested = set(case_ids) if case_ids is not None else None
    found: dict[str, BenchmarkCase] = {}
    for path in sorted(root.glob("*.json")):
        case = load_case(path, schema_path=schema_path)
        if case.case_id in found:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        found[case.case_id] = case
    if requested is not None:
        missing = requested - found.keys()
        if missing:
            raise ValueError(f"unknown case_ids: {sorted(missing)}")
    selected = [case for key, case in sorted(found.items()) if requested is None or key in requested]
    ready: list[BenchmarkCase] = []
    for case in selected:
        if case.status == "draft" and on_draft == "skip":
            continue
        require_ready(case.raw)
        ready.append(case)
    return ready


def write_suite_jsonl(
    cases: Iterable[BenchmarkCase],
    output: str | Path,
    *,
    schema_path: str | Path | None = None,
) -> Path:
    """Write a derived JSONL suite; source case files stay untouched.

    Args:
        cases: Validated cases to serialize, one JSON object per line.
        output: Destination ``.jsonl`` path. Must not overwrite a source case.
        schema_path: Optional override used to re-validate before write.

    Returns:
        Absolute path of the written suite.

    Raises:
        ValueError: If the destination is unsafe or the suite has duplicate ids.
    """
    selected = list(cases)
    destination = Path(output).resolve()
    if destination in {case.source_path for case in selected}:
        raise ValueError("suite output cannot overwrite a source case")
    if destination.suffix != ".jsonl":
        raise ValueError("suite output must have .jsonl extension")
    ids = [case.case_id for case in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("suite contains duplicate case_ids")
    for case in selected:
        validate_case(case.raw, schema_path=schema_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(case.raw, ensure_ascii=False, sort_keys=True) + "\n" for case in selected),
        encoding="utf-8",
    )
    return destination
