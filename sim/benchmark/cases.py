"""Authorial v2 case JSON to validated, verified-only suites.

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
from urllib.parse import unquote, urlsplit

from jsonschema import Draft202012Validator

from .contracts import response_schema, validate_gold_contract
from .path_lib import DEFAULT_SCHEMA_PATH
_EVALUATION_OUTCOMES = frozenset({
    "answer", "access_control", "no_data", "missing_skill", "out_of_scope",
})
_COMPARISON_DEFAULTS = {
    "ordered": False,
    "row_key": [],
    "allow_extra_rows": False,
    "numeric_absolute_tolerance": 0,
}


@dataclass(frozen=True, slots=True)
class _CaseSchema:
    fields: frozenset[str]
    categories: frozenset[str]
    case_id: re.Pattern[str]
    sha256: re.Pattern[str]
    version: str


@lru_cache(maxsize=8)
def _case_schema(schema_path: str) -> _CaseSchema:
    contract = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return _CaseSchema(
        fields=frozenset(contract["required"]),
        categories=frozenset(contract["properties"]["category"]["enum"]),
        case_id=re.compile(contract["properties"]["case_id"]["pattern"]),
        sha256=re.compile(contract["properties"]["skill_registry_hash"]["pattern"]),
        version=str(contract["properties"]["schema_version"]["const"]),
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


def _resolve_comparison(comparison: Any) -> dict[str, Any]:
    """Expand the v2 ``simple_comparison`` shorthand without mutating input."""
    if not isinstance(comparison, dict):
        raise ValueError("gold_comparison must be an object")
    if "template" not in comparison:
        return dict(comparison)
    if comparison["template"] != "simple_comparison":
        raise ValueError("gold_comparison.template is unknown")
    overrides = {key: value for key, value in comparison.items() if key != "template"}
    if set(overrides) - set(_COMPARISON_DEFAULTS):
        raise ValueError("gold_comparison has unknown fields")
    return {**_COMPARISON_DEFAULTS, **overrides}


def _resolve_schema(value: Any, base: Path, stack: tuple = ()) -> Any:
    """Inline local JSON Schema references relative to the authorial case."""
    if isinstance(value, list):
        return [_resolve_schema(item, base, stack) for item in value]
    if not isinstance(value, dict):
        return value
    resolved_siblings = {
        key: (item if key in {"const", "enum", "default", "examples"}
              else _resolve_schema(item, base, stack))
        for key, item in value.items() if key != "$ref"
    }
    if "$ref" not in value:
        return resolved_siblings
    ref = urlsplit(value["$ref"])
    if ref.scheme or ref.netloc or ref.query:
        raise ValueError("gold_contract supports only local JSON Schema references")
    path = (base.parent / unquote(ref.path)).resolve() if ref.path else base.resolve()
    key = (path, ref.fragment)
    if key in stack:
        raise ValueError(f"cyclic JSON Schema reference: {value['$ref']}")
    target = json.loads(path.read_text(encoding="utf-8-sig"))
    pointer = unquote(ref.fragment)
    if pointer:
        if not pointer.startswith("/"):
            raise ValueError("only JSON Pointer fragments are supported")
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            target = target[int(token)] if isinstance(target, list) else target[token]
    resolved = _resolve_schema(target, path, stack + (key,))
    return {"allOf": [resolved, resolved_siblings]} if resolved_siblings else resolved


def _validate_comparison(comparison: Any) -> None:
    fields = {
        "ordered", "row_key", "allow_extra_rows", "numeric_absolute_tolerance",
    }
    if not isinstance(comparison, dict) or set(comparison) != fields:
        raise ValueError("gold_comparison has missing or unknown fields")
    for field in ("ordered", "allow_extra_rows"):
        if not isinstance(comparison[field], bool):
            raise ValueError(f"gold_comparison.{field} must be boolean")
    keys = comparison["row_key"]
    if not isinstance(keys, list) or any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError("gold_comparison.row_key must be an array of strings")
    if len(keys) != len(set(keys)):
        raise ValueError("gold_comparison.row_key contains duplicates")
    tolerance = comparison["numeric_absolute_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ValueError(
            "gold_comparison.numeric_absolute_tolerance must be >= 0"
        )


def validate_case(raw: Any, *, schema_path: str | Path | None = None) -> None:
    """Validate the v2.0 authorial shape used by the benchmark repository.

    Draft cases may retain incomplete gold. Verified cases must pass
    ``require_ready`` as well.

    Args:
        raw: Parsed JSON object for one case.
        schema_path: Optional override for the published v2 schema.

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
    if raw["schema_version"] != schema.version:
        raise ValueError(f"schema_version must be {schema.version!r}")
    if not schema.case_id.fullmatch(_nonempty(raw["case_id"], "case_id")):
        raise ValueError("case_id contains unsafe characters")
    if not isinstance(raw["status"], str) or raw["status"] not in {"draft", "verified"}:
        raise ValueError("status must be draft or verified")
    if not isinstance(raw["category"], str) or raw["category"] not in schema.categories:
        raise ValueError(f"unknown category: {raw['category']!r}")
    for field in ("query", "source_type", "source_business_process"):
        _nonempty(raw[field], field)
    if raw["snapshot_id"] is not None:
        _nonempty(raw["snapshot_id"], "snapshot_id")
    if not isinstance(raw["employee_role"], str) or raw["employee_role"] not in {"self", "manager", "hr"}:
        raise ValueError("employee_role must be self, manager or hr")
    if raw["employee_id"] is not None and (
        isinstance(raw["employee_id"], bool) or not isinstance(raw["employee_id"], int)
    ):
        raise ValueError("employee_id must be an integer or null")
    if raw["skill_registry_hash"] is not None and not schema.sha256.fullmatch(
        _nonempty(raw["skill_registry_hash"], "skill_registry_hash")
    ):
        raise ValueError("skill_registry_hash must be sha256:<64 lowercase hex digits> or null")
    skills = raw["expected_skills"]
    if not isinstance(skills, list) or any(not isinstance(s, str) or not s.strip() for s in skills):
        raise ValueError("expected_skills must be an array of non-empty strings")
    if len(set(skills)) != len(skills):
        raise ValueError("expected_skills contains duplicates")
    if not isinstance(raw["gold_contract"], dict):
        raise ValueError("gold_contract must be an object")
    if raw["gold_contract"]:
        validate_gold_contract(raw["gold_contract"])
    comparison = _resolve_comparison(raw["gold_comparison"])
    _validate_comparison(comparison)
    if raw["status"] == "verified":
        require_ready(raw)


def require_ready(raw: dict[str, Any]) -> None:
    """Refuse incomplete cases; never fill actor or gold by guessing.

    Args:
        raw: Already structurally validated case object.

    Raises:
        ValueError: If status is draft, required verified fields are missing, or
            gold violates the public response contract.
    """
    if raw["status"] != "verified":
        raise ValueError(f"{raw['case_id']}: draft case is not runnable")
    if (
        raw["employee_id"] is None
        or raw["snapshot_id"] is None
        or raw["skill_registry_hash"] is None
        or not raw["gold_contract"]
        or not isinstance(raw["gold_answer"], dict)
    ):
        raise ValueError(
            f"{raw['case_id']}: verified requires actor, snapshot, contract and gold_answer"
        )
    expected = raw["gold_answer"].get("outcome")
    if expected not in _EVALUATION_OUTCOMES:
        raise ValueError(f"{raw['case_id']}: gold_answer.outcome is unknown")
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
            f"{raw['case_id']}: gold_answer outcome {expected!r} is incompatible "
            f"with category {raw['category']!r}"
        )
    errors = list(
        Draft202012Validator(response_schema(raw["gold_contract"])).iter_errors(
            raw["gold_answer"]
        )
    )
    if errors:
        raise ValueError(
            f"{raw['case_id']}: gold_answer violates gold_contract: "
            f"{errors[0].message}"
        )
    comparison = _resolve_comparison(raw["gold_comparison"])
    if comparison["row_key"] and not isinstance(raw["gold_answer"].get("rows"), list):
        raise ValueError("gold_comparison.row_key is only valid for array rows")


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
        """Authorial status: ``draft`` or ``verified``."""
        return self.raw["status"]


def load_case(path: str | Path, *, schema_path: str | Path | None = None) -> BenchmarkCase:
    """Load and validate one authorial JSON case.

    Args:
        path: Path to a ``.json`` case file.
        schema_path: Optional override for the published v2 schema.

    Returns:
        Validated case wrapper. Draft files are allowed.

    Raises:
        ValueError: If the file is not a valid v2 case.
    """
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("gold_contract"), dict):
        raw["gold_contract"] = _resolve_schema(raw["gold_contract"], source)
    if isinstance(raw, dict) and isinstance(raw.get("gold_comparison"), dict):
        raw["gold_comparison"] = _resolve_comparison(raw["gold_comparison"])
    validate_case(raw, schema_path=schema_path)
    return BenchmarkCase(source_path=source, raw=raw)


def load_suite(
    cases_dir: str | Path,
    case_ids: Iterable[str] | None = None,
    *,
    on_draft: str = "skip",
    schema_path: str | Path | None = None,
) -> list[BenchmarkCase]:
    """Load runnable cases recursively from a benchmark repository.

    Args:
        cases_dir: Root containing authorial JSON cases. JSON records outside
            the runner's ``draft``/``ready`` lifecycle are ignored.
        case_ids: If set, only these ids are returned; unknown ids fail.
        on_draft: ``skip`` drops drafts, ``error`` refuses them.
        schema_path: Optional override for the published v2 schema.

    Returns:
        Verified cases sorted by ``case_id``. Source files are never rewritten.

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
    for path in sorted(root.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or "case_id" not in raw
            or raw.get("status") not in {"draft", "verified"}
        ):
            continue
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
