"""Per-employee permission scoping.

Why this exists
---------------
The spec requires that a researcher-selected employee identity "must actually
get 403s". That is not decoration. One of the mandatory question categories is
access control, and a question like "show me my manager's salary" only measures
anything if the API genuinely refuses. An emulator that authorises everything
turns every access-control question into a test of the model's politeness rather
than of the system's boundary.

Where the scope comes from
--------------------------
Two sources, both server-side:

* ``truth/people.json`` — ``person_id``, ``employee_id``, ``unit_id``, ``is_head``.
  This file is never served through the API (``test_latent_factors_are_not_served``
  guards that), but using it for *authorisation* is correct: in a real deployment
  the identity service knows the org chart too. Nothing derived here reaches the
  agent except an allow/deny decision.
* The org tree, rebuilt from the snapshot seed. It is deterministic, so it costs
  a few hundred milliseconds at startup instead of a file on disk that could
  drift from the corpus it describes.

Roles
-----
``self``     sees only their own record. The default for a non-manager.
``manager``  sees their own unit and every unit below it. The default for a head.
``hr``       sees everything, including the restricted schemas.

Restricted schemas are the ones an ordinary employee has no business reading:
recruitment pipelines, external candidate profiles, and the talent radar mirror.
Those return 403 for ``self`` and ``manager`` regardless of row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

#: Schemas an ordinary employee cannot read at all. Chosen because each holds
#: data about people other than the requester by construction.
HR_ONLY_SCHEMAS = frozenset({"recruitment", "anagent"})

#: Individual models that are HR-only even though their schema is not. The
#: talent radar deliberately mirrors ~10% of employees under a foreign key, so
#: it is a re-identification surface (see docs/research-agenda.md, RQ3).
HR_ONLY_MODELS = frozenset({"dm_special.talent_radar_people"})

#: Columns that carry a person identity. A filter on one of these is how a query
#: asks about a specific human, so it is where row-level scope is enforced.
PERSON_KEY_COLUMNS = frozenset({
    "person_id", "employee_id", "emp_key", "tn", "tab_num", "employee_number",
})

ROLES = ("self", "manager", "hr")


class AccessDenied(Exception):
    """Raised when the requesting identity may not see what it asked for.

    Carries a Heimdall error code so the app layer can render the real envelope
    rather than inventing a shape that the production API never emits.
    """

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        self.code = code
        self.detail = detail
        self.hint = hint
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class Scope:
    """What one identity is allowed to see."""

    employee_id: str
    person_id: str
    role: str
    unit_id: int
    #: Every reference this identity may read, in either key space. Used for the
    #: "did the query name someone forbidden" check, where the caller may have
    #: written either kind of id.
    visible_person_ids: frozenset[str]
    #: The two key spaces kept apart, because a row-scope filter goes against a
    #: specific column and the columns have different types: `person_id` is a
    #: UUID and `employee_id` is numeric. Filtering one column with a mixed set
    #: is rejected outright by the engine — `filter-value-invalid`, "колонка
    #: person_id — UUID, а значение '2457060' не UUID" — which would turn every
    #: scoped read into a 400 rather than an authorised answer.
    visible_uuids: frozenset[str] = frozenset()
    visible_employee_ids: frozenset[str] = frozenset()

    def visible_for(self, column: str) -> frozenset[str]:
        return (self.visible_employee_ids if column == "employee_id"
                else self.visible_uuids)

    @property
    def sees_everything(self) -> bool:
        return self.role == "hr"

    def may_read_model(self, schema: str, model: str) -> bool:
        if self.sees_everything:
            return True
        if schema in HR_ONLY_SCHEMAS:
            return False
        return f"{schema}.{model}" not in HR_ONLY_MODELS

    def may_read_person(self, person_ref: str) -> bool:
        if self.sees_everything:
            return True
        return str(person_ref) in self.visible_person_ids


class IdentityIndex:
    """Resolves an employee id to a :class:`Scope`.

    Built once at emulator start. On the 2 741-person corpus this is
    milliseconds; on 294 000 it is a few seconds and a couple of int64 arrays.
    """

    def __init__(self, snapshot_root: str | Path, *, hr_employee_ids: Iterable[str] = ()) -> None:
        self.root = Path(snapshot_root)
        self.hr_ids = {str(x) for x in hr_employee_ids}

        truth = json.loads((self.root / "truth" / "people.json").read_text("utf-8"))
        self.person_id = [str(x) for x in truth["person_id"]]
        self.employee_id = [str(x) for x in truth["employee_id"]]
        self.unit_id = np.asarray(truth["unit_id"], dtype=np.int64)
        self.is_head = np.asarray(truth["is_head"]).astype(bool)

        manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
        self.seed = int(manifest["seed"])
        self.snapshot_id = str(manifest["snapshot_id"])
        self.n_units = int(manifest["units"])

        self._by_employee = {e: i for i, e in enumerate(self.employee_id)}
        self._descendants = self._build_descendant_map()
        self._cache: dict[str, Scope] = {}

    # ------------------------------------------------------------------ tree

    def _build_descendant_map(self) -> dict[int, set[int]]:
        """unit_id -> that unit and every unit beneath it.

        Rebuilt from the seed rather than read from disk: the tree is a pure
        function of (seed, requested headcount), and a persisted copy could
        silently disagree with the corpus it is supposed to describe.

        The *requested* headcount has to be recovered, not assumed. ``org.build``
        takes the number asked for, but the tree it produces has a different
        actual headcount because the recursive split rounds. The manifest records
        only the actual figure, so building with ``len(self.person_id)`` yields a
        **different, smaller organisation** — 2 494 people in 503 units where the
        corpus has 2 741 in 540. That was the original bug here, and it was not
        loud: unit ids below the smaller tree's size still resolved, so managers
        in low-numbered units got plausible scopes while everyone above the cut
        silently collapsed to self-only. Wrong 403s are worse than no 403s,
        because the access-control question category then measures nothing.
        """
        from sim.oracle.labels import rebuild_org_tree

        tree = rebuild_org_tree(self.seed, len(self.person_id), self.n_units)

        parent = np.asarray(tree.parent, dtype=np.int64)
        children: dict[int, list[int]] = {}
        for node, par in enumerate(parent):
            if par >= 0:
                children.setdefault(int(par), []).append(node)

        out: dict[int, set[int]] = {}
        for node in range(len(parent)):
            seen: set[int] = set()
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(children.get(cur, ()))
            out[node] = seen
        return out

    # ----------------------------------------------------------------- scope

    def role_for(self, employee_id: str) -> str:
        if str(employee_id) in self.hr_ids:
            return "hr"
        idx = self._by_employee.get(str(employee_id))
        if idx is None:
            return "self"
        return "manager" if bool(self.is_head[idx]) else "self"

    def scope_for(self, employee_id: str) -> Scope:
        employee_id = str(employee_id)
        if employee_id in self._cache:
            return self._cache[employee_id]

        idx = self._by_employee.get(employee_id)
        if idx is None:
            raise AccessDenied(
                "auth-failed",
                f"identity {employee_id} is not an employee in this snapshot",
                "the researcher must select an employee_id present in truth/people.json",
            )

        role = self.role_for(employee_id)
        unit = int(self.unit_id[idx])

        if role == "hr":
            uuids: frozenset[str] = frozenset()
            employees: frozenset[str] = frozenset()
        elif role == "manager":
            units = self._descendants.get(unit, {unit})
            mask = np.isin(self.unit_id,
                           np.fromiter(units, dtype=np.int64, count=len(units)))
            rows = np.nonzero(mask)[0]
            uuids = frozenset(self.person_id[i] for i in rows)
            employees = frozenset(self.employee_id[i] for i in rows)
        else:
            uuids = frozenset({self.person_id[idx]})
            employees = frozenset({employee_id})

        scope = Scope(employee_id=employee_id, person_id=self.person_id[idx],
                      role=role, unit_id=unit,
                      visible_person_ids=uuids | employees,
                      visible_uuids=uuids, visible_employee_ids=employees)
        self._cache[employee_id] = scope
        return scope

    # ------------------------------------------------------------- sampling

    def sample_identities(self, n: int = 5) -> list[dict[str, Any]]:
        """A few identities of each role, for researchers picking a subject."""
        out: list[dict[str, Any]] = []
        heads = [i for i in range(len(self.person_id)) if self.is_head[i]][:n]
        plain = [i for i in range(len(self.person_id)) if not self.is_head[i]][:n]
        for i in heads + plain:
            scope = self.scope_for(self.employee_id[i])
            out.append({
                "employee_id": self.employee_id[i],
                "person_id": self.person_id[i],
                "role": scope.role,
                "unit_id": scope.unit_id,
                "visible_people": ("all" if scope.sees_everything
                                   else len(scope.visible_person_ids)),
            })
        return out


# ------------------------------------------------------------------ checking


def collect_person_filters(filters: Any) -> list[str]:
    """Pull every literal value filtered against a person-key column.

    Walks the filter tree the same way the engine does. Used to decide whether a
    query is asking about a specific person, and if so, whom.
    """
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            column = node.get("column")
            if isinstance(column, str) and column.lower() in PERSON_KEY_COLUMNS:
                for key in ("value", "values", "pattern"):
                    val = node.get(key)
                    if isinstance(val, (str, int)):
                        found.append(str(val))
                    elif isinstance(val, list):
                        found.extend(str(v) for v in val if isinstance(v, (str, int)))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(filters)
    return found


def scope_filter(scope: Scope, column: str = "person_id") -> dict[str, Any]:
    """A predicate restricting rows to what this identity may see.

    Node shape follows the engine exactly: ``condition_in`` takes ``value`` (not
    ``values``) and the operator is upper-case ``IN``. The engine enforces
    ``additionalProperties:false``, so a near-miss here is not a soft failure —
    it rejects the whole request with 400 and the identity sees nothing at all.
    """
    return {"type": "condition_in", "column": column, "operator": "IN",
            "value": sorted(scope.visible_for(column))}


def restrict(scope: Scope, body: dict[str, Any], *,
             column: str = "person_id") -> dict[str, Any]:
    """Return ``body`` with a mandatory scope predicate ANDed onto its filters.

    Without this, :func:`enforce` is advisory rather than enforcing. It refuses a
    query that *names* a forbidden person, but a query with no person filter at
    all passed straight through and returned the whole mart — measured: an
    identity entitled to see 2 records received 50 rows, 49 of them other
    people's. Any agent that pages a mart and filters client-side sidestepped the
    boundary entirely, and the access-control question category was measuring
    the agent's phrasing habits rather than the permission system.

    The explicit-reference 403 in :func:`enforce` is kept alongside this on
    purpose: asking for a specific forbidden person should be refused, not
    silently answered with an empty result, because an empty result asserts that
    the person does not exist.
    """
    if scope.sees_everything:
        return body

    predicate = scope_filter(scope, column)
    existing = body.get("filters")
    if not existing:
        return {**body, "filters": predicate}
    # The engine's `and` node holds its children under `conditions`.
    return {**body, "filters": {"type": "and", "conditions": [existing, predicate]}}


def enforce(scope: Scope, schema: str, model: str, body: dict[str, Any]) -> None:
    """Raise :class:`AccessDenied` if this identity may not run this query.

    Two checks, in the order a real system would apply them:

    1. Model-level. Some schemas are HR-only; asking at all is a 403.
    2. Row-level. If the query names a specific person outside the scope, that
       is a 403 rather than an empty result — an empty result would teach the
       agent that the person does not exist, which is a different and wrong fact.
    """
    if not scope.may_read_model(schema, model):
        raise AccessDenied(
            "forbidden",
            f"identity {scope.employee_id} (role={scope.role}) may not read "
            f"{schema}.{model}",
            "this model is restricted to the HR role",
        )

    if scope.sees_everything:
        return

    for ref in collect_person_filters(body.get("filters")):
        if not scope.may_read_person(ref):
            raise AccessDenied(
                "forbidden",
                f"identity {scope.employee_id} (role={scope.role}) may not read "
                f"records for {ref}",
                "an employee sees their own record; a head also sees their org subtree",
            )
