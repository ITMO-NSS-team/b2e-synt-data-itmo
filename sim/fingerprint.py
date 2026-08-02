"""Run fingerprint — the identity of an experimental run.

Why this module refuses rather than defaults
--------------------------------------------
Every comparison this environment exists to support is a comparison *between
two runs*. "Configuration A hallucinates less than configuration B" is only a
claim if both runs recorded what they actually were. A run that silently
defaulted one field is not a weaker data point — it is an unusable one, because
you cannot tell afterwards which side of the comparison it belongs to.

So the fingerprint is constructed once, validated eagerly, and any missing field
raises before a single token is spent. There is deliberately no default for any
field, and no partial construction: a half-built fingerprint cannot exist.

The fields are fixed by the research questions:

``agent_config_version``   which agent config produced this run
``prompt_registry_version``which system prompt revision was rendered
``skill_registry_hash``    which skills were executable
``model_id``               which model answered
``temperature``            sampling temperature
``data_snapshot_hash``     which corpus snapshot was served
``traps_enabled``          RQ1's independent variable
``latency_profile``        RQ2's independent variable
``hr_employee_ids``        who held the HR role, and therefore saw everything

``traps_enabled`` and ``data_snapshot_hash`` are both present on purpose. Traps
live at two layers: channel quirks are toggled at request time, but the data
traps in ``b2e/traps.py`` are baked into the snapshot at build time. A run with
traps off is therefore served from a *different snapshot*, and only the pair of
fields identifies the condition unambiguously.

``hr_employee_ids`` was added late, after it caused exactly the failure this
module exists to prevent. The HR grant decides whether the API answers a
company-wide question or refuses it — in the deployed corpus a manager sees 21
people and an HR identity sees 294 000 — so it is one of the largest effects
available on any metric. It was configured in ``.env``, read only by the
emulator, and absent from the fingerprint, which meant two runs that differed by
the single most consequential permission setting carried the same
``condition_id`` and would have been pooled as one condition.

It is normalised to a string rather than kept as a list: OTel attribute values
are scalars, and a stable sorted rendering makes the grouping key exact instead
of order-dependent. Empty means nobody, and is written ``"none"`` — a real
condition with a real name, not an unset field.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

#: Every field here must be set on every root span. Order is the documented
#: order in docs/span-schema.md; keep the two in step.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "agent_config_version",
    "prompt_registry_version",
    "skill_registry_hash",
    "model_id",
    "temperature",
    "data_snapshot_hash",
    "traps_enabled",
    "latency_profile",
    "hr_employee_ids",
)

#: What ``hr_employee_ids`` says when nobody holds the role. Spelled rather than
#: left empty, because :meth:`RunFingerprint.create` treats an empty string as
#: unset — and "nobody has HR" is the default experimental condition, not a
#: missing value.
NO_HR = "none"


def canonical_hr_ids(ids: Any) -> str:
    """Normalise an HR grant to one stable string.

    Sorted and de-duplicated, so ``["7", "9"]`` and ``["9", "7", "9"]`` are the
    same condition and hash to the same ``condition_id``. Accepts the list the
    emulator reports or an already-rendered string, because this is called on
    both sides of an HTTP boundary.
    """
    if ids is None:
        return NO_HR
    if isinstance(ids, str):
        parts = [p.strip() for p in ids.split(",")]
    else:
        parts = [str(p).strip() for p in ids]
    kept = sorted({p for p in parts if p and p != NO_HR})
    return ",".join(kept) if kept else NO_HR

#: Span attribute prefix. Flat scalar attributes, not a nested object: OTel
#: attribute values are scalars, and Phoenix filters on flat keys.
ATTR_PREFIX = "b2e.run."

VALID_LATENCY_PROFILES = ("instant", "realistic", "degraded")


class IncompleteFingerprint(RuntimeError):
    """Raised when a run is dispatched without a complete fingerprint.

    This is the enforcement point the spec asks for. It is an error, not a
    warning, because a warning would produce traces that look fine and compare
    wrong.
    """

    def __init__(self, missing: Iterable[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(
            "refusing to start run: fingerprint fields unset: "
            + ", ".join(self.missing)
            + ". Every root span must carry the full fingerprint or "
              "cross-experiment comparison is invalid."
        )


@dataclass(frozen=True, slots=True)
class RunFingerprint:
    """Immutable identity of a run. Construct via :meth:`create`."""

    agent_config_version: str
    prompt_registry_version: str
    skill_registry_hash: str
    model_id: str
    temperature: float
    data_snapshot_hash: str
    traps_enabled: bool
    latency_profile: str
    hr_employee_ids: str

    # -------------------------------------------------------------- build

    @classmethod
    def create(cls, **fields: Any) -> "RunFingerprint":
        """Build and validate. Raises :class:`IncompleteFingerprint` if unusable.

        Treats ``None`` and empty string as unset. A field that is present but
        empty is the common real failure — an unset environment variable that
        arrived as ``""`` — and it must fail exactly like a missing key.
        """
        # Normalised before the completeness check, and only when the key is
        # actually present. The distinction matters: *not declaring* the HR
        # grant is a missing field and must raise, but declaring it as empty is
        # the default condition — nobody holds HR — and has to survive a check
        # whose whole job is to reject empty values.
        fields = dict(fields)
        if "hr_employee_ids" in fields:
            fields["hr_employee_ids"] = canonical_hr_ids(fields["hr_employee_ids"])

        missing = [
            name for name in FINGERPRINT_FIELDS
            if fields.get(name) is None
            or (isinstance(fields.get(name), str) and not fields[name].strip())
        ]
        if missing:
            raise IncompleteFingerprint(missing)

        unknown = set(fields) - set(FINGERPRINT_FIELDS)
        if unknown:
            raise ValueError(f"unknown fingerprint fields: {sorted(unknown)}")

        profile = fields["latency_profile"]
        if profile not in VALID_LATENCY_PROFILES:
            raise ValueError(
                f"latency_profile {profile!r} not one of {VALID_LATENCY_PROFILES}"
            )

        return cls(
            agent_config_version=str(fields["agent_config_version"]),
            prompt_registry_version=str(fields["prompt_registry_version"]),
            skill_registry_hash=str(fields["skill_registry_hash"]),
            model_id=str(fields["model_id"]),
            temperature=float(fields["temperature"]),
            data_snapshot_hash=str(fields["data_snapshot_hash"]),
            traps_enabled=bool(fields["traps_enabled"]),
            latency_profile=str(profile),
            hr_employee_ids=str(fields["hr_employee_ids"]),
        )

    # -------------------------------------------------------------- emit

    def as_span_attributes(self) -> dict[str, Any]:
        """Flat OTel attributes for the root span, one key per field."""
        return {ATTR_PREFIX + name: getattr(self, name) for name in FINGERPRINT_FIELDS}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def condition_id(self) -> str:
        """Stable short id for "the same experimental condition".

        Two runs share a ``condition_id`` exactly when every field matches,
        which is the definition of "comparable" in this environment. Used to
        group runs in the research API without re-deriving the rule.

        Note that adding a field necessarily changes every id. That is correct
        rather than unfortunate: runs recorded before ``hr_employee_ids`` existed
        did not record who could see the whole company, so they are genuinely
        not comparable with runs that did.
        """
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def require_complete(fingerprint: RunFingerprint | None) -> RunFingerprint:
    """Guard for dispatch paths. Call before spending a single token."""
    if fingerprint is None:
        raise IncompleteFingerprint(FINGERPRINT_FIELDS)
    return fingerprint
