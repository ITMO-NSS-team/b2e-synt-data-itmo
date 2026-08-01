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

``traps_enabled`` and ``data_snapshot_hash`` are both present on purpose. Traps
live at two layers: channel quirks are toggled at request time, but the data
traps in ``b2e/traps.py`` are baked into the snapshot at build time. A run with
traps off is therefore served from a *different snapshot*, and only the pair of
fields identifies the condition unambiguously.
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
)

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

    # -------------------------------------------------------------- build

    @classmethod
    def create(cls, **fields: Any) -> "RunFingerprint":
        """Build and validate. Raises :class:`IncompleteFingerprint` if unusable.

        Treats ``None`` and empty string as unset. A field that is present but
        empty is the common real failure — an unset environment variable that
        arrived as ``""`` — and it must fail exactly like a missing key.
        """
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

        Two runs share a ``condition_id`` exactly when all eight fields match,
        which is the definition of "comparable" in this environment. Used to
        group runs in the research API without re-deriving the rule.
        """
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def require_complete(fingerprint: RunFingerprint | None) -> RunFingerprint:
    """Guard for dispatch paths. Call before spending a single token."""
    if fingerprint is None:
        raise IncompleteFingerprint(FINGERPRINT_FIELDS)
    return fingerprint
