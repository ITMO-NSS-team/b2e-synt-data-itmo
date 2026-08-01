"""Emulator configuration — the two independent variables, and where data lives.

``traps_enabled`` is not one switch over one mechanism. Traps live at two layers:

* **Channel quirks** (``heimdall/engine/quirks.py``) — silent case folding, bool
  string coercion, silent limit clamping, nulls-first ordering. These are applied
  at request time and can be toggled in place.
* **Data traps** (``b2e/traps.py``) — uppercase surnames, NULL-means-unscored,
  the duplicated signal, raw region codes, the stale key-employee flag. These are
  baked into the snapshot when the corpus is built, because a trap keyed to a
  *person* has to be stable across all 37 marts. They cannot be toggled at
  request time; a traps-off run needs a traps-off corpus.

So ``traps_enabled=false`` means: clear the quirks *and* serve the traps-off
snapshot. If that snapshot has not been built, the emulator refuses rather than
serving traps-on data under a traps-off label — which would silently corrupt
exactly the RQ1 comparison the flag exists to enable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from sim.latency import DEFAULT_PROFILE, PROFILES

#: Built by `make seed-traps-off`; see docs/deployment.md.
DEFAULT_SNAPSHOT_ON = "data-small"
DEFAULT_SNAPSHOT_OFF = "data-small-notraps"


class SnapshotUnavailable(RuntimeError):
    """A requested condition has no corpus behind it."""


@dataclass
class EmulatorConfig:
    snapshot_traps_on: Path = field(default_factory=lambda: Path(DEFAULT_SNAPSHOT_ON))
    snapshot_traps_off: Path = field(default_factory=lambda: Path(DEFAULT_SNAPSHOT_OFF))
    catalog_path: Path = field(default_factory=lambda: Path("catalog/snapshot.json"))
    skills_root: Path = field(default_factory=lambda: Path("heimdall-skills"))

    traps_enabled: bool = True
    latency_profile: str = DEFAULT_PROFILE

    #: Employee ids granted the HR role. Empty by default: HR access is a
    #: deliberate experimental condition, not a convenience.
    hr_employee_ids: tuple[str, ...] = ()

    #: The dev router can create skills over HTTP. It is off in any deployment
    #: the agent can reach; see docs/skill-execution-threat-model.md.
    enable_dev_router: bool = False

    def __post_init__(self) -> None:
        self.snapshot_traps_on = Path(self.snapshot_traps_on)
        self.snapshot_traps_off = Path(self.snapshot_traps_off)
        self.catalog_path = Path(self.catalog_path)
        self.skills_root = Path(self.skills_root)
        if self.latency_profile not in PROFILES:
            raise ValueError(
                f"latency_profile {self.latency_profile!r} not one of {PROFILES}")

    @classmethod
    def from_env(cls) -> "EmulatorConfig":
        env = os.environ
        hr = tuple(x for x in env.get("HEIMDALL_HR_EMPLOYEE_IDS", "").split(",") if x)
        return cls(
            snapshot_traps_on=env.get("HEIMDALL_SNAPSHOT_ON", DEFAULT_SNAPSHOT_ON),
            snapshot_traps_off=env.get("HEIMDALL_SNAPSHOT_OFF", DEFAULT_SNAPSHOT_OFF),
            catalog_path=env.get("HEIMDALL_CATALOG", "catalog/snapshot.json"),
            skills_root=env.get("HEIMDALL_SKILLS", "heimdall-skills"),
            traps_enabled=env.get("HEIMDALL_TRAPS_ENABLED", "true").lower() != "false",
            latency_profile=env.get("HEIMDALL_LATENCY_PROFILE", DEFAULT_PROFILE),
            hr_employee_ids=hr,
            enable_dev_router=env.get("HEIMDALL_ENABLE_DEV_ROUTER", "false").lower() == "true",
        )

    # ----------------------------------------------------------------- paths

    def snapshot_for(self, traps_enabled: bool) -> Path:
        path = self.snapshot_traps_on if traps_enabled else self.snapshot_traps_off
        if not (path / "manifest.json").exists():
            raise SnapshotUnavailable(
                f"no corpus at {path} for traps_enabled={traps_enabled}. "
                f"Build it first: make seed-traps-off"
                if not traps_enabled else
                f"no corpus at {path}. Build it first: make seed"
            )
        return path

    def snapshot_id(self, traps_enabled: bool) -> str:
        manifest = json.loads(
            (self.snapshot_for(traps_enabled) / "manifest.json").read_text("utf-8"))
        return str(manifest["snapshot_id"])
