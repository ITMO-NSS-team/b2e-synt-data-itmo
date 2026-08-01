"""Latency profiles for the Heimdall emulator.

Why not a fixed sleep
---------------------
RQ2 asks how to reduce end-to-end answer latency when the agent cannot compute
and must instead compose API calls. The only levers the agent has are *how many*
calls it makes and *what shape* each one is. A fixed sleep per call makes the
first lever measurable and the second invisible: a query for four columns and a
query for all 642 would cost the same, so "ask for fewer columns" would score as
a free win. That would be a measurement artefact, not a finding.

So latency is drawn per endpoint from a distribution, and ``mcp_query`` is
additionally sensitive to the shape of the request — column count and row count
— because a columnar store genuinely is.

Distributions are log-normal. Service latency is positive, right-skewed and has
a long tail; a normal distribution would produce negative draws and no tail, and
a fixed value would produce no p95 at all. The assumed parameters are stated in
``docs/latency-profiles.md`` — they are assumptions about a system we are not
measuring, and reporting p95 without disclosing them would be dishonest.

Determinism
-----------
The same request, in the same profile, at the same occurrence index, gets the
same latency. This is deliberate: when a researcher changes the system prompt and
re-runs a question, the latency difference should be attributable to the change
in agent behaviour, not to fresh noise. Repeat calls within a session differ,
because the occurrence index is part of the seed.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping

PROFILES = ("instant", "realistic", "degraded")
DEFAULT_PROFILE = "realistic"


@dataclass(frozen=True, slots=True)
class Dist:
    """Log-normal latency in milliseconds, described by median and spread.

    ``median_ms`` is the 50th percentile. ``sigma`` is the standard deviation of
    the underlying normal in log space: 0.25 is a tight service, 0.7 is a noisy
    one. ``floor_ms`` models fixed overhead that no amount of luck removes
    (TLS, framing, the process actually waking up).
    """

    median_ms: float
    sigma: float
    floor_ms: float = 0.0

    def draw(self, u_normal: float) -> float:
        """Draw given a standard-normal sample."""
        return self.floor_ms + self.median_ms * math.exp(self.sigma * u_normal)


@dataclass(frozen=True, slots=True)
class Profile:
    """A named set of per-endpoint distributions."""

    name: str
    endpoints: Mapping[str, Dist]
    default: Dist
    #: Extra cost per requested column on mcp_query, before the log-normal draw.
    ms_per_column: float = 0.0
    #: Extra cost per returned row on mcp_query.
    ms_per_row: float = 0.0

    def dist_for(self, endpoint: str) -> Dist:
        return self.endpoints.get(endpoint, self.default)


# ------------------------------------------------------------------ profiles
#
# Endpoint keys are the emulator's logical operation names, not URL paths, so
# that a route rename does not silently drop an endpoint back to `default`.

_REALISTIC = Profile(
    name="realistic",
    default=Dist(median_ms=45, sigma=0.45, floor_ms=6),
    endpoints={
        # Catalogue reads. Served from memory on the real system too.
        "list_models":    Dist(median_ms=22, sigma=0.35, floor_ms=5),
        "describe_model": Dist(median_ms=38, sigma=0.40, floor_ms=5),
        "get_docs":       Dist(median_ms=28, sigma=0.35, floor_ms=5),
        # v2 skill surface.
        "overview":       Dist(median_ms=30, sigma=0.35, floor_ms=5),
        "find_skills":    Dist(median_ms=55, sigma=0.50, floor_ms=6),
        "get_skill":      Dist(median_ms=25, sigma=0.35, floor_ms=5),
        # The data path. Base cost only; width and height are added below.
        "mcp_query":      Dist(median_ms=70, sigma=0.55, floor_ms=8),
        # Recruitment RPC stubs reach a separate system in production.
        "rpc":            Dist(median_ms=240, sigma=0.60, floor_ms=15),
    },
    ms_per_column=0.55,
    ms_per_row=0.09,
)

_INSTANT = Profile(
    name="instant",
    default=Dist(median_ms=0, sigma=0.0, floor_ms=0),
    endpoints={},
    ms_per_column=0.0,
    ms_per_row=0.0,
)

# Degraded is not "realistic times a constant". A loaded database degrades at the
# tail first: the median moves somewhat, the spread moves a lot. Scaling the
# median alone would produce a slower system with the same *shape*, and the
# p95/p50 ratio — the thing that actually hurts an agent making ten sequential
# calls — would not move at all.
_DEGRADED = Profile(
    name="degraded",
    default=Dist(median_ms=140, sigma=0.90, floor_ms=12),
    endpoints={
        "list_models":    Dist(median_ms=60, sigma=0.70, floor_ms=10),
        "describe_model": Dist(median_ms=110, sigma=0.80, floor_ms=10),
        "get_docs":       Dist(median_ms=80, sigma=0.70, floor_ms=10),
        "overview":       Dist(median_ms=90, sigma=0.75, floor_ms=10),
        "find_skills":    Dist(median_ms=190, sigma=0.95, floor_ms=12),
        "get_skill":      Dist(median_ms=75, sigma=0.70, floor_ms=10),
        "mcp_query":      Dist(median_ms=320, sigma=1.10, floor_ms=18),
        "rpc":            Dist(median_ms=900, sigma=1.20, floor_ms=40),
    },
    ms_per_column=1.8,
    ms_per_row=0.35,
)

_BY_NAME: dict[str, Profile] = {p.name: p for p in (_INSTANT, _REALISTIC, _DEGRADED)}


def get_profile(name: str) -> Profile:
    if name not in _BY_NAME:
        raise ValueError(f"unknown latency profile {name!r}; expected one of {PROFILES}")
    return _BY_NAME[name]


# ------------------------------------------------------------------ sampling


def _standard_normal(seed_bytes: bytes) -> float:
    """Box-Muller from a hash. No global RNG state, so it is safe under uvicorn
    workers and reproducible regardless of request interleaving."""
    digest = hashlib.sha256(seed_bytes).digest()
    a, b = struct.unpack_from("<QQ", digest)
    # Open interval (0, 1): u1 == 0 would make log(u1) infinite.
    u1 = (a + 1) / (2**64 + 1)
    u2 = (b + 1) / (2**64 + 1)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _canonical(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(payload)


def sample_ms(
    profile: Profile,
    endpoint: str,
    *,
    request: Any = None,
    occurrence: int = 0,
    n_columns: int = 0,
    n_rows: int = 0,
) -> float:
    """Latency in milliseconds for one call.

    ``occurrence`` distinguishes repeated identical requests within a session so
    that a retry loop does not get a suspiciously flat latency profile, while
    keeping the whole sequence reproducible across runs.
    """
    if profile.name == "instant":
        return 0.0

    seed = "|".join((profile.name, endpoint, _canonical(request), str(occurrence)))
    u = _standard_normal(seed.encode("utf-8"))

    dist = profile.dist_for(endpoint)
    shape_ms = n_columns * profile.ms_per_column + n_rows * profile.ms_per_row
    base = Dist(dist.median_ms + shape_ms, dist.sigma, dist.floor_ms)
    return max(0.0, base.draw(u))


def describe_profiles() -> list[dict[str, Any]]:
    """Machine-readable profile description, served by the emulator and used to
    generate docs/latency-profiles.md so the doc cannot drift from the code."""
    out: list[dict[str, Any]] = []
    for profile in (_INSTANT, _REALISTIC, _DEGRADED):
        rows = []
        for name in sorted(set(profile.endpoints) | {"<default>"}):
            dist = profile.default if name == "<default>" else profile.endpoints[name]
            rows.append({
                "endpoint": name,
                "median_ms": dist.median_ms,
                "sigma": dist.sigma,
                "floor_ms": dist.floor_ms,
                # p95 of a log-normal is median * exp(1.645 * sigma).
                "p95_ms": round(dist.floor_ms + dist.median_ms * math.exp(1.645 * dist.sigma), 1),
            })
        out.append({
            "profile": profile.name,
            "ms_per_column": profile.ms_per_column,
            "ms_per_row": profile.ms_per_row,
            "endpoints": rows,
        })
    return out
