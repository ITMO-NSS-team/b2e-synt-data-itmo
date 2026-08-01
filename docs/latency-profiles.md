# Latency profiles: the assumed distributions

RQ2 ("Скорость ответа", `docs/research-agenda.md` §2) is measured against a
simulated Heimdall. Every latency number the simulator produces — p50, p95, the
cost of `columns: ["*"]`, the penalty for N round-trips instead of one — is a
consequence of parameters we chose, not of a system anybody profiled. Reporting
a p95 without publishing those parameters would let a reader mistake our
assumptions for a measurement. This document publishes them.

Source of truth is `sim/latency.py`. The tables below were produced by running
`sim.latency.describe_profiles()` in the repo venv on 2026-08-01, not typed by
hand, and the same function is served live at `GET /control/latency-profiles`
on the emulator, so a reviewer can check the running condition against this page.

---

## 1. Why a distribution at all, and why log-normal

**Why not a fixed sleep.** The agent cannot execute code at query time. Its only
levers are *how many* calls it makes and *what shape* each call has. A constant
delay per call makes the first lever visible and the second one invisible: a
four-column projection and a 642-column `["*"]` scan would cost exactly the same
millisecond count, so "ask for fewer columns" would score as a free win with no
downside. That would be an artefact of the harness, not a finding about agents.
A constant also has no p95 — every percentile equals the mean — which silently
deletes the failure mode that actually hurts a ten-call plan: one unlucky call.

**Why not a normal distribution.** Service latency is positive, right-skewed and
heavy in the upper tail. A Gaussian centred at 70 ms with enough spread to
produce an interesting tail also produces negative draws, which have to be
clipped — and clipping at zero quietly changes the mean and destroys the
symmetry that was the only reason to pick a Gaussian. A Gaussian narrow enough
to avoid negatives has no tail worth measuring.

**Why log-normal.** It is positive by construction, right-skewed, has a
parameter (`sigma`) that controls the tail independently of the centre
(`median_ms`), and its percentiles are closed-form, so the tables here can be
computed rather than simulated:

```
latency_ms = floor_ms + median_ms * exp(sigma * Z),   Z ~ N(0, 1)
p50        = floor_ms + median_ms
p95        = floor_ms + median_ms * exp(1.645 * sigma)
```

`floor_ms` is fixed overhead no draw can remove — TLS, framing, the process
waking up. It is kept outside the log-normal on purpose: multiplicative noise
belongs to the work, not to the constant cost of being a network service. It
also damps the p95/p50 ratio slightly for cheap endpoints, which is the correct
behaviour — a 5 ms catalogue read is not six times slower on a bad day.

---

## 2. The three profiles

`instant`, `realistic`, `degraded`. The profile name is a field of the run
fingerprint (`sim/fingerprint.py`, `latency_profile`), so no two conditions can
be compared without the profile being recorded alongside the numbers.

| Profile | What it is for |
|---|---|
| `instant` | All latency is exactly 0.0 ms. For correctness runs (RQ1, RQ3) where waiting adds nothing, and for CI. `sample_ms` short-circuits before drawing, so `instant` is not "a very fast distribution" — it is no distribution at all. |
| `realistic` | The default. A healthy in-memory/columnar service on a quiet host. This is the baseline every RQ2 comparison is reported against. |
| `degraded` | The same service under load. Not `realistic` scaled by a constant — see §5. |

Endpoint keys are the emulator's *logical operation names*, not URL paths
(`_classify` in `sim/emulator/app.py`), so renaming a route cannot silently drop
an endpoint back onto `<default>` and make it look faster.

### 2.1 `instant`

`ms_per_column = 0.0`, `ms_per_row = 0.0`.

| Endpoint | median ms | sigma | floor ms | p95 ms |
|---|---:|---:|---:|---:|
| `<default>` | 0 | 0.00 | 0 | 0.0 |

### 2.2 `realistic`

`ms_per_column = 0.55`, `ms_per_row = 0.09`.

| Endpoint | median ms | sigma | floor ms | p95 ms |
|---|---:|---:|---:|---:|
| `<default>` | 45 | 0.45 | 6 | 100.3 |
| `describe_model` | 38 | 0.40 | 5 | 78.4 |
| `find_skills` | 55 | 0.50 | 6 | 131.2 |
| `get_docs` | 28 | 0.35 | 5 | 54.8 |
| `get_skill` | 25 | 0.35 | 5 | 49.5 |
| `list_models` | 22 | 0.35 | 5 | 44.1 |
| `mcp_query` | 70 | 0.55 | 8 | 181.0 |
| `overview` | 30 | 0.35 | 5 | 58.4 |
| `rpc` | 240 | 0.60 | 15 | 659.0 |

The shape of the ordering is itself an assumption, and it is the one worth
arguing about: catalogue reads (`list_models`, `get_skill`, `get_docs`) are
cheap because on the real system they are served from memory; `mcp_query` is
the data path and carries both a higher median and a wider `sigma`; `rpc` is an
order of magnitude worse because the recruitment stubs reach a different system
in production (`sim/emulator/rpc.py`), which is exactly why the agenda treats a
recruitment detour as expensive.

### 2.3 `degraded`

`ms_per_column = 1.8`, `ms_per_row = 0.35`.

| Endpoint | median ms | sigma | floor ms | p95 ms |
|---|---:|---:|---:|---:|
| `<default>` | 140 | 0.90 | 12 | 627.3 |
| `describe_model` | 110 | 0.80 | 10 | 420.1 |
| `find_skills` | 190 | 0.95 | 12 | 918.7 |
| `get_docs` | 80 | 0.70 | 10 | 263.0 |
| `get_skill` | 75 | 0.70 | 10 | 247.2 |
| `list_models` | 60 | 0.70 | 10 | 199.8 |
| `mcp_query` | 320 | 1.10 | 18 | 1972.4 |
| `overview` | 90 | 0.75 | 10 | 319.1 |
| `rpc` | 900 | 1.20 | 40 | 6519.5 |

---

## 3. Why `mcp_query` is additionally shape-sensitive

The corpus is a hybrid columnar store: ~800 of 4 599 catalogue columns are
materialised, the rest are computed on read (`docs/assumptions.md` A-1). Under
both storage regimes, a wide projection genuinely costs more than a narrow one —
more columns to touch, more values to compute, more JSON to serialise. If the
emulator ignored that, the corpus's central property would be unobservable and
the RQ2 hypothesis "project only what you need" would be untestable.

So `mcp_query` gets a shape term before the draw:

```python
shape_ms = n_columns * profile.ms_per_column + n_rows * profile.ms_per_row
base     = Dist(dist.median_ms + shape_ms, dist.sigma, dist.floor_ms)
```

Two design details that are easy to get wrong and were decided deliberately:

* **The shape term is added to the median, not to the result.** It is inside the
  log-normal, so a wide query is not "a narrow query plus a constant" — its
  whole distribution, tail included, scales. A wide query on a bad day is
  catastrophic, not merely late. Adding `shape_ms` after the draw would have
  produced a wide query whose *absolute* tail spread equalled a narrow one's,
  which is not how a loaded columnar scan behaves.
* **`columns: ["*"]` counts as 642 columns**, the width of the widest mart in
  the catalogue (`_count_columns`, `sim/emulator/app.py`). The agent's laziest
  possible request is charged as the most expensive one, which is the point.

Row count is charged from the *actual response body*, counted in the ASGI send
wrapper (`_count_rows`), not from the requested `limit`. A filter that matches
nothing is cheap even if it asked for 5 000 rows. This is also why latency is
injected **after** the handler runs rather than before: the delay depends on how
much data came back, and sleeping first would have to guess. The middleware
sleeps only `target_ms - already_spent_ms`, so real compute time counts toward
the budget instead of stacking on top of it.

### 3.1 Measured: narrow vs wide, per profile

2 000 draws per cell from `sim.latency.sample_ms` itself (varying `occurrence`,
which is the same knob that gives repeated calls their spread inside a session).
Reproduce with the snippet in §7.

| Profile | Shape | p50 ms | p95 ms | p95/p50 | mean ms | max ms |
|---|---|---:|---:|---:|---:|---:|
| `instant` | 4 cols × 100 rows | 0.0 | 0.0 | — | 0.0 | 0.0 |
| `instant` | 642 cols × 100 rows | 0.0 | 0.0 | — | 0.0 | 0.0 |
| `instant` | 642 cols × 2 741 rows | 0.0 | 0.0 | — | 0.0 | 0.0 |
| `realistic` | 4 cols × 100 rows | 89.4 | 220.3 | 2.47 | 104.6 | 477.8 |
| `realistic` | 642 cols × 100 rows | 440.6 | 1 080.8 | 2.45 | 511.9 | 4 566.8 |
| `realistic` | 642 cols × 2 741 rows | 667.4 | 1 655.2 | 2.48 | 786.3 | 4 430.8 |
| `degraded` | 4 cols × 100 rows | 394.3 | 2 307.6 | 5.85 | 684.4 | 9 727.6 |
| `degraded` | 642 cols × 100 rows | 1 624.8 | 9 027.2 | 5.56 | 2 801.4 | 78 352.6 |
| `degraded` | 642 cols × 2 741 rows | 2 502.2 | 14 464.5 | 5.78 | 4 566.7 | 142 300.0 |

What this buys the research programme, in numbers rather than adjectives:

* Under `realistic`, asking for `["*"]` instead of four columns on the same
  100 rows costs **4.9× at p50** (440.6 vs 89.4 ms) and **4.9× at p95**. That is
  a large enough effect that an agent trained to project narrowly will show a
  visible win, and small enough that a single extra round-trip (~90 ms) can
  still be the cheaper trade — which is the comparison RQ2 is actually about.
* Under `degraded`, the same laziness costs **4.1× at p50** and 3.9× at p95 —
  but in absolute terms the p95 gap is **6.7 seconds** (9 027 vs 2 308 ms),
  where under `realistic` it was 0.86 s. Sequential plans compound it: ten wide
  calls at p50 is ~16 s of pure waiting.
* Full-scan (`["*"]` with no limit, 2 741 rows) adds a further 1.5× over the
  wide-but-limited case under both profiles, so `limit` is separately visible
  from column choice. The two levers do not confound each other.

The empirical p50 tracks the closed form closely (`realistic` narrow: theory
`8 + (70 + 4·0.55 + 100·0.09) = 89.2` ms vs measured 89.4). The empirical p95
runs a few percent above the closed form because 2 000 hash-derived draws give a
p95 estimate with several percent of standard error; use the closed-form column
in §2 when an exact number matters, and this table when the question is what a
run will feel like.

---

## 4. A consequence worth stating: `degraded` can exceed the agent's timeout

The agent's HTTP client to Heimdall uses a 60 s timeout
(`sim/agent/tools.py`). Measured over 20 000 draws per cell:

| Profile | Shape | draws > 60 s |
|---|---|---:|
| `realistic` | all three shapes | 0 / 20 000 (0.000 %) |
| `degraded` | 4 cols × 100 rows | 0 / 20 000 (0.000 %) |
| `degraded` | 642 cols × 100 rows | 6 / 20 000 (0.030 %) |
| `degraded` | 642 cols × 2 741 rows | 36 / 20 000 (0.180 %) |

This is a property of the assumption set, not a bug: a heavy-tailed model with
`sigma = 1.10` on top of a 1.5 s effective median *will* occasionally produce a
minute-long call. It means that under `degraded`, a wide query carries a small
but non-zero probability of a tool timeout, and any RQ2 run on `degraded` must
report timeout counts alongside latency, or the reported p95 will be censored at
60 s without saying so. Under `realistic` the question does not arise.

---

## 5. Why `degraded` widens the tail instead of scaling the median

The tempting implementation is `degraded = realistic × k`. It is wrong for the
purpose, and the reason is the p95/p50 ratio.

Multiplying every median by `k` and leaving `sigma` alone produces a system that
is uniformly slower **with exactly the same shape**. Its p95/p50 ratio is
unchanged, because for a log-normal that ratio is `exp(1.645 · sigma)` and does
not depend on the median at all. But the p95/p50 ratio is precisely what hurts
an agent making ten sequential calls: with a stable ratio, an agent's plan
degrades predictably and every strategy degrades by the same factor, so the A/B
between "one aggregate call" and "N row-wise calls" would come out *identical*
under load and under calm. There would have been no reason to have a `degraded`
profile at all.

Real systems do not degrade that way. A loaded database degrades at the tail
first: the median moves somewhat (queueing), the spread moves a lot (contention,
GC pauses, cache eviction, retry storms). So `degraded` moves both, and moves
`sigma` harder:

| Endpoint | median ×  | sigma: realistic → degraded | p95/p50, realistic → degraded |
|---|---:|---|---|
| `list_models` | 2.7× | 0.35 → 0.70 | 1.63 → 2.85 |
| `describe_model` | 2.9× | 0.40 → 0.80 | 1.82 → 3.50 |
| `get_docs` | 2.9× | 0.35 → 0.70 | 1.66 → 2.92 |
| `overview` | 3.0× | 0.35 → 0.75 | 1.67 → 3.19 |
| `get_skill` | 3.0× | 0.35 → 0.70 | 1.65 → 2.91 |
| `find_skills` | 3.5× | 0.50 → 0.95 | 2.15 → 4.55 |
| `mcp_query` | 4.6× | 0.55 → 1.10 | 2.32 → 5.84 |
| `rpc` | 3.8× | 0.60 → 1.20 | 2.58 → 6.94 |

(Ratios are the published p95 over the published p50 = `floor + median`, so they
include the floor's damping effect; the pure log-normal ratios
`exp(1.645·sigma)` are slightly higher.)

Read this as: under load, the *median* gets about 3× worse across the board,
but the *tail relative to the median* roughly doubles, and it doubles most on
the two endpoints an agent leans on hardest — `mcp_query` and `rpc`. The
measured `mcp_query` ratios in §3.1 confirm it end-to-end: 2.47 under
`realistic`, 5.85 under `degraded`, for identical requests.

The practical consequence for RQ2 is the interesting one. Under `realistic`, a
plan's expected total is a decent predictor of its actual total, so minimising
call count is close to optimal. Under `degraded`, a plan with more calls has
more chances to draw from a fat tail, so the *variance* of the total grows
faster than the count — strategies that were near-equivalent on the mean
separate sharply on p95. A profile that only scaled the median could not have
shown that.

---

## 6. Determinism, and why it beats fresh noise

Latency is not drawn from an RNG. It is derived by hashing:

```
seed = "profile | endpoint | canonical(request) | occurrence"
Z    = Box-Muller over SHA-256(seed)
```

Consequences, and the reasons they were wanted:

* **No global RNG state.** The draw depends only on its own seed, so it is
  identical under uvicorn with one worker or four, and unaffected by how
  requests interleave. A shared `random.Random` would make latency depend on
  concurrency, i.e. on the machine, i.e. not reproducible.
* **Same condition + same call ⇒ same latency.** When a researcher edits the
  system prompt and re-runs a question, the latency delta must be attributable
  to the agent behaving differently, not to a different roll. Fresh noise would
  require averaging many runs to see an effect that determinism exposes in one —
  and on a 2 vCPU host, run budget is the binding constraint.
* **The request payload is canonicalised** (`json.dumps(sort_keys=True)`), so
  `{"model": m, "columns": c}` and `{"columns": c, "model": m}` are the same
  call. Key order is not a research variable and must not perturb a measurement.
* **`occurrence` is part of the seed**, so a retry loop does not get a
  suspiciously flat latency trace. The Nth identical call in a session differs
  from the first, and the *sequence* is still reproducible across runs.

Measured (`realistic`, `mcp_query`, 2 columns, 100 rows):

```
repeat same call, same occurrence : 165.068  165.068   identical: True
key order does not matter         : 165.068            identical: True
occurrence 0..4                   : [165.1, 97.9, 98.0, 147.8, 56.0]
different columns, occurrence 0   : 157.8
same request under degraded       : 1838.7
```

The trade this makes: a single (condition, question) pair yields one latency
sample, not a distribution, so per-question error bars have to come from
varying the question or the agent, not from re-running the same trace. That is
the right trade for an A/B design and the wrong one for characterising the
emulator itself; if you need the latter, sweep `occurrence` as §3.1 does.

Every injected value is also written to the span as `b2e.latency.injected_ms`
(`sim/telemetry.py`), separate from wall-clock duration, so an analysis can
always tell simulated delay from real compute.

---

## 7. Reproducing every number on this page

```bash
cd /home/mosyamac/b2e-synt-data
PYTHONPATH=. .venv/bin/python -c "
import json; from sim.latency import describe_profiles
print(json.dumps(describe_profiles(), indent=1))"
```

And the shape experiment in §3.1:

```python
from sim.latency import get_profile, sample_ms

N = 2000
req = {"model": "employee_actual", "columns": ["*"], "limit": 100}
draws = [sample_ms(get_profile("realistic"), "mcp_query", request=req,
                   occurrence=i, n_columns=642, n_rows=100) for i in range(N)]
draws.sort()
print(draws[N // 2], draws[int(0.95 * N)])
```

Against a running emulator: `curl -s localhost:8081/control/latency-profiles`.

---

## 8. These are assumptions, not measurements

**Nobody profiled the real Heimdall.** No production traces were available for
this work; ClickHouse is not even installed on this host (`docs/assumptions.md`
A-1). Every median, every `sigma`, every `floor_ms`, `ms_per_column` and
`ms_per_row` on this page was chosen to be *plausible and internally
consistent*, not to match an observation. Specifically, the following are
guesses:

1. That catalogue reads are ~20–40 ms and the data path is ~70 ms — i.e. the
   ratio between metadata and data, which sets how expensive `describe_model`
   caching looks.
2. That `sigma` for a healthy service is ~0.35–0.55. This single number decides
   every p95 in §2.2.
3. That the cost of a column is linear at 0.55 ms and the cost of a row is
   linear at 0.09 ms. A real columnar store is closer to linear in columns but
   often sublinear in rows after a fixed setup cost; we did not model that.
4. That `["*"]` should be charged as 642 columns rather than the actual width of
   the mart being queried.
5. That load triples the median and doubles the tail ratio (§5).
6. That `rpc` is ~3.4× the data path, standing in for a network hop to a
   different system.

**What is not an assumption:** the *ordering* — catalogue < data path < rpc,
narrow < wide, healthy < loaded — and the *mechanism* — right-skewed, positive,
tail-widening under load, shape-sensitive on the data path. Those are structural
claims that would survive recalibration.

**What a reader may conclude.** Relative results are safe: "strategy A is 2.3×
faster than strategy B at p95 under `realistic`" is a statement about agent
behaviour under a disclosed cost model, and it holds as long as the cost model
is monotone in call count and request width. Absolute results are not: "the
agent answers in 1.4 s" is a statement about our parameters, and must never be
quoted without the profile name and a pointer to this file.

**What would change if real numbers arrived.** Only the constants in
`sim/latency.py` — the `Dist(...)` literals in `_REALISTIC` and `_DEGRADED`, and
the two `ms_per_*` fields. No code path, no test, and no experiment design
depends on their values. Concretely:

* Measured per-endpoint p50 and p95 pin down the parameters directly:
  `floor_ms` from the minimum observed latency, then
  `median_ms = p50 − floor_ms` and
  `sigma = [ln(p95 − floor_ms) − ln(median_ms)] / 1.645`. Three observed numbers
  per endpoint are enough; nothing has to be fitted iteratively.
* A measured cost curve for column count would replace `ms_per_column` and, if
  the curve is not linear, would justify promoting `shape_ms` from two scalars
  to a callable on `Profile`. That is the one structural change real data could
  force, and it is confined to `sample_ms`.
* Real load-test data would replace the `_DEGRADED` block wholesale. If it
  turned out that the real system degrades by scaling the median with a stable
  tail ratio, §5's argument would be wrong and `degraded` would stop being a
  useful second condition — that is a falsifiable prediction of this design.
* Any recalibration must bump the profile constants **and regenerate this
  document**, because `describe_profiles()` is what the emulator serves at
  `/control/latency-profiles`, and a doc that disagrees with the served values
  is worse than no doc.
