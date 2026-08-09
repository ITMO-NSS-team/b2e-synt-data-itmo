# RQ4 Driver and Analysis Implementation Plan (Plan C)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Build the seeded question schedule, the experiment driver that runs three arms through nine epochs, and the paired analysis that turns the results into a report.

**Architecture:** One agent container serves all three logical instances. An instance is a `(employee_id, config_ref)` pair — `POST /sessions` accepts both per session, and `config_ref` pins the `agent_config` version that carries that instance's `memory_ref`. No per-instance processes, no per-instance Phoenix projects: the driver knows which session belongs to which instance because it created it, and it stamps `metadata` with arm, epoch, instance and question id.

**Depends on:** Plan A (merged, `sim/oracle/qgen.py`, `sim/oracle/reference.py`, `sim/research/answer.py`, `evaluate.py`, `judge.py`) and Plan B (merged, `sim/reflection/*`). Suite green at 670.

## Global Constraints

- Python ≥ 3.10, `from __future__ import annotations`, tests via `B2E_LLM_MODE=replay PYTHONPATH=. .venv/bin/python -m pytest tests -q`, numpy only in `.venv`.
- Frozen dataclasses with `slots=True`; docstrings explain why; no new dependency.
- **The schedule is drawn once per replication and replayed byte-identically in every arm.** At every epoch all three arms have answered exactly the same 30 questions with the same bindings and the same acting employees. This is what makes the comparison paired and what makes arm A1 the epoch-difficulty calibrator.
- **Entity pools are disjoint across epochs**, so a memorised fact about one person cannot help a later question.
- Corpora and scratch go in `.pytest-*` or `.analysis-*` directories — both gitignored.

---

### Task 1: The seeded schedule

**Files:** create `sim/oracle/schedule.py`, `tests/test_oracle_schedule.py`.

**Produces:**
- `ScheduledItem(question_id, text, category, family, question_class, reference_spec, canaries, epoch, instance, pair_id)` — frozen.
- `build_schedule(gold, *, seed, replication) -> tuple[ScheduledItem, ...]` — exactly 270 items: 180 from `qgen.generate` plus 90 caution items drawn from `sim.oracle.basket` (18 each of `out_of_scope`, `ambiguous`, `no_data`, `access_control`, `prompt_injection`), slots bound from the snapshot.
- `commit_design(registry, items, *, actor) -> str` — the `run_design` artefact, so a run is reproducible and re-scorable without re-running the agent.

**Required properties, each its own test:**
- exactly 270 items, 9 epochs × 30, 10 per instance per epoch;
- each epoch's 30 carry the same family/category mix (stratified draw), so an epoch cannot be accidentally easy;
- every question appears exactly once — sampling without replacement is the anti-leakage guarantee, so assert it rather than assume it;
- `pair_id` is unique within a replication and identical across arms, since the paired test keys on it;
- entity pools disjoint across epochs — assert no `person:` or `unit:` entity recurs in two epochs;
- no unbound `{` survives in any dispatched text;
- deterministic in `(seed, replication)`.

Caution items need their canaries: use `sim.oracle.basket.injection_canaries(question)`. Recall `sim/research/evaluate.py` now **raises** on an empty canary tuple for a `prompt_injection` item, so an item whose `gold_ref` has no canary mapping must be filtered out at schedule time rather than exploding mid-run.

---

### Task 2: The driver

**Files:** create `scripts/run_rq4.py`, `tests/test_run_rq4.py`.

**Shape of a run.** For each replication, for each epoch 0..8:
1. For each arm in a seeded random order, for each instance, ask that instance's 10 questions for this epoch via `POST /sessions` then `POST /sessions/{id}/messages`, with `metadata = {arm, epoch, instance, replication, question_id, pair_id}`.
2. Score each answer immediately: `sim.research.answer.parse_answer`, `sim.research.evaluate.score_correctness` against `reference.evaluate`, and `turn_facts` from the session's Phoenix spans for `api_validity` / `efficiency`. Append one JSON line per turn to `var/rq4/<run_id>/turns.jsonl`.
3. After the epoch closes for an arm:
   - **A1** does nothing.
   - **A2** runs reflection three times, once per instance over that instance's own 10 episodes, writing `memory_A2_i{1,2,3}`.
   - **A3** runs reflection **once** over all 30 pooled episodes, writing `memory_A3_fleet`, read by all three instances next epoch. Use `sim.reflection.reflect.merge_shared` on the mechanical facts *before* the curator call, so the model never sees three copies of one API error and cannot triple-count it.
   - Commit a new `agent_config_rq4_<arm>_<instance>` version pinning the new `memory_ref`, and use that `config_ref` for the next epoch's sessions.

**Non-negotiable operational properties** — this runs unattended for hours:
- **Resumable.** A checkpoint file records every completed `(replication, arm, epoch, instance, question_id)`. On restart the driver skips them. A crash at hour four must not restart from zero.
- **Rate-limit tolerant.** A Max subscription enforces rolling-window limits and this run makes tens of thousands of model calls. On a 429 or a CLI failure, back off exponentially with jitter and retry; after N failures record the turn as `dispatch_failed` and continue rather than aborting the arm. A turn that never ran is `scored=false` with a reason, which the report already handles — an aborted arm is not.
- **Arms interleaved, never run arm-by-arm**, so model or CLI drift over a long run is balanced across arms instead of confounded with them.
- **Bounded concurrency**, default 3, `--concurrency` to change it. The host has 3.8 GiB and no swap while the emulator alone is capped at 5g, so more will thrash.
- `--replications` (default 1), `--epochs` (default 9), `--dry-run` that walks the whole schedule without calling the model, so the wiring can be proven for free.

**Tests** run against a fake agent endpoint, never a live model: resume skips completed work; a 429 retries then continues; arms interleave; the schedule replays identically across arms; a failed turn is recorded rather than aborting.

---

### Task 3: The report

**Files:** create `sim/research/report.py`, `tests/test_research_report.py`.

Reads `turns.jsonl` and emits both JSON and a readable Markdown summary:

- **Primary:** `pass_rate` per arm at the final epoch, and the paired difference `A2 − A1` and `A3 − A1` over `pair_id`, tested by permutation (10 000 draws, two-sided) with a bootstrap CI. Report percentage points, not only a p-value.
- **Learning curve:** per epoch, `A2 − A1` and `A3 − A1`. A1 removes epoch difficulty because all three arms answered the same questions that epoch. The hypothesis "A3 learns faster" is the slope of that difference across epochs.
- **Guard, printed before any headline:** caution-category pass rate per arm, `followed_injection_rate`, `fabricated_id_rate`. An arm that gains on the 180 deterministic questions while losing the 90 caution ones has traded caution for accuracy, and the report must say so rather than reporting a win.
- **Cost:** tokens per correct answer, seconds per correct answer (wall-clock and modelled), Heimdall calls per answer, and memory size in tokens per epoch.
- **Health, printed first:** `scored=false` rate per arm with reasons, and `dispatch_failed` counts. Every rate downstream depends on the denominator, and a differential dropout between arms invalidates the comparison before any quality number is read.

State the honest limit in the output itself: with one replication there is one memory trajectory per arm, so the result is a described difference with a confidence interval, not a significance claim about reflection in general.
