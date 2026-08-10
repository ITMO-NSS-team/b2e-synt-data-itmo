# RQ4 run `rq4-2026-08-09` — results

One complete run of the nightly-reflection experiment: three arms, nine epochs,
270 questions each, 810 agent turns, finished 2026-08-10 05:40 UTC.

These files live here rather than under `var/` because `var/` is gitignored — it
holds rebuildable things (corpora, databases, scratch). A run's results are not
rebuildable: re-running the experiment does not reproduce them, because the
agent is not deterministic on this harness. They are evidence, so they are
committed.

| file | what it is |
|---|---|
| `turns.jsonl` | 810 rows, one per attempted turn. The primary record; everything else is derived from it. Schema is documented in `scripts/run_rq4.py` (`TURNS_JSONL_SCHEMA`). |
| `summary.md` / `summary.json` | the generated report — health, guards, primary contrast, learning curve, cost |
| `memory-dynamics.md` | how each arm's memory changed epoch to epoch: item counts, and the added/removed lesson text per epoch |
| `memory-artifacts.json` | every version 1→8 of all four memory packs, with per-epoch diffs, plus the 270-question run design |

## Arms

| arm | memory | reflections per epoch |
|---|---|---|
| A1 | none | 0 |
| A2 | private per instance | 3 — one per instance, over its own 10 episodes |
| A3 | one pack shared by all three instances | 1 — over all 30 pooled episodes |

All three arms answered the **same** 270 questions with the same bindings in the
same epochs, so the comparison is paired per question and A1 calibrates epoch
difficulty.

## Headline

Pooled over the eight post-treatment epochs, paired against A1:

| contrast | difference | 95% CI | p | n pairs |
|---|---:|---|---:|---:|
| A2 − A1 (isolated memory) | +3.90pp | [−0.9, +8.7] | 0.151 | 231 |
| A3 − A1 (shared memory) | +0.43pp | [−3.9, +4.3] | 1.000 | 233 |
| A3 − A2 (the sharing test) | −3.91pp | [−9.1, +1.3] | 0.202 | 230 |

Pass rate, epochs 1–8: **A1 31.8%, A2 36.1%, A3 32.2%**.

Nothing clears significance. The predicted ordering was A3 > A2 > A1; observed is
A2 > A3 ≈ A1, with the sharing contrast pointing negative. Read as: no evidence
that shared reflection beats isolated reflection, and a weak non-significant hint
that isolated reflection beats none.

## Caveats that belong with any citation of these numbers

* **One replication, so one memory trajectory per arm.** The intervals describe
  which 270 questions were drawn and how they fell on the epoch grid — not
  run-to-run variance in what reflection writes. A second seed could curate
  different lessons and move the curve for reasons this analysis cannot see.
* **The design resolves about ±5pp** at n≈230 paired questions. Effects smaller
  than that are invisible here regardless of whether they are real.
* **Identity model.** The 180 answerable questions are asked under a shared HR
  identity, not three distinct employees — Heimdall's row-level scoping makes
  one-identity-per-instance structurally impossible for org-wide questions. So
  A3 tested pooling across *memory states*, not across users with different data
  visibility. See `docs/rq4-methodology.md` §4.
* **Spans were not collected** (`--skip-spans`): Phoenix took ~20s per query and
  degraded as the run grew. Correctness, tokens, Heimdall call counts and wall
  time all come from the agent's own stats; `api_validity` therefore sits at its
  neutral value and `memory_tokens` is 0 in the report — an artefact of the
  export path, not evidence that memory was absent. The registry shows the packs
  were pinned and non-empty throughout.
* **`summary.md`'s learning-curve slope includes epoch 0**, where no arm had
  memory. The post-treatment slopes are +0.48pp/epoch (A2) and +0.43 (A3).

Methodology: `docs/rq4-methodology.md`. Design: `docs/superpowers/specs/2026-08-09-rq4-reflection-memory-design.md`.
