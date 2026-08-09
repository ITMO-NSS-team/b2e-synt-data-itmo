# RQ4 Memory and Reflection Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the long-term memory artefact, deliver it into the agent's system prompt, and implement the nightly reflection that writes it — both the isolated per-instance form and the shared cross-instance form.

**Architecture:** Memory is a registry artefact (`kind="memory"`), not a skill and not a prompt edit, so `sim/skills.py`'s human approval gate is untouched. One reflection procedure serves both arms; only its input differs. Everything before the single LLM call is deterministic code — span extraction, a pre-aggregation table, value-free episode cards — so the model proposes lesson text but never owns an evidence counter. Five mechanical guard rules reject any lesson that could carry an answer, a name, or a question paraphrase.

**Tech Stack:** Python 3.10+, stdlib + numpy. Reflection's LLM call goes through the `claude -p` CLI, because `ANTHROPIC_API_KEY` is empty in this deployment and the OAuth token 403s on `/v1/messages` (`docs/assumptions.md` A-8).

**Spec:** `docs/superpowers/specs/2026-08-09-rq4-reflection-memory-design.md` §4, §5.

**Depends on (already merged to main):** Plan A — `sim/oracle/reference.py`, `sim/oracle/qgen.py`, `sim/research/answer.py`, `sim/research/evaluate.py`, `sim/research/judge.py`. Suite green at 582.

**Followed by:** Plan C — seeded schedule, three-instance driver, paired analysis.

## Global Constraints

- Python ≥ 3.10. Every module starts `from __future__ import annotations`.
- Tests run as `B2E_LLM_MODE=replay PYTHONPATH=. .venv/bin/python -m pytest tests -q` from the repo root. numpy is only in `.venv`; never invoke bare `python3`.
- Agent-facing text (memory lesson text, prompt fragments) is **Russian**. Code comments are English.
- Frozen dataclasses with `slots=True` for every value type.
- **No memory record may contain a person id, a surname, a unit name, a question paraphrase, or a numeric literal absent from the evidence.** This is enforced by `sim/reflection/guard.py` and is the property that keeps the shared arm from being an answer-transport channel.
- The reflection worker must never commit a registry name matching `^agent_config` or `^system_prompt`, and `is_human_action=True` must continue to appear only in `sim/admin/app.py`. Both are asserted by tests.
- Docstrings explain *why*, not *what*. `sim/oracle/labels.py` is the style reference.
- No new runtime dependency in `pyproject.toml`.
- Memory pack hard caps: `K_MAX = 24` records, `T_MAX = 1200` tokens rendered, **≥300 tokens reserved for `pitfall`**. Token estimation uses `sim.agent.loop.estimate_tokens` (the repo's own, `CHARS_PER_TOKEN_ESTIMATE = 3.2`), never a second opinion.

---

### Task 1: Memory record, pack, budget and renderer

The artefact itself. Pure functions plus a registry read/write; no spans, no LLM.

**Files:**
- Create: `sim/reflection/__init__.py`, `sim/reflection/memory.py`
- Test: `tests/test_reflection_memory.py`

**Interfaces:**
- Consumes: `sim.registry.Registry` (`commit`, `load`, `head`), `sim.agent.loop.estimate_tokens`.
- Produces:
  - `MemoryItem(id, scope, kind, trigger, text, support, refute, episodes_support, episodes_refute, created_epoch, last_useful_epoch, epochs_idle, origin_instances, tokens)` — frozen, `slots=True`. `kind ∈ {"api_mechanic", "method", "pitfall"}`.
  - `MemoryPack(arm, scope, epoch, items, rendered, digest, parent_digest)` — frozen, `slots=True`.
  - `score(item: MemoryItem) -> float` — Laplace `(support+1)/(support+refute+2)`, rounded 4dp.
  - `select(items, *, k_max=K_MAX, t_max=T_MAX, pitfall_reserve=PITFALL_RESERVE) -> list[MemoryItem]`
  - `render(items) -> str`
  - `build_pack(arm, scope, epoch, items, parent_digest) -> MemoryPack`
  - `commit_pack(registry, pack, *, actor) -> str` — returns the pinned ref `name@N`; asserts the name is neither `agent_config*` nor `system_prompt*`.
  - `load_pack(registry, ref) -> MemoryPack`
  - `K_MAX`, `T_MAX`, `PITFALL_RESERVE`, `IDLE_RETIRE = 3`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reflection_memory.py`. Cover, each with a distinct assertion:

```python
def test_score_is_laplace_so_one_lucky_hit_cannot_outrank_nine_of_ten():
    lucky = _item(support=1, refute=0)
    solid = _item(support=9, refute=1)
    assert memory.score(solid) > memory.score(lucky)

def test_select_reserves_capacity_for_pitfalls_even_when_they_score_lower():
    # 30 high-scoring api_mechanic items and 2 low-scoring pitfalls.
    # The pitfalls must still appear: a pack that can only say "do more" is a
    # refusal suppressor, which is the failure the reserve exists to prevent.

def test_select_never_exceeds_the_hard_caps():
    # 100 items, assert len(chosen) <= K_MAX and estimate_tokens(render(chosen)) <= T_MAX

def test_render_is_deterministic_for_the_same_items():
    # same items, two calls, byte-identical

def test_render_contains_no_counters_or_identifiers():
    # the rendered block carries lesson text only — no support counts, no ids

def test_commit_pack_refuses_to_write_an_agent_config_name():
    with pytest.raises(AssertionError):
        memory.commit_pack(reg, _pack(arm="agent_config_rq4"), actor="t")

def test_commit_pack_is_idempotent_for_identical_content():
    # registry.commit returns the existing head unchanged, so a night that
    # learned nothing must not bump the version — otherwise the loop
    # manufactures a condition change out of a no-op
    ref1 = memory.commit_pack(reg, pack, actor="t")
    ref2 = memory.commit_pack(reg, pack, actor="t")
    assert ref1 == ref2

def test_load_pack_round_trips_every_field():
```

- [ ] **Step 2: Run the test, watch it fail** — `ModuleNotFoundError: sim.reflection`
- [ ] **Step 3: Implement `sim/reflection/memory.py`.**

Key decisions to honour, each with the reason in a docstring:
- `select` is a tiered greedy pass: reserve `PITFALL_RESERVE` tokens for `pitfall` items first, then fill the remainder by descending `score`, breaking ties by `id` so a rerun reproduces the pack.
- `render` is a deterministic f-string with three Russian sections — `### Чего не делать`, `### Приёмы`, `### Заметки о витринах` — and a fixed closing line stating these are observations about how to work with the API, not about what any answer should be. Never an LLM.
- `commit_pack` opens with `assert not re.match(r"^(agent_config|system_prompt)", name)`.

- [ ] **Step 4: Run the tests, all pass**
- [ ] **Step 5: Commit** — `feat(reflection): память как артефакт реестра, а не как скилл и не как правка промпта`

---

### Task 2: Deliver memory into the agent's prompt

Four small edits across three files. Highest blast radius in the plan: `sim/agent/prompt.py` and `sim/agent/app.py` are load-bearing, and the suite must stay green at 582.

**Files:**
- Modify: `sim/agent/config.py`, `sim/agent/app.py`, `sim/telemetry.py`
- Test: `tests/test_reflection_delivery.py`

**Interfaces:**
- `AgentConfig` gains `memory_ref: str = ""`; `MEMORY_STRATEGIES` gains `"reflected"`.
- `_memory_block` changes signature from `(config, session)` to `(state, config, session)`; both call sites (`sim/agent/app.py:356` and `:718`) pass `state`, which is already in scope at both.
- `sim/telemetry.py` `start_run` gains flat root-span attributes `b2e.memory.ref`, `b2e.memory.digest`, `b2e.memory.epoch`, `b2e.memory.tokens`, `b2e.run.arm`, `b2e.run.epoch`, `b2e.run.replication`, `b2e.role` — following the `b2e.code_execution` precedent at `sim/agent/claude_code.py:1771`.

- [ ] **Step 1: Write the failing test.** Assert: `AgentConfig(memory_strategy="reflected")` with an empty `memory_ref` raises; `memory_ref` set while strategy is not `reflected` raises; a `memory_ref` not of the pinned `name@N` form raises (a floating head would let the pack change under a running experiment); `_memory_block` with `reflected` returns the pack's exact `rendered` string; `render(DEFAULT_SYSTEM_PROMPT, ...)` still succeeds with and without a memory block; existing configs stored without the new field still load (`from_dict` rejects only unknown keys).
- [ ] **Step 2: Run it, watch it fail**
- [ ] **Step 3: Implement.** Add the three `__post_init__` refusals in the established style — the module's own comment says a declared-but-inert condition is worse than a crash.
- [ ] **Step 4: Run the FULL suite** — `B2E_LLM_MODE=replay PYTHONPATH=. .venv/bin/python -m pytest tests -q`. It was 582; it must be ≥582 with no failures. If an existing test asserts a prompt property, update the expectation and say so explicitly in the report with old and new values.
- [ ] **Step 5: Commit** — `feat(agent): память доезжает до промпта закреплённой версией, а не плавающей головой`

---

### Task 3: Span extraction and the privacy boundary

Turn one turn's Phoenix spans into value-free call records. `query_shape()` is simultaneously the anti-memorisation boundary and the PII boundary; it is the single most important function in Plan B.

**Files:**
- Create: `sim/reflection/extract.py`
- Test: `tests/test_reflection_extract.py`

**Interfaces:**
- Consumes: `sim.traceview` (`normalise_rest_span`, `unflatten`, `attr`, `enrich`), `sim.research.phoenix_client.PhoenixClient.spans_for_session`.
- Produces:
  - `CallRecord(seq, tool, schema, logic_model, columns, filter_nodes, order_by, limit_bucket, http_status, error_code, rows, argument_keys)` — frozen, `slots=True`.
  - `query_shape(params: dict) -> dict` — value-free.
  - `call_records(spans: list[dict]) -> list[CallRecord]`
  - `turn_facts(spans) -> TraceFacts` — the Plan A type, populated from real spans, including `memory_tokens`.

- [ ] **Step 1: Write the failing test.** The critical assertions:

```python
def test_query_shape_discards_every_filter_value():
    """Person ids, surnames and unit names live only in filter values and in
    output.value. If they survive this function, a shared memory becomes a
    channel for one employee's data to reach another's agent."""
    params = {"schema": "dm_core", "logic_model": "employee_actual",
              "columns": ["grade_level"],
              "filters": {"node": "and", "args": [
                  {"node": "condition", "column": "person_id", "op": "eq",
                   "value": "8f14e45f-ceea-467a-9d0f-2b4c3a1e5b77"},
                  {"node": "condition_like", "column": "employee_full_name",
                   "pattern": "Иванов%"}]}}
    shape = extract.query_shape(params)
    blob = json.dumps(shape, ensure_ascii=False)
    assert "8f14e45f" not in blob and "Иванов" not in blob
    assert ("condition", "person_id", "eq") in [tuple(n) for n in shape["filter_nodes"]]

def test_query_shape_walks_the_whole_filter_ast_not_a_flat_dict():
    """The filter body is a nine-node tree (heimdall/engine/filters.py:30-52).
    An implementation iterating dict.items() silently loses nested nodes, and a
    lost node is a lesson keyed on a shape that never occurred."""
```

Plus: `limit_bucket` maps to `none|1-10|11-100|101-1000|>1000`; `call_records` pairs each TOOL span with at most one child Heimdall CHAIN span; a TOOL span whose child is unmatched keeps its params but reports `http_status=None`.

- [ ] **Step 2: Run it, watch it fail**
- [ ] **Step 3: Implement.** `walk()` recurses `condition, condition_in, condition_like, condition_null, condition_array, condition_param, and, or, not`, emitting `(node_type, column, operator)` per leaf and discarding every `value`, `pattern`, `args`, `expr`. Read the real node list from `heimdall/engine/filters.py` rather than trusting this plan.
- [ ] **Step 4: Add a property test** over a corpus of real spans: assert no `person_id` from `truth/people.json` and no surname from the snapshot survives `query_shape` for any recorded call. Build the corpus in a `.pytest-*` directory — that pattern is gitignored.
- [ ] **Step 5: Run tests, commit** — `feat(reflection): форма запроса без значений — одна функция, две границы`

---

### Task 4: Pre-aggregation, episode cards, and the guard

What the LLM is allowed to see, and what it is allowed to say.

**Files:**
- Create: `sim/reflection/aggregate.py`, `sim/reflection/guard.py`
- Test: `tests/test_reflection_aggregate.py`, `tests/test_reflection_guard.py`

**Interfaces:**
- Produces:
  - `EpisodeCard(question_class, family, category, plan, api_errors, verdict, failure_class, calls, tokens, seconds)` — frozen. **No answer text, no gold value, ever.**
  - `aggregate(episodes) -> dict` — per-class pass rate and cost; API error histogram keyed `(endpoint, http_status, error_code, argument_keys)` each with the fix diff to the next succeeding call of the same shape; waste counts (duplicate calls, pagination walks, over-fetch); and the memory audit — for each existing item, which episodes its trigger matched, whether the plan followed it, and how those episodes scored.
  - `check(text, *, evidence, basket_texts, held_out_texts) -> str | None` — returns the violated rule id or `None`.

- [ ] **Step 1: Write the failing tests.** For `guard.check`, one test per rule, each asserting a realistic violation is caught AND a legitimate lesson passes:

| Rule | Rejects |
|---|---|
| G1 | any UUID, any surname from the snapshot's 474 forms, any org unit name |
| G2 | a numeric literal with ≥3 significant digits absent from the evidence object (1–99 allowed, so «не более 2 запросов» survives) |
| G3 | >240 characters or >2 sentences |
| G4 | word 5-gram Jaccard ≥ 0.30 against any basket question text |
| G5 | an imperative naming a tool permission, a refusal policy, or the evaluator («эталон», «gold», «правильный ответ») |

For `aggregate`, assert that an `EpisodeCard` never carries answer text or a gold value — serialise the card and assert the gold value's string form is absent.

- [ ] **Step 2: Run, watch fail. Step 3: Implement. Step 4: Run, pass.**
- [ ] **Step 5: Commit** — `feat(reflection): модель предлагает формулировку, счётчики ведёт код`

---

### Task 5: The reflection procedure, isolated and shared

One procedure, two inputs. Everything after the merge is byte-identical between the arms.

**Files:**
- Create: `sim/reflection/curate.py`, `sim/reflection/reflect.py`
- Test: `tests/test_reflection_reflect.py`

**Interfaces:**
- `curate(cards, table, memory, *, client) -> list[Op]` where `Op` is `ADD(kind, trigger, text)` / `REVISE(id, text)` / `DROP(id, reason)`, at most 8. `support` and `refute` are **not** in the output schema — the model cannot write its own evidence.
- `reflect(episodes, prior_items, *, arm, scope, epoch, client, registry) -> MemoryPack`
- `merge_shared(per_instance_candidates) -> list[MemoryItem]` — exact-key merge by `sha256(canonical_bytes({miner_key}))`, set-union of episode ids, cross-instance rank bonus for items whose evidence spans ≥2 instances.

- [ ] **Step 1: Write the failing test.** The load-bearing assertions:

```python
def test_the_model_cannot_write_its_own_evidence():
    """A curator that both discovers a pattern and counts it will assert
    'this fails often' from two episodes. Counters are owned by code."""
    ops = curate.parse_ops('[{"op":"ADD","kind":"method","trigger":{},'
                           '"text":"x","support":99}]')
    assert all(not hasattr(o, "support") for o in ops)

def test_shared_merge_unions_episodes_rather_than_summing_them():
    """Three instances that saw the SAME episode must count once. Summing is
    the 3x-data confound arriving through the counter."""

def test_shared_and_isolated_differ_only_in_input():
    """Same episodes, same prior memory, same fake client: reflect(scope='fleet')
    and reflect(scope='i1') must produce identical packs. If they differ, the
    shared arm has a second treatment nobody declared."""

def test_a_night_that_learns_nothing_does_not_bump_the_version():
def test_reflection_refuses_to_run_mid_epoch():
def test_guard_rejections_are_recorded_not_silently_dropped():
```

- [ ] **Step 2: Run, watch fail.**
- [ ] **Step 3: Implement.** The LLM call goes through `claude -p --output-format json` with `--disallowed-tools` covering everything — reflection has no business reading files or calling Heimdall. Cache responses by content hash so re-deriving an epoch is cheap. Take the CLI path from `sim.agent.claude_code` rather than reinventing it, and read the credential the same way `child_env` does.
- [ ] **Step 4: Run the full suite.**
- [ ] **Step 5: Commit** — `feat(reflection): ночная рефлексия — изолированная и общая различаются только входом`

---

## Self-review

**Spec coverage.** §4's record schema, storage, delivery and token budget → Tasks 1–2. §5's six steps → Tasks 3–5: collect and extract (3), pre-aggregate and cards (4), curate and guard (4–5), counters/decay/budget/commit (1, 5). The isolated/shared difference table → Task 5's `merge_shared` and its equality test.

**Deferred to Plan C, deliberately:** the barrier that waits for all three instances (it belongs to the driver, which owns epoch boundaries), and populating `memory_tokens` on the root span (the driver knows which pack a turn used).

**Type consistency.** `MemoryItem` is produced by Task 1 and consumed by Tasks 4 and 5. `CallRecord` and `TraceFacts` cross Task 3 → Task 4. `EpisodeCard` crosses Task 4 → Task 5. `TraceFacts` is Plan A's type, already merged, and Task 3 populates it rather than redefining it.

**No placeholders.** Where this plan gives a spec rather than code — `aggregate`, `curate`, `merge_shared` — it is because the shape depends on what the real spans contain, and five of six Plan A briefs contained a defect precisely where I guessed at data I had not looked at. Implementers are instructed to read the real span schema first.
