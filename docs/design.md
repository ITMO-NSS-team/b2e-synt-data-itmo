# B2E Synthetic HR Data Engine — design

**Status:** approved for implementation (author's discretion, 2026-08-01)
**Target:** a Heimdall-shaped synthetic HR corpus at Sberbank scale (~300 000 people,
all 37 marts, 4 599 columns) good enough to run agent simulation experiments against.

---

## 1. Why a rewrite and not a patch

The predecessor engine (`skill-factory-hr-sandbox/hr_sandbox_generate`) got the hard
parts right: catalog-driven generation, latent factors, positional array alignment by
construction, deterministic snapshot ids, deliberate traps. It is reused here.

What it cannot do is carry a *simulation*. Measured on its own output (3 000 people,
seed 20260726) the following defects are not bugs to fix one by one — they are
consequences of one architectural choice: **each mart is filled independently, and
only a hand-listed set of columns is "known"**. Everything else is noise that happens
to have the right type.

| # | Defect | Evidence | Consequence for a B2E agent |
|---|---|---|---|
| W1 | Same person, different identity per mart | `employee_competence_actual.employee_full_name` agrees with `employee_actual` on **0/3000** rows; `employee_id` **0/3000** | Any join on anything but `person_id` silently produces a wrong person. The agent cannot detect it. |
| W2 | Intra-row contradiction | `employee_hist.employee_last_name` vs its own `employee_full_name`: **30/3000**; `birthday` there is filler → employees born in 2022 | Agent quoting a field it read verbatim looks like it hallucinated |
| W3 | Implicit array groups unaligned | lengths of `educational_institution_name` / `educational_speciality` / `education_type_name` agree on **4/3000** rows (explicit `educ.*` block: 3000/3000) | "Where did she study and in what field" has no correct answer |
| W4 | The ability signal never reaches the observable columns | corr(estimation mark, grade) = **0.035**, corr(mark, competence) = **−0.013**, while corr(age, grade) = **0.881** | Ranking questions are unanswerable from performance data; grade is a proxy for age |
| W5 | Nine competence columns are byte-identical per person | within-person spread **exactly 0.0** on all 3 000 | "Which competency is weakest" has no answer |
| W6 | Implausible marginals | 62.5% managers, 12.8% piled at the G18 clamp ceiling, `is_probation` constant 0, median time-in-role 0 yrs, 5.3% gen X and no boomers | Every aggregate the agent reports is obviously wrong to a domain reader |
| W7 | No referential integrity | manager / successor / predecessor / children names are invented strings; `successors.person_id` points at no row; org unit ids unrelated to employees | Team analysis — the core B2E use case — is impossible |
| W8 | Vocabulary too small and uniform | 144 surnames → **290 exact full-name collisions**; 13 universities, 10 courses, 10 specialties, all equiprobable | Name lookup is unrealistically ambiguous; category counts carry no signal |
| W9 | Placeholder text leaks into content columns | `"промежуточный institute_name 61"` in `candidate_*.institute_name`, `address`, `telegram`, `scientific_degree` | Agent must either quote garbage or silently drop fields |
| W10 | 19 of 37 marts are empty | whole `recruitment` schema, `recruitment_actual`, `evolution_item` content, `memai_qa`, … | Entire question classes unanswerable — cannot measure refusal vs hallucination |
| W11 | History is decorative | 1 540 events / 3 000 people, 58.8% with none, 2.5-year window; the base row *is* the current state, events do not fold into it | Trend, tenure and "what changed" questions have no ground truth |
| W12 | Ground truth is discarded | `attrition_risk_hidden` computed, never written anywhere, not even to a sidecar | Nothing to score agent answers against |
| W13 | Does not scale | per-person Python dicts, 41 KB/person on disk → 12 GB and far over RAM at 300 k | Cannot reach the target population |

## 2. Design principles

**P1 — One entity, many projections.** A single columnar `Population` is the sole
source of person facts. A mart is a *view*, never an independent generator. This
removes W1 and W2 by construction rather than by adding per-mart mappings.

**P2 — Resolver chain, not hand-listed mappings.** Every catalog column resolves
through an ordered chain; the first provider that claims the name wins:

```
1. mart-specific override      (position_actual.vacancy, …)
2. person attribute            (canonical name or alias: employee_full_name, grade_level …)
3. org attribute via unit_id   (block_name, oshs_level_7_unit_name, tb_name …)
4. declared array group        (educ.*, estimation.*, educational_*, career_* …)
5. semantic filler by name     (typed, dictionary-backed, plausible)
6. typed filler                (last resort, ClickHouse type only)
```

Adding a mart costs zero mapping code: it inherits every name it shares with the
population. A new alias fixes a whole class of columns across all 37 marts at once.

**P3 — Referential integrity is built, not asserted.** Order of construction:
org tree → positions → people → assignments → relations. A manager is the
`person_id` of a real employee who heads the unit. A successor is a real employee.
A candidate applies to a real requisition for a real position in a real unit.

**P4 — One clock.** `AS_OF = 2026-07-01`. Every date is derived from it. `age`
follows from `birthday`, tenure from `last_hire_date`, role tenure from
`position_start_date`. History is an event log that **folds to the current state**:
`replay(base, events) == employee_actual`. Checked by the validation gate.

**P5 — Factor model with distinct loadings.** Four latent factors
(`ability`, `potential`, `engagement`, `conscientiousness`) drive every observable
through explicit loadings plus per-column idiosyncratic noise. Competencies get
per-competency offsets, so they correlate ~0.6 with each other but differ within a
person. Performance marks are drawn from an ordinal model whose cutpoints move with
ability, so `corr(mark, ability) ≈ 0.55` instead of 0.03.

**P6 — Hybrid storage.** ~800 "semantic surface" columns are materialized in a
chunked columnar store. The remaining ~3 800 long-tail columns are **procedural**:
the value is `f(hash(seed, mart, column, row_index))`, produced on read, zero bytes on
disk. Deterministic across processes, so an API served from two workers agrees. This
is what makes 300 000 × 4 599 fit in ~1 GB and 3 GB of RAM.

**P7 — Truth is a sidecar, never a column.** Latent factors and gold labels
(attrition risk, readiness, nine-box cell, key-employee status, best-fit candidate per
requisition) live in `truth/`, are never exposed by the API, and exist solely to score
agent answers.

**P8 — Traps are declarative and switchable.** Every silent-failure trap
(uppercase Cyrillic, NULL-means-unscored, duplicated signal columns, raw region codes,
string-typed dates) is declared in `traps.yaml` with an on/off flag, so a hallucination
experiment can be run with and without them and the difference attributed.

## 3. Components

```
catalog/         snapshot.json — derived from the Heimdall OpenAPI (37 models, 4599 cols)
b2e/gen/rng.py         deterministic hash-based RNG: value = f(seed, ns, index)
b2e/gen/dicts.py       Russian vocabularies with realistic frequency weights
b2e/gen/names.py       name grammar: surname stems with m/f forms, Zipf sampling
b2e/gen/org.py         Sberbank-like org tree: 11 blocks → 12 TB → units → positions
b2e/gen/population.py  vectorized person core (numpy, chunked)
b2e/gen/blocks.py      array groups, aligned by construction
b2e/gen/resolve.py     the resolver chain (P2)
b2e/gen/marts.py       projections for all 37 marts
b2e/gen/history.py     event log + fold-to-current invariant
b2e/store/             chunked columnar writer + procedural reader
b2e/validate.py        consistency gate — fails the build, not a report
b2e/htmldoc.py         static HTML documentation of every mart and column
b2e/cli.py             build / validate / serve / doc
heimdall/              the API emulator, reused as-is (v1, v2, orion, REST, MCP bridge)
truth/                 hidden factors and gold labels (not served)
```

## 4. Stages

| Stage | Deliverable | Done when |
|---|---|---|
| S1 | Catalog from OpenAPI, project skeleton | 37/7/4599/148 reproduced |
| S2 | Org tree + population core, vectorized | 300 k people, plausible pyramids |
| S3 | Domain blocks: performance, goals, competencies, learning, absence, succession, career | aligned groups, factor loadings verified |
| S4 | Projection layer — all 37 marts non-empty | every mart has rows; no mart-local identity |
| S5 | Hybrid storage; 300 k build under RAM budget | build completes, ≤ 2 GB on disk |
| S6 | API emulator wired to the new store; HTML docs | `mcp_query` answers on every mart |
| S7 | Validation gate + realism report | all invariants green, marginals in range |
| S8 | Truth sidecar + research harness hooks | gold labels for 5 task families |

## 5. What this unlocks for the B2E research questions

The simulator is not the research; it is the instrument. Mapping:

- **(1) Hallucination reduction.** Needs traps that are switchable (P8), ground truth
  to score against (P7), and *unanswerable* questions — hence the deliberately empty
  marts (W10 becomes a feature once it is declared rather than accidental). Refusal is
  measurable only when the corpus is known to lack the answer.
- **(2) Latency.** Needs a cost model that is honest: the columnar store makes wide
  selects genuinely more expensive than narrow ones, so "fewer, better-shaped API
  calls" shows up as measured time, not as a proxy metric.
- **(3) Memory.** Needs multi-system identity: the same person appears under different
  keys in different marts (`person_id`, `emp_key`, `tn`, external `id` in
  `talent_radar_people`). Whether cross-system carryover helps or poisons context is
  testable only if the same human genuinely exists in several places with partially
  inconsistent attributes — which is now modelled deliberately instead of accidentally.
- **(4) Fleet-level reflection.** Needs many distinguishable employees with stable
  attributes over time so that sessions are comparable, and a population large enough
  that skill reuse across users is a real question (300 k, ~30 k managers).

## 6. Explicit non-goals

- Not a Heimdall reimplementation beyond the query contract already emulated.
- No real personal data of any kind; every vocabulary is hand-authored.
- The B2E agent itself is out of scope — this repository ships only the instrument.
