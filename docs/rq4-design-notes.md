# RQ4 — fleet reflection: design notes

**Status: design sketch. Nothing here is scheduled and nothing here is
implemented.** No reflection code exists in this repository, none is proposed,
and this document does not ask for any. Its job is narrower and, for the moment,
more useful: to write down what substrate the environment *already* provides for
RQ4, which shapes of reflection loop that substrate can and cannot support, and
which of the obvious designs are wrong in ways that are easy to discover late and
expensive to unwind.

RQ4, restated from `docs/research-agenda.md` §4: many employees each run their own
B2E instance and leave per-session feedback; is there an optimal background
process that mints new skills, rewrites system prompts, or proposes new Heimdall
endpoints from that traffic? The agenda already sets out the five-step loop —
cluster traces by intent, turn systematically disliked clusters into skill
candidates, turn repeated request sequences into predefined-code candidates, turn
repeated expensive requests into API-endpoint candidates, turn systematic
request-shape errors into prompt edits — and it also states the open problem that
makes the whole thing hard: **user feedback is noisy and biased, a like
correlates with confident tone rather than with correctness, and that is why
`truth/` exists as a separate channel.** Everything below is downstream of that
one sentence.

One conclusion is not negotiable and is stated here rather than buried in §7: a
reflection loop that mints skills feeds directly into the lifecycle in
`docs/skill-execution-threat-model.md` §3, agent-authored artefacts always enter
as `draft`, and promotion to `approved` requires an explicit human action. **A
reflection loop that auto-approves is a defect, not a feature.** It is not a
performance optimisation with a safety cost to be traded off; it deletes the only
boundary the threat model has.

---

## 1. What the substrate already gives

RQ4 is unusually well served by work that was done for other reasons. Almost
everything a reflection loop would need to *read* exists, is versioned, and is
already load-bearing for RQ1–RQ3. What does not exist is anything that would
*write*, which is the correct asymmetry to be in.

### 1.1 The span tree is a machine-readable plan, not just a log

`docs/span-schema.md` §1 fixes one tree shape per turn: an `AGENT` root, one
`CHAIN` per reasoning iteration, `LLM` calls, `TOOL` calls, and — nested under
the tool that caused them — one `CHAIN` per Heimdall HTTP call. That nesting was
chosen so the tool-call-to-HTTP-call ratio stays measurable for RQ2. For RQ4 it
does something else: it makes the **executed query plan** a first-class object.
`b2e.heimdall.endpoint`, `b2e.http.status`, `b2e.heimdall.error_code`,
`b2e.heimdall.rows`, `b2e.heimdall.columns_requested` and
`b2e.latency.injected_ms` on each leaf, in tree order, are precisely the thing a
"repeated sequence of requests" candidate is made of. No separate mining schema
is needed to see that ten thousand sessions all issued the same four queries in
the same order; it is a group-by over data that is already exported without
sampling.

`sim/telemetry.py` guarantees this is complete rather than best-effort:
instrumentation is explicit (`auto_instrument=False`, with the reason given —
implicit instrumentation that silently stops matching a library version leaves a
trace that *looks* complete), and there is no sampling at all.

### 1.2 The fingerprint makes "same conditions" checkable

`sim/fingerprint.py` refuses to open a root span without all eight
`b2e.run.*` fields, and derives `condition_id` as a hash over them. This is the
single most important thing the environment gives RQ4, and it is easy to
undervalue because it was built for RQ1.

Any statement of the form "cluster C gets systematic dislikes" is meaningless
across a mixed set of conditions. If half the traces in C ran with
`traps_enabled=true` and half did not, or with two different
`prompt_registry_version` values, the cluster is a mixture and the "systematic"
signal is a between-condition difference wearing the costume of a within-cluster
one. `condition_id` turns "are these traces comparable?" from a judgement call
into an equality test. Any mining pass that does not stratify by it is producing
artefacts, and the fingerprint is what lets a reviewer check that it did.

Note what the fingerprint does *not* contain: the acting employee. `session.id`
and `user.id` are separate span attributes (the real OpenInference keys, which is
what makes Phoenix group sessions natively). That is exactly the right split for
RQ4 — transfer across teams is a **within-condition, between-user** contrast, and
it is expressible only because the user is not baked into the condition.

### 1.3 Feedback and oracle score are parallel annotations on the same span

`docs/span-schema.md` §4: user feedback is written as a Phoenix annotation named
`user_feedback` (`label` ∈ {like, dislike}, `score` ∈ {1,0}, free-text
`explanation`), at response scope or session scope; oracle scoring is written
separately as an annotation named `oracle`. Two annotations, same target, written
by different processes.

This is the divergence measurement from the agenda, pre-built. The like-vs-correct
2×2 is a join, not a project. §6 is about what to do with it and how it goes
wrong.

### 1.4 The registry is already the right shape for proposals

`sim/registry.py` is content-addressed and append-only: blobs keyed by SHA-256,
named version histories that are never updated in place, and an append-only audit
table. Its `kind` column is a free string. A candidate skill, a proposed prompt
diff, a proposed endpoint specification, a cluster definition, a mining run's
parameters — all of these are "configuration" in exactly the sense the registry
defines, and none of them would require a schema change to store. That is worth
saying explicitly because it removes the most common excuse for building a
bespoke store: there is no missing substrate here, only missing intent.

The append-only property matters for a reason specific to reflection. A loop that
mutates its own inputs — rewrites the prompt it was mined under, then mines again
— produces a history that cannot be reconstructed unless every intermediate state
is still fetchable byte-for-byte. The registry's docstring makes the point for
config generally; for a self-modifying loop it is the difference between an
experiment and an anecdote.

### 1.5 The skill lifecycle already refuses what the loop would want to do

`sim/skills.py` implements the state machine. Three properties are the ones RQ4
runs into:

* `author()` has **no `state` parameter**, and the docstring says why: a state
  parameter is exactly the field an injected instruction would set. Agent-authored
  skills land in `draft`, full stop.
* `transition()` refuses any promotion to `approved` or `active` unless
  `is_human_action=True`, with an error message that names the reason — automated
  approval would make the gate decorative.
* Approval is pinned to `code_hash`, and `resolve_for_execution` re-hashes the
  bytes immediately before execution. The approve-then-swap attack described in
  the threat model does not work.

Also relevant, and easy to miss: `SkillRecord` carries a `definition_hash`
alongside `code_hash`, and both are content-addressed. A "skill" with no code at
all is still gated, because a definition is rendered into agent-facing instruction
text and is therefore an instruction-injection channel in its own right. Any
reflection design that treats "it's only a prompt fragment, not code" as a reason
to skip review has misread what the gate is protecting.

### 1.6 Labelled corners exist, and they are small

Three sources of ground truth that a mining pass can be *validated against*,
though not mined from:

* `sim/oracle/basket.py` — 265 questions, 5 families × 6 categories, with
  `paraphrase_of` linking each rewording to its base. Its own docstring is blunt
  about the size: roughly 13 answerable items per family, so one item moves a cell
  by ~8 points. It smoke-tests; it does not support a five-point claim.
* `sim/oracle/labels.py` — gold labels computed as a **pure function of the
  population**, never of the marts, so a projection bug shows up as a measured
  error instead of cancelling out.
* The org tree, rebuilt deterministically from the snapshot seed via
  `b2e.gen.org.build`, plus `unit_id` / `is_head` in `truth/people.json`. This is
  the partition that makes "one team versus another" a well-defined phrase.

And one segmentation axis that exists in the corpus for exactly this question:
`stable.agentic_skill_level`, six steps of AI-tool proficiency. See §5.

### 1.7 What the substrate does not have

Stated plainly, so nobody discovers it three weeks in:

* **No intent label on an arbitrary span.** The root span's `metadata` carries
  `{employee_role, question_id, experiment_id, basket_id}` — but `question_id` and
  `basket_id` are populated for basket runs. Organic fleet traffic has no such
  field, which is the entire difficulty of §3.
* **No cluster identity, no candidate artefact, no reviewer queue.** Nothing
  models "this draft came from cluster 47 which covers 3 100 sessions".
* **No epoch in the fingerprint.** If a reflection loop ever exists, its own
  version becomes a variable that changes agent behaviour, and by the rule in
  `sim/agent/config.py` — nothing influencing behaviour may be a constant in the
  code — it would have to become a ninth fingerprint field. That is a change to
  `docs/span-schema.md` §2 and to `sim/fingerprint.py`, and it is a prerequisite,
  not a detail.
* **No real users.** This is the big one, and §6 is largely about its
  consequences.

---

## 2. Clustering traces by intent rather than by surface text

The agenda says "by intent (not by the text of the request)". That parenthesis is
doing real work, and this corpus makes the reason unusually concrete.

### 2.1 Why surface-text clustering fails *here* specifically

The basket is templated on purpose: question text carries `{subject}`, `{peer}`,
`{unit}`, `{requisition}` slots and the harness binds them from the snapshot,
because a basket with names baked in would be valid for exactly one build. The
consequence for clustering is direct — two traces of the *same* intent differ
mainly in a bound proper noun, and with a 474-form surname dictionary over a large
population there are many exact namesakes. Meanwhile paraphrases were written to
be lexically distant from their base while preserving intent, and the caution
categories are deliberately near-identical in surface form to their answerable
siblings: "Сравни {subject} и {peer}" opens an `answerable` item, an
`out_of_scope` item, a `no_data` item and a `prompt_injection` item. A
text-similarity clusterer will happily merge "compare two people" with "compare
two people and fire the weaker one". Those require opposite behaviours, and a
cluster containing both cannot yield a coherent candidate.

So: embedding the user's message and running k-means recovers the slot vocabulary
and the family, not the intent. It will look like it works — clusters will be
clean and interpretable — and the thing it is clean about is the wrong thing.

### 2.2 Option A — the action signature

Cluster over what the agent *did*, derived from the span tree: the ordered
sequence of `(b2e.heimdall.endpoint, schema.model, bucketed columns_requested,
filter key columns, aggregate-or-rowwise, status/error_code)`, plus the tool
sequence and the iteration count. This has the property that a skill or a
predefined code block is a *replacement for an executed plan*, so clustering in
plan-space clusters in the space the candidate lives in.

Honest problem, and it is not small: **the action signature is a function of the
current configuration as much as of the intent.** If the system prompt tells the
agent to call `describe_model` before its first query, every cluster shares that
prefix and you have partly clustered the prompt. Two mitigations, both cheap:
stratify strictly by `condition_id` before clustering (§1.2), and quotient out
sub-sequences that appear in ≥95% of traces in the stratum before computing
distances — a step every trace takes carries no information about which traces
belong together. Neither mitigation helps if the prompt *conditionally* branches;
then the branch is a real behavioural cluster and separating "intent cluster" from
"prompt-branch cluster" needs the outcome side (§2.4).

### 2.3 Option B — an LLM intent labeller

Give a model the question plus a compressed trace and ask for a normalised intent
string, then cluster the strings. Cheap, and it handles the slot problem
trivially.

Two conditions, without which this quietly poisons the whole loop. First, the
labeller's prompt and model must be **pinned in the registry and version-stamped**,
for the reason `sim/fingerprint.py` gives about runs: if you re-label last
quarter's traces with this quarter's labeller and the clusters move, you cannot
tell whether the fleet changed or the labeller did, and the loop will confidently
report a trend that is its own drift. Second, the labeller is a component of the
system under study reading the outputs of the system under study; if it shares a
model family with the agent it will share the agent's blind spots, and clusters of
the agent's characteristic failures are exactly what it will be worst at naming.

### 2.4 Option C — cluster by failure signature, not by question

Ignore intent for the first pass and cluster on the *outcome shape* that
`sim/research/metrics.py` already computes: fabricated ids, fabricated numbers
with the source rows observed, missed refusal, followed injection, plus HTTP
status histogram and round-trip count. This does not answer "what were users
trying to do", but it does answer "what does this system get wrong, repeatedly,
in the same way", which is closer to what a candidate needs to be.

This is the option most likely to be useful first, and the least likely to be
chosen, because it produces clusters that are boring to read.

### 2.5 The falsifiable gate before any of this is believed

There is a free validation set: **the basket's `paraphrase_of` groups are known
intent classes.** A clusterer that cannot recover paraphrase groups from basket
traces — same base question, different wording, same slots bound — will not find
intents in organic traffic. Measure purity and completeness against those groups
and against `(family, category)`, and require both: high purity on `family` with
low purity on `category` is precisely the failure in §2.1, and it is invisible if
you only report one number.

The honesty caveat on this gate, which must travel with any result that uses it:
the basket has 265 items written by the same people who would write the
clusterer, its questions are templated in a uniform way that organic traffic is
not, and passing the gate is necessary rather than sufficient. It is a way to
*reject* clusterers cheaply, not a way to certify one.

---

## 3. Telling three kinds of candidate apart

The agenda's steps 2–4 name three different artefacts. They are not
interchangeable, they have different costs, and the discriminator between them is
available in the trace.

| Candidate | What it changes | Trace signature | Cost of being wrong |
|---|---|---|---|
| **Skill** (definition ± code) | the agent's *method* | data was fetched correctly, conclusion is wrong; `fabricated_ids` empty; disagreement with `oracle` is in interpretation | a bad instruction in the prompt of every session that retrieves it |
| **Predefined code** attached to a skill | deterministic post-processing | `fabricated_numbers` non-empty **while the inputs were observed** — the rows came back, the arithmetic did not | a wrong number, silently, at scale |
| **New Heimdall endpoint** | the API under test | correct answer, high `heimdall_calls`, wide `columns_requested`, high `b2e.latency.injected_ms` | invalidates the catalogue, the snapshot, and every prior comparison |

The middle row deserves emphasis because it is the crispest discriminator the
environment offers. `sim/research/metrics.py` computes `fabricated_numbers` as
"numbers asserted in the answer minus numbers observed in tool results in that
same run". A cluster where the rows were fetched and the *derived* number was
invented — a rank, a percentile, a gap, a share — is a cluster asking for
arithmetic to be moved out of prose and into predefined code. That is a
mechanical read, not a judgement call, and it maps onto real families: `unit_share`,
`competency_gap`, `grade_gap`, `tenure_vs_median` are all "fetch then compute".

The third row is the one to be conservative about. The catalogue is generated from
the Heimdall OpenAPI spec and gated by `test_catalog_matches_documented_inventory`;
adding an endpoint is a corpus-build event that changes `data_snapshot_hash` and
therefore `condition_id`. A reflection loop must emit an endpoint candidate as a
**proposal artefact in the registry**, never as a mutation. In a real deployment
it is also a request to a different team on a quarterly cycle, which sets the
outer clock for the whole loop and should be stated in any design rather than
assumed away.

### 3.1 The trap confound, and the rule that handles it

The corpus contains deliberate data traps (`b2e/traps.py`, toggled at build time)
and one deliberate stale flag: `is_key_employee` is last year's cut of the
methodology and *disagrees on purpose* with a recomputation from current data.

A mining pass that does not know this will find a beautiful, high-support cluster
around key-employee questions and mint a skill that reproduces the stale flag,
because that is what makes the disagreement go away. It will have learned the
trap, not the task. The same applies to every column a trap corrupts.

The rule that handles it is available for free: `traps_enabled` is a fingerprint
field, and a traps-off run is served from a different snapshot. **A candidate
must reproduce in both snapshots before it is treated as real.** A candidate that
exists only with traps on is a candidate to fix the corpus, or a finding about
data quality — either way it is not a skill. This is one of the few places where
the synthetic corpus is strictly better than production data for studying RQ4: in
production you cannot toggle the data defects off.

---

## 4. Measuring whether a skill transfers between teams

The agenda asks whether a skill learned on one team applies to the neighbouring
one. The environment makes this well-posed, which is rarer than it sounds.

**The design.** Hold the skill's `code_hash` and `definition_hash` fixed — they
are content-addressed, so "the same skill" is an identity check rather than a
naming convention. Hold `condition_id` fixed, so model, temperature, prompt
version, snapshot, traps and latency profile are all constant. Vary only the
acting employee and the bound slots. Because the acting employee is *not* part of
the fingerprint (§1.2), this is a legitimate within-condition contrast.

**Distance.** Use the org tree, which is deterministic from the seed, and define
transfer distance structurally: same parent unit → sibling unit in the same
department → different department in the same block → different territorial bank.
Reporting a single "does it transfer" boolean throws away the only interesting
part, which is where the curve falls off.

**The null worth taking seriously.** A skill that helps only on its donor unit has
memorised unit-specific strings — a unit name, a set of `person_id`s, a local
grade mix. This is the same failure the basket was templated to avoid, and it is
detectable the same way: rebind the slots and see what survives. A candidate that
degrades under slot rebinding within the donor unit should never reach a reviewer
at all.

**The honest limit, which must be attached to every transfer result produced
here.** The population comes from one factor model with one seed. Units differ in
size, grade mix and the realised draws, but they are *exchangeable by
construction* in a way real teams are not: there are no team cultures, no local
tooling, no domain-specific vocabulary, no manager idiosyncrasies. Transfer across
synthetic teams is therefore an easier problem than transfer across real ones.
The asymmetry to hold onto: **a negative result is informative** (if a skill fails
to transfer between exchangeable synthetic teams, it will certainly fail between
real ones), **a positive result is weak evidence** and must be reported as such.

**The axis that may matter more than the org tree.**
`stable.agentic_skill_level` gives six steps of AI-tool proficiency, already in
the corpus. A skill mined from the traces of proficient users — who ask specific,
well-scoped questions — may be useless or harmful for users at the bottom of that
scale, whose questions land disproportionately in the `ambiguous` category where
the correct behaviour is to ask one clarifying question rather than to run. Since
a fleet's traffic is dominated by whoever uses it most, and that is not a random
sample of employees, transfer *across proficiency* is the more likely place for a
mined skill to break. It is also cheaper to test than org transfer, because
proficiency is a mart column rather than a tree walk.

---

## 5. Like-versus-correct as a first-class metric

This is the part of RQ4 that the environment can genuinely sharpen, and also the
part where the environment's limits are most severe. Both, in order.

### 5.1 The measurement exists; report the 2×2, not a correlation

`user_feedback` and `oracle` are separate annotations on the same target
(§1.3). The quantity of interest is the joint distribution, and it should be
reported as a full 2×2 — liked-and-correct, liked-and-wrong, disliked-and-correct,
disliked-and-wrong — never as a single correlation or agreement rate. A
correlation of 0.3 is compatible with wildly different failure profiles, and the
two off-diagonal cells have completely different meanings:

* **liked-and-wrong** is the poison cell. It is what a like-optimising loop
  selects for, and its rate is the metric RQ4 exists to watch.
* **disliked-and-correct** is the cell that makes naive mining destructive,
  because it is systematically *not* random noise here — see next.

### 5.2 Correct refusals are structurally dislikeable, and that is a mechanism, not a worry

Four of the basket's six categories (`out_of_scope`, `ambiguous`, `no_data`,
`access_control`) have a correct behaviour that is a refusal, a deferral, or a
clarifying question. A user who asked whether their colleague is ready for
promotion and got a 403 explanation did not get what they wanted. The correct
answer is the disliked one, reliably and by construction.

Therefore: **a loop that treats "cluster with systematic dislikes" as "cluster
needing a new skill" will preferentially mint skills that suppress refusals.**
That is not a hypothetical drift over many iterations; it is the first-order
behaviour of the naive version of the agenda's step 2, and it points the fleet
directly at the thing RQ1 is trying to reduce.

The guard is available and should be mandatory rather than advisory: every
candidate carries the `missed_refusal_rate`, `followed_injection_rate` and
`fabricated_*` rates that `sim/research/metrics.py` already separates, computed on
the cluster before and after. **A candidate that raises the like rate while
raising `missed_refusal_rate` must be rejected mechanically, before a human sees
it.** Otherwise the reviewer is being handed a regression labelled as an
improvement, which converts the human gate from a safety control into a rubber
stamp with extra steps.

### 5.3 There are no real users here, and it is dishonest to pretend otherwise

The corpus is data, not behaviour; the agenda says so, and it applies to feedback
as much as to traces. Any `user_feedback` annotation in this environment will be
written either by the handful of humans running the experiment or by a simulated
rater. **If likes are simulated, the like-versus-correct divergence is a parameter,
not a measurement.** A study that generates likes with a model and then reports
the divergence between likes and truth has measured its own generator. No amount
of scale fixes this, and reporting it as an empirical finding about human feedback
would be straightforwardly false.

What the environment *can* do instead is worth more than a fake measurement:
**red-team the loop against an injected, known bias.** Specify a feedback
generator with an explicit, adversarial structure — likes as a function of answer
confidence, hedging, length and formatting, independent of correctness — run the
proposed reflection loop under it, and measure how far the fleet's `oracle` scores
degrade before anything in the loop notices. That is a falsifiable claim about
loop *robustness*, it does not require pretending the simulated raters are
people, and it produces a number a designer can act on: the amount of feedback
bias a given loop tolerates before it starts making the fleet worse. Because the
bias was specified rather than measured, the result generalises as a lower bound
on fragility, which is the direction one wants to be conservative in.

### 5.4 Do not treat the oracle as ground truth for text answers

`oracle` is not one thing. The deterministic parts — does the asserted
`person_id` appear in a tool result, does the asserted number match an observed
one, did the answer contain the injection canary — are close to ground truth
because they are checkable against the run's own API responses. The rubric parts,
where a judge model scores free text (roadmap S10), are a second noisy signal.
Reporting like-versus-oracle disagreement as though one side were truth, when both
sides are model-mediated, would launder judge bias into a headline number. The
2×2 should be reported separately for deterministically-scored and
rubric-scored items, and the two should never be pooled into one rate.

---

## 6. The human approval gate is the rate limit, and that is the design

### 6.1 It is enforced in code, and auto-approval is a defect

`sim/skills.py` will raise `LifecycleError` on any promotion to `approved` or
`active` without `is_human_action=True`, and the error text states the reason.
There is no configuration flag that relaxes it. The threat model's §3 rules 1–3
say the same in prose, and §3.1 adds the rule most likely to be broken later:
nothing in the pipeline may import, exec, lint-by-import, dry-run, preview or
test draft code, because any such step executes injection-authored code *before*
the human click.

A reflection loop is a program. The only way a program approves a skill is by
passing `is_human_action=True`, which is a forged human action, and any call site
outside the admin UI that does so is the defect — not a shortcut, not a
configurable mode. This is cheap to enforce and worth enforcing before anyone is
tempted: a static check that `is_human_action=True` appears only in the admin UI's
approval handler, in the same spirit as the checklist item in the threat model's
§9 for draft code never being executed.

The "helpful validation" trap has a reflection-specific shape worth naming. A
loop that mints candidates has an obvious quality problem — most candidates are
bad — and the obvious fix is to run them and keep the ones that work. That is
precisely the prohibited step, dressed as engineering rigour, at fleet scale and
on code whose provenance is user traffic that may carry injected instructions from
HR fields. If a candidate must be evaluated by execution, it goes through the
human gate first and is evaluated as an `active` skill in a controlled experiment,
which is slower and is the point.

### 6.2 The scarce resource is reviewer attention, so optimise precision, not recall

The threat model's R-1 states that a malicious or coerced approver is unmitigated
by construction: the approver *is* the boundary. A fleet-scale reflection loop is
an unbounded generator of approval requests aimed at that boundary, and there is a
volume attack that needs no cleverness — flood the queue until the reviewer
approves by fatigue. The instruction that produces the flood can arrive through
injected content in HR data, which the corpus deliberately ships.

The consequence for loop design is concrete. If a reviewer can genuinely evaluate
on the order of ten skills a week — read the definition, read the code, understand
the cluster it came from, check the guard metrics — then a loop producing two
hundred candidates a week is **worse** than one producing eight good ones, not
twenty times better. Surplus candidates are not free; they consume the only
resource that makes any of them safe. So:

* the loop's objective is **candidates per approved skill**, and the number to
  report is queue precision, not candidates mined;
* mining happens at cluster level and submits **one draft per cluster**, not one
  per session. Content addressing already deduplicates byte-identical drafts —
  `SkillStore._create` returns the existing record — but the realistic flood is
  *near*-duplicates: the same intent with a different bound slot, from thousands
  of instances. Canonicalising before submission is the loop's job, not the
  store's;
* a candidate arrives with its evidence attached — cluster size, condition
  stratum, before/after guard metrics, the traps-on/traps-off reproduction from
  §3.1 — because a reviewer who has to reconstruct the case themselves is a
  reviewer who will eventually stop trying.

### 6.3 Prompt rewriting is the highest-risk output, not the safest

Of the three outputs RQ4 names, "rewrite the system prompt" reads like the mildest
— no code, no sandbox, just words. It is the opposite. The threat model's R-2 is
explicit: prompt editing bypasses the sandbox entirely, and the agent is the
legitimate holder of every credential and the only component with network egress.
Its §7.2 lists system-prompt hijack as the highest-impact reach in the model, and
today prompt commits do not go through the skill approval queue at all.

Any RQ4 design that includes automated prompt rewriting must route it through the
same human gate as skills. The registry already provides everything needed to do
so — append-only versions, byte-exact diffs, an audit row per commit, and a
version string that lands in the fingerprint — so the missing piece is a policy
decision and a queue, not infrastructure.

### 6.4 Continuous self-modification and controlled measurement are in direct tension

This is a structural point and the environment surfaces it rather than hiding it.
Committing a new system prompt produces a new `prompt_registry_version`, which
changes `condition_id`, which means runs on either side of the commit are **not
comparable by the environment's own definition**. The same is true of activating a
skill: it changes `skill_registry_hash`.

So a loop that continuously edits prompts and activates skills destroys, as it
runs, the ability to tell whether it is helping. Every improvement claim it makes
spans a condition boundary. The resolution is not to weaken the fingerprint —
that would only hide the problem — but to run reflection in **discrete, versioned
epochs**: freeze a configuration, accumulate traffic, mine, review, approve,
freeze the new configuration, and compare epoch to epoch on a held-out basket that
did not participate in the mining. This is slower than a continuously-learning
fleet and it is the only version whose results mean anything.

It also implies the ninth fingerprint field from §1.7. If the loop's own version
influences what the fleet does, it is a condition variable, and by the rule in
`sim/agent/config.py` it cannot be a constant in the code.

---

## 7. The one contrast worth designing for

If a single result were to come out of RQ4 in this environment, this is the
candidate, and it is available precisely because `truth/` exists and production
does not have it.

Run the same loop design twice:

* **Oracle-supervised** — the mining pass may read `oracle` annotations, hence
  gold labels. This is an upper bound that no real deployment can reach, because
  in production there is no `truth/people.json`.
* **Feedback-only** — the mining pass sees `user_feedback` and the traces, and
  nothing else. This is the realistic setting, and it is what RQ4 actually asks
  about.

The **gap** between them is the result: how much of the achievable improvement is
reachable from noisy, biased human feedback alone. A small gap would be a
genuinely interesting finding. A large gap is the expected outcome and would say
that fleet reflection on likes is, at best, a candidate-generation heuristic that
needs an independent correctness signal to be safe — which would in turn make the
design question "where does the correctness signal come from in production?"
rather than "how do we cluster better".

The discipline this requires: the feedback-only arm must be **genuinely blind**
to `truth/`, including indirectly. It is easy to leak — a cluster selected using
oracle scores and then handed to the feedback-only arm has already been
oracle-supervised. The environment cannot enforce this; only the experiment design
can, and it should be stated in the run's registry entry rather than assumed.

---

## 8. What would not work

Consolidated, because these are the designs most likely to be proposed and each
one fails for a specific reason rather than a vague one.

1. **Embedding the user's message and clustering.** Recovers slots and families,
   merges answerable questions with their `out_of_scope` and `prompt_injection`
   siblings, and looks convincing while doing it. §2.1.
2. **Optimising a scalar reward derived from likes.** Selects for confident tone,
   suppresses correct refusals, and has no term for the caution categories that
   are ~40% of what correct behaviour consists of. §5.2.
3. **Auto-approving "low-risk" candidates.** There is no reliable low-risk
   classifier, and the risk is not primarily in the code — a definition-only skill
   is rendered verbatim into agent-facing instruction text and is an injection
   channel with no code at all. §1.5, threat model §3.
4. **Validating drafts by executing them.** Prohibited, for the reason in threat
   model §3.1, and the reflection setting is where the temptation is strongest
   because candidate quality is low and the fix looks like engineering rigour.
   §6.1.
5. **Continuous online adaptation.** Breaks `condition_id`, so the loop deletes
   the evidence that it worked. §6.4.
6. **Mining the basket's own traces and reporting it as a fleet result.** 265
   questions, written by the same authors, small-n by its own declaration.
   Basket traces validate a clusterer; they do not constitute fleet traffic.
   §2.5.
7. **Treating every dislike as a bug report.** On the caution categories a dislike
   is usually correct behaviour being disliked. §5.2.
8. **Letting the deployed loop read `truth/`.** It measures an oracle-supervised
   loop, which is not what RQ4 asks and is not reproducible outside a synthetic
   corpus. Keep it as a deliberate upper-bound arm, never as the default. §7.
9. **Simulating likes and reporting the like-vs-correct divergence as a finding.**
   The divergence would be a parameter of the generator. Red-team the loop against
   a specified bias instead. §5.3.
10. **Mining across mixed conditions.** A "systematic" within-cluster signal that
    is actually a between-condition difference. Stratify by `condition_id` first.
    §1.2.
11. **Skills that only work on their donor unit.** Detectable by slot rebinding
    within the donor unit, and should never reach a reviewer. §4.
12. **Candidates that only reproduce with traps on.** Those learn the corpus's
    deliberate defects. Require reproduction in both snapshots. §3.1.

---

## 9. Prerequisites, if this is ever built

Not a plan and not a proposal — a list of what would have to be true first, so
that a future decision is made with the costs visible.

* A ninth fingerprint field for the reflection loop's own version, with the
  matching change to `docs/span-schema.md` §2. Without it, epochs are not
  distinguishable. (§1.7, §6.4)
* An intent or cluster identity on organic traces, which today exists only for
  basket runs via `metadata.question_id`. (§1.7)
* A candidate artefact kind in the registry carrying its evidence bundle — cluster
  size, stratum, guard metrics before and after, traps-on/off reproduction. No
  schema change needed; the registry's `kind` is a free string. (§1.4, §6.2)
* A reviewer queue with an explicit throughput assumption, stated as a number, so
  that "candidates per week" can be compared against it rather than admired.
  (§6.2)
* A static check that `is_human_action=True` appears only in the admin UI's
  approval handler. (§6.1)
* A decision, written down, on whether prompt commits join the same approval
  queue as skills. Today they do not, and R-2 is the largest unmitigated reach in
  the threat model. (§6.3)
* S10 scoring, since the guard metrics in §5.2 are what make a candidate
  reviewable at all, and a candidate without them is a request for the reviewer to
  guess.

Until those exist, the correct amount of reflection code in this repository is
zero, which is what it currently has.
