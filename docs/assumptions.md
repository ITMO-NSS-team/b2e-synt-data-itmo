# Assumptions

Running log. Every entry is something the spec left open, or something the spec
asserted that turned out not to hold on this server. Each has a decision and a
reversal cost, so a reviewer can overturn any of them cheaply.

Status legend: **OPEN** — needs your ruling · **TAKEN** — decided, proceeding ·
**BLOCKED** — cannot proceed until resolved.

---

## A-1 · Storage backend is the hybrid columnar store, not ClickHouse — **TAKEN**

**Spec said:** C1 is "backed by the local ClickHouse holding the generated data."

**Measured:** ClickHouse is not installed on this host (`systemctl is-active
clickhouse-server` → `inactive`, no `clickhouse` binary). The corpus does not live
in ClickHouse and never has.

**Why ClickHouse is the wrong backend here, not just an absent one.** The corpus is
deliberately hybrid (`docs/design.md`, `b2e/store.py`): ~800 of 4 599 catalogue
columns are materialised on disk, and the remaining ~3 800 are computed on read
from the coordinate `(seed, mart, column, row)`. Loading it into ClickHouse would
require materialising all 4 599 columns for 294 000 people. That inverts the
central design decision of the dataset, multiplies disk by roughly two orders of
magnitude, and would not fit in 3.8 GiB of RAM alongside Phoenix and Postgres.

**Decision:** `heimdall-emulator` reads through the existing
`heimdall/store/columnar.py` + `ProceduralSnapshot` path, which already serves
materialised and procedural columns indistinguishably through the API.

**Reversal cost:** low-moderate. The store is behind an interface; a ClickHouse
backend can be added later for the materialised subset only. It would not be able
to serve the procedural tail.

---

## A-2 · There is no existing oracle question bank — **BLOCKED → scoped**

**Spec said:** C4 `GET /experiments/{id}` reports metrics "vs the existing oracle
question bank: hallucination rate, ...".

**Measured:** it does not exist. `docs/roadmap.md` lists S8 (gold labels for five
task families) and S9 (question basket, 120–200 per family with paraphrases) as
open, not closed. `truth/people.json` holds latent factors and basic labels only —
not reference answers.

**Consequence:** hallucination rate is undefined until an oracle exists. Without
it, `GET /experiments/{id}` can report latency, API calls per answer and token
cost, but not correctness — which is the RQ1 metric.

**Decision:** build a minimal S8/S9 subset as a first-class deliverable rather than
pretending the metric works. Minimal means: gold labels computed as a **pure
function of the population, never of the marts** (a mart projection bug must not
leak into the reference), plus a question basket that includes the five mandatory
cautious categories — out-of-scope, ambiguous, no-data, access-control,
prompt-injection. Without those five the harness measures accuracy but not
caution, and RQ1 is about caution.

**Scope taken:** enough questions per family to make the metric move, not the full
120–200×5. Stated explicitly in the experiment output so nobody reads a small-n
result as a large-n one.

---

## A-3 · Memory budget is ~2.5 GiB, not 4 GiB — **TAKEN**

**Measured:** 3.8 GiB total. Non-negotiable resident load already present: Acronis
backup agent (~190 MiB across three processes), dockerd (~49 MiB), an existing
`valoboros` container (190 MiB), sshd/systemd. A 300 000-person corpus build was
still running at recon time holding 1.9 GiB RSS; that is transient.

**Working budget for the new stack: ~2.5 GiB across Phoenix, Postgres,
heimdall-emulator, b2e-agent, research-api, admin-ui, reverse proxy, and the skill
sandbox tier.** 2 vCPU.

**Consequences that follow, and are not negotiable at this size:**
- Postgres tuned small (`shared_buffers` in the low hundreds of MiB), not default.
- The emulator holds one snapshot reader, not one per worker; worker count is
  bounded by RAM, not by CPU.
- The sandbox tier cannot be "a container per execution kept warm."
- No swap exists (`Swap: 0B`), so an OOM is a kill, not a slowdown. Every service
  gets an explicit compose memory limit so the kernel kills the intended victim.

**Reversal cost:** none, if the box grows. All limits are compose values.

---

## A-4 · Sandbox is built from Docker + seccomp + rlimits — **TAKEN**

**Measured:** no gVisor (`runsc`), no bubblewrap, no firejail, no nsjail. Docker
29.3.1, Compose v5.1.1, `unprivileged_userns_clone=1`, kernel 6.8 with
`CONFIG_SECCOMP=y` and `CONFIG_HAVE_ARCH_SECCOMP_FILTER=y`.

**Consequence:** the isolation boundary available is a container with dropped
capabilities and a seccomp profile — a shared-kernel boundary. This is weaker than
a virtualised one and that limit is stated plainly in
`docs/skill-execution-threat-model.md` rather than glossed.

**Reversal cost:** installing gVisor would strengthen the boundary without changing
the architecture, since skill execution is already isolated to its own service.

---

## A-5 · Documentation language — **OPEN**

The existing corpus documentation (`design.md`, `research-agenda.md`,
`roadmap.md`, `HANDOFF.md`) is written in Russian. This task was specified in
English and names its deliverables in English.

**Decision taken provisionally:** new simulation-environment documents are written
in English, matching the language the requirements were given in. The pre-existing
Russian documents are left untouched.

**Say the word and I will switch the new documents to Russian** for continuity with
the corpus docs. Cost is a translation pass, no code impact.

---

## A-6 · Credentials supplied inline are treated as compromised — **TAKEN**

Three live credentials were pasted into the task prompt: an Anthropic OAuth token,
a Telegram bot token, and a GitHub PAT. The spec itself requires refusing inline
credentials. All three are treated as burned and are not written to any file,
commit, or configuration.

The stack reads them from environment variables only, names fixed as
`CLAUDE_CODE_OAUTH_TOKEN`, `TELEGRAM_BOT_TOKEN`, `GITHUB_TOKEN`, supplied via a
gitignored `.env`. Rotate all three before use.

Note this is a repeat: `HANDOFF.md` §9 records the same failure mode in the
previous session.

---

## A-7 · Git remote already exists — **OPEN**

The spec says "create a remote git repo and PUSH." One already exists:
`https://github.com/Mosyamac2/b2e-synt-data` (public, per `HANDOFF.md`).

**Provisional decision:** push the simulation environment to the existing
repository, since it is the same project.

**Recommendation before pushing:** make it private. The simulation environment adds
service topology, prompt registry, and skill definitions to a repository that
already exposes the Heimdall mart and column taxonomy. Corpus data itself stays out
of git (already gitignored, reproducible from seed).
