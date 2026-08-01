# Skill execution: threat model and sandbox design

**Gate G1.** Written before any skill-execution code, as required.

The premise this document defends: the B2E agent may only (a) call Heimdall API
endpoints and (b) execute predefined code attached to an **approved** skill. It
may author a new skill, definition and companion code — but that artefact becomes
executable only after an explicit human action. Any design in which
agent-authored code reaches execution without that action is a defect.

Everything below was verified on this host rather than assumed. Where a control
is not a real boundary, it is labelled as not a real boundary. Section 8 lists
what is *not* mitigated.

---

## 1. What is already broken

Before designing anything, the existing code was audited. Three findings are
live defects in `heimdall/`, not hypotheticals, and they shape the design.

### 1.1 The dev router is unauthenticated — **fixed in C1**

`heimdall/app.py` mounts `_dev_router` unconditionally, and none of its four
routes carry `Depends(_require_bearer)`:

| Route | Effect |
|---|---|
| `POST /api/v2/dev/add_skill_for_debug/` | writes a skill file |
| `POST /api/v2/dev/skills/save` | writes a skill file |
| `GET /api/v2/dev/skills/raw?path=` | reads a file under the skills root |
| `DELETE /api/v2/dev/skills?path=` | deletes a file |

Every data route is bearer-gated; these are not. Anyone who can reach the port
can write a skill, and `state.invalidate()` makes it immediately visible to
`find_skills` / `get_skill`. Since skill markdown is rendered verbatim into
agent-facing instruction text, this is a direct prompt-injection channel.

**Status: fixed.** `sim/emulator/app.py` refuses `/api/v2/dev/*` with 404 unless
`enable_dev_router=true`. It is enforced in middleware, not by removing routes —
the first attempt did the latter and *silently did nothing*, because this FastAPI
version stores included routers as `_IncludedRouter` objects with no `.path`
attribute, so a path-based filter matched nothing while appearing to succeed. The
regression test asserts 404 by making the request, not by inspecting the table.

### 1.2 Arbitrary file write via `_save` — confirmed exploitable

`_save` builds its target as `skills_root / skill.domain / f"{skill.name}{suffix}"`
where **`domain` and `name` are read out of the uploaded file** and never
sanitised or passed through `_safe_path`. A recipe declaring
`domain: "../../../../tmp/..."` writes outside the skills root. This was
demonstrated end-to-end during the audit, producing a file at `/tmp/`.

The blast radius is bounded — `_parse` looks up `EXTENSIONS[path.suffix]` and
raises for anything other than `.yaml/.yml/.md`, so you cannot drop a `.py`, a
`.so`, or a crontab line — but arbitrary YAML/Markdown overwrite anywhere the
process can write is still a defect.

### 1.3 `_safe_path` has no separator boundary

It checks `str(target).startswith(str(root.resolve()))`. `../heimdall-skills-evil/x`
passes, because `/…/heimdall-skills-evil` starts with `/…/heimdall-skills`. Plain
`../` escapes are caught by the prior `resolve()`; the sibling-prefix case is not.

### 1.4 Consequence for the design

`heimdall-skills/` is **attacker-writable prose**. It must never be the execution
code store. If the sandbox ever mounted it, an unauthenticated HTTP POST would
become code execution. The executable store lives on a different path, with
different ownership, and is written only by the approval action.

A related, separate issue: **recipe skills have no approval gate at all today.**
A recipe carries a ready `mcp_query` body that `render.py` presents to the agent
as "executes as is". That is not Python execution, so it cannot reach the syscall
table, but it is a data-plane capability plus an instruction-smuggling channel,
and it is ungated. Recipes are brought under the same lifecycle in §3.

---

## 2. Host reality

Measured, not assumed:

| Property | Value | Consequence |
|---|---|---|
| CPU / RAM | 2 vCPU Icelake, 3.8 GiB, **zero swap** | one sandbox tier, not a container per call |
| Docker | 29.3.1, runc 1.3.4 | patched against CVE-2025-31133 / 52565 / 52881 *today* |
| gVisor / Kata / bwrap / nsjail | **absent** | no virtualised boundary available |
| `userns-remap` | **not enabled** (no `/etc/docker/daemon.json`) | container uid == host uid |
| Rootless Docker | **cannot start** — `newuidmap`/`newgidmap` absent | rules out the rootless-broker design |
| Seccomp | kernel supports it; Docker's builtin profile active | builtin blocks ~44 of 300+ syscalls |
| `docker` group | contains the host user | group membership is root-equivalent |

The boundary available is a **shared-kernel container**. That is weaker than a
virtualised one. It is stated here rather than implied.

---

## 3. Lifecycle state machine

```
                    agent authors                 human uploads
                          │                             │
                          ▼                             ▼
                      ┌───────┐                  ┌──────────────┐
                      │ draft │─── submit ──────▶│pending_review│
                      └───────┘                  └──────────────┘
                          │                          │      │
                          │                  approve │      │ reject
                          │                (HUMAN)   ▼      ▼
                          │                   ┌──────────┐ ┌──────────┐
                          │                   │ approved │ │ rejected │
                          │                   └──────────┘ └──────────┘
                          │                          │
                          │                   enable │ (HUMAN)
                          │                          ▼
                          │                    ┌────────┐
                          │                    │ active │──── retire ───▶ ┌─────────┐
                          │                    └────────┘                 │ retired │
                          │                          ▲                    └─────────┘
                          └── any edit ──────────────┘  invalidates: new hash, no approval
```

Rules, enforced in code and not by convention:

1. **Agent-authored skills always enter as `draft`.** There is no field in the
   agent's tool schema that can carry a state. Authoring is a write of inert
   bytes to a content-addressed blob store.
2. **Uploaded skills always enter as `pending_review`.** Upload is a human
   action; it still is not approval.
3. **Only an explicit human action in the admin UI promotes to `approved`.**
   No API path, no agent tool, no automation performs this transition.
4. **Only `active` is executable.** `approved` means "a human blessed these
   bytes"; `active` means "and it is currently switched on". The split exists so
   disabling a skill during an incident does not destroy its approval history.
5. **Any edit invalidates approval**, because approval is pinned to the hash of
   the exact bytes (§4), and edited bytes have a different hash and therefore no
   approval record.
6. **Retirement is terminal.** A retired hash is never re-activated; a
   resurrection is a new approval of the same bytes, separately audited.

Every transition writes an append-only audit row (actor, timestamp, from-state,
to-state, hash) via `sim.registry`.

### 3.1 The rule that matters most

**Nothing in the pipeline may import, exec, lint-by-import, "dry-run", "preview",
or "test" draft code.** The audit specifically probed for this and it is the
single most likely way this design gets defeated later — by someone adding a
helpful "validate the skill before showing it to the reviewer" step. Any such
step executes injection-authored code *before* the human click, which is the
whole boundary.

Validation of a draft is therefore restricted to operations that never execute:
`ast.parse` for syntax, static import extraction from the AST, byte-size and
shape checks. `compile()` is permitted (it does not execute); `exec`, `eval`,
`__import__`, and `importlib` on draft code are not.

---

## 4. Content addressing

* Every skill artefact — definition and companion code — is stored as an
  immutable blob keyed by **SHA-256 of the exact bytes** (`sim/registry.py`).
* Approval attaches to the **hash**, never to the name.
* At execution time the runner resolves **by hash**, re-computes the digest of
  the bytes it is about to run, and refuses on mismatch.

That last point is not redundant. The audit found a working attack against
name-based resolution: get a benign `attrition_by_dept` approved, then write
malicious bytes for that name into the store, and a name-resolving executor runs
the swapped code with no re-approval. Content addressing at *storage* time does
not stop it; content addressing at *execution* time does.

The approved store is root-owned and is not writable by the emulator uid, the
agent uid, or the sandbox uid. The only writer is the approval action.

---

## 5. Sandbox design

Three designs were developed independently and scored by three judges under
distinct lenses (isolation strength, fit to 2 vCPU / 2.5 GiB, operability). The
**two-container spool split** won 2–1. The rootless-broker design was eliminated
on a hard fact: rootless Docker cannot start on this host. The prefork pool has
the best latency (~4 ms) but places every execution in one container sharing a
tmpfs, a uid, and a cgroup with its siblings.

### 5.1 Topology

```
   ┌────────────────────────┐         ┌──────────────────────────────┐
   │ gateway  (edge network)│         │ worker   network_mode: none  │
   │ holds NO agent secrets │         │ read_only, cap_drop ALL      │
   │ writes spool/new/<id>  │────────▶│ claims job, execs by hash    │
   │ reads  spool/out/<id>  │◀────────│ writes spool/out/<id>        │
   └────────────────────────┘  spool  └──────────────────────────────┘
                              (volume)          │ ro mount
                                                ▼
                                        /skills  (root-owned, approved)
```

`spool-init` runs first as a one-shot root container to `chown` the named volume
— named volumes are created `root:root` and uid 65534 cannot otherwise write
them. This was a real ordering bug found in testing; the compose file uses
`depends_on: {condition: service_completed_successfully}`.

Job handoff is write-to-temp plus `os.rename`, which is atomic on the shared
volume. Measured: 60 concurrent submits produced 60 unique results, zero lost,
zero duplicated.

Cost: **~25 MB steady state, ~160 ms cold start** per execution on this box.

### 5.2 Enforced limits

| Control | Mechanism | Verified |
|---|---|---|
| **Network** | `network_mode: none` — empty netns, loopback only | `socket.create_connection` to the gateway and to `1.1.1.1:53` both fail; `if_nameindex() == [(1,'lo')]`; holds after the skill removes the import allowlist entirely |
| **Filesystem** | `read_only: true` rootfs; sole writable path is tmpfs `/tmp` `rw,noexec,nosuid,nodev,size=64m` | writes to `/`, `/usr`, `/skills` raise; `/skills` mounted `:ro` so running code cannot mutate the approved corpus |
| **CPU** | cgroup v2 `cpu.max`, plus `RLIMIT_CPU` soft **==** hard | throttling observed in `cpu.stat`; soft-only is useless — `SIGXCPU` is catchable |
| **Memory** | `mem_limit=320m` with `memswap_limit=320m` (equal ⇒ swap off); inner `RLIMIT_AS=192m` soft==hard | OOM kill observed, exit 137, container survived; ordinary overuse gives a clean `MemoryError` |
| **Wall clock** | supervisor `communicate(timeout=…)`, then `killpg(SIGKILL)` on a `setsid()` group | 2 s limit fired at ~2001 ms; independent of CPU quota, so a *sleeping* job still dies |
| **Environment** | no inheritance; env built explicitly as `{PATH, HOME, LC_ALL, PYTHONDONTWRITEBYTECODE}` | a hostile skill dumping its own `environ` saw exactly those four keys; scanning `/proc/*/environ` found zero secrets |
| **PIDs** | cgroup `pids_limit`, plus `RLIMIT_NPROC=0` | fork/subprocess from the runner raises `BlockingIOError` |
| **JSON in / JSON out** | one JSON document in, one out, size-capped | see §8 residual R-6 on strictness |

### 5.3 The import allowlist is not a boundary

A `sys.meta_path` finder allowlists a fixed stdlib subset. **This is telemetry,
not security**, and it is labelled that way everywhere it appears.

It was defeated trivially during the audit: `sys.meta_path[:] = []` removes it,
`sys.modules['os']` is already populated, and any function's `__globals__` chain
reaches whatever is loaded. In one test `ctypes` reached a raw `syscall(41)`
directly — only seccomp stopped it.

Its real jobs are: logging attempted imports, catching honest mistakes in
agent-authored skills, and forcing skill authors to declare a capability surface
so human review is tractable. The controls that actually hold are the empty
network namespace, the read-only filesystem, the scrubbed environment, the
separate PID namespace, and the cgroup limits.

---

## 6. Attacker: compromised admin UI

Assume full control of C5 — can click approve, edit prompts, upload skills,
toggle traps and latency.

**Reaches:**
- **Code execution in the worker, by design.** The approver *is* the boundary;
  an attacker holding the approve button has RCE in the sandbox by definition.
  Its blast radius is bounded to the worker by §5.2.
- **The agent's behaviour, with no code at all**, by committing a new system
  prompt. This is the highest-impact reach in the entire model and the sandbox
  does not touch it — see §7.2.
- **Phoenix traces** (agent prompts, tool calls, HR data in transit), the
  `sim.registry` history, and research-api, over the compose network.

**Does not reach:** the agent's OAuth/Telegram credentials directly (different
container, different PID namespace, secrets never in the sandbox tier), and raw
Heimdall data routes without the agent's bearer token.

**Stopped by:** nothing, within this scenario. That is the definition of the
scenario, and it is why admin-UI authentication, CSRF protection, and audit
integrity are load-bearing rather than hygienic.

---

## 7. Attacker: the agent-authoring path

Assume control of the agent's output via prompt injection carried in HR data —
realistic, because the corpus deliberately ships injection-style content.

### 7.1 What is genuinely blocked

- **Same-session execution.** The agent's tool surface has no code-execution
  tool. Authoring is `registry.put_object` — inert bytes. Nothing reads and runs
  a stored blob. This is enforced by the harness, not by the prompt.
- **YAML deserialisation RCE.** `yaml.safe_load` throughout;
  `!!python/object/apply:os.system` raises `ConstructorError`.
- **Writing executable file types** through the dev routes: the extension gate
  admits only `.yaml/.yml/.md`.
- **Inheriting approval across an edit:** content addressing, §4.

### 7.2 What succeeds

- **Recipe injection (present in current code).** The dev routes need no token
  and recipes have no approval gate, so an attacker registers a recipe whose
  markdown the agent is told "executes as is". Ceiling: a crafted query plan plus
  smuggled instructions — no arbitrary code. *Mitigated here* by blocking the dev
  routes (§1.1) and bringing recipes under the §3 lifecycle.
- **Stored XSS into the approver's browser.** `render.py` emits
  `description`/`body`/`notes` unescaped. If the admin UI renders that markdown
  as HTML, an injected payload runs in the approver's authenticated session and
  issues the approve POST itself — the human made no decision. *Mitigated* by:
  autoescaping renderer, no raw HTML, strict CSP, CSRF-token-gated approve, and
  showing the exact bytes being approved.
- **System-prompt hijack.** Not code, not sandboxed. Committing a prompt that
  instructs the agent to enumerate employees or emit its token drives the one
  component that holds every secret and has network egress. *Partially mitigated*
  by treating prompt commits as privileged, audited, and diffed; **not fully
  mitigated** — see R-2.

---

## 8. Residual risks — explicitly NOT mitigated

These are accepted, not solved. Listing them is the point.

**R-1 · A malicious or coerced approver is unmitigated by construction.**
Approver equals RCE-in-sandbox. Every downstream control assumes the approval
decision was honest. There is no second-person rule, no quorum, no signing.

**R-2 · System-prompt editing bypasses the sandbox entirely.** The threat model
covers code, not prompts. The agent is the legitimate holder of every credential
and the only component with egress. Highest-impact reach in the model. A future
mitigation would gate prompt commits behind the same approval queue as skills;
it is not built here.

**R-3 · The shared spool volume is a bidirectional trust channel.** A compromised
worker can read sibling jobs' inputs from `spool/new/` and write attacker-chosen
results to `spool/out/`. The gateway then relays that content to the agent, which
*does* have network — so `network_mode: none` is bypassed by confused deputy.
Schema validation does not help when the payload's *content* is the injection.
Partially reduced by treating all skill output as untrusted data that is never
interpreted as instructions, but not eliminated.

**R-4 · No custom seccomp profile.** Only Docker's builtin. Hostile approved code
reaches most of the syscall surface via `ctypes`. A tight allowlist (~120 calls,
`SCMP_ACT_ERRNO` default) would materially shrink this and is the highest-value
unimplemented hardening.

**R-5 · Shared-kernel escape.** No gVisor, no Kata, no `userns-remap`; container
uid maps 1:1 to host uid. Worker RCE is one Linux LPE or runc CVE away from host
root. runc 1.3.4 is patched *today* — that is a standing patch obligation, not a
boundary.

**R-6 · JSON-out is not strictly validated.** `json.dump(default=str)` silently
coerces a live object to `'<…object…>'` rather than rejecting it. Strict
serialisation would refuse.

**R-7 · Audit integrity against a compromised admin UI.** `sim.registry.audit` is
append-only, but the approve-time audit row is written *by* the admin UI. A fully
controlled admin UI can approve while omitting or forging the row.

**R-8 · Cross-execution side channels.** Shared physical cores, no Spectre
mitigation, readable `/proc`, timing observable through the response. Not
addressed.

**R-9 · Import-allowlist friction will cause drift.** Real stdlib closures pull
transitive imports (`statistics` → `random` → `_random`, `fractions`), so the
allowlist gets widened repeatedly. Each widening is a small, unreviewed erosion.

**R-10 · Worker crash recovery has a gap.** Slot identity is the container
hostname, stable across restart-policy restarts but *not* across
`compose up --force-recreate`, which strands the predecessor's `spool/run` entry.
A time-based reaper in the gateway covers it, at the cost of a delayed failure.

**R-11 · The corpus is synthetic.** "Other employees' HR data" is generated, so
the real-world confidentiality impact of cross-job disclosure is low *in this
deployment*. It would not be in a production one, and the design should not be
lifted without revisiting that.

---

## 9. Verification checklist

Claims in this document that are covered by automated tests:

- [x] `/api/v2/dev/*` returns 404 when the dev router is disabled — asserted by
      request, not by route inspection
- [x] Editing an approved skill produces a different hash with no approval record
- [x] Identical content does not create a spurious new version
- [x] Old config versions remain fetchable after a change
- [ ] Runner refuses a hash whose recomputed digest does not match *(with the sandbox)*
- [ ] Draft code is never imported, exec'd, or preview-run *(static check in CI)*
- [ ] Network, filesystem, env, memory, CPU and wall-clock limits hold *(sandbox suite)*
