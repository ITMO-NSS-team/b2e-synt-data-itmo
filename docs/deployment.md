# Deployment

One host, Docker Compose, a single reverse proxy in front of everything.

## 1 · Exposed surface

**Exactly one container publishes a host port**: the proxy. Everything else is on
an internal compose network with `internal: true`, which has no gateway to the
outside — not merely "unpublished", but unroutable.

| Port | Open to | Service | Why it is open |
|---|---|---|---|
| 443/tcp | internet | `proxy` (Caddy) | The requirement is that researchers reach the API and admin UI from the internet via the server's public IP. This is the only way in. TLS terminates here. |
| 80/tcp | internet | `proxy` | ACME HTTP-01 challenge and a redirect to 443. Serves no application content. |

**Nothing else is published.** Specifically not open:

| Service | Port | Reachable from |
|---|---|---|
| `heimdall-emulator` | 8081 | internal network only — never the proxy |
| `b2e-agent` | 8082 | proxy, at `/agent/*` |
| `research-api` | 8083 | proxy, at `/research/*` |
| `admin-ui` | 8084 | proxy, at `/admin/*` |
| `phoenix` | 6006 | proxy, at `/phoenix/*` |
| `postgres` | 5432 | internal only |
| `sandbox-worker` | — | `network_mode: none`; no network stack at all |

The emulator is deliberately not published. It is the data plane: it will answer
any HR query the acting identity is entitled to, and the bearer check accepts any
non-empty token. The permission boundary that makes it safe is the acting-employee
header, which only the agent sets correctly. Publishing it would let anyone pick
their own identity.

## 2 · Authentication

Every path behind the proxy requires HTTP Basic — including health endpoints,
because a health endpoint that names the snapshot hash and the trap state is
itself information about the experiment.

The admin UI performs its **own** Basic auth in addition to the proxy's. The
duplication is deliberate: the admin UI holds the skill approval button, and a
proxy misconfiguration should not expose it. It refuses to start if `ADMIN_USER`
or `ADMIN_PASSWORD` is unset.

```bash
make hash-password        # prints a bcrypt hash; the plaintext touches no file
```

Put the hash in `BASIC_AUTH_HASH`. `PUBLIC_HOST` decides the certificate: a real
DNS name gets a public Let's Encrypt certificate automatically; a bare IP gets
Caddy's internal CA and browsers will warn. **Prefer a DNS name** — a browser
warning that researchers learn to click through is a habit that costs more later.

## 3 · Secrets

All via `deploy/.env`, gitignored, never in code and never in a commit.

```bash
cp deploy/.env.example deploy/.env
$EDITOR deploy/.env
```

| Variable | Notes |
|---|---|
| `BASIC_AUTH_HASH` | from `make hash-password` |
| `ADMIN_USER` / `ADMIN_PASSWORD` | admin UI refuses to start without both |
| `POSTGRES_PASSWORD` | Phoenix's backing store |
| `CLAUDE_CODE_OAUTH_TOKEN` | subscription token; preferred, cheaper for simulation |
| `ANTHROPIC_API_KEY` | fallback if no OAuth token |
| `TELEGRAM_BOT_TOKEN` | only needed with `PROFILE=telegram` |
| `TELEGRAM_DEFAULT_EMPLOYEE` | identity a chat acts as until `/employee` changes it |
| `B2E_TELEGRAM_CONFIG_REF` | `agent_config_interactive` (default) gives a chat one resumable session; `agent_config` makes every message start cold |

If a credential is ever pasted into a chat, an issue, or a commit, treat it as
burned and rotate it. This has already happened twice on this project
(`HANDOFF.md` §9, and `docs/assumptions.md` A-6).

## 4 · Memory budget

The host has **3.8 GiB and zero swap**, so an OOM is a kill rather than a
slowdown. Every service therefore carries an explicit limit, so the kernel kills
the intended victim instead of choosing.

| Service | Limit |
|---|---|
| `phoenix` | 640m |
| `heimdall-emulator` | 512m |
| `b2e-agent` | 384m |
| `postgres` | 320m |
| `sandbox-worker` | 320m |
| `research-api` | 256m |
| `admin-ui` | 192m |
| `telegram-bot` (optional) | 128m |
| `proxy` | 96m |
| **total** | **~2.7 GiB** (2.8 with Telegram) |

That leaves roughly 1 GiB for the host. It is tight. Check it with:

```bash
make ps      # compose state plus real `docker stats` memory usage
```

Postgres is tuned down (`shared_buffers=96MB`, `max_connections=32`) because the
defaults assume a machine that is not this one.

**Do not run a 300 000-person corpus build while the stack is up.** That build
peaks around 1.9 GiB RSS and will push the box into OOM kills. Use `make seed`
(3 000 people, ~25 s) for the stack; the large corpus is for offline analysis.

## 5 · Bring-up

```bash
cp deploy/.env.example deploy/.env && $EDITOR deploy/.env
make seed            # corpus, ~25 s
make up              # whole stack
make smoke           # end-to-end gate; non-zero exit on any failure
```

Optional:

```bash
make seed-traps-off  # required before traps_enabled=false (RQ1)
make up PROFILE=telegram
make openapi         # regenerate docs/openapi/*.json from the live apps
```

Without a traps-off corpus the emulator **refuses** to switch rather than serve
traps-on data under a traps-off label, which would silently corrupt the RQ1
comparison the flag exists to enable.

### The Telegram bridge

A chat is a conversation, not a sequence of unrelated questions. The bridge
opens one B2E session per chat against `agent_config_interactive`, whose
`conversation_mode` is `resume`, so consecutive messages continue the same
headless Claude Code session and the agent sees its own earlier tool calls and
their results.

| Command | Effect |
|---|---|
| *(any text)* | forwarded to the agent verbatim, in the current session |
| `/start_new_session` | drops the session; the next message opens a fresh one with no history |

| `/new` | same thing, kept for muscle memory |
| `/whoami` | acting employee, current session id, config ref |
| `/employee <id>` | switch identity; resets the session, since it carries a permission scope |

While a turn runs, the bridge posts one status message and rewrites it in place
— *«запрашиваю данные → изучаю структуру витрины… 34 с, шагов: 6»* — from
`GET /sessions/{id}/progress` on the agent. That endpoint is a view of one
in-flight turn: nothing it returns is persisted, polling it cannot change the
turn, and the question and answer still pass through untouched. Every failure in
that path is swallowed, so a broken status display costs the status line and
never the answer. The `messages_api` harness registers no progress, so a chat on
that config simply shows the initial *«принял вопрос»* until the answer lands.

Two consequences worth knowing before reading transcripts. A follow-up costs
context tokens — turn *n* pays to re-read turns 1..n-1, so a long thread gets
steadily more expensive per answer; `/start_new_session` is how you stop paying
for a thread you are done with. And because the session persists, the agent may
end a turn with a clarifying question and expect an answer. That is deliberate:
Claude Code has no `AskUserQuestion` in headless mode (probed on 2.1.220 — the
tool is absent from the session's surface even when granted), so the turn
boundary is the only asking mechanism there is.

`agent_config_interactive` is created by **admin-ui**, not by the agent. The
agent mounts the registry read-only — the threat model requires that the agent
uid cannot reach the approval store — so it can read a config and never create
one. In practice this is invisible: admin-ui seeds the shipped set at startup
and the agent picks it up. It matters if you run the stack without admin-ui, in
which case `/healthz` reports the missing ref and the first message returns 503
with the seeding command rather than a bare 500. To seed by hand:

```bash
docker compose -f deploy/docker-compose.yml exec admin-ui python -c \
  "from sim.agent.shipped import bootstrap; from sim.registry import Registry; \
   print(bootstrap(Registry('/app/registry/registry.db')))"
```

Session transcripts live on the agent's volume under `B2E_CLAUDE_HOME` and
`B2E_SESSION_ROOT`, so a conversation survives `up --build`. If a transcript is
lost anyway, the turn falls back to a fresh session, records
`b2e.resume_failed` on the span, and says so in `errors` — an answer that
quietly lost its context is worse than one that admits it.

## 6 · Hardening applied

- `cap_drop: ALL` on every service; the only capability granted anywhere is
  `NET_BIND_SERVICE` on the proxy, to bind 80/443.
- `no-new-privileges: true` everywhere.
- All Python services run as uid 10001, never root.
- `sandbox-worker`: `network_mode: none`, `read_only: true`, tmpfs `/tmp` with
  `noexec,nosuid,nodev`, `pids_limit: 64`, `mem_limit == memswap_limit` (swap
  off), `cpus: 0.5`.
- **No service has access to the Docker socket.** That is the reason for the
  two-container spool split in `docs/skill-execution-threat-model.md` §5: a
  container that can create containers is root-equivalent on this host.
- The emulator's `/api/v2/dev/*` routes are refused with 404 in the deployed
  profile. They have no authentication and can write skill files; see the threat
  model §1.1.

## 7 · Known operational limits

- **Shared kernel.** No gVisor, no Kata, and `userns-remap` is not enabled, so a
  container uid maps 1:1 to a host uid. runc 1.3.4 is patched against
  CVE-2025-31133 / 52565 / 52881 as of writing — that is a standing patch
  obligation, not a boundary. Residual risk R-5.
- **The host user is in the `docker` group**, which is root-equivalent. Anyone
  with shell access to this host already has everything; the controls here defend
  against network attackers and against the agent, not against a local operator.
- **Single host, no redundancy.** Restarting the stack drops in-flight
  experiments; jobs are recorded but not resumed.
- **Phoenix runs with auth disabled** and relies entirely on the proxy. If the
  proxy is bypassed — for instance by publishing a port during debugging — every
  trace, including full prompts and HR payloads in transit, is readable.
