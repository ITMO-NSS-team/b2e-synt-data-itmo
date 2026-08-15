# ITMO stand — what differs from the original

This is a copy of `b2e-synt-data`, renamespaced so it runs **alongside** the
original stack on the same host. Only deployment identity changed; no
application code was touched.

## Access

Everything is behind one Caddy proxy with HTTP Basic. Nothing is unauthenticated.

| Surface        | URL                                  |
|----------------|--------------------------------------|
| landing        | `https://<host>:8443/`               |
| research API   | `https://<host>:8443/research/docs`  |
| agent API      | `https://<host>:8443/agent/docs`     |
| admin UI       | `https://<host>:8443/admin/`         |
| Phoenix traces | `https://<host>:8443/phoenix/`       |

The edge user is `researcher`. Read the generated passwords with:

```sh
grep -E '^(ADMIN_USER|ADMIN_PASSWORD)=' deploy/.env
```

The `researcher` plaintext is **not** stored — only its bcrypt hash is, in
`BASIC_AUTH_HASH`. Re-issue it with `make hash-password` if it is lost.

TLS is Caddy's internal CA (no DNS name for this host), so browsers warn. That
is expected, not a misconfiguration.

## Deltas from the original stack

| Thing | original | here |
|---|---|---|
| compose project | `b2e-sim` | `b2e-itmo` |
| images | `b2e-sim/{python,agent}:local` | `b2e-itmo/{python,agent}:local` |
| host ports | 80, 443 | 8080, 8443 |
| edge network | 172.19.0.0/16 | 172.30.0.0/16 (pinned) |
| internal network | 172.20.0.0/16 | 172.31.0.0/16 (pinned) |
| egress relay | 172.19.0.1:10810 | 172.30.0.1:10811 |
| Phoenix project | `b2e-sim` | `b2e-itmo` |
| corpus (`DATA_DIR`) | `../data` (300 000) | `../data-small` (3 000) |
| Telegram bot | original bot | `@B2E_ITMO_test_bot` |
| passwords | — | all regenerated, shared with nothing |

Subnets are **pinned** here on purpose: `proxy-relay` binds its gateway address
by literal value from `RELAY_BIND`, so an auto-assigned subnet would silently
break agent egress on some later `up`.

`DATA_DIR` points at the small corpus because Heimdall holds the snapshot
resident — ~3.7 GiB on the full corpus. This host has 15 GiB and **zero swap**
while the original stack is also running; two full-corpus emulators would not
both survive. To use the full corpus here, either stop the other stack or run
`make data` and repoint `DATA_DIR`.

## Two fixes that were required to boot from scratch

Both are latent bugs in the original compose file. They are invisible there
because that stack's volumes were fixed by hand long ago; they bite any *fresh*
deployment.

1. **`registry_data` was never initialised.** The Dockerfile creates and chowns
   `/app/var` and `/spool` but not `/app/registry`, so a new named volume mounts
   `root:root` and uid 10001 cannot create `registry.db`. Both `admin-ui` and
   `b2e-agent` died with `sqlite3.OperationalError: unable to open database
   file`, and `/admin` and `/agent` answered 502.

2. **Nothing could create that database in the right order.** `b2e-agent` mounts
   the volume read-only by design — the threat model requires that the agent uid
   cannot self-approve a skill — so it can neither create the file nor run the
   `PRAGMA journal_mode=WAL` its `Registry` issues on open. Only `admin-ui`
   mounts it rw, and `admin-ui depends_on b2e-agent`, so the agent always started
   first and crash-looped.

Fix: a `registry-init` service (mirroring the existing `spool-init`) creates the
schema and chowns the volume before either service starts. The read-only mount
is unchanged.

**These fixes are worth porting back to the original repo** — it cannot currently
be redeployed from clean volumes.

## Running it

```sh
make up PROFILE=telegram   # or: docker compose -f deploy/docker-compose.yml \
                           #       --env-file deploy/.env --profile telegram up -d
make ps
make logs
make test                  # 724 tests, replay mode, spends nothing
```

`make up` does not start the relay. The agent needs it to reach Anthropic, so
bring the stack up with both profiles:

```sh
docker compose -f deploy/docker-compose.yml --env-file deploy/.env \
  --profile telegram --profile relay up -d
```

## Git

The `origin` remote was **removed** so this copy cannot accidentally push to the
original team's repository. Add the ITMO remote before the first push. Full
history from the original project is retained.

The changes to `deploy/docker-compose.yml` are uncommitted — review them, then
commit under the new team's authorship.
