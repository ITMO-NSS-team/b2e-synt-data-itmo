"""The configs a deployment must have before it can answer anything.

Why this is its own module
--------------------------
Because the service that *needs* these configs is not allowed to write them.

``b2e-agent`` mounts the registry read-only, on purpose: the threat model says
the agent uid must not be able to touch the approval store, and an earlier
read-write mount made that false — in the code-execution arm a single UPDATE
self-approves every draft skill. So the agent can read a config and can never
create one.

That was invisible for as long as the shipped set never changed. The refs were
written once, during an era when the mount was still read-write, and
``_bootstrap_registry`` has been dead code ever since: every ``head()`` returns
a version, so no commit is ever attempted. Adding a *new* shipped config is what
wakes it up — and the write it then attempts raises
``sqlite3.OperationalError: attempt to write a readonly database`` from
``AgentState.__init__``, which takes the service down at boot.

The split here follows the mount. ``bootstrap`` is called by whoever holds a
writable registry — that is ``admin-ui``, the operator surface — and ``audit``
is called by the agent, which reports what is missing instead of trying to fix
it. A service that refuses to start because a second config is absent would be
making the same mistake in the other direction.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from sim.agent.config import AgentConfig, DEFAULT_OPENROUTER_MODEL
from sim.agent.prompt import DEFAULT_SYSTEM_PROMPT

#: Registry ref of the shipped conversational config. One name, imported by the
#: bootstrap, the agent and the Telegram bridge, so the three cannot drift onto
#: three different strings.
INTERACTIVE_CONFIG_REF = "agent_config_interactive"

#: Registry ref of the shipped OpenRouter / messages_api config used by the
#: laptop golden path. Separate from ``agent_config`` so the original Claude
#: Code experiment arm keeps its condition_id.
OPENROUTER_CONFIG_REF = "agent_config_openrouter"


def shipped_configs() -> dict[str, tuple[str, dict[str, Any], str]]:
    """ref -> (kind, body, note). Built fresh so no caller can mutate it."""
    return {
        "system_prompt": (
            "prompt", {"template": DEFAULT_SYSTEM_PROMPT}, "shipped default"),
        "agent_config": (
            "agent", AgentConfig().as_dict(), "shipped default"),
        # Differs from `agent_config` in exactly one field. The Telegram bridge
        # opens sessions against it, so a researcher gets a conversation while
        # batch arms keep stateless turns — and because the two are separate
        # refs, the fingerprint gives them separate condition_ids rather than
        # blurring one condition into the other.
        INTERACTIVE_CONFIG_REF: (
            "agent", AgentConfig(conversation_mode="resume").as_dict(),
            "shipped default, resumable sessions"),
        OPENROUTER_CONFIG_REF: (
            "agent",
            AgentConfig(
                harness="messages_api",
                model_id=DEFAULT_OPENROUTER_MODEL,
                max_output_tokens=4096,
            ).as_dict(),
            "laptop golden path: OpenRouter via messages_api",
        ),
        "skill_registry": ("skills", {"active": []}, "empty registry"),
    }


def bootstrap(registry) -> list[str]:
    """Create any shipped config that is absent. Returns the refs created.

    Idempotent, and safe to call from more than one service: a ref that already
    exists is left exactly as it is, including any operator edits on top of it.
    Only the write-owning service should call this — see the module docstring.
    """
    created = []
    for ref, (kind, body, note) in shipped_configs().items():
        if registry.head(ref) is None:
            registry.commit(ref, kind, body, actor="bootstrap", note=note)
            created.append(ref)
    return created


def audit(registry) -> tuple[list[str], str]:
    """Report which shipped configs are missing, without writing anything.

    Returns ``(missing_refs, human_status)``. The status is meant for
    ``/healthz``: a missing ref does not stop the service, but it does break
    every request that names it, and "the bot returns 500 on the first message"
    is a far worse way to discover that than a health check saying so.
    """
    missing = [ref for ref in shipped_configs() if registry.head(ref) is None]
    if not missing:
        return [], "complete"
    return missing, (
        f"missing {missing}; the registry is read-only to this service. "
        f"Start admin-ui, which owns registry writes, or seed it once with: "
        f"docker compose -f deploy/docker-compose.yml exec admin-ui "
        f"python -c \"from sim.agent.shipped import bootstrap; "
        f"from sim.registry import Registry; "
        f"print(bootstrap(Registry('/app/registry/registry.db')))\"")


def bootstrap_if_writable(registry) -> str:
    """Best-effort bootstrap for a service that may or may not hold the pen.

    Used by the agent. It will normally do nothing at all — the registry is
    mounted read-only there — but a host dev run points the same code at a
    writable file, and refusing to seed it would make `make serve` fail for a
    reason that has nothing to do with what the developer was testing.
    """
    try:
        created = bootstrap(registry)
    except sqlite3.OperationalError:
        # Read-only: expected in the container. Say what is missing rather than
        # pretending the call succeeded.
        _missing, status = audit(registry)
        return status
    _missing, status = audit(registry)
    return f"seeded {created}" if created else status
