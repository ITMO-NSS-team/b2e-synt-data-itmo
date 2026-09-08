"""Agent configuration — everything the agent's behaviour depends on.

The rule from the spec is that *nothing* influencing behaviour may be a constant
in the code. If it is not in this object, it cannot be varied, and if it cannot
be varied it cannot be an experimental condition. Every field here is therefore
part of a versioned config committed to ``sim.registry``, and its version string
goes into the run fingerprint.

The context window is read from config rather than hardcoded, for the reason the
spec gives: a hardcoded 200 000 becomes a lie the moment the model changes, and
the failure is silent — the agent simply starts truncating differently than the
trace claims.

Some fields constrain each other, and ``__post_init__`` refuses the impossible
combinations instead of ignoring them — see ``code_execution`` and
``context_strategy`` below. A declared-but-inert condition is worse than a
crash: it still gets its own ``condition_id``, so the run *looks* like an arm
that was measured. Note where that check lives: ``sim.registry`` is a dumb
versioned store and validates nothing, so a caller writing a body straight
through ``registry.commit`` can still persist an illegal pair. The refusal
happens when the blob is loaded into this dataclass, which is the only path any
harness takes to read a config.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

#: Default model under test.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

#: Strategies for the live remaining-token-budget signal that Haiku 4.5 returns
#: after each tool call. Whether the agent *uses* it is a research variable, so
#: it is switchable rather than assumed.
BUDGET_STRATEGIES = (
    "ignore",        # never look at it — the baseline
    "reactive",      # compact context once the remaining budget drops below a floor
    "planning",      # let remaining budget cap how many more tool calls to attempt
)

#: How the agent assembles context for each model call.
#:
#: Only ``sim.agent.loop.pack_context`` implements these, and that packer is
#: reached only through ``harness="messages_api"``. So:
#:
#: * ``full`` — no loop-level packing at all. Honoured everywhere, because it is
#:   the honest label for "the harness applies no strategy of its own", which is
#:   exactly what ``claude_code`` does: the CLI owns its own window and compacts
#:   on its own terms.
#: * ``windowed`` / ``summarised`` — implemented by ``pack_context``, reachable
#:   only under ``messages_api``. Declaring either on ``claude_code`` used to be
#:   accepted and then do nothing: ``sim/agent/claude_code.py`` never reads the
#:   field, in either conversation mode. A sweep over the three values on the
#:   default harness would therefore have produced three identical arms with
#:   three different ``condition_id``s and supported the conclusion "context
#:   handling does not matter". ``__post_init__`` refuses that pair now.
CONTEXT_STRATEGIES = (
    "full",          # every prior message, until the window forces truncation
    "windowed",      # last N turns only
    "summarised",    # older turns replaced by a running summary
)

#: Whether the agent may write code and run it in the same session.
#:
#: ``forbidden`` is the project's premise and the default: the agent may only
#: call Heimdall and execute predefined code attached to an approved skill.
#:
#: ``allowed`` exists to test the obvious rival hypothesis — that the constraint
#: is what costs the agent its accuracy, and lifting it would improve every
#: metric on the question basket. That hypothesis deserves a measurement rather
#: than an assumption, and without this arm the whole project is arguing with
#: itself. Two warnings attach to it, both in docs/skill-execution-threat-model.md §11:
#:
#: * The comparison is only sound where the agent **cannot reach the corpus
#:   files**. Once arbitrary Python runs, no tool policy constrains ``open()``,
#:   and ``truth/people.json`` is the answer key. In the compose deployment the
#:   corpus is not mounted into the agent, so the arm is fair. On a host dev run
#:   it is not, and the result would be a measurement of cheating.
#: * The agent's process holds the model credential. An arm that runs arbitrary
#:   code can read it. Use a dedicated low-quota token for this arm.
CODE_EXECUTION_MODES = ("forbidden", "allowed")

#: How a turn is actually executed.
#:
#: ``claude_code`` runs a headless `claude -p` session. It is the default because
#: the subscription OAuth token is scoped to Claude Code, and because a Claude
#: Code session already *is* a ReAct agent with a fixed tool surface — the thing
#: under test — rather than a reimplementation of one.
#:
#: ``messages_api`` drives the loop in ``sim.agent.loop`` against the Messages
#: API. Kept because it is deterministic under replay, which CI needs, and
#: because it gives finer-grained spans when a study is about the loop itself.
HARNESSES = ("claude_code", "messages_api")

#: Whether a session is a conversation or a sequence of independent turns.
#:
#: ``stateless`` is the default and the measurement baseline: every turn starts
#: cold. A batch arm must stay here — runs that share context are not
#: independent samples, and the per-turn token counts stop being comparable once
#: turn *n* is paying to re-read turns 1..n-1.
#:
#: What enforces it on ``claude_code`` is the absence of ``--resume``; for
#: ``messages_api`` it is an empty history. This comment used to say
#: ``--no-session-persistence``, and that was worth more than a stale word: the
#: flag conflated *sharing context*, which is this experimental variable, with
#: *filing a transcript*, which is the CLI's own bookkeeping. Believing the two
#: were one thing cost the stand every per-model-call measurement for fifty
#: traces and sent two expensive alternatives — the CLI's OTel exporter, a
#: TLS-terminating egress proxy — to be designed before anyone tried dropping
#: the flag. Verified on the live stack when it was dropped: two turns in the
#: same working directory, persistence on, no ``--resume``, and the second had
#: no memory of the first.
#:
#: ``resume`` makes the session an actual dialogue. ``claude_code`` reopens the
#: same headless session with ``--resume``, so the model sees its own prior tool
#: calls and results, not a flattened transcript. This exists for the Telegram
#: bridge, where a researcher asks a follow-up and means it as a follow-up.
#:
#: It is a config field rather than a property of the calling surface because
#: the spec's rule applies: it changes behaviour, so it must be versioned and
#: must reach the fingerprint. Two runs that differ only in this are two
#: conditions, and ``condition_id`` has to say so.
CONVERSATION_MODES = ("stateless", "resume")

#: Long-horizon memory shape. This is RQ3's independent variable: does importing
#: an employee's data from other systems help, or pollute?
MEMORY_STRATEGIES = (
    "none",              # no memory between sessions
    "current_mart",      # only the primary HR mart
    "hr_plus_external",  # HR mart plus the external talent-radar mirror
    "everything",        # including the stale _dep replica
    "reflected",         # RQ4: a sim.reflection memory pack, pinned by version
)

#: A registry ref of the form ``name@N`` — never a floating head. Matched
#: against ``memory_ref`` because a floating ``memory_isolated_i1`` would let
#: the pack a running experiment reads change under it the moment reflection
#: commits a new epoch, which destroys comparability between two turns that
#: are supposed to be the same condition.
_PINNED_REF = re.compile(r"^[^@\s]+@[1-9][0-9]*$")


@dataclass
class AgentConfig:
    """A complete, versionable description of one agent configuration."""

    # ---- model
    model_id: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_output_tokens: int = 64_000
    context_window_tokens: int = 200_000

    # ---- prompt
    system_prompt_ref: str = "system_prompt"        # registry ref, head or pinned
    prompt_variables: dict[str, Any] = field(default_factory=dict)

    # ---- capability surface
    tool_subset: tuple[str, ...] = (
        "list_models", "describe_model", "get_docs",
        "mcp_query", "find_skills", "get_skill",
    )
    skill_registry_ref: str = "skill_registry"

    # ---- execution
    harness: str = "claude_code"
    code_execution: str = "forbidden"
    conversation_mode: str = "stateless"

    # ---- behaviour
    budget_strategy: str = "ignore"
    context_strategy: str = "full"
    memory_strategy: str = "none"
    #: Pinned ``name@N`` ref of a ``sim.reflection.memory.MemoryPack``. Only
    #: meaningful — and only legal — when ``memory_strategy == "reflected"``;
    #: see the refusals in ``__post_init__``.
    memory_ref: str = ""

    # ---- loop control
    max_tool_iterations: int = 12
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.5

    # ---- packing
    reserve_output_tokens: int = 8_000
    windowed_turns: int = 6

    def __post_init__(self) -> None:
        if self.harness not in HARNESSES:
            raise ValueError(f"harness {self.harness!r} not in {HARNESSES}")
        if self.code_execution not in CODE_EXECUTION_MODES:
            raise ValueError(
                f"code_execution {self.code_execution!r} not in {CODE_EXECUTION_MODES}")
        if self.code_execution == "allowed" and self.harness != "claude_code":
            # Refuse rather than ignore. The messages_api loop has no tool that
            # can execute code, so this combination would produce runs labelled
            # "code allowed" in which no code could ever run — a whole arm of
            # the experiment quietly measuring the control condition.
            raise ValueError(
                "code_execution='allowed' requires harness='claude_code'; the "
                "messages_api loop exposes no tool capable of executing code, "
                "so the combination would mislabel the control arm")
        if self.conversation_mode not in CONVERSATION_MODES:
            raise ValueError(
                f"conversation_mode {self.conversation_mode!r} not in "
                f"{CONVERSATION_MODES}")
        if self.budget_strategy not in BUDGET_STRATEGIES:
            raise ValueError(
                f"budget_strategy {self.budget_strategy!r} not in {BUDGET_STRATEGIES}")
        if self.context_strategy not in CONTEXT_STRATEGIES:
            raise ValueError(
                f"context_strategy {self.context_strategy!r} not in {CONTEXT_STRATEGIES}")
        if self.context_strategy != "full" and self.harness != "messages_api":
            # Same argument as the code_execution refusal above, other axis.
            # Only sim.agent.loop.pack_context implements windowing and
            # summarisation, and only the messages_api path calls it; the Claude
            # Code CLI owns its own context window. So this pair used to name a
            # condition that ran identically to `full` — an experiment sweeping
            # the axis on the default harness would have measured nothing and
            # concluded that context handling does not matter.
            #
            # `full` is deliberately exempt: it is the default, every stored
            # config carries it, and on claude_code it is the true description
            # of what the harness does — no loop-level packing.
            raise ValueError(
                f"context_strategy={self.context_strategy!r} requires "
                f"harness='messages_api'; only the sim.agent.loop packer "
                f"implements windowing and summarisation, and the Claude Code "
                f"CLI owns its own context window, so the combination would "
                f"label an arm that runs identically to 'full'. Set "
                f"harness='messages_api', or leave context_strategy='full'.")
        if self.memory_strategy not in MEMORY_STRATEGIES:
            raise ValueError(
                f"memory_strategy {self.memory_strategy!r} not in {MEMORY_STRATEGIES}")
        if self.memory_strategy == "reflected" and not self.memory_ref:
            # Same argument as the code_execution and context_strategy
            # refusals above: a declared-but-inert condition is worse than a
            # crash. "reflected" with no ref would render an empty memory
            # section and silently collapse into the "none" condition while
            # the fingerprint kept claiming otherwise.
            raise ValueError(
                "memory_strategy='reflected' requires memory_ref to be set to "
                "a pinned sim.reflection memory pack ref")
        if self.memory_ref and self.memory_strategy != "reflected":
            raise ValueError(
                f"memory_ref is set ({self.memory_ref!r}) but "
                f"memory_strategy={self.memory_strategy!r} is not 'reflected'; "
                f"a ref that nothing reads is a condition nobody applied")
        if self.memory_ref and not _PINNED_REF.match(self.memory_ref):
            # A floating head (bare "memory_isolated_i1", no "@N") would let
            # the pack a running experiment reads change the moment reflection
            # commits the next epoch — two turns recorded under the same
            # condition_id would then have been served different memory.
            raise ValueError(
                f"memory_ref {self.memory_ref!r} is not pinned to a version; "
                f"expected the form 'name@N', e.g. 'memory_isolated_i1@3'")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("max_output_tokens must be smaller than the context window")
        if not self.tool_subset:
            raise ValueError("tool_subset must not be empty: an agent with no tools "
                             "cannot reach Heimdall and every answer would be invented")
        # Membership, not just non-emptiness. A typo'd tool name used to pass
        # here and then be silently dropped by both harnesses, producing exactly
        # the empty surface the check above refuses — while the fingerprint
        # claimed the full config. This is the same argument `from_dict` already
        # makes for unknown keys, applied to values.
        from sim.agent.tools import KNOWN_TOOLS

        unknown = sorted(set(self.tool_subset) - set(KNOWN_TOOLS))
        if unknown:
            raise ValueError(
                f"unknown tools in tool_subset: {unknown}; known: {list(KNOWN_TOOLS)}")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_subset"] = list(self.tool_subset)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        payload = dict(data)
        if "tool_subset" in payload:
            payload["tool_subset"] = tuple(payload["tool_subset"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            # Refuse rather than ignore: a typo'd key that is silently dropped
            # produces a run that claims a condition it did not apply.
            raise ValueError(f"unknown agent config fields: {sorted(unknown)}")
        return cls(**payload)

    @property
    def input_token_budget(self) -> int:
        """How much of the window may be spent on input."""
        return self.context_window_tokens - self.reserve_output_tokens
