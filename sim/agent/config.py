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
"""
from __future__ import annotations

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
#: cold. For ``claude_code`` that is ``--no-session-persistence``; for
#: ``messages_api`` it is an empty history. A batch arm must stay here — runs
#: that share context are not independent samples, and the per-turn token counts
#: stop being comparable once turn *n* is paying to re-read turns 1..n-1.
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
)


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
        if self.memory_strategy not in MEMORY_STRATEGIES:
            raise ValueError(
                f"memory_strategy {self.memory_strategy!r} not in {MEMORY_STRATEGIES}")
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
