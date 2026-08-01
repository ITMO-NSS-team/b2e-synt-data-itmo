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
