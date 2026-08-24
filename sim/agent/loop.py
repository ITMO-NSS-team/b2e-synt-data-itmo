"""The agent loop.

One turn is: render the prompt, call the model, run whatever tools it asked for,
feed the results back, repeat until it answers or the iteration cap is reached.

Three things here are experimental conditions rather than implementation detail,
and each is switchable from config:

* **context strategy** — how prior turns are packed into the window
* **budget strategy** — whether the live remaining-token-budget signal is used
* **retry policy** — how a failed tool call is handled

The loop never executes agent-authored code. It cannot: the tool surface has no
parameter that carries source text to an interpreter (see ``sim/agent/tools.py``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sim import telemetry
from sim.agent.config import AgentConfig
from sim.agent.llm import LLMClient, LLMResponse
from sim.agent.tools import HeimdallTools, render_tool_result, tool_schemas
from sim.costguard import CostGuard
from sim.fingerprint import RunFingerprint

#: Rough token estimate. Deliberately crude and clearly named: the real count
#: comes back from the API, and pretending to a precise local tokenizer would
#: invite trusting a number that is not the one the model used.
CHARS_PER_TOKEN_ESTIMATE = 3.2


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        total += len(content) if isinstance(content, str) else len(str(content))
    return int(total / CHARS_PER_TOKEN_ESTIMATE)


@dataclass
class TurnResult:
    answer: str
    iterations: int
    tool_calls: int
    heimdall_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    stop_reason: str
    trace_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def pack_context(messages: list[dict[str, Any]], config: AgentConfig) -> list[dict[str, Any]]:
    """Apply the configured context strategy.

    ``summarised`` is intentionally not a model call. Summarising with a second
    LLM request would add latency and tokens to every turn and make the RQ2
    latency numbers depend on a hidden call the trace would have to account for
    separately. The cheap structural summary here keeps the comparison clean; a
    model-based summariser is a separate condition worth adding deliberately.
    """
    if config.context_strategy == "full":
        return messages

    if config.context_strategy == "windowed":
        head = [m for m in messages[:1] if m.get("role") == "user"]
        tail = messages[-(config.windowed_turns * 2):]
        return (head + tail) if head and head[0] not in tail else tail

    if config.context_strategy == "summarised":
        if len(messages) <= config.windowed_turns * 2:
            return messages
        older = messages[:-(config.windowed_turns * 2)]
        recent = messages[-(config.windowed_turns * 2):]
        digest = "; ".join(
            f"{m.get('role')}: {str(m.get('content'))[:160]}" for m in older)
        return [{"role": "user",
                 "content": f"[сводка предыдущих шагов] {digest}"}] + recent

    return messages


def _remaining_budget(config: AgentConfig, messages: list[dict[str, Any]]) -> int:
    return max(0, config.input_token_budget - estimate_tokens(messages))


def run_turn(
    *,
    question: str,
    history: list[dict[str, Any]],
    config: AgentConfig,
    system_prompt: str,
    client: LLMClient,
    tools: HeimdallTools,
    fingerprint: RunFingerprint,
    session_id: str,
    employee_id: str,
    guard: CostGuard | None = None,
    metadata: dict[str, Any] | None = None,
) -> TurnResult:
    """Run one user turn to completion."""
    schemas = tool_schemas(config.tool_subset)
    messages: list[dict[str, Any]] = list(history) + [
        {"role": "user", "content": question}]

    iterations = 0
    tool_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0
    errors: list[str] = []
    answer = ""
    stop_reason = "completed"
    heimdall_before = tools.call_count

    with telemetry.start_run(
        "b2e.turn",
        fingerprint=fingerprint,
        session_id=session_id,
        employee_id=employee_id,
        metadata=metadata,
        question=question,
    ) as root:
        trace_id = telemetry.current_trace_id()

        while iterations < config.max_tool_iterations:
            iterations += 1

            if guard is not None:
                guard.check_before_call()

            packed = pack_context(messages, config)
            remaining = _remaining_budget(config, packed)

            # The budget signal is only *acted on* when the strategy says so.
            # Recording it either way lets a researcher compare "had the signal
            # and ignored it" against "never had it".
            if config.budget_strategy == "reactive" and remaining < config.reserve_output_tokens:
                packed = pack_context(messages, config) if config.context_strategy != "full" else packed[-8:]
            if config.budget_strategy == "planning" and remaining < config.reserve_output_tokens:
                stop_reason = "budget_exhausted"
                answer = answer or ("Не хватает контекста, чтобы продолжить "
                                    "сбор данных по этому вопросу.")
                break

            with telemetry.start_chain(f"iteration.{iterations}",
                                       **{"b2e.iteration": iterations}):
                response = _call_model(
                    client=client, config=config, system_prompt=system_prompt,
                    messages=packed, schemas=schemas, remaining=remaining,
                )

            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            call_cost = response.cost_usd(config.model_id)
            cost_usd += call_cost
            if guard is not None:
                guard.record(tokens=response.total_tokens, usd=call_cost)

            uses = response.tool_uses()
            text = (response.text() or "").strip()
            messages.append({"role": "assistant", "content": response.content})

            if uses:
                results = []
                for use in uses:
                    tool_calls += 1
                    name = use.get("name", "")
                    arguments = use.get("input") or {}
                    with telemetry.start_tool(name, parameters=arguments,
                                              tool_call_id=use.get("id")) as span:
                        payload = _dispatch_with_retry(
                            tools, name, arguments, config, errors)
                        telemetry.set_io(span, output_value=payload)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": use.get("id"),
                        "content": render_tool_result(payload),
                    })
                messages.append({"role": "user", "content": results})
                continue

            if text:
                answer = text
                stop_reason = response.stop_reason or "end_turn"
                break

            # Empty content after a tool round is not a finished answer.
            # Ask for a visible reply instead of returning answer:"".
            errors.append("empty completion; requesting a visible answer")
            messages.append({
                "role": "user",
                "content": (
                    "По данным уже полученных результатов инструментов "
                    "сформулируй итоговый ответ текстом. Если есть число — "
                    "назови его явно."
                ),
            })
        else:
            stop_reason = "max_iterations"
            answer = answer or ("Не удалось собрать ответ за отведённое число "
                                "обращений к API.")

        telemetry.set_io(root, output_value=answer)
        root.set_attribute("b2e.turn.iterations", iterations)
        root.set_attribute("b2e.turn.tool_calls", tool_calls)
        root.set_attribute("b2e.turn.heimdall_calls", tools.call_count - heimdall_before)
        root.set_attribute("b2e.turn.stop_reason", stop_reason)
        root.set_attribute("b2e.turn.cost_usd", round(cost_usd, 6))

    return TurnResult(
        answer=answer, iterations=iterations, tool_calls=tool_calls,
        heimdall_calls=tools.call_count - heimdall_before,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cost_usd=cost_usd, stop_reason=stop_reason, trace_id=trace_id,
        messages=messages, errors=errors,
    )


def _call_model(*, client: LLMClient, config: AgentConfig, system_prompt: str,
                messages: list[dict[str, Any]], schemas: list[dict[str, Any]],
                remaining: int) -> LLMResponse:
    params = {"temperature": config.temperature,
              "max_tokens": config.max_output_tokens}
    with telemetry.start_llm(
        "llm.messages.create", model=config.model_id,
        invocation_parameters=params, messages=messages,
        system=system_prompt, tools=schemas,
    ) as span:
        response = client.complete(
            model=config.model_id, system=system_prompt, messages=messages,
            tools=schemas, temperature=config.temperature,
            max_tokens=config.max_output_tokens)
        telemetry.record_llm_result(
            span,
            output_messages=response.content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cache_read=response.cache_read_tokens,
            cache_write=response.cache_write_tokens,
            stop_reason=response.stop_reason,
            remaining_budget=response.remaining_token_budget
            if response.remaining_token_budget is not None else remaining,
            budget_strategy=config.budget_strategy,
            cost_usd=response.cost_usd(config.model_id),
        )
    return response


def _dispatch_with_retry(tools: HeimdallTools, name: str, arguments: dict[str, Any],
                         config: AgentConfig, errors: list[str]) -> Any:
    """Retry transport failures only.

    A 4xx from Heimdall is *information* — the agent asked for something the API
    refuses, and learning that is the point. Retrying it would hide a
    correctable mistake behind an identical second failure and inflate the
    per-answer call count that RQ2 measures.
    """
    attempt = 0
    while True:
        payload = tools.dispatch(name, arguments)
        transport_failure = (isinstance(payload, dict)
                             and payload.get("error") == "transport")
        if not transport_failure or attempt >= config.retry_attempts:
            if isinstance(payload, dict) and "error" in payload:
                errors.append(f"{name}: {payload.get('error')}")
            return payload
        attempt += 1
        time.sleep(config.retry_backoff_seconds * attempt)
