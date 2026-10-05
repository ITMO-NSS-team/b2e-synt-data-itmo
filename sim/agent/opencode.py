"""OpenCode CLI harness with the same isolated Heimdall surface as Claude Code."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from sim import telemetry
from sim.agent.claude_code import (
    HEIMDALL_TOOLS,
    MCP_SERVER_NAME,
    SKILL_CHANNEL_TOOLS,
    ClaudeCodeResult,
    LlmCall,
    _read_bridge_log,
    harness_policy_note,
)
from sim.agent.config import AgentConfig
from sim.agent.opencode_timing import collect_native_timings
from sim.agent.provider import is_zai, llm_provider
from sim.tool_outcomes import tool_outcome


API_PRICE_ENV = {
    "input": "B2E_API_PRICE_INPUT_USD_PER_MTOK",
    "output": "B2E_API_PRICE_OUTPUT_USD_PER_MTOK",
    "cache_read": "B2E_API_PRICE_CACHE_READ_USD_PER_MTOK",
    "cache_write": "B2E_API_PRICE_CACHE_WRITE_USD_PER_MTOK",
}


def api_token_prices_from_env() -> dict[str, float] | None:
    """Read a complete, explicit API-equivalent price set, or no prices."""
    raw = {name: (os.environ.get(variable) or "").strip()
           for name, variable in API_PRICE_ENV.items()}
    if not any(raw.values()):
        return None
    missing = [API_PRICE_ENV[name] for name, value in raw.items() if not value]
    if missing:
        raise ValueError(
            "API-equivalent pricing requires all four rates; missing "
            + ", ".join(missing)
        )
    try:
        prices = {name: float(value) for name, value in raw.items()}
    except ValueError as exc:
        raise ValueError("API-equivalent token prices must be numbers") from exc
    if any(not math.isfinite(value) or value < 0 for value in prices.values()):
        raise ValueError("API-equivalent token prices must be finite and non-negative")
    return prices


def _api_cost(*, input_tokens: int, output_tokens: int,
              reasoning_tokens: int, cache_read_tokens: int,
              cache_write_tokens: int, prices: dict[str, float]) -> float:
    return (
        input_tokens * prices["input"]
        + (output_tokens + reasoning_tokens) * prices["output"]
        + cache_read_tokens * prices["cache_read"]
        + cache_write_tokens * prices["cache_write"]
    ) / 1_000_000


def _unpriced_cost_mode(model: str) -> str:
    return "subscription" if model.startswith("zai-coding-plan/") else "unavailable"


class OpenCodeHarness:

    def __init__(
        self,
        *,
        heimdall_url: str,
        heimdall_token: str,
        bridge_path: str = "/app/heimdall/bridge.py",
        runner_path: str = "/opt/skills/run",
        opencode_bin: str = "opencode",
        python_bin: str = "python3.12",
        proxy: dict[str, str] | None = None,
        workdir: str | None = None,
        session_root: str = "var/sessions",
        opencode_home: str | None = None,
        timeout_seconds: int = 600,
    ) -> None:
        self.heimdall_url = heimdall_url
        self.heimdall_token = heimdall_token
        self.bridge_path = bridge_path
        self.runner_path = runner_path
        self.opencode_bin = opencode_bin
        self.python_bin = python_bin
        self.proxy = proxy or {}
        self.workdir = workdir
        self.session_root = session_root
        self.opencode_home = opencode_home
        self.timeout = timeout_seconds

    def model_name(self, model_id: str) -> str:
        if "/" in model_id:
            return model_id
        provider = "zai-coding-plan" if is_zai() else llm_provider()
        return f"{provider}/{model_id}"

    def session_workdir(self, config: AgentConfig,
                        b2e_session_id: str | None) -> Path | None:
        if config.conversation_mode != "resume" or not b2e_session_id:
            return None
        safe = "".join(c for c in str(b2e_session_id) if c.isalnum() or c in "-_")
        return Path(self.session_root) / safe

    def _bridge_environment(self, employee_id: str, config: AgentConfig,
                            trace_log: str) -> dict[str, str]:
        environment = {
            "HEIMDALL_URL": self.heimdall_url,
            "HEIMDALL_TOKEN": self.heimdall_token,
            "HEIMDALL_EMPLOYEE_ID": str(employee_id),
            "HEIMDALL_CHANNEL": "v2",
            "HEIMDALL_TOOL_SUBSET": ",".join(
                name for name in HEIMDALL_TOOLS if name in config.tool_subset
            ),
            "HR_TRACE_LOG": trace_log,
        }
        traceparent = telemetry.current_traceparent()
        if traceparent:
            environment["TRACEPARENT"] = traceparent
        return environment

    def _permissions(self, config: AgentConfig) -> dict[str, Any]:
        permissions: dict[str, Any] = {"*": "deny"}
        for name in HEIMDALL_TOOLS:
            if name in config.tool_subset:
                permissions[f"{MCP_SERVER_NAME}_{name}"] = "allow"
        if config.code_execution == "allowed":
            permissions.update({
                "bash": "allow", "edit": "allow", "read": "allow",
                "glob": "allow", "grep": "allow",
            })
        elif set(config.tool_subset) & SKILL_CHANNEL_TOOLS:
            permissions["bash"] = {
                "*": "deny",
                f"{self.runner_path} *": "allow",
            }
        return permissions

    def build_config(self, *, config: AgentConfig, system_prompt: str,
                     employee_id: str, trace_log: str) -> dict[str, Any]:
        model = self.model_name(config.model_id)
        provider_id, model_id = model.split("/", 1)
        provider: dict[str, Any] = {
            provider_id: {
                "models": {
                    model_id: {
                        "name": model_id,
                        "limit": {
                            "context": config.context_window_tokens,
                            "output": config.max_output_tokens,
                        },
                    }
                }
            }
        }
        base_url = (os.environ.get("ZAI_OPENAI_BASE_URL") or "").strip()
        if is_zai() and base_url:
            provider[provider_id]["options"] = {"baseURL": base_url}
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": model,
            "share": "disabled",
            "autoupdate": False,
            "experimental": {"openTelemetry": True},
            "provider": provider,
            "mcp": {
                MCP_SERVER_NAME: {
                    "type": "local",
                    "command": [self.python_bin, self.bridge_path],
                    "enabled": True,
                    "environment": self._bridge_environment(
                        employee_id, config, trace_log
                    ),
                }
            },
            "agent": {
                "b2e": {
                    "description": "Isolated B2E benchmark agent",
                    "mode": "primary",
                    "model": model,
                    "prompt": system_prompt + "\n" + self.harness_note(config),
                    "temperature": config.temperature,
                    "permission": self._permissions(config),
                }
            },
            "default_agent": "b2e",
            # Model descriptions can exceed 100 KB on one line. The default
            # 50 KB cap spills them to files the isolated agent cannot read.
            "tool_output": {"max_bytes": 512 * 1024, "max_lines": 10_000},
        }

    def harness_note(self, config: AgentConfig) -> str:
        """Shared policy with directly available MCP tools, not Claude's ToolSearch."""
        tools = ", ".join(
            f"`{name}`" for name, grant in self._permissions(config).items()
            if name.startswith(f"{MCP_SERVER_NAME}_") and grant == "allow"
        )
        return harness_policy_note(
            config, runner=self.runner_path,
            tool_discovery=(f"* Инструменты Heimdall доступны напрямую: {tools}. "
                            "Других инструментов Heimdall у тебя нет."),
        )

    def child_env(self, config_path: Path) -> dict[str, str]:
        home = Path(self.opencode_home or os.environ.get("HOME", "/tmp"))
        home.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_STATE_HOME": str(home / ".local" / "state"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        }
        if is_zai():
            token = ((os.environ.get("ZAI_API_KEY") or "").strip()
                     or (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip())
            if not token:
                raise RuntimeError(
                    "OpenCode with LLM_PROVIDER=zai requires ZAI_API_KEY or "
                    "ANTHROPIC_AUTH_TOKEN"
                )
            env["ZHIPU_API_KEY"] = token
        else:
            api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError(
                    "OpenCode with LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY"
                )
            env["ANTHROPIC_API_KEY"] = api_key
        env.update(self.proxy)
        return env

    def build_argv(self, question: str, *, config: AgentConfig,
                   resume_session_id: str | None = None) -> list[str]:
        argv = [
            self.opencode_bin, "run", "--pure", "--format", "json",
            "--model", self.model_name(config.model_id), "--agent", "b2e",
        ]
        if config.conversation_mode == "resume" and resume_session_id:
            argv += ["--session", resume_session_id]
        return argv + ["--", question]

    def run(self, *, question: str, config: AgentConfig, system_prompt: str,
            employee_id: str, keep_stream: bool = True,
            b2e_session_id: str | None = None,
            resume_session_id: str | None = None,
            on_event: Callable[[dict[str, Any]], None] | None = None,
            ) -> ClaudeCodeResult:
        persistent = self.session_workdir(config, b2e_session_id)
        workdir = Path(self.workdir or persistent
                       or tempfile.mkdtemp(prefix="b2e-opencode-"))
        workdir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:16]
        bridge_log = workdir / f"heimdall-{run_id}.jsonl"
        config_path = workdir / f"opencode-{run_id}.json"
        config_path.write_text(json.dumps(self.build_config(
            config=config, system_prompt=system_prompt,
            employee_id=employee_id, trace_log=str(bridge_log),
        )), encoding="utf-8")
        if config.conversation_mode != "resume":
            resume_session_id = None
        attempts_started = time.perf_counter()
        result = self._invoke(
            question, config=config, config_path=config_path, workdir=workdir,
            resume_session_id=resume_session_id, keep_stream=keep_stream,
            on_event=on_event,
        )
        if resume_session_id and result.is_error and not result.answer:
            failed = result
            retry_delay_ms = (time.perf_counter() - attempts_started) * 1000
            result = self._invoke(
                question, config=config, config_path=config_path, workdir=workdir,
                resume_session_id=None, keep_stream=keep_stream, on_event=on_event,
            )
            result.resumed_failed = True
            warning = (f"resume of {resume_session_id} failed ({failed.error}); "
                       "retried in a new session without history")
            result.error = warning + (f"; {result.error}" if result.error else "")
            before, after = failed.api_duration_ms, result.api_duration_ms
            result.api_duration_ms = (before + after
                                      if before is not None and after is not None else None)
            if failed.llm_calls:
                result.ttft_stream_ms = failed.ttft_stream_ms
                result.time_to_request_ms = failed.time_to_request_ms
            elif result.time_to_request_ms is not None:
                result.time_to_request_ms += retry_delay_ms
            result.timing_calls += failed.timing_calls
            # Both invocations belong to this turn, not just the successful one.
            # Do not silently discard paid work from a failed resume.
            for field in ("num_turns", "input_tokens", "output_tokens",
                          "cache_read_tokens", "cache_creation_tokens", "duration_ms"):
                setattr(result, field, getattr(failed, field) + getattr(result, field))
            if failed.reasoning_tokens is not None:
                result.reasoning_tokens = (
                    (result.reasoning_tokens or 0) + failed.reasoning_tokens)
            for field in ("llm_calls", "tool_calls", "permission_denials"):
                setattr(result, field, getattr(failed, field) + getattr(result, field))
            if failed.cost_usd is not None and result.cost_usd is not None:
                if failed.cost_mode == result.cost_mode:
                    result.cost_usd += failed.cost_usd
                else:
                    result.cost_usd, result.cost_mode = None, "unavailable"
            elif failed.total_tokens or failed.llm_calls or failed.cost_usd is not None:
                result.cost_usd = None
                if failed.cost_mode != result.cost_mode:
                    result.cost_mode = "unavailable"
        result.system_suffix = system_prompt + "\n" + self.harness_note(config)
        result.runner_path = self.runner_path
        result.bridge_calls = _read_bridge_log(bridge_log)
        config_path.unlink(missing_ok=True)
        if not keep_stream:
            bridge_log.unlink(missing_ok=True)
        if self.workdir is None and persistent is None:
            shutil.rmtree(workdir, ignore_errors=True)
        return result

    def _invoke(self, question: str, *, config: AgentConfig, config_path: Path,
                workdir: Path, resume_session_id: str | None, keep_stream: bool,
                on_event: Callable[[dict[str, Any]], None] | None,
                ) -> ClaudeCodeResult:
        started = time.perf_counter()
        started_ns = time.time_ns()
        token_prices = api_token_prices_from_env()
        lines: list[str] = []
        timed_out = False
        err_path = workdir / "opencode.stderr"
        with collect_native_timings() as timings, open(err_path, "w+", encoding="utf-8") as err:
            env = self.child_env(config_path)
            env["OTEL_EXPORTER_OTLP_ENDPOINT"] = timings.endpoint
            # `opencode run` may exit without flushing the SDK's default 5s
            # batch. Drain locally while it is alive; missing batches remain
            # explicitly unavailable, never replaced with estimated timings.
            env["OTEL_BSP_SCHEDULE_DELAY"] = "10"
            env["NO_PROXY"] = env.get("NO_PROXY", "") + ",127.0.0.1,localhost"
            proc = subprocess.Popen(
                self.build_argv(question, config=config,
                                resume_session_id=resume_session_id),
                stdout=subprocess.PIPE, stderr=err, text=True,
                env=env, cwd=str(workdir), bufsize=1,
            )

            def kill() -> None:
                nonlocal timed_out
                timed_out = True
                proc.kill()

            watchdog = threading.Timer(self.timeout, kill)
            watchdog.start()
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    lines.append(line)
                    if on_event is not None:
                        _observe_open_code(line, on_event)
                proc.wait()
            finally:
                watchdog.cancel()
                if proc.stdout is not None:
                    proc.stdout.close()
            err.seek(0)
            stderr_text = err.read()
        err_path.unlink(missing_ok=True)
        duration_ms = int((time.perf_counter() - started) * 1000)
        if timed_out:
            model = self.model_name(config.model_id)
            return ClaudeCodeResult(
                answer="", session_id=None, num_turns=0, input_tokens=0,
                output_tokens=0, cache_read_tokens=0, cache_creation_tokens=0,
                cost_usd=None, cost_mode=_unpriced_cost_mode(model),
                duration_ms=duration_ms, is_error=True,
                error=f"opencode timed out after {self.timeout}s",
            )
        stream = "".join(lines)
        result = parse_stream(stream, model=self.model_name(config.model_id),
                              duration_ms=duration_ms,
                              token_prices=token_prices)
        timings.apply(result, started_ns=started_ns)
        if keep_stream:
            path = workdir / f"opencode-{uuid.uuid4().hex[:16]}.stream.jsonl"
            path.write_text(stream, encoding="utf-8")
            result.stream_path = str(path)
        if proc.returncode != 0:
            result.is_error = True
            result.error = result.error or stderr_text[:2000]
        return result


def _tool_name(name: str) -> str:
    prefix = f"{MCP_SERVER_NAME}_"
    if name.startswith(prefix):
        return f"mcp__{MCP_SERVER_NAME}__{name.removeprefix(prefix)}"
    return "Bash" if name == "bash" else name


def _observe_open_code(line: str,
                       callback: Callable[[dict[str, Any]], None]) -> None:
    try:
        event = json.loads(line.strip() or "{}")
    except ValueError:
        return
    if event.get("type") != "tool_use":
        return
    part = event.get("part") or {}
    state = part.get("state") or {}
    if state.get("status") not in ("completed", "error"):
        return
    output = state.get("output") if state.get("output") is not None else state.get("error")
    outcome = tool_outcome(output, is_error=state.get("status") == "error")
    timing = state.get("time") or {}
    call_id = str(part.get("callID") or part.get("id") or "")
    name = _tool_name(str(part.get("tool") or ""))
    callback({
        "type": "assistant",
        "message": {"id": str(part.get("messageID") or ""), "content": [{
            "type": "tool_use", "id": call_id, "name": name,
            "input": state.get("input") or {},
            "_b2e_started_ns": int(timing.get("start") or
                                    event.get("timestamp") or 0) * 1_000_000,
        }]},
    })
    callback({
        "type": "user",
        "message": {"content": [{
            "type": "tool_result", "tool_use_id": call_id,
            "content": output,
            "is_error": outcome["is_error"],
            "permission_denied": outcome["permission_denied"],
            "_b2e_ended_ns": int(timing.get("end") or
                                  event.get("timestamp") or 0) * 1_000_000,
        }]},
    })


def parse_stream(stream: str, *, model: str, duration_ms: int = 0,
                 token_prices: dict[str, float] | None = None
                 ) -> ClaudeCodeResult:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    for line in stream.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
            if event.get("type") == "error":
                errors.append(json.dumps(event.get("error"), ensure_ascii=False))

    session_id = next((str(e["sessionID"]) for e in events
                       if e.get("sessionID")), None)
    content_parts = [
        (str(e.get("type")), e.get("part") or {})
        for e in events if e.get("type") in ("reasoning", "text")
    ]
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        if event.get("type") != "tool_use":
            continue
        part = event.get("part") or {}
        state = part.get("state") or {}
        output = state.get("output") if state.get("output") is not None else state.get("error")
        call_id = str(part.get("callID") or part.get("id") or "")
        tool_calls_by_id[call_id or f"event-{index}"] = {
            "id": call_id,
            "name": _tool_name(str(part.get("tool") or "")),
            "input": state.get("input") or {},
            "output": output,
            **tool_outcome(output, is_error=state.get("status") == "error",
                           unfinished=state.get("status") not in ("completed", "error")),
        }
    tool_calls = list(tool_calls_by_id.values())

    starts = {
        str((e.get("part") or {}).get("messageID") or ""): int(e.get("timestamp") or 0)
        for e in events if e.get("type") == "step_start"
    }
    finishes = [e for e in events if e.get("type") == "step_finish"]
    final_message_id = str(
        ((finishes[-1].get("part") or {}).get("messageID") or "")
    ) if finishes else ""
    answer = "\n".join(
        str(part.get("text") or "")
        for event_type, part in content_parts
        if event_type == "text"
        and str(part.get("messageID") or "") == final_message_id
    ).strip()
    llm_calls: list[LlmCall] = []
    input_tokens = output_tokens = reasoning_tokens = cache_read = cache_write = 0
    raw_costs = [float((event.get("part") or {}).get("cost") or 0.0)
                 for event in finishes]
    has_reported_cost = any(cost > 0 for cost in raw_costs)
    cost_mode = ("reported" if has_reported_cost else
                 "api_estimate" if token_prices is not None else
                 _unpriced_cost_mode(model))
    costs: list[float] = []
    for event in finishes:
        part = event.get("part") or {}
        tokens = part.get("tokens") or {}
        cache = tokens.get("cache") or {}
        call_input = int(tokens.get("input") or 0)
        call_output = int(tokens.get("output") or 0)
        call_reasoning = int(tokens.get("reasoning") or 0)
        call_read = int(cache.get("read") or 0)
        call_write = int(cache.get("write") or 0)
        input_tokens += call_input
        output_tokens += call_output
        reasoning_tokens += call_reasoning
        cache_read += call_read
        cache_write += call_write
        if has_reported_cost:
            call_cost: float | None = float(part.get("cost") or 0.0)
        elif token_prices is not None:
            call_cost = _api_cost(
                input_tokens=call_input, output_tokens=call_output,
                reasoning_tokens=call_reasoning,
                cache_read_tokens=call_read, cache_write_tokens=call_write,
                prices=token_prices,
            )
        else:
            call_cost = None
        if call_cost is not None:
            costs.append(call_cost)
        message_id = str(part.get("messageID") or "")
        ended_ms = int(event.get("timestamp") or 0)
        contents = [
            {"type": (telemetry.REASONING if event_type == "reasoning"
                      else telemetry.TEXT),
             "text": str(item.get("text") or "")}
            for event_type, item in content_parts
            if str(item.get("messageID") or "") == message_id
        ]
        llm_calls.append(LlmCall(
            message_id=message_id, model=model,
            input_tokens=call_input, output_tokens=call_output,
            cache_read_tokens=call_read, cache_creation_tokens=call_write,
            reasoning_tokens=call_reasoning, cost_usd=call_cost,
            cost_mode=cost_mode,
            stop_reason=str(part.get("reason") or ""),
            started_ns=starts.get(message_id, ended_ms) * 1_000_000,
            ended_ns=ended_ms * 1_000_000,
            contents=contents,
            tool_calls=[call for call in tool_calls
                        if any(str((e.get("part") or {}).get("messageID") or "") == message_id
                               and str((e.get("part") or {}).get("callID") or
                                       (e.get("part") or {}).get("id") or "") == call["id"]
                               for e in events if e.get("type") == "tool_use")],
        ))

    return ClaudeCodeResult(
        answer=answer, session_id=session_id,
        num_turns=len(finishes) or sum(e.get("type") == "step_start" for e in events),
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_read_tokens=cache_read, cache_creation_tokens=cache_write,
        reasoning_tokens=reasoning_tokens,
        cost_usd=(sum(costs) if finishes and len(costs) == len(finishes)
                  else None),
        cost_mode=cost_mode, duration_ms=duration_ms,
        tool_calls=tool_calls, llm_calls=llm_calls,
        permission_denials=[{"tool_name": call["name"], "tool_use_id": call["id"],
                            "tool_input": call["input"]}
                           for call in tool_calls if call["permission_denied"]],
        transcript_status="missing" if not finishes else "",
        is_error=bool(errors), error="\n".join(errors),
    )
