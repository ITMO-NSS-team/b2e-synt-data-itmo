"""C2 tests: the agent loop, run against a real emulator on a real socket.

A real socket rather than an in-process transport, because the thing being
verified includes the HTTP boundary: the acting-employee header, the 403s, and
the per-call spans that RQ2's "API calls per answer" is counted from.

The model is scripted, so these tests make no API call and spend nothing.
"""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from sim.agent.config import AgentConfig
from sim.agent.llm import LLMResponse, ScriptedClient
from sim.agent.loop import pack_context, run_turn
from sim.agent.prompt import DEFAULT_SYSTEM_PROMPT, PromptRenderError, render
from sim.agent.tools import HeimdallTools, tool_schemas
from sim.emulator.app import create_app as create_emulator
from sim.emulator.config import EmulatorConfig
from sim.fingerprint import RunFingerprint

pytestmark = pytest.mark.skipif(
    not Path("data-small/manifest.json").exists(),
    reason="needs the data-small corpus; run `make seed`",
)

FINGERPRINT = RunFingerprint.create(
    agent_config_version="agent_config@1",
    prompt_registry_version="system_prompt@1",
    skill_registry_hash="sha256:" + "00" * 32,
    model_id="claude-haiku-4-5-20251001",
    temperature=0.0,
    data_snapshot_hash="heimdall-sandbox@78d53675db91e17f",
    traps_enabled=True,
    latency_profile="instant",
    hr_employee_ids=[],
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def emulator_url():
    port = _free_port()
    app = create_emulator(EmulatorConfig(snapshot_traps_on="data-small",
                                         latency_profile="instant"))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "emulator did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def employee(emulator_url):
    import httpx
    ids = httpx.get(f"{emulator_url}/control/identities?n=3", timeout=10,
                    trust_env=False).json()
    return next(i for i in ids if i["role"] == "self")


def _tools(emulator_url, employee_id):
    return HeimdallTools(emulator_url, employee_id=employee_id, token="t")


def _text(text: str) -> LLMResponse:
    return LLMResponse(content=[{"type": "text", "text": text}],
                       stop_reason="end_turn", prompt_tokens=100,
                       completion_tokens=20)


def _tool_use(name: str, args: dict, call_id: str = "tu_1") -> LLMResponse:
    return LLMResponse(
        content=[{"type": "tool_use", "id": call_id, "name": name, "input": args}],
        stop_reason="tool_use", prompt_tokens=120, completion_tokens=30)


# ------------------------------------------------------------------- prompts


def test_prompt_renders_with_supplied_variables():
    out = render(DEFAULT_SYSTEM_PROMPT, {"employee_id": "123", "memory_block": ""})
    assert "123" in out


def test_missing_variable_is_an_error_not_an_empty_string():
    """StrictUndefined: a silently-empty variable would change the prompt the
    fingerprint claims was used."""
    with pytest.raises(PromptRenderError):
        render(DEFAULT_SYSTEM_PROMPT, {"employee_id": "123"})


# --------------------------------------------------------------------- tools


def test_tool_subset_controls_the_exposed_surface():
    assert [t["name"] for t in tool_schemas(("mcp_query",))] == ["mcp_query"]
    assert len(tool_schemas(("mcp_query", "list_models"))) == 2


def test_messages_api_mcp_query_exposes_metrics_and_id_types():
    """Without metrics in the schema the model cannot send fact_count and falls
    back to asking the user whether an org mart exists."""
    query = next(t for t in tool_schemas(AgentConfig().tool_subset)
                 if t["name"] == "mcp_query")
    assert "metrics" in query["input_schema"]["properties"]
    text = query["description"]
    assert "fact_count" in text
    assert "employee_id" in text and "UUID" in text
    listed = next(t for t in tool_schemas(AgentConfig().tool_subset)
                  if t["name"] == "list_models")
    assert "employee_actual" in listed["description"]


def test_no_tool_accepts_executable_code():
    """The no-code-execution premise is a property of the tool surface.

    If any tool ever grows a parameter that carries source text, this test fails
    and the premise has to be re-argued rather than quietly lost.
    """
    banned = {"code", "script", "source", "exec", "eval", "command",
              "python", "expression", "program"}
    for schema in tool_schemas(AgentConfig().tool_subset):
        properties = set(schema["input_schema"].get("properties", {}))
        assert not (properties & banned), (schema["name"], properties & banned)


# ---------------------------------------------------------------------- loop


def test_direct_answer_without_tools(emulator_url, employee):
    client = ScriptedClient([_text("Ответ без обращения к API.")])
    tools = _tools(emulator_url, employee["employee_id"])
    try:
        result = run_turn(
            question="Привет", history=[], config=AgentConfig(),
            system_prompt="sys", client=client, tools=tools,
            fingerprint=FINGERPRINT, session_id="ses_test",
            employee_id=employee["employee_id"])
    finally:
        tools.close()
    assert result.iterations == 1
    assert result.tool_calls == 0
    assert result.heimdall_calls == 0
    assert "Ответ без" in result.answer


def test_tool_call_reaches_heimdall_and_is_counted(emulator_url, employee):
    client = ScriptedClient([
        _tool_use("mcp_query", {"schema": "dm_core",
                                "logic_model": "employee_actual",
                                "columns": ["person_id"], "limit": 2}),
        _text("Готово."),
    ])
    tools = _tools(emulator_url, employee["employee_id"])
    try:
        result = run_turn(
            question="Сколько людей?", history=[], config=AgentConfig(),
            system_prompt="sys", client=client, tools=tools,
            fingerprint=FINGERPRINT, session_id="ses_test2",
            employee_id=employee["employee_id"])
    finally:
        tools.close()
    assert result.tool_calls == 1
    assert result.heimdall_calls == 1
    assert result.iterations == 2


def test_permission_denial_is_returned_to_the_agent_not_hidden(emulator_url, employee):
    """A 403 must reach the model as information. Swallowing it would leave the
    agent free to invent the answer it could not fetch."""
    client = ScriptedClient([
        _tool_use("mcp_query", {"schema": "recruitment",
                                "logic_model": "job_requisition_large",
                                "columns": ["id"], "limit": 1}),
        _text("Доступ закрыт."),
    ])
    tools = _tools(emulator_url, employee["employee_id"])
    try:
        result = run_turn(
            question="Покажи заявки на найм", history=[], config=AgentConfig(),
            system_prompt="sys", client=client, tools=tools,
            fingerprint=FINGERPRINT, session_id="ses_test3",
            employee_id=employee["employee_id"])
    finally:
        tools.close()
    tool_result = result.messages[-2]["content"][0]["content"]
    assert "forbidden" in tool_result
    assert result.heimdall_calls == 1


def test_iteration_cap_stops_a_runaway_loop(emulator_url, employee):
    config = AgentConfig(max_tool_iterations=3)
    client = ScriptedClient([
        _tool_use("list_models", {}, f"tu_{i}") for i in range(6)])
    tools = _tools(emulator_url, employee["employee_id"])
    try:
        result = run_turn(
            question="Зациклись", history=[], config=config,
            system_prompt="sys", client=client, tools=tools,
            fingerprint=FINGERPRINT, session_id="ses_test4",
            employee_id=employee["employee_id"])
    finally:
        tools.close()
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3


def test_unknown_tool_is_reported_not_crashed(emulator_url, employee):
    client = ScriptedClient([_tool_use("delete_everything", {}), _text("ок")])
    tools = _tools(emulator_url, employee["employee_id"])
    try:
        result = run_turn(
            question="?", history=[], config=AgentConfig(),
            system_prompt="sys", client=client, tools=tools,
            fingerprint=FINGERPRINT, session_id="ses_test5",
            employee_id=employee["employee_id"])
    finally:
        tools.close()
    assert result.heimdall_calls == 0
    assert any("unknown tool" in e for e in result.errors)


# ------------------------------------------------------------ context packing


def _history(n: int):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(n)]


def test_full_strategy_keeps_everything():
    messages = _history(30)
    assert pack_context(messages, AgentConfig(context_strategy="full")) == messages


def test_windowed_strategy_shrinks_context():
    messages = _history(30)
    packed = pack_context(messages, AgentConfig(context_strategy="windowed",
                                                harness="messages_api",
                                                windowed_turns=3))
    assert len(packed) < len(messages)


def test_summarised_strategy_preserves_recent_turns_verbatim():
    messages = _history(30)
    packed = pack_context(messages, AgentConfig(context_strategy="summarised",
                                                harness="messages_api",
                                                windowed_turns=3))
    assert packed[0]["content"].startswith("[сводка")
    assert packed[-1] == messages[-1]


# -------------------------------------------------------------------- config


def test_unknown_config_field_is_refused():
    """A typo'd key silently dropped would produce a run claiming a condition it
    never applied."""
    with pytest.raises(ValueError):
        AgentConfig.from_dict({**AgentConfig().as_dict(), "temperture": 0.7})


def test_invalid_strategy_names_are_refused():
    with pytest.raises(ValueError):
        AgentConfig(budget_strategy="guess")
    with pytest.raises(ValueError):
        AgentConfig(memory_strategy="telepathy")
    with pytest.raises(ValueError):
        AgentConfig(conversation_mode="remembers_everything")


def test_conversation_mode_reaches_the_fingerprinted_config():
    """It changes behaviour, so the spec's rule applies: it must be versionable
    and it must land in the config blob the fingerprint hashes. A mode that
    lived only on the calling surface would make two different conditions share
    one condition_id."""
    body = AgentConfig(conversation_mode="resume").as_dict()
    assert body["conversation_mode"] == "resume"
    assert AgentConfig.from_dict(body).conversation_mode == "resume"


def test_a_config_predating_conversation_mode_still_loads():
    """`_bootstrap_registry` only commits when a ref is absent, so a deployment
    that has ever started keeps its original `agent_config` blob. If that blob
    could not be loaded, the upgrade would take the service down on boot."""
    legacy = AgentConfig().as_dict()
    del legacy["conversation_mode"]
    assert AgentConfig.from_dict(legacy).conversation_mode == "stateless"


@pytest.mark.parametrize("strategy", ["windowed", "summarised"])
@pytest.mark.parametrize("mode", ["stateless", "resume"])
def test_a_context_strategy_the_default_harness_cannot_honour_is_refused(
        strategy, mode):
    """Only sim.agent.loop.pack_context implements these, and only messages_api
    calls it — claude_code.py never reads the field in either conversation mode.
    Accepting the pair gave three condition_ids to one behaviour, so a sweep
    over the axis would have concluded that context handling does not matter."""
    with pytest.raises(ValueError) as exc:
        AgentConfig(context_strategy=strategy, harness="claude_code",
                    conversation_mode=mode)
    assert "messages_api" in str(exc.value)


@pytest.mark.parametrize("mode", ["stateless", "resume"])
def test_context_strategy_full_stays_legal_on_the_default_harness(mode):
    """`full` is the default and every stored config carries it; on claude_code
    it is also the honest label, since the loop applies no packing of its own.
    Refusing it would make the shipped default unconstructible."""
    config = AgentConfig(context_strategy="full", conversation_mode=mode)
    assert config.harness == "claude_code"


@pytest.mark.parametrize("strategy", ["full", "windowed", "summarised"])
@pytest.mark.parametrize("mode", ["stateless", "resume"])
def test_every_context_strategy_is_legal_on_messages_api(strategy, mode):
    """The packer runs inside run_turn, on the messages the loop appends as it
    goes, so it bites in both conversation modes — the constraint is on the
    harness alone."""
    assert AgentConfig(context_strategy=strategy, harness="messages_api",
                       conversation_mode=mode).context_strategy == strategy


def test_a_config_predating_the_context_strategy_constraint_still_loads():
    """Same argument as the conversation_mode case above: a deployment that has
    ever started keeps its original blob, and blobs written before `harness`
    existed carry no such key. `context_strategy='full'` is what all of them
    say, so the new check must not make them unloadable on boot."""
    legacy = AgentConfig().as_dict()
    del legacy["harness"]
    loaded = AgentConfig.from_dict(legacy)
    assert loaded.harness == "claude_code"
    assert loaded.context_strategy == "full"


def test_both_shipped_configs_still_construct():
    """sim.agent.shipped commits these at bootstrap, from admin-ui, which holds
    the only writable registry mount. A constraint that refused one of them
    would take the operator surface down at boot."""
    from sim.agent.shipped import shipped_configs

    for ref, (kind, body, _note) in shipped_configs().items():
        if kind == "agent":
            assert AgentConfig.from_dict(body).context_strategy == "full", ref


def test_a_stored_config_the_constraint_refuses_is_a_503_not_a_bare_500(
        tmp_path, monkeypatch):
    """The other half of the admin-UI repair path, on the side that reads.

    `windowed` + `claude_code` was legal when registries in the field were
    written, and `sim.registry` validates nothing, so the blob survives the
    upgrade and is read here — on every session and every batch cell. A bare
    ValueError out of `from_dict` is an "Internal Server Error" naming neither
    the ref nor the field, and the agent mounts the registry read-only, so the
    caller also has to be told where the repair lives."""
    monkeypatch.setenv("B2E_REGISTRY_DB", str(tmp_path / "registry.db"))
    monkeypatch.setenv("B2E_AGENT_DB", str(tmp_path / "agent.db"))
    monkeypatch.setenv("B2E_LLM_MODE", "replay")
    from fastapi import HTTPException

    from sim.agent.app import AgentState

    state = AgentState()
    stale = {**AgentConfig(harness="messages_api",
                           context_strategy="windowed").as_dict(),
             "harness": "claude_code"}
    state.registry.commit("agent_config", "agent", stale, actor="test-seed")

    with pytest.raises(HTTPException) as exc:
        state.build_fingerprint("agent_config")
    assert exc.value.status_code == 503
    assert "agent_config" in exc.value.detail
    assert "messages_api" in exc.value.detail
    assert "/config" in exc.value.detail


def test_the_shipped_interactive_config_differs_in_exactly_one_field():
    """The Telegram bridge opens sessions against it, and the fingerprint has to
    be able to attribute any difference in results to the one variable that
    changed — not to a second edit that came along for the ride."""
    from sim.agent.app import INTERACTIVE_CONFIG_REF

    default = AgentConfig().as_dict()
    interactive = AgentConfig(conversation_mode="resume").as_dict()
    differing = {k for k in default if default[k] != interactive[k]}
    assert differing == {"conversation_mode"}
    assert INTERACTIVE_CONFIG_REF == "agent_config_interactive"


def test_empty_tool_subset_is_allowed():
    assert AgentConfig(tool_subset=()).tool_subset == ()
