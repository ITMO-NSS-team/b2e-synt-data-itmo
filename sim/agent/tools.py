"""The agent's tool surface: Heimdall calls and approved-skill execution.

The hard constraint lives here, not in the prompt
-------------------------------------------------
There is no tool that accepts code and runs it. ``author_skill`` writes inert
bytes to a content-addressed store and returns a hash; ``run_skill`` takes a hash
and refuses anything the approval store does not mark ``active``. There is no
argument anywhere in these schemas through which source text can reach an
interpreter in the same session.

That is deliberate and it is the difference between a rule and a boundary. A
system prompt saying "do not execute code you wrote" is a request the model can
be talked out of by injected content in an HR field. A tool surface with no
execute-this-string parameter cannot be talked out of anything.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from sim import telemetry

#: Logical endpoint names, matching sim.latency's keys and the span schema.
ENDPOINT_LIST_MODELS = "list_models"
ENDPOINT_DESCRIBE = "describe_model"
ENDPOINT_DOCS = "get_docs"
ENDPOINT_QUERY = "mcp_query"
ENDPOINT_FIND_SKILLS = "find_skills"
ENDPOINT_GET_SKILL = "get_skill"
ENDPOINT_OVERVIEW = "get_overview"

#: The canonical tool surface. Both harnesses and AgentConfig validate against
#: this one tuple so they cannot drift apart again.
KNOWN_TOOLS: tuple[str, ...] = (
    ENDPOINT_LIST_MODELS, ENDPOINT_DESCRIBE, ENDPOINT_DOCS, ENDPOINT_QUERY,
    ENDPOINT_FIND_SKILLS, ENDPOINT_GET_SKILL, ENDPOINT_OVERVIEW,
)


def tool_schemas(subset: tuple[str, ...]) -> list[dict[str, Any]]:
    """Anthropic tool definitions for the configured subset.

    Restricting the subset is an experimental lever: fewer tools means fewer
    round trips available to the agent, which is one of the few things it can
    vary under the no-code constraint (RQ2).
    """
    all_tools: dict[str, dict[str, Any]] = {
        ENDPOINT_LIST_MODELS: {
            "name": "list_models",
            "description": (
                "Список витрин каталога: схема, модель, описание, признак "
                "истории, число колонок и метрик. Не спрашивай пользователя, "
                "есть ли витрина — вызови этот инструмент. Это каталог, не ACL: "
                "схемы anagent и recruitment и модель "
                "dm_special.talent_radar_people отвечают 403, если роль не hr. "
                "Оргструктура и численность подразделения для руководителя и "
                "сотрудника — dm_core.employee_actual; сервер сам ограничит "
                "строки видимым поддеревом."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        ENDPOINT_DESCRIBE: {
            "name": "describe_model",
            "description": (
                "Точные имена колонок, метрик и параметрических членов одной "
                "витрины. Имена берутся ТОЛЬКО отсюда: сервер сверяет каждое с "
                "каталогом и отклоняет незнакомое. columns и metrics — разные "
                "списки: колонку нельзя запросить в metrics, метрику — в "
                "columns. Дорого по токенам — вызывай точечно."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "schema": {"type": "string"},
                    "logic_model": {"type": "string"},
                },
                "required": ["schema", "logic_model"],
            },
        },
        ENDPOINT_DOCS: {
            "name": "get_docs",
            "description": (
                "Справочник по контракту канала. Дешевле, чем describe_model, "
                "когда нужно понять правило, а не перечень колонок. Темы: "
                "getting_started, rules, rows, aggregate, limits, filters, "
                "history, param_metrics, top_n, compare_people, errors. "
                "Численность — get_docs('aggregate') и метрика fact_count, "
                "не подсчёт строк страницы."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
        ENDPOINT_QUERY: {
            "name": "mcp_query",
            "description": (
                "Запрос к витрине. Режим выводится из структуры тела: metrics "
                "или param_metrics — агрегат, только columns — плоские строки. "
                "Лишнее поле отклоняет всё тело. Идентификаторы: число в "
                "системном промпте — employee_id (табельный номер, integer). "
                "person_id — UUID. Фильтр person_id по табельному номеру даёт "
                "filter-value-invalid. Чтобы посчитать людей — не вычитывай "
                "строки, а запроси metrics: [\"fact_count\"] к "
                "dm_core.employee_actual; без группировки вернётся одна строка "
                "итога по видимому поддереву. len(data) — размер страницы "
                "(лимит 100, потолок 1000), не численность."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "string",
                        "description": "Схема витрины, напр. dm_core",
                    },
                    "logic_model": {
                        "type": "string",
                        "description": "Имя витрины, напр. employee_actual",
                    },
                    "columns": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Имена колонок из describe_model. В агрегате это "
                            "ключи группировки; без них — одна строка итога."
                        ),
                    },
                    "metrics": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Имена метрик из describe_model. Включает режим "
                            "агрегата. Для подсчёта людей обычно fact_count."
                        ),
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Дерево фильтров: узлы and/or/not и листья "
                            "condition. Грамматика — get_docs('filters')."
                        ),
                    },
                    "order_by": {
                        "type": "array", "items": {"type": "object"},
                        "description": (
                            "[{field, kind: column|metric, direction: "
                            "asc|desc}]. Нужен для устойчивой пагинации."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Строк на страницу. По умолчанию 100, потолок "
                            "1000; выше — молча ужимается."
                        ),
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Сдвиг следующей страницы, пока has_next_page. "
                            "Только вместе с order_by."
                        ),
                    },
                    "time_dimensions": {
                        "type": "array", "items": {"type": "object"},
                        "description": "Включает режим истории (SCD2).",
                    },
                    "limit_by": {
                        "type": "object",
                        "description": "{limit, by: [колонки]} — N строк на группу.",
                    },
                    "param_metrics": {
                        "type": "array", "items": {"type": "object"},
                        "description": "[{name, args}] для метрик с parameters.",
                    },
                    "param_columns": {
                        "type": "array", "items": {"type": "object"},
                        "description": "[{name, args}] для параметрических колонок.",
                    },
                },
                "required": ["schema", "logic_model"],
            },
        },
        ENDPOINT_FIND_SKILLS: {
            "name": "find_skills",
            "description": (
                "Найти рецепт или справочный приём под задачу. Значение «*» "
                "вернёт всё. Сначала ищи рецепт, list_models — запасной путь."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"},
                               "limit": {"type": "integer"}},
                "required": ["query"],
            },
        },
        ENDPOINT_GET_SKILL: {
            "name": "get_skill",
            "description": (
                "Взять скилл целиком: готовый запрос, что подставлять и "
                "чего делать нельзя. Рецепт исполняется как есть."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        ENDPOINT_OVERVIEW: {
            "name": "get_overview",
            "description": (
                "Обзор каталога: домены, активные скиллы, грамматика "
                "фильтров. Начинай отсюда, если не знаешь, что вообще есть."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    }
    unknown = [name for name in subset if name not in all_tools]
    if unknown:
        # Filtering silently is how the two harnesses drifted apart. A name that
        # cannot be mapped means the run would proceed with a smaller tool
        # surface than the config it is fingerprinted against claims.
        raise ValueError(
            f"unknown tools in tool_subset: {unknown}; known: {sorted(all_tools)}")
    return [all_tools[name] for name in subset]


class HeimdallTools:
    """HTTP client for the emulator, one instance per session.

    Acts for a specific employee: the acting identity is a header on every call,
    so a permission refusal is a real 403 rather than a policy the agent is
    trusted to respect.
    """

    def __init__(self, base_url: str, *, employee_id: str, token: str,
                 channel: str = "v2", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.employee_id = str(employee_id)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            # Internal service-to-service traffic must not traverse a user proxy.
            # Without this, httpx honours HTTP_PROXY/ALL_PROXY from the ambient
            # environment and tries to route a compose-network call through it —
            # which on this host is a SOCKS proxy that is not even installed.
            trust_env=False,
            headers={
                "Authorization": f"Bearer {token}",
                "X-Employee-Id": self.employee_id,
                "x-heimdall-mcp-version": channel,
            },
        )
        self.call_count = 0

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------- dispatch

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "list_models": self.list_models,
            "describe_model": self.describe_model,
            "get_docs": self.get_docs,
            "mcp_query": self.mcp_query,
            "find_skills": self.find_skills,
            "get_skill": self.get_skill,
            "get_overview": self.get_overview,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool {name!r}",
                    "hint": "this tool is not in the configured subset"}
        try:
            return handler(**arguments)
        except TypeError as exc:
            return {"error": "bad arguments", "detail": str(exc)}
        except httpx.HTTPError as exc:
            return {"error": "transport", "detail": str(exc)}

    # ---------------------------------------------------------------- calls

    def _call(self, method: str, path: str, endpoint: str, *,
              json_body: Any = None, params: dict[str, Any] | None = None,
              columns_requested: int = 0) -> dict[str, Any]:
        self.call_count += 1
        with telemetry.start_heimdall_call(
            method=method, path=path, endpoint=endpoint,
            request_body=json_body if json_body is not None else params,
        ) as span:
            response = self._client.request(method, path, json=json_body, params=params)
            try:
                body = response.json()
            except ValueError:
                body = {"error": "non-json response", "text": response.text[:2000]}

            rows = 0
            if isinstance(body, dict):
                for key in ("data", "rows", "items"):
                    if isinstance(body.get(key), list):
                        rows = len(body[key])
                        break

            telemetry.record_heimdall_result(
                span, status=response.status_code, body=body, rows=rows,
                columns_requested=columns_requested)
            if response.status_code >= 400 and isinstance(body, dict):
                body = {**body, "_http_status": response.status_code}
            return body

    def list_models(self) -> dict[str, Any]:
        return self._call("GET", "/api/v1/mcp/models/", ENDPOINT_LIST_MODELS)

    def describe_model(self, schema: str, logic_model: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/mcp/models/{schema}/{logic_model}/",
                          ENDPOINT_DESCRIBE)

    def get_docs(self, topic: str) -> dict[str, Any]:
        return self._call("GET", "/api/v1/mcp/docs/", ENDPOINT_DOCS,
                          params={"topic": topic})

    def mcp_query(self, **body: Any) -> dict[str, Any]:
        columns = body.get("columns")
        n_columns = 642 if (isinstance(columns, list) and "*" in columns) else \
            (len(columns) if isinstance(columns, list) else 0)
        return self._call("POST", "/api/v1/mcp/query/", ENDPOINT_QUERY,
                          json_body=body, columns_requested=n_columns)

    def find_skills(self, query: str, limit: int = 10) -> dict[str, Any]:
        return self._call("GET", "/api/v2/mcp/skills/", ENDPOINT_FIND_SKILLS,
                          params={"query": query, "limit": limit})

    def get_skill(self, name: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v2/mcp/skills/{name}/", ENDPOINT_GET_SKILL)

    def get_overview(self) -> dict[str, Any]:
        return self._call("GET", "/api/v2/mcp/overview/", ENDPOINT_OVERVIEW)


def render_tool_result(payload: Any, *, max_chars: int = 60_000) -> str:
    """Serialise a tool result for the model.

    Capped because a 642-column response can exhaust the window in one call, and
    an agent that blows its context on one query is a confound in every latency
    measurement that follows.
    """
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n…[truncated at {max_chars} chars]"
    return text
