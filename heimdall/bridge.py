#!/usr/bin/env python3
"""stdio-мост MCP → HTTP-эмулятор Heimdall. Без сторонних зависимостей.

Зачем он есть
-------------
Герметичность прогона в этом проекте держится на ``manifest_hash`` от хэшей
смонтированных ФАЙЛОВ. Сетевой клиент хэшировать нечем, и весь оффлайн-контур
воспроизводимости — кэш прогонов, фикстуры replay — рассыпался бы. Файл-мост
хэшируется как обычный артефакт, а движок при этом остаётся в основном
окружении и волен пользоваться зависимостями.

Отсюда жёсткое ограничение: **только стандартная библиотека**. В RUN_DIR нет
виртуального окружения, мост запускается голым ``python3``, и лишний импорт
проявится не при сборке, а посреди прогона.

Окружение
---------
``HEIMDALL_URL``     адрес эмулятора, по умолчанию http://127.0.0.1:8080
``HEIMDALL_TOKEN``   токен технической учётной записи (Authorization: Bearer)
``HEIMDALL_CHANNEL`` канал: v1 | v2 | orion, по умолчанию v2
``HR_TRACE_LOG``     журнал вызовов (JSONL) — независимый от агента источник трейса
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "heimdall-sandbox"
SERVER_VERSION = "01.002.00"

BASE_URL = os.environ.get("HEIMDALL_URL", "http://127.0.0.1:8080").rstrip("/")
TOKEN = os.environ.get("HEIMDALL_TOKEN", "")
CHANNEL = os.environ.get("HEIMDALL_CHANNEL", "v2")
EMPLOYEE_ID = os.environ.get("HEIMDALL_EMPLOYEE_ID", "")
TRACE_LOG = os.environ.get("HR_TRACE_LOG", "")
TIMEOUT = float(os.environ.get("HEIMDALL_TIMEOUT", "60"))

_STR = {"type": "string"}

#: Инструменты канала v2. В v1 и orion каталога скиллов нет, но список
#: инструментов моста от этого не меняется: сервис сам отвечает 404, и агент
#: должен это увидеть, а не гадать, почему инструмент пропал.
TOOLS = [
    {
        "name": "get_overview",
        "description": "Обзор каталога: домены с числом рецептов, полный список "
                       "активных скиллов с описаниями и грамматика фильтров. "
                       "Начинай отсюда, если не знаешь, что вообще есть.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_docs",
        "description": "Документация по теме: описание витрины, правил фильтрации "
                       "или конкретного приёма. Дешевле, чем describe_model, "
                       "когда нужно понять смысл, а не перечень колонок.",
        "inputSchema": {"type": "object", "required": ["topic"],
                        "properties": {"topic": dict(_STR, description="Тема")}},
    },
    {
        "name": "find_skills",
        "description": "Найти рецепт или справочный приём под задачу. Возвращает "
                       "карточки с relevance 0..1. Значение «*» вернёт всё. "
                       "Выбирай по title и description — тегов в выдаче нет.",
        "inputSchema": {
            "type": "object", "required": ["query"],
            "properties": {
                "query": dict(_STR, description="Задача своими словами или «*»"),
                "domain": dict(_STR, description="Ограничить доменом"),
                "kind": dict(_STR, description="recipe | reference"),
                "limit": {"type": "integer", "description": "По умолчанию 10"},
                "offset": {"type": "integer"},
                "include_deprecated": {"type": "boolean"},
            },
        },
    },
    {
        "name": "get_skill",
        "description": "Взять скилл целиком: готовый запрос, что подставлять, что "
                       "вернётся и чего делать нельзя. Рецепт исполняется как есть — "
                       "меняются только поля из params, обязательные фильтры не трогай.",
        "inputSchema": {"type": "object", "required": ["name"],
                        "properties": {"name": dict(_STR, description="Slug скилла")}},
    },
    {
        "name": "list_models",
        "description": "Список витрин каталога: схема, модель, описание, признак "
                       "истории, число колонок и метрик. Запасной путь, когда "
                       "готового рецепта под задачу нет.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "describe_model",
        "description": "Точные имена колонок, метрик и параметрических членов одной "
                       "витрины. Имена берутся только отсюда: сервер сверяет каждое "
                       "с каталогом и отклоняет незнакомое.",
        "inputSchema": {"type": "object", "required": ["schema", "logic_model"],
                        "properties": {"schema": _STR, "logic_model": _STR}},
    },
    {
        "name": "mcp_query",
        "description": "Запрос к витрине. Режим выводится из структуры тела: "
                       "time_dimensions — история, metrics — агрегат, только columns — "
                       "плоские строки. Лишнее поле отклоняет всё тело.",
        "inputSchema": {
            "type": "object", "required": ["schema", "logic_model"],
            "properties": {
                "schema": _STR, "logic_model": _STR,
                "columns": {"type": "array", "items": _STR},
                "metrics": {"type": "array", "items": _STR},
                "filters": {"type": "object"},
                "order_by": {"type": "array", "items": {"type": "object"}},
                "time_dimensions": {"type": "array", "items": {"type": "object"}},
                "limit": {"type": "integer"}, "offset": {"type": "integer"},
                "limit_by": {"type": "object"},
                "param_metrics": {"type": "array", "items": {"type": "object"}},
                "param_columns": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
]

TOOL_INDEX = {tool["name"]: tool for tool in TOOLS}


# ------------------------------------------------------------------- запрос

def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json",
               "x-heimdall-mcp-version": CHANNEL}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    # Действующий сотрудник. Токен аутентифицирует *сервис*, а этот заголовок
    # говорит, от чьего имени идёт запрос, — и именно по нему эмулятор решает,
    # ответить данными или 403. Без него мультиарендность неотличима от её
    # отсутствия: все сессии видели бы одно и то же.
    if EMPLOYEE_ID:
        headers["X-Employee-Id"] = EMPLOYEE_ID
    return headers


def _http(method: str, path: str, *, params: dict | None = None,
          body: dict | None = None) -> tuple[int, dict]:
    """Один HTTP-вызов. Доменная ошибка — не исключение, а такой же ответ."""
    url = BASE_URL + path
    if params:
        clean = {k: v for k, v in params.items() if v not in (None, "")}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=_headers())
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"code": "internal-error", "detail": raw[:400],
                              "hint": "сервис ответил не-JSON"}
    except urllib.error.URLError as exc:
        return 503, {"code": "internal-error", "detail": f"эмулятор недоступен: {exc.reason}",
                     "hint": f"проверь, что сервис слушает {BASE_URL}"}


def _dispatch(tool: str, args: dict) -> tuple[int, dict]:
    if tool == "list_models":
        return _http("GET", "/api/v1/mcp/models/")
    if tool == "describe_model":
        schema = urllib.parse.quote(str(args.get("schema", "")))
        model = urllib.parse.quote(str(args.get("logic_model", "")))
        return _http("GET", f"/api/v1/mcp/models/{schema}/{model}/")
    if tool == "mcp_query":
        return _http("POST", "/api/v1/mcp/query/", body=args)
    if tool == "get_overview":
        return _http("GET", "/api/v2/mcp/overview/")
    if tool == "find_skills":
        return _http("GET", "/api/v2/mcp/skills/", params=args)
    if tool == "get_skill":
        name = urllib.parse.quote(str(args.get("name", "")))
        return _http("GET", f"/api/v2/mcp/skills/{name}/")
    if tool == "get_docs":
        return _http("GET", "/api/v1/mcp/docs/", params=args)
    raise KeyError(tool)


def _log(tool: str, args: dict, status: int, payload: dict, elapsed_ms: float) -> None:
    """Журнал вызовов: источник трейса, не зависящий от того, что скажет агент."""
    if not TRACE_LOG:
        return
    record = {
        "tool": tool,
        "args_keys": sorted(args),
        "status": status,
        "duration_ms": round(elapsed_ms, 2),
        "code": payload.get("code") if isinstance(payload, dict) else None,
        "rows": len(payload.get("data", [])) if isinstance(payload, dict) else None,
        "bytes": len(json.dumps(payload, ensure_ascii=False)),
    }
    try:
        with open(TRACE_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass                                    # журнал не должен ронять прогон


# ------------------------------------------------------------------ JSON-RPC

def _result(request_id, payload) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def _error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(request: dict) -> dict | None:
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    if method in ("notifications/initialized", "initialized"):
        return None                             # уведомление, ответа не требует
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})
    if method == "tools/call":
        params = request.get("params") or {}
        tool = params.get("name", "")
        args = params.get("arguments") or {}
        if tool not in TOOL_INDEX:
            return _error(request_id, -32602,
                          f"неизвестный инструмент {tool}; доступны "
                          f"{', '.join(sorted(TOOL_INDEX))}")
        started = time.monotonic()
        status, payload = _dispatch(tool, args)
        _log(tool, args, status, payload, (time.monotonic() - started) * 1000)
        return _result(request_id, {
            "content": [{"type": "text",
                         "text": json.dumps(payload, ensure_ascii=False)}],
            # Доменная ошибка приходит телом с code/detail/hint: агент обязан её
            # прочитать, а не гадать по молчанию. Флаг помечает её как отказ.
            "isError": status >= 400,
        })
    return _error(request_id, -32601, f"метод {method} не поддерживается")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(_error(None, -32700, "не JSON")) + "\n")
            sys.stdout.flush()
            continue
        response = handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
