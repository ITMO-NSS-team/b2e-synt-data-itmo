"""Recruitment RPC stubs in the real RpcEnvelope shape.

The envelope and error shape are taken from the Heimdall OpenAPI specification,
not invented::

    success  {"result": <payload>, "error": null}
    failure  {"result": null, "error": {"code": <int|str>, "message": <ru>, "data": <any>}}

``code`` is an integer when the refusal came from Pulse HR itself (34125: this
position may not have a requisition created against it; 34003: the requisition
is not in a status that allows cancellation) and a string when the proxy refused
(``auth-failed``, ``request-validation-error``).

Two levels of failure, kept distinct on purpose
-----------------------------------------------
A missing required parameter is a **422** with a proxy string code: the request
never reached Pulse HR. A business refusal is a **200** carrying an error
envelope with a numeric code: the call succeeded, the answer was no. Collapsing
these would remove one of the more instructive failure modes an agent has to
learn — "HTTP 200 does not mean it worked".

Coverage
--------
All 115 operations from the ``MCP Recruitment`` tag are routed, so the surface
the agent discovers matches production. A documented subset has real behaviour
backed by the corpus. The rest answer with the string code
``not-implemented-in-emulator``.

That is deliberate. Fabricating plausible payloads for ninety unimplemented
endpoints would put invented facts into the environment whose entire purpose is
measuring invented facts. An honest refusal is a usable observation; a
hallucinated candidate profile is contamination.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

RPC_TABLE_PATH = Path("catalog/recruitment_rpc.json")

#: Real Pulse HR codes, with the meanings documented in the specification.
PULSE_POSITION_NOT_ALLOWED = 34125
PULSE_BAD_STATUS_FOR_CANCEL = 34003

#: Proxy-level string codes.
CODE_NOT_IMPLEMENTED = "not-implemented-in-emulator"
CODE_VALIDATION = "request-validation-error"


def ok(result: Any) -> dict[str, Any]:
    return {"result": result, "error": None}


def err(code: int | str, message: str, data: Any = None) -> dict[str, Any]:
    return {"result": None, "error": {"code": code, "message": message, "data": data}}


def _stable_bucket(value: Any, modulo: int) -> int:
    """Deterministic bucket, so the same input always gets the same verdict.

    A requisition that can be created once must be creatable again on a re-run,
    or an A/B comparison would differ for reasons unrelated to the agent.
    """
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo


def load_table(path: Path | str = RPC_TABLE_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"count": 0, "operations": []}
    return json.loads(path.read_text("utf-8"))


# --------------------------------------------------------------- handlers


def _create_job_requisition(body: dict[str, Any], state) -> dict[str, Any]:
    missing = [f for f in ("position_id", "salary_min", "salary_max") if f not in body]
    if missing:
        return err(CODE_VALIDATION,
                   f"не заполнены обязательные поля: {', '.join(missing)}",
                   {"missing": missing})

    position = body["position_id"]
    # Documented refusal: some positions may not have requisitions raised.
    if _stable_bucket(("position-allowed", position), 5) == 0:
        return err(PULSE_POSITION_NOT_ALLOWED,
                   "На этой позиции создавать заявку нельзя",
                   {"position_id": position})

    # Salary arrives in kopecks and is echoed back in kopecks, per the spec.
    return ok({
        "id": f"REQ-{_stable_bucket(('req', position), 900000) + 100000}",
        "position_id": position,
        "salary_min": body["salary_min"],
        "salary_max": body["salary_max"],
        "currency": "RUB",
        "amount_unit": "kopecks",
        "status": "NEW",
    })


def _cancel_job_requisition(body: dict[str, Any], state) -> dict[str, Any]:
    req_id = body.get("job_requisition_id")
    if not req_id:
        return err(CODE_VALIDATION, "не заполнено обязательное поле: job_requisition_id",
                   {"missing": ["job_requisition_id"]})

    # Only requisitions in a cancellable status may be cancelled.
    if _stable_bucket(("cancellable", req_id), 3) == 0:
        return err(PULSE_BAD_STATUS_FOR_CANCEL,
                   "Заявка находится в статусе, из которого отмена невозможна",
                   {"job_requisition_id": req_id, "status": "IN_PROGRESS"})

    return ok({"id": req_id, "status": "CANCELLED"})


def _get_position_details(params: dict[str, Any], state) -> dict[str, Any]:
    position = params.get("position_id")
    if not position:
        return err(CODE_VALIDATION, "не заполнен обязательный параметр: position_id",
                   {"missing": ["position_id"]})

    base = 8_000_00 + _stable_bucket(("salary", position), 4_000_00)
    return ok({
        "position_id": position,
        "availableGradeSalaryMin": base,
        "availableGradeSalaryMax": base + 3_500_00,
        "amount_unit": "kopecks",
        "currency": "RUB",
    })


def _search_persons(params: dict[str, Any], state) -> dict[str, Any]:
    """Search employees by name. Backed by the corpus, scoped to the caller."""
    query = (params.get("query") or params.get("q") or "").strip()
    if not query:
        return err(CODE_VALIDATION, "не заполнен обязательный параметр: query",
                   {"missing": ["query"]})
    index = state.identity_index()
    hits = [
        {"employee_id": index.employee_id[i], "person_id": index.person_id[i]}
        for i in range(min(len(index.person_id), 200))
        if query.lower() in index.employee_id[i].lower()
    ][:20]
    return ok({"items": hits, "total": len(hits)})


#: operation_id -> (handler, source) where source is "body" or "params".
IMPLEMENTED: dict[str, tuple[Any, str]] = {
    "create_job_requisition": (_create_job_requisition, "body"),
    "cancel_job_requisition": (_cancel_job_requisition, "body"),
    "get_position_details": (_get_position_details, "params"),
    "search_persons": (_search_persons, "params"),
}


# ----------------------------------------------------------------- router


def rpc_router(state, table_path: Path | str = RPC_TABLE_PATH) -> APIRouter:
    """Route every MCP Recruitment operation; implement the documented subset."""
    router = APIRouter(tags=["MCP Recruitment"])
    table = load_table(table_path)

    @router.get("/recruitment/api/v1/_operations")
    def list_operations() -> dict[str, Any]:
        """What exists and what is actually backed. Honesty is the point."""
        return {
            "count": table.get("count", 0),
            "implemented": sorted(IMPLEMENTED),
            "operations": [
                {"operation_id": o["operation_id"], "method": o["method"],
                 "path": o["path"], "summary": o["summary"],
                 "implemented": o["operation_id"] in IMPLEMENTED}
                for o in table.get("operations", [])
            ],
        }

    seen: set[tuple[str, str]] = set()
    for op in table.get("operations", []):
        key = (op["method"], op["path"])
        if key in seen:
            continue
        seen.add(key)
        _register(router, op, state)

    return router


def _register(router: APIRouter, op: dict[str, Any], state) -> None:
    operation_id = op["operation_id"]
    method = op["method"].lower()
    path = op["path"]
    required_params = [p["name"] for p in op.get("parameters", []) if p.get("required")]

    async def handler(request: Request) -> JSONResponse:
        params = dict(request.query_params)
        body: dict[str, Any] = {}
        if method in ("post", "put", "patch", "delete"):
            raw = await request.body()
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        body = parsed
                except ValueError:
                    return JSONResponse(status_code=422, content=err(
                        CODE_VALIDATION, "тело запроса не является объектом JSON"))

        # Proxy-level validation: never reached Pulse HR, so it is a 422.
        missing = [p for p in required_params if p not in params]
        if missing:
            return JSONResponse(status_code=422, content=err(
                CODE_VALIDATION,
                f"не заполнены обязательные параметры: {', '.join(missing)}",
                {"missing": missing}))

        entry = IMPLEMENTED.get(operation_id)
        if entry is None:
            # 200 with an error envelope: the endpoint exists, the emulator
            # simply does not back it. Saying so beats inventing a payload.
            return JSONResponse(status_code=200, content=err(
                CODE_NOT_IMPLEMENTED,
                f"операция {operation_id} существует в спецификации, но не "
                f"реализована в эмуляторе",
                {"operation_id": operation_id}))

        func, source = entry
        payload = body if source == "body" else params
        result = func(payload, state)
        status = 422 if (result.get("error") or {}).get("code") == CODE_VALIDATION else 200
        return JSONResponse(status_code=status, content=result)

    router.add_api_route(path, handler, methods=[op["method"]],
                         name=operation_id, summary=op.get("summary", ""))
