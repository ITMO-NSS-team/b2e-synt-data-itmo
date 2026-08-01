"""C1 tests: permission scoping, blocked surfaces, RPC envelope, toggles.

The dev-router test asserts by *making the request*. An earlier version of the
block inspected ``app.router.routes`` instead, and passed while the routes were
still live: this FastAPI version stores included routers as ``_IncludedRouter``
objects with no ``.path``, so a path-based filter matched nothing. Testing the
observable behaviour rather than the internal table is the whole lesson.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sim.emulator.app import create_app
from sim.emulator.config import EmulatorConfig
from sim.emulator.identity import (
    AccessDenied, IdentityIndex, collect_person_filters, enforce,
)

pytestmark = pytest.mark.skipif(
    not (__import__("pathlib").Path("data-small/manifest.json").exists()),
    reason="needs the data-small corpus; run `make seed`",
)

BEARER = {"Authorization": "Bearer test-token"}


@pytest.fixture(scope="module")
def app():
    return create_app(EmulatorConfig(snapshot_traps_on="data-small",
                                     latency_profile="instant"))


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


@pytest.fixture(scope="module")
def actors(client):
    ids = client.get("/control/identities?n=3").json()
    return {
        "manager": next(i for i in ids if i["role"] == "manager"),
        "employee": next(i for i in ids if i["role"] == "self"),
    }


# ------------------------------------------------------------------ blocked


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/v2/dev/skills"),
    ("GET", "/api/v2/dev/skills/raw?path=x"),
    ("POST", "/api/v2/dev/skills/save"),
    ("POST", "/api/v2/dev/add_skill_for_debug/"),
    ("DELETE", "/api/v2/dev/skills?path=x"),
])
def test_dev_router_is_unreachable(client, method, path):
    """These routes have no auth in heimdall and can write skill files.

    Reachable by the agent they are a direct path from "the agent emitted text"
    to "the registry serves it as instructions".
    """
    assert client.request(method, path).status_code == 404


def test_dev_router_can_be_enabled_deliberately():
    """The block is a deployment choice, not a code deletion."""
    dev = TestClient(create_app(EmulatorConfig(
        snapshot_traps_on="data-small", latency_profile="instant",
        enable_dev_router=True)))
    assert dev.get("/api/v2/dev/skills").status_code != 404


# --------------------------------------------------------------- permissions


def test_data_request_without_identity_is_refused(client):
    r = client.post("/api/v1/mcp/query/", headers=BEARER,
                    json={"schema": "dm_core", "logic_model": "employee_actual",
                          "columns": ["person_id"], "limit": 1})
    assert r.status_code == 403
    assert r.json()["code"] == "forbidden"


def test_employee_may_read_permitted_model(client, actors):
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    r = client.post("/api/v1/mcp/query/", headers=h,
                    json={"schema": "dm_core", "logic_model": "employee_actual",
                          "columns": ["person_id"], "limit": 2})
    assert r.status_code == 200


def test_employee_may_not_read_hr_only_schema(client, actors):
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    r = client.post("/api/v1/mcp/query/", headers=h,
                    json={"schema": "recruitment",
                          "logic_model": "job_requisition_large",
                          "columns": ["id"], "limit": 1})
    assert r.status_code == 403


def test_employee_may_not_read_a_foreign_person(client, actors):
    """403, not an empty result: an empty result would teach the agent that the
    person does not exist, which is a different and wrong fact."""
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    r = client.post("/api/v1/mcp/query/", headers=h,
                    json={"schema": "dm_core", "logic_model": "employee_actual",
                          "columns": ["person_id"],
                          "filters": {"type": "condition", "column": "person_id",
                                      "operator": "=",
                                      "value": actors["manager"]["person_id"]}})
    assert r.status_code == 403


def test_manager_sees_more_than_an_individual_contributor(actors):
    index = IdentityIndex("data-small")
    mgr = index.scope_for(actors["manager"]["employee_id"])
    emp = index.scope_for(actors["employee"]["employee_id"])
    assert len(mgr.visible_person_ids) > len(emp.visible_person_ids)


def test_unknown_identity_is_rejected():
    with pytest.raises(AccessDenied) as exc:
        IdentityIndex("data-small").scope_for("000000000")
    assert exc.value.code == "auth-failed"


def test_person_filters_are_extracted_from_nested_trees():
    refs = collect_person_filters({"type": "and", "nodes": [
        {"type": "or", "nodes": [
            {"type": "condition", "column": "person_id",
             "operator": "in", "values": ["A", "B"]}]},
        {"type": "condition", "column": "grade", "operator": ">", "value": 3},
    ]})
    assert set(refs) == {"A", "B"}


def test_hr_role_bypasses_model_acl(actors):
    index = IdentityIndex("data-small",
                          hr_employee_ids=[actors["employee"]["employee_id"]])
    scope = index.scope_for(actors["employee"]["employee_id"])
    assert scope.role == "hr"
    enforce(scope, "recruitment", "job_requisition_large", {"filters": None})


# ------------------------------------------------------------------ fidelity


def test_unknown_body_field_rejects_the_whole_request(client, actors):
    """additionalProperties:false — one stray key kills the whole body."""
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    r = client.post("/api/v1/mcp/query/", headers=h,
                    json={"schema": "dm_core", "logic_model": "employee_actual",
                          "columns": ["person_id"], "bogus_field": 1})
    assert r.status_code == 400
    assert r.json()["code"] == "request-validation-error"


# ----------------------------------------------------------------------- rpc


def test_all_recruitment_operations_are_routed(client, actors):
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    body = client.get("/recruitment/api/v1/_operations", headers=h).json()
    assert body["count"] == 115


def test_rpc_success_uses_the_real_envelope(client, actors):
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    r = client.post("/recruitment/api/v1/requisitions/", headers=h,
                    json={"position_id": "POS-1", "salary_min": 8_000_00,
                          "salary_max": 12_000_00})
    assert r.status_code == 200
    assert set(r.json()) == {"result", "error"}
    assert r.json()["error"] is None
    assert r.json()["result"]["status"] == "NEW"


def test_business_refusal_is_http_200_with_a_numeric_code(client, actors):
    """"HTTP 200 does not mean it worked" is a lesson the agent must learn."""
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    for i in range(40):
        r = client.post("/recruitment/api/v1/requisitions/", headers=h,
                        json={"position_id": f"POS-{i}", "salary_min": 1,
                              "salary_max": 2})
        error = r.json()["error"]
        if error and error["code"] == 34125:
            assert r.status_code == 200
            return
    pytest.fail("no 34125 refusal produced across 40 positions")


def test_missing_required_field_is_422_not_200(client, actors):
    """Proxy-level validation never reached Pulse HR, so it is not a 200."""
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    r = client.post("/recruitment/api/v1/requisitions/", headers=h,
                    json={"position_id": "POS-1"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "request-validation-error"


def test_unimplemented_operation_says_so_rather_than_inventing(client, actors):
    """Fabricating payloads would contaminate the thing we measure."""
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    r = client.get("/recruitment/api/v1/interviews/notification-intervals/",
                   headers=h)
    assert r.json()["error"]["code"] == "not-implemented-in-emulator"


# ------------------------------------------------------------------- toggles


def test_latency_profile_can_be_switched(client):
    assert client.put("/control/config",
                      json={"latency_profile": "degraded"}
                      ).json()["latency_profile"] == "degraded"
    client.put("/control/config", json={"latency_profile": "instant"})


def test_unknown_latency_profile_is_rejected(client):
    assert client.put("/control/config",
                      json={"latency_profile": "warp"}).status_code == 422


def test_traps_off_without_a_corpus_refuses(client):
    """Refuse rather than serve traps-on data under a traps-off label — that
    would silently corrupt the RQ1 comparison the flag exists to enable."""
    r = client.put("/control/config", json={"traps_enabled": False})
    assert r.status_code == 409


def test_healthz_reports_the_condition(client):
    body = client.get("/control/healthz").json()
    assert body["status"] == "ok"
    assert "data_snapshot_hash" in body
    assert body["traps_enabled"] is True
