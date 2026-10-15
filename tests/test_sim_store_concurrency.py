"""Committed session and job rows stay readable during concurrent SQL writes."""
from concurrent.futures import ThreadPoolExecutor
import threading

from sim.agent.store import Store


def test_concurrent_committed_sessions_messages_and_jobs_remain_readable(tmp_path):
    store = Store(tmp_path / "agent.db")
    start = threading.Barrier(16)

    def worker(employee):
        start.wait(timeout=5)
        created_sessions = []
        for turn in range(64):
            fingerprint = {"employee": employee, "turn": turn}
            sid = store.create_session(employee_id=str(employee), config_ref="config",
                                       fingerprint=fingerprint)
            created_sessions.append(sid)
            session = store.get_session(sid)
            assert session is not None
            assert session["id"] == sid
            assert session["employee_id"] == str(employee)
            assert session["fingerprint"] == fingerprint
            assert store.append_message(session_id=sid, role="user", content=f"q{turn}") == 1
            assert store.append_message(session_id=sid, role="assistant", content=f"a{turn}") == 2
            assert store.history_for_model(sid) == [
                {"role": "user", "content": f"q{turn}"},
                {"role": "assistant", "content": f"a{turn}"},
            ]
            sessions = store.list_sessions(employee_id=str(employee), limit=1000)
            assert {row["id"] for row in sessions} == set(created_sessions)
            assert all(row["employee_id"] == str(employee) for row in sessions)
            jid, created = store.create_job(request=fingerprint, idempotency_key=f"{employee}-{turn}")
            assert created is True
            store.update_job(jid, status="completed", result={"session": sid})
            job = store.get_job(jid)
            assert job is not None
            assert (job["id"], job["request"], job["status"], job["result"]) == (
                jid, fingerprint, "completed", {"session": sid})
            assert jid in {row["id"] for row in store.list_jobs(limit=2000)}
        return created_sessions

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            sessions = [sid for batch in pool.map(worker, range(16)) for sid in batch]
        assert len(set(sessions)) == 1024
        assert len(store.list_sessions(limit=2000)) == 1024
        assert len(store.list_jobs(limit=2000)) == 1024
    finally:
        store.close()


def test_parallel_fingerprint_reads_and_skill_store_open_keep_configs_available(tmp_path):
    from sim.registry import Registry, canonical_bytes, sha256_hex
    from sim.skills import SkillStore
    registry = Registry(tmp_path / "registry.db")
    config = {"system_prompt_ref": "system_prompt", "skill_registry_ref": "skills"}
    version = registry.commit("agent_config_interactive", "config", config, actor="test")
    registry.commit("system_prompt", "prompt", {"template": "system"}, actor="test")
    start = threading.Barrier(16)

    def fingerprint_reads(employee):
        start.wait(timeout=5)
        for _ in range(128):
            observed, body = registry.load("agent_config_interactive")
            assert observed == version
            assert body == config
            assert registry.load("system_prompt")[1] == {"template": "system"}
            assert SkillStore(registry).registry_hash() == "sha256:" + sha256_hex(canonical_bytes([]))
            assert registry.history("agent_config_interactive") == [version]
            assert registry.names("config") == ["agent_config_interactive"]
            assert registry.get_version("agent_config_interactive", 1) == version
            assert registry.get_json(version.hash) == config
            assert len(registry.audit_read()) == 2
            assert len(list(registry.iter_audit())) == 2

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(fingerprint_reads, range(16)))
    finally:
        registry.close()
