"""Reproduce overlapping SQLite cursor access, then verify shared read locking."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
import time
from sim.registry import Registry
from sim.skills import SkillStore
from sim.agent.store import Store


class GuardedCursor:
    def __init__(self, cursor, gate):
        self.cursor, self.gate = cursor, gate
    def fetchone(self):
        try:
            time.sleep(.001)
            return self.cursor.fetchone()
        finally:
            self.gate.release()
    def fetchall(self):
        try:
            time.sleep(.001)
            return self.cursor.fetchall()
        finally:
            self.gate.release()


class GuardedConnection:
    """SQLite misuse is reproducible if execute/consume overlap on one connection."""
    def __init__(self, conn):
        self.conn, self.gate = conn, threading.Lock()
    def execute(self, *args):
        if not self.gate.acquire(blocking=False):
            raise sqlite3.InterfaceError("overlapping shared connection access")
        try:
            return GuardedCursor(self.conn.execute(*args), self.gate)
        except BaseException:
            self.gate.release()
            raise
    def __getattr__(self, key):
        return getattr(self.conn, key)


def test_parallel_readonly_registry_and_skill_fingerprints(tmp_path):
    path = tmp_path / "registry.db"
    writer = Registry(path)
    version = writer.commit("config", "agent", {"model_id": "one"}, actor="test")
    skill = SkillStore(writer).upload(name="test", definition={"x": 1}, code="def run(payload): return payload", actor="test")
    writer.close()
    registry = Registry(path, readonly=True)
    skills = SkillStore(registry)
    registry._conn = GuardedConnection(registry._conn)
    def fingerprint(_):
        for _ in range(10):
            assert registry.load(version.ref)[1] == {"model_id": "one"}
            assert registry.names("agent") == ["config"]
            assert skills.get(skill.code_hash).name == "test"
            assert len(skills.list()) == 1
            assert skills.registry_hash().startswith("sha256:")
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fingerprint, range(8)))
    finally:
        registry.close()


def test_parallel_session_and_job_reads_use_the_same_connection_lock(tmp_path):
    store = Store(tmp_path / "agent.db")
    session = store.create_session(employee_id="1", config_ref="config@1", fingerprint={}, metadata={"run": "one"})
    store.append_message(session_id=session, role="user", content="question")
    job, _ = store.create_job(request={}, idempotency_key="one")
    store._conn = GuardedConnection(store._conn)
    def read(_):
        for _ in range(10):
            assert store.get_session(session)["metadata"] == {"run": "one"}
            assert len(store.list_sessions()) == 1
            assert store.history_for_model(session) == [{"role": "user", "content": "question"}]
            assert store.get_job(job)["status"] == "queued"
            assert len(store.list_jobs()) == 1
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(read, range(8)))
    finally:
        store.close()
