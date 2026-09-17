from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import quote

import sim.skill_eval._path  # noqa: F401
from benchmarking.models import HeimdallSkill
from benchmarking.serialization import write_skill

_DOCKER_SOCK = "/var/run/docker.sock"


class SkillInstaller(ABC):
    @abstractmethod
    def install(self, skill: HeimdallSkill | None) -> Path | None:
        ...

    @abstractmethod
    def cleanup(self) -> None:
        ...


class BindMountRestartInstaller(SkillInstaller):
    def __init__(
        self,
        skills_root: str = "heimdall-skills",
        overlay_dirname: str = "_factory",
        compose_file: str = "deploy/docker-compose.yml",
        env_file: str = "deploy/.env",
        service: str = "heimdall-emulator",
        project: str = "b2e-itmo",
        cleanup_after: bool = True,
        use_sudo: bool = False,
    ) -> None:
        self.skills_root = Path(skills_root)
        self.overlay = self.skills_root / overlay_dirname
        self.compose_file = compose_file
        self.env_file = env_file
        self.service = service
        self.project = project
        self.cleanup_after = cleanup_after
        self.use_sudo = use_sudo

    def install(self, skill: HeimdallSkill | None) -> Path | None:
        if self.overlay.exists():
            shutil.rmtree(self.overlay)
        self.overlay.mkdir(parents=True, exist_ok=True)
        if skill is None:
            self._restart()
            return None
        path = write_skill(skill, self.overlay)
        self._restart()
        return path

    def cleanup(self) -> None:
        if self.cleanup_after and self.overlay.exists():
            shutil.rmtree(self.overlay)
            self._restart()

    def _restart(self) -> None:
        if _docker_cli_ok(self.use_sudo):
            subprocess.run(
                self._compose() + ["restart", self.service], check=True)
            self._wait_cli_health()
            return
        container_id = _find_container(self.project, self.service)
        _docker_api("POST", f"/containers/{container_id}/restart")
        self._wait_api_health(container_id)

    def _compose(self) -> list[str]:
        prefix = ["sudo"] if self.use_sudo or not _docker_ok() else []
        return prefix + [
            "docker", "compose", "-f", self.compose_file,
            "--env-file", self.env_file,
        ]

    def _wait_cli_health(self, attempts: int = 40) -> None:
        probe = (
            "import urllib.request,sys;"
            "sys.exit(0 if urllib.request.urlopen("
            "'http://127.0.0.1:8081/control/healthz',timeout=5).status==200 else 1)"
        )
        last = "no attempt"
        for _ in range(attempts):
            result = subprocess.run(
                self._compose() + ["exec", "-T", self.service, "python", "-c", probe],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return
            last = result.stderr or result.stdout or ""
            time.sleep(2.0)
        raise RuntimeError(
            f"heimdall-emulator did not recover after restart: {last}")

    def _wait_api_health(self, container_id: str, attempts: int = 40) -> None:
        last = "no attempt"
        for _ in range(attempts):
            body = json.loads(
                _docker_api("GET", f"/containers/{container_id}/json"))
            health = ((body.get("State") or {}).get("Health") or {}).get("Status")
            running = (body.get("State") or {}).get("Running")
            if health == "healthy" or (health is None and running):
                time.sleep(1.0)
                return
            last = str(health or body.get("State"))
            time.sleep(2.0)
        raise RuntimeError(
            f"heimdall-emulator did not recover after restart: {last}")


class _UnixHTTPConnection(HTTPConnection):
    def __init__(self, path: str) -> None:
        super().__init__("localhost")
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self._path)
        self.sock = sock


def _docker_ok() -> bool:
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True)
    return result.returncode == 0


def _docker_cli_ok(use_sudo: bool) -> bool:
    cmd = ["sudo", "docker", "info"] if use_sudo else ["docker", "info"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    except FileNotFoundError:
        return False


def _docker_api(method: str, path: str, *, timeout: int = 60) -> str:
    conn = _UnixHTTPConnection(_DOCKER_SOCK)
    conn.timeout = timeout
    conn.request(method, f"/v1.41{path}")
    response = conn.getresponse()
    body = response.read().decode("utf-8")
    conn.close()
    if response.status >= 400:
        raise RuntimeError(f"docker API {method} {path}: {response.status} {body}")
    return body


def _find_container(project: str, service: str) -> str:
    filters = json.dumps({"label": [
        f"com.docker.compose.project={project}",
        f"com.docker.compose.service={service}",
    ]})
    payload = json.loads(_docker_api(
        "GET", f"/containers/json?filters={quote(filters)}"))
    if not payload:
        raise RuntimeError(
            f"no running container for {project}/{service}; is the stack up?")
    return payload[0]["Id"]
