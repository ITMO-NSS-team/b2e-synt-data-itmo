from __future__ import annotations

from sim.agent.config import AgentConfig
from sim.registry import Registry

ON_REF = "agent_config_benchmark_skills_on"
OFF_REF = "agent_config_benchmark_skills_off"
SOURCE_REF = "agent_config_openrouter"


def pin(registry: Registry, source_ref: str = SOURCE_REF) -> dict[str, str]:
    _, raw = registry.load(source_ref)
    on = AgentConfig.from_dict(raw).as_dict()
    off = dict(on)
    off["tool_subset"] = [
        name for name in on["tool_subset"]
        if name not in {"find_skills", "get_skill"}
    ]
    on_version = registry.commit(
        ON_REF, "agent", on, actor="skill_eval",
        note="benchmark: skills enabled",
    )
    off_version = registry.commit(
        OFF_REF, "agent", off, actor="skill_eval",
        note="benchmark: find_skills/get_skill disabled",
    )
    return {"skills_on": on_version.ref, "skills_off": off_version.ref}


def main() -> None:
    import os
    path = os.environ.get("B2E_REGISTRY_DB", "/app/registry/registry.db")
    registry = Registry(path)
    refs = pin(registry)
    print(f"skills_on={refs['skills_on']}")
    print(f"skills_off={refs['skills_off']}")


if __name__ == "__main__":
    main()
