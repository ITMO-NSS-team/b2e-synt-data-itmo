from __future__ import annotations

from sim.agent.config import AgentConfig
from sim.benchmark.modes import DATA_TOOLS
from sim.registry import Registry

ON_REF = "agent_config_benchmark_skills_on"
DISABLED_REF = "agent_config_benchmark_skills_disabled"
GENERAL_REF = "agent_config_benchmark_general_knowledge"
SOURCE_REF = "agent_config"


def pin(registry: Registry, source_ref: str = SOURCE_REF) -> dict[str, str]:
    _, raw = registry.load(source_ref)
    on = AgentConfig.from_dict(raw).as_dict()
    disabled = dict(on)
    disabled["tool_subset"] = list(DATA_TOOLS)
    general = dict(on)
    general["tool_subset"] = []
    on_version = registry.commit(
        ON_REF, "agent", on, actor="benchmark",
        note="benchmark: skills enabled",
    )
    disabled_version = registry.commit(
        DISABLED_REF, "agent", disabled, actor="benchmark",
        note="benchmark: skills disabled; data tools only",
    )
    general_version = registry.commit(
        GENERAL_REF, "agent", general, actor="benchmark",
        note="benchmark: general knowledge; no Heimdall tools",
    )
    return {
        "existing_skills": on_version.ref,
        "skills_disabled": disabled_version.ref,
        "general_knowledge": general_version.ref,
    }


def main() -> None:
    from sim.benchmark.env import load_env, require_env

    load_env()
    path = require_env("B2E_REGISTRY_DB")
    registry = Registry(path)
    refs = pin(registry)
    print(f"existing_skills={refs['existing_skills']}")
    print(f"skills_disabled={refs['skills_disabled']}")
    print(f"general_knowledge={refs['general_knowledge']}")


if __name__ == "__main__":
    main()
