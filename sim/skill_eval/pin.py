from __future__ import annotations

from sim.agent.config import AgentConfig
from sim.registry import Registry

ON_REF = "agent_config_benchmark_skills_on"
GENERAL_REF = "agent_config_benchmark_general_knowledge"
SOURCE_REF = "agent_config"


def pin(registry: Registry, source_ref: str = SOURCE_REF) -> dict[str, str]:
    _, raw = registry.load(source_ref)
    on = AgentConfig.from_dict(raw).as_dict()
    general = dict(on)
    general["tool_subset"] = []
    on_version = registry.commit(
        ON_REF, "agent", on, actor="skill_eval",
        note="benchmark: skills enabled",
    )
    general_version = registry.commit(
        GENERAL_REF, "agent", general, actor="skill_eval",
        note="benchmark: general knowledge; no Heimdall tools",
    )
    return {
        "existing_skills": on_version.ref,
        "general_knowledge": general_version.ref,
    }


def main() -> None:
    from sim.benchmark.env import load_env, require_env

    load_env()
    path = require_env("B2E_REGISTRY_DB")
    registry = Registry(path)
    refs = pin(registry)
    print(f"existing_skills={refs['existing_skills']}")
    print(f"general_knowledge={refs['general_knowledge']}")


if __name__ == "__main__":
    main()
