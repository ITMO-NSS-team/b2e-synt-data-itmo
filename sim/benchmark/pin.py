from __future__ import annotations

from sim.agent.config import AgentConfig
from sim.benchmark.modes import DATA_TOOLS, DOCS_ONLY_TOOLS, GENERATED_SKILL_TOOLS
from sim.registry import Registry

ON_REF = "agent_config_benchmark_skills_on"
DISABLED_REF = "agent_config_benchmark_skills_disabled"
GENERAL_REF = "agent_config_benchmark_general_knowledge"
DOCS_ONLY_REF = "agent_config_benchmark_docs_only"
EXISTING_NO_DOCS_REF = "agent_config_benchmark_existing_skills_no_docs"
GENERATED_REF = "agent_config_benchmark_generated_skills"
SOURCE_REF = "agent_config"


def pin(registry: Registry, source_ref: str = SOURCE_REF) -> dict[str, str]:
    _, raw = registry.load(source_ref)
    on = AgentConfig.from_dict(raw).as_dict()
    disabled = dict(on)
    disabled["tool_subset"] = list(DATA_TOOLS)
    general = dict(on)
    general["tool_subset"] = []
    docs_only = dict(on)
    docs_only["tool_subset"] = list(DOCS_ONLY_TOOLS)
    existing_no_docs = dict(on)
    existing_no_docs["tool_subset"] = list(GENERATED_SKILL_TOOLS)
    generated = dict(on)
    generated["tool_subset"] = list(GENERATED_SKILL_TOOLS)
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
    docs_only_version = registry.commit(
        DOCS_ONLY_REF, "agent", docs_only, actor="benchmark",
        note="benchmark: data tools and procedural docs; no skills",
    )
    existing_no_docs_version = registry.commit(
        EXISTING_NO_DOCS_REF, "agent", existing_no_docs, actor="benchmark",
        note="benchmark: existing skills without procedural docs",
    )
    generated_version = registry.commit(
        GENERATED_REF, "agent", generated, actor="benchmark",
        note="benchmark: generated catalog only; no Heimdall docs",
    )
    return {
        "existing_skills": on_version.ref,
        "existing_plus_generated": on_version.ref,
        "generated_skills": generated_version.ref,
        "skills_disabled": disabled_version.ref,
        "general_knowledge": general_version.ref,
        "docs_only": docs_only_version.ref,
        "existing_skills_no_docs": existing_no_docs_version.ref,
    }


def main() -> None:
    from sim.benchmark.env import load_env, require_env

    load_env()
    path = require_env("B2E_REGISTRY_DB")
    registry = Registry(path)
    refs = pin(registry)
    print(f"existing_skills={refs['existing_skills']}")
    print(f"existing_plus_generated={refs['existing_plus_generated']}")
    print(f"generated_skills={refs['generated_skills']}")
    print(f"skills_disabled={refs['skills_disabled']}")
    print(f"general_knowledge={refs['general_knowledge']}")
    print(f"docs_only={refs['docs_only']}")
    print(f"existing_skills_no_docs={refs['existing_skills_no_docs']}")


if __name__ == "__main__":
    main()
