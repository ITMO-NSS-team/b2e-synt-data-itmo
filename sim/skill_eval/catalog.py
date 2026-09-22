from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import sim.skill_eval._path  # noqa: F401
from benchmarking.models import BusinessTask
from factory.base import SkillCreateRequest, SkillFactory
from sim.skill_eval.types import EvalCase, SessionSpec


class CatalogStrategy(ABC):
    name: str
    config_ref: str

    @abstractmethod
    def prepare(self, case: EvalCase) -> None:
        ...

    def session_spec(
        self, case: EvalCase, *, eval_id: str, hydra_run: str,
    ) -> SessionSpec:
        return SessionSpec(
            employee_id=case.runtime_actor_employee_id,
            config_ref=self.config_ref,
            metadata={
                "eval_id": eval_id,
                "catalog": self.name,
                "case_id": case.case_id,
                "hydra_run": hydra_run,
            },
        )

    def teardown(self) -> None:
        return None


class GeneralKnowledgeCatalog(CatalogStrategy):
    name = "general_knowledge"

    def __init__(
        self, config_ref: str = "agent_config_benchmark_general_knowledge",
    ) -> None:
        self.config_ref = config_ref

    def prepare(self, case: EvalCase) -> None:
        del case


class ExistingSkillsCatalog(CatalogStrategy):
    name = "existing_skills"

    def __init__(self, config_ref: str = "agent_config_benchmark_skills_on") -> None:
        self.config_ref = config_ref

    def prepare(self, case: EvalCase) -> None:
        del case


class FactoryCatalog(CatalogStrategy):
    name = "factory"

    def __init__(
        self,
        factory: SkillFactory,
        installer: Any,
        config_ref: str = "agent_config_benchmark_skills_on",
        tool_subset: list[str] | None = None,
    ) -> None:
        self.factory = factory
        self.installer = installer
        self.config_ref = config_ref
        self.tool_subset = tuple(tool_subset or (
            "list_models", "describe_model", "get_docs",
            "mcp_query", "find_skills", "get_skill",
        ))

    def prepare(self, case: EvalCase) -> None:
        skill = self.factory.generate(SkillCreateRequest(
            business_task=BusinessTask.model_validate(case.business_task),
            tool_subset=self.tool_subset,
            expected_skill=case.expected_skill,
            expected_skill_kind=case.expected_skill_kind,
        ))
        self.installer.install(skill)

    def teardown(self) -> None:
        self.installer.cleanup()
