from __future__ import annotations

import sim.skill_eval._path  # noqa: F401

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from sim.skill_eval.cases import load_cases
from sim.skill_eval.config import register_config_store
from sim.skill_eval.loggers import bind_stand
from sim.skill_eval.runner import EvalRunner

register_config_store()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    hydra_run = HydraConfig.get().runtime.output_dir
    catalog = instantiate(cfg.catalog)
    scorers = instantiate(cfg.scorers)
    stand = instantiate(cfg.stand)
    loggers = instantiate(cfg.logging)
    bind_stand(loggers, stand)
    runner = EvalRunner(
        catalog=catalog,
        scorers=scorers,
        loggers=loggers,
        stand=stand,
        eval_id=str(cfg.eval_id),
        hydra_run=str(hydra_run),
        ignore_snapshot=bool(cfg.ignore_snapshot),
    )
    case_ids = OmegaConf.to_container(cfg.case_ids, resolve=True)
    if case_ids in (None, "all"):
        case_ids = None
    elif isinstance(case_ids, str):
        case_ids = [item.strip() for item in case_ids.split(",") if item.strip()]
    cases = load_cases(cfg.cases_path, case_ids)
    print(
        f"skill-eval {cfg.eval_id} catalog={catalog.name} "
        f"cases={len(cases)}"
    )
    print(
        "Numeric gold is scored only when live snapshot_id matches the case "
        "(or ignore_snapshot=true). Gold was built on data-previous; compose "
        "defaults to data-small."
    )
    summary = runner.run(cases)
    print(
        f"task_success_rate={summary.task_success_rate:.3f} "
        f"routing_accuracy={summary.routing_accuracy} "
        f"n={summary.n_cases}"
    )


if __name__ == "__main__":
    main()
