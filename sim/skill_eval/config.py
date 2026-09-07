from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EvalConfig:
    cases_path: str = "benchmark-60-cases.jsonl"
    case_ids: Any = field(default_factory=lambda: ["001-answerable"])
    eval_id: str = MISSING
    results_dir: str = "results"
    mlflow_tracking_uri: str = "var/mlruns"
    ignore_snapshot: bool = False
    phoenix_project: str = "b2e-itmo"
    catalog: Any = MISSING
    logging: Any = MISSING
    scorers: Any = MISSING
    stand: Any = MISSING
    factory: Any = MISSING
    installer: Any = MISSING


def register_config_store() -> ConfigStore:
    store = ConfigStore.instance()
    store.store(name="config_schema", node=EvalConfig)
    return store
