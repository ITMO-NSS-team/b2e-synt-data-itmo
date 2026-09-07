from typing import Any

from sim.skill_eval.loggers.base import CompositeLogger, EvalLogger
from sim.skill_eval.loggers.jsonl import JsonlLogger
from sim.skill_eval.loggers.mlflow import MLflowLogger
from sim.skill_eval.loggers.phoenix import PhoenixLogger

__all__ = [
    "CompositeLogger",
    "EvalLogger",
    "JsonlLogger",
    "MLflowLogger",
    "PhoenixLogger",
    "bind_stand",
]


def bind_stand(logger: EvalLogger, stand: Any) -> None:
    if isinstance(logger, CompositeLogger):
        for child in logger.loggers:
            bind_stand(child, stand)
        return
    if isinstance(logger, PhoenixLogger):
        logger.bind_http(stand.phoenix_http())
