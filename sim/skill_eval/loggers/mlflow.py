from __future__ import annotations

from pathlib import Path
from typing import Any

from sim.skill_eval.loggers.base import EvalLogger
from sim.skill_eval.types import CaseScore, EvalCase, RunSummary, TurnResult


class MLflowLogger(EvalLogger):
    def __init__(
        self,
        tracking_uri: str = "var/mlruns",
        experiment: str = "skill-eval",
        eval_id: str = "",
        jsonl_path: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "mlflow is required when logging includes MLflow. "
                "Install deploy/requirements-eval.txt in the eval environment."
            ) from exc
        self._mlflow = mlflow
        uri = tracking_uri
        if not uri.startswith("file:") and "://" not in uri:
            Path(uri).mkdir(parents=True, exist_ok=True)
            uri = Path(uri).resolve().as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self._run = mlflow.start_run(run_name=eval_id or None)
        self._jsonl_path = jsonl_path
        self._params = dict(params or {})
        if self._params:
            mlflow.log_params(_flat(self._params))

    def log_case(
        self, case: EvalCase, turn: TurnResult, score: CaseScore,
        *, record: dict[str, Any],
    ) -> None:
        del case, turn, score, record

    def log_run(self, summary: RunSummary) -> None:
        metrics: dict[str, float] = {
            "task_success_rate": summary.task_success_rate,
            "mean_heimdall_calls": summary.mean_heimdall_calls,
            "mean_tokens": summary.mean_tokens,
            "mean_latency_ms": summary.mean_latency_ms,
            "n_cases": float(summary.n_cases),
        }
        if summary.routing_accuracy is not None:
            metrics["routing_accuracy"] = summary.routing_accuracy
        for category, values in summary.per_category.items():
            for key, value in values.items():
                metrics[f"{category}_{key}"] = float(value)
        self._mlflow.log_metrics(metrics)
        if self._jsonl_path and Path(self._jsonl_path).exists():
            self._mlflow.log_artifact(self._jsonl_path)
        summary_path = Path(self._jsonl_path).with_suffix(".summary.json")
        if summary_path.exists():
            self._mlflow.log_artifact(str(summary_path))
        self._mlflow.end_run()


def _flat(values: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        text = str(value)
        out[key] = text[:250]
    return out
