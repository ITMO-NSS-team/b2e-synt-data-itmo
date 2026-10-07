"""Publish one final telemetry summary for a merged benchmark evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim import telemetry

from .cli import configure_metric_export


def publish_summary(result_dir: str | Path) -> bool:
    """Copy an already calculated merged summary to Phoenix/OpenLIT."""
    root = Path(result_dir)
    manifest = _read_object(root / "run-manifest.json")
    summary = _read_object(root / "summary.json")
    if manifest.get("check_only"):
        raise ValueError("cannot publish a check-only benchmark")

    provider = configure_metric_export()
    if provider is None:
        return False
    eval_id = str(manifest["eval_id"])
    modes = tuple(str(item["name"]) for item in manifest["modes"])
    try:
        with telemetry.benchmark_run(
            eval_id=eval_id,
            modes=modes,
            repetitions=int(manifest["repetitions"]),
            case_count=len(manifest["case_ids"]),
            final=True,
        ) as span:
            telemetry.set_benchmark_summary(
                span, summary, eval_id=eval_id, final=True,
            )
    finally:
        if provider.force_flush() is False:
            raise RuntimeError("merged benchmark telemetry export did not flush")
    return True


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing merged benchmark artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"merged benchmark artifact must be an object: {path}")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Publish the final summary of a merged benchmark run",
    )
    result.add_argument("--result-dir", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        exported = publish_summary(args.result_dir)
        print(json.dumps({
            "result_dir": str(Path(args.result_dir).resolve()),
            "telemetry_exported": exported,
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"benchmark summary publishing failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
