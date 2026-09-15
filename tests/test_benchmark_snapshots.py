"""Unit tests for content-addressed skill catalog snapshots."""
from __future__ import annotations

from pathlib import Path

import pytest

from sim.benchmark.catalog_snapshots import snapshot_catalog, verify_snapshot


def skill_yaml(name: str) -> str:
    return (
        f"name: {name}\n"
        f"title: {name}\n"
        "kind: recipe\nversion: 1.0.0\ndomain: org\n"
        f"description: Skill {name}\n"
        "model: {schema: dm_core, logic_model: employee_actual}\n"
        "query: {schema: dm_core, logic_model: employee_actual, metrics: [fact_count]}\n"
    )


def test_standard_snapshot_is_content_addressed_and_detects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "org").mkdir(parents=True)
    (source / "org" / "standard.yaml").write_text(skill_yaml("standard"), encoding="utf-8")
    snapshot = snapshot_catalog(source, tmp_path / "snapshots")
    assert snapshot == snapshot_catalog(source, tmp_path / "snapshots")
    assert snapshot.name == verify_snapshot(snapshot).removeprefix("sha256:")
    (snapshot / "org" / "standard.yaml").write_text(
        skill_yaml("standard") + "notes: [changed]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="modified"):
        verify_snapshot(snapshot)
