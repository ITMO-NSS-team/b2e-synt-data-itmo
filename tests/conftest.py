"""Shared test fixtures.

The tracer provider lives here rather than in each test module because OpenTelemetry
allows exactly one global provider per process: a second
``trace.set_tracer_provider`` call is ignored with a warning, so whichever module
registered second would silently collect nothing and every one of its span
assertions would fail for a reason that has nothing to do with the code under
test. One provider, one exporter, cleared per test.
"""
from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sim.fingerprint import RunFingerprint


@pytest.fixture(scope="session")
def _exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # Simple, not batched: a test that has to wait for a flush interval to see
    # its own span is a test that will eventually be flaky.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture()
def spans(_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    _exporter.clear()
    return _exporter


@pytest.fixture()
def fingerprint() -> RunFingerprint:
    return RunFingerprint.create(
        agent_config_version="agent_config@3",
        prompt_registry_version="system_prompt@1",
        skill_registry_hash="sha256:" + "0" * 64,
        model_id="claude-haiku-4-5-20251001",
        temperature=0.0,
        data_snapshot_hash="heimdall-sandbox@test",
        traps_enabled=True,
        latency_profile="realistic",
        hr_employee_ids="",
    )
