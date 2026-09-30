from __future__ import annotations


def test_configure_adds_optional_secondary_otlp_exporter(monkeypatch) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace import export
    from phoenix import otel

    from sim import telemetry

    class Provider:
        def __init__(self) -> None:
            self.processors = []
            self.options = []

        def add_span_processor(self, processor, **options) -> None:
            self.processors.append(processor)
            self.options.append(options)

    class Exporter:
        def __init__(self, *, endpoint: str) -> None:
            self.endpoint = endpoint

    class Processor:
        def __init__(self, exporter: Exporter) -> None:
            self.exporter = exporter

    provider = Provider()
    register_options = {}

    def register(**options):
        register_options.update(options)
        return provider

    monkeypatch.setattr(otel, "register", register)
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", Exporter)
    monkeypatch.setattr(export, "BatchSpanProcessor", Processor)

    configured = telemetry.configure(
        endpoint="http://phoenix:6006/v1/traces",
        project_name="b2e-itmo",
        secondary_endpoint="http://openlit:4318",
    )

    assert configured is provider
    assert len(provider.processors) == 1
    assert provider.processors[0].exporter.endpoint == "http://openlit:4318/v1/traces"
    assert provider.options == [{"replace_default_processor": False}]
    assert register_options["resource"].attributes["service.name"] == "b2e-agent"


def test_benchmark_metrics_are_exported_as_a_correlated_evaluator_span(spans) -> None:
    from sim import telemetry

    telemetry.emit_benchmark_metrics(
        {"exact_match": 1, "latency_ms": 1250.5, "cost_usd": None},
        eval_id="eval-1", run_id="run-1", case_id="case-1",
        mode="existing_skills", repetition=1, status="completed",
        scorer_version="benchmark-scorer@1",
        trace_id="1" * 32, parent_span_id="2" * 16,
    )

    span = spans.get_finished_spans()[0]
    assert span.name == "b2e.benchmark.score"
    assert span.context.trace_id == int("1" * 32, 16)
    assert span.parent.span_id == int("2" * 16, 16)
    assert span.attributes["openinference.span.kind"] == "EVALUATOR"
    assert span.attributes["b2e.metric.exact_match"] == 1
    assert span.attributes["b2e.metric.latency_ms"] == 1250.5
    assert "b2e.metric.cost_usd" not in span.attributes
