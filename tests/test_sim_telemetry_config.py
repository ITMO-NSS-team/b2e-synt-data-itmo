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
