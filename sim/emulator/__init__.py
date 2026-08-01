"""C1 — the Heimdall API emulator as a service.

This is deliberately a thin layer over ``heimdall/``. That package already
reproduces the parts of the real API surface that make agents fail in
interesting ways: ``additionalProperties:false`` rejecting a whole body over one
stray filter key, ``condition_like`` wanting ``pattern`` and not ``value``,
granularity validation, SCD2 history semantics, and thirty-seven real Pulse HR
error codes. Rewriting any of that here would produce a second, subtly different
implementation of the thing under test.

What this package adds is what an *experiment* needs and a library does not:

``identity``  per-employee permission scoping, so 403 is a real outcome
``rpc``       recruitment RPC stubs in the RpcEnvelope shape
``config``    the two independent variables — traps_enabled, latency_profile
``app``       assembly, latency injection, health and config endpoints
"""

__all__ = ["create_app"]


def create_app(*args, **kwargs):  # pragma: no cover - thin re-export
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
