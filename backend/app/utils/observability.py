"""Observability helpers for configuring tracing providers."""

from __future__ import annotations

import logging

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.semconv.resource import ResourceAttributes

    _OTEL_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    trace = None  # type: ignore[assignment]
    JaegerExporter = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    TraceIdRatioBased = None  # type: ignore[assignment]
    ResourceAttributes = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False

_logger = logging.getLogger(__name__)

_TRACING_INITIALIZED = False


def _parse_otlp_headers(header_str: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for pair in header_str.split(","):
        if not pair.strip():
            continue
        key, _, value = pair.partition("=")
        if not key:
            continue
        headers[key.strip()] = value.strip()
    return headers


def configure_tracing(
    settings,
    service_name: str,
    service_version: str | None = None,
) -> object | None:
    """Configure and register the OpenTelemetry tracer provider."""

    global _TRACING_INITIALIZED

    if _TRACING_INITIALIZED:
        return trace.get_tracer_provider() if trace is not None else None

    if not _OTEL_AVAILABLE:
        _logger.debug("opentelemetry_not_installed")
        return None

    if not getattr(settings, "TRACING_ENABLED", True):
        return None

    exporter_choice = getattr(settings, "TRACING_EXPORTER", "otlp").lower()

    resource_attrs = {
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: getattr(settings, "APP_ENV", "development"),
    }
    if service_version:
        resource_attrs[ResourceAttributes.SERVICE_VERSION] = service_version

    resource = Resource.create(resource_attrs)

    sample_rate = float(getattr(settings, "TRACING_SAMPLE_RATE", 1.0))
    sampler = TraceIdRatioBased(sample_rate)
    provider = TracerProvider(resource=resource, sampler=sampler)

    exporter = None

    try:
        if exporter_choice == "jaeger":
            exporter = JaegerExporter(
                agent_host_name=getattr(settings, "JAEGER_AGENT_HOST", "localhost"),
                agent_port=int(getattr(settings, "JAEGER_AGENT_PORT", 6831)),
            )
        else:
            headers = _parse_otlp_headers(getattr(settings, "OTEL_EXPORTER_OTLP_HEADERS", ""))
            exporter = OTLPSpanExporter(
                endpoint=getattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"),
                insecure=bool(getattr(settings, "OTEL_EXPORTER_OTLP_INSECURE", True)),
                headers=headers if headers else None,
            )
    except Exception:  # noqa: BLE001
        _logger.exception(
            "failed_to_configure_tracing_exporter", extra={"exporter": exporter_choice}
        )
        return None

    if exporter is None:
        _logger.warning("no_tracing_exporter_configured", extra={"exporter": exporter_choice})
        return None

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACING_INITIALIZED = True
    return provider


__all__ = ["configure_tracing"]
