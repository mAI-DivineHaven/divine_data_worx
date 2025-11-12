"""
Health check router for API uptime monitoring.

Provides a simple health check endpoint for load balancers, monitoring tools,
and orchestration platforms (e.g., Kubernetes liveness/readiness probes).

The endpoint returns a minimal JSON response with no database dependencies
for fast response times even under heavy load or database issues.

Example Usage:
    ```bash
    curl http://localhost:8000/v1/healthz
    # Response: {"ok": true}
    ```

Integration with monitoring:
    ```yaml
    # Kubernetes liveness probe example
    livenessProbe:
      httpGet:
        path: /v1/healthz
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 10
    ```
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz", response_model=dict[str, bool])
def healthz() -> dict[str, bool]:
    """
    Health check endpoint for API uptime monitoring.

    Returns a simple JSON response indicating the API is responsive.
    Does not check database connectivity or other dependencies to ensure
    fast response even during partial system degradation.

    Returns:
        Dictionary with single "ok" key set to True

    Response:
        ```json
        {"ok": true}
        ```

    Status Codes:
        200: API is healthy and responsive
    """
    return {"ok": True}
