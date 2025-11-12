"""JWT authentication middleware."""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.jwt import (
    JWTAlgorithmError,
    JWTClaimError,
    JWTError,
    JWTExpiredError,
    JWTSignatureError,
    decode_jwt,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer JWT tokens and attach the payload to request state."""

    def __init__(
        self,
        app,
        secret_key: str,
        algorithm: str = "HS256",
        audience: str | None = None,
        issuer: str | None = None,
        exempt_paths: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.audience = audience
        self.issuer = issuer
        self.exempt_paths = set(exempt_paths or [])
        self.enabled = bool(secret_key)
        if not self.enabled:
            logger.warning("JWTAuthMiddleware disabled - missing secret key")

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request.state.user = None
        request.state.authenticated = False

        if self._should_skip(request):
            return await call_next(request)

        if not self.enabled:
            return await call_next(request)

        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.lower().startswith("bearer "):
            return _unauthorized_response("Unauthorized")

        token = authorization.split(" ", 1)[1]

        try:
            payload = decode_jwt(
                token,
                secret_key=self.secret_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
            )
        except (JWTSignatureError, JWTExpiredError) as exc:
            logger.info("JWT validation failed", extra={"reason": str(exc)})
            return _unauthorized_response("Invalid token")
        except (JWTAlgorithmError, JWTClaimError, JWTError) as exc:
            logger.warning("JWT parsing error", extra={"error": str(exc)})
            return _unauthorized_response("Invalid token")

        request.state.user = payload
        request.state.authenticated = True
        return await call_next(request)

    def _should_skip(self, request: Request) -> bool:
        if request.method.upper() == "OPTIONS":
            return True
        path = request.url.path
        if path in self.exempt_paths:
            return True
        return False


__all__ = ["JWTAuthMiddleware"]


def _unauthorized_response(detail: str) -> JSONResponse:
    """Return a RFC-compliant 401 response for Bearer authentication."""

    return JSONResponse(
        status_code=401,
        content={"detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )
