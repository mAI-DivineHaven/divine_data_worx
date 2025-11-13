"""Minimal JWT decoding utilities supporting HS* algorithms."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Iterable
from typing import Any


class JWTError(Exception):
    """Base JWT error."""


class JWTSignatureError(JWTError):
    """Raised when signature verification fails."""


class JWTAlgorithmError(JWTError):
    """Raised when an unsupported algorithm is requested."""


class JWTExpiredError(JWTError):
    """Raised when the token is expired."""


class JWTClaimError(JWTError):
    """Raised when a required claim is missing or invalid."""


def _b64decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def decode_jwt(
    token: str,
    secret_key: str,
    algorithms: Iterable[str] | None = None,
    audience: str | None = None,
    issuer: str | None = None,
) -> dict[str, Any]:
    """Decode and validate a JWT payload without external dependencies."""

    if not secret_key:
        raise JWTSignatureError("Secret key required for JWT validation")

    algorithms = list(algorithms or ["HS256"])
    segments = token.split(".")
    if len(segments) != 3:
        raise JWTError("Invalid token format")

    header_segment, payload_segment, signature_segment = segments
    header = json.loads(_b64decode(header_segment).decode("utf-8"))

    algorithm = header.get("alg")
    if algorithm not in algorithms:
        raise JWTAlgorithmError(f"Unsupported JWT algorithm: {algorithm}")

    signing_input = f"{header_segment}.{payload_segment}".encode()
    expected_signature = _sign(signing_input, secret_key, algorithm)
    actual_signature = _b64decode(signature_segment)

    if not hmac.compare_digest(expected_signature, actual_signature):
        raise JWTSignatureError("Invalid JWT signature")

    payload = json.loads(_b64decode(payload_segment).decode("utf-8"))

    now = int(time.time())
    exp = payload.get("exp")
    if exp is not None and now >= int(exp):
        raise JWTExpiredError("Token has expired")

    nbf = payload.get("nbf")
    if nbf is not None and now < int(nbf):
        raise JWTClaimError("Token not yet valid (nbf)")

    if audience is not None:
        aud = payload.get("aud")
        if isinstance(aud, list):
            if audience not in aud:
                raise JWTClaimError("Audience claim mismatch")
        elif aud != audience:
            raise JWTClaimError("Audience claim mismatch")

    if issuer is not None:
        iss = payload.get("iss")
        if iss != issuer:
            raise JWTClaimError("Issuer claim mismatch")

    return payload


def _sign(message: bytes, secret_key: str, algorithm: str) -> bytes:
    if algorithm == "HS256":
        digestmod = hashlib.sha256
    elif algorithm == "HS384":
        digestmod = hashlib.sha384
    elif algorithm == "HS512":
        digestmod = hashlib.sha512
    else:
        raise JWTAlgorithmError(f"Unsupported JWT algorithm: {algorithm}")

    return hmac.new(secret_key.encode("utf-8"), message, digestmod).digest()
