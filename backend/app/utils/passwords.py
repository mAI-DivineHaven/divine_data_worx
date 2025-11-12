"""Password hashing utilities using PBKDF2."""

from __future__ import annotations

import binascii
import os
from dataclasses import dataclass
from hashlib import pbkdf2_hmac
from typing import Final

DEFAULT_ITERATIONS: Final[int] = 150_000
ALGORITHM: Final[str] = "sha256"


@dataclass(frozen=True)
class PasswordHash:
    """Structured representation of a stored password hash."""

    algorithm: str
    iterations: int
    salt: bytes
    digest: bytes

    def encode(self) -> str:
        """Serialize the hash to a persistent string format."""

        salt_hex = binascii.hexlify(self.salt).decode("ascii")
        digest_hex = binascii.hexlify(self.digest).decode("ascii")
        return f"pbkdf2_{self.algorithm}${self.iterations}${salt_hex}${digest_hex}"


def _derive(password: str, salt: bytes, iterations: int) -> bytes:
    if not password:
        raise ValueError("password must not be empty")
    return pbkdf2_hmac(ALGORITHM, password.encode("utf-8"), salt, iterations)


def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Generate a PBKDF2 hash for the provided password."""

    salt = os.urandom(16)
    digest = _derive(password, salt, iterations)
    return PasswordHash(ALGORITHM, iterations, salt, digest).encode()


def verify_password(password: str, encoded: str) -> bool:
    """Verify a candidate password against a stored PBKDF2 hash."""

    try:
        scheme, iterations_str, salt_hex, digest_hex = encoded.split("$")
    except ValueError:
        return False

    if not scheme.startswith("pbkdf2_"):
        return False

    algorithm = scheme.split("_", 1)[1]
    if algorithm != ALGORITHM:
        return False

    try:
        iterations = int(iterations_str)
        salt = binascii.unhexlify(salt_hex)
        digest = binascii.unhexlify(digest_hex)
    except (ValueError, binascii.Error):
        return False

    candidate = pbkdf2_hmac(algorithm, password.encode("utf-8"), salt, iterations)
    # Constant time compare
    if len(candidate) != len(digest):
        return False

    result = 0
    for x, y in zip(candidate, digest):
        result |= x ^ y
    return result == 0


__all__ = ["hash_password", "verify_password"]
