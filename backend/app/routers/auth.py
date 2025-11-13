"""Authentication endpoints for issuing JWT access tokens."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..config import settings
from ..db.postgres_async import get_pg
from ..repositories.users import UserRepository
from ..schemas.users import UserRole, UserSummary
from ..utils.jwt import encode_jwt
from ..utils.passwords import verify_password


class LoginRequest(BaseModel):
    email: str = Field(..., examples=["user@example.com"])
    password: str = Field(..., min_length=1, examples=["secret"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = Field(default="bearer", example="bearer")
    expires_in: int = Field(..., description="Token lifetime in seconds")
    user: UserSummary


router = APIRouter(prefix="/auth", tags=["auth"])


def _build_user_summary(row) -> UserSummary:
    return UserSummary(
        user_id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        role=UserRole(row["role"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    conn=Depends(get_pg),
):
    """Authenticate a user via email/password and issue a JWT access token."""

    record = await UserRepository.fetch_user_credentials(conn, payload.email)
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    password_hash: str | None = record.get("password_hash")
    if not password_hash or not verify_password(payload.password, password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    user = _build_user_summary(record)

    if not settings.JWT_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured",
        )

    expires_in_seconds = settings.JWT_ACCESS_TOKEN_EXPIRES_MINUTES * 60
    token_payload = {
        "sub": str(user.user_id),
        "email": user.email,
        "role": user.role.value,
    }

    token = encode_jwt(
        token_payload,
        secret_key=settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
        expires_in=expires_in_seconds,
        audience=settings.JWT_AUDIENCE,
        issuer=settings.JWT_ISSUER,
    )

    return TokenResponse(access_token=token, expires_in=expires_in_seconds, user=user)


__all__ = ["router", "LoginRequest", "TokenResponse"]
