"""Tests for authentication endpoints."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from backend.app.config import settings
from backend.app.utils.passwords import hash_password


@pytest.mark.asyncio
async def test_login_success(async_client, mock_pg_conn):
    user_id = uuid4()
    password = "Testing123!"
    settings.JWT_SECRET_KEY = "test-secret"
    mock_pg_conn.fetchrow = AsyncMock(
        return_value={
            "id": user_id,
            "email": "user@example.com",
            "display_name": "Test User",
            "role": "member",
            "password_hash": hash_password(password),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    )

    with patch("backend.app.routers.auth.verify_password", return_value=True) as verify_mock:
        response = await async_client.post(
            "/v1/auth/login",
            json={"email": "user@example.com", "password": password},
        )
        verify_mock.assert_called_once()

    data = response.json()
    assert mock_pg_conn.fetchrow.await_count == 1
    assert response.status_code == 200, data
    assert "access_token" in data and data["access_token"]
    assert data["token_type"] == "bearer"
    assert isinstance(data["expires_in"], int) and data["expires_in"] > 0
    assert data["user"]["user_id"] == str(user_id)
    assert data["user"]["email"] == "user@example.com"


@pytest.mark.asyncio
async def test_login_invalid_password(async_client, mock_pg_conn):
    settings.JWT_SECRET_KEY = "test-secret"
    mock_pg_conn.fetchrow = AsyncMock(
        return_value={
            "id": uuid4(),
            "email": "user@example.com",
            "display_name": "Test User",
            "role": "member",
            "password_hash": hash_password("CorrectPass123"),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    )

    response = await async_client.post(
        "/v1/auth/login",
        json={"email": "user@example.com", "password": "WrongPass"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


@pytest.mark.asyncio
async def test_login_unknown_user(async_client, mock_pg_conn):
    settings.JWT_SECRET_KEY = "test-secret"
    mock_pg_conn.fetchrow = AsyncMock(return_value=None)

    response = await async_client.post(
        "/v1/auth/login",
        json={"email": "missing@example.com", "password": "irrelevant"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"
