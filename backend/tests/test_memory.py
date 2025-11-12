"""Integration-style tests for the session memory API layer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from backend.tests.conftest import create_mock_record


@pytest.mark.asyncio
async def test_session_memory_crud(async_client, mock_pg_conn):
    """Exercise session memory CRUD endpoints with mocked persistence."""

    session_id = "session-unit"
    messages: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    next_message_id = 1
    next_citation_id = 1

    def now() -> datetime:
        return datetime.now(UTC)

    async def fetchrow_side_effect(query, *args, **kwargs):
        nonlocal next_message_id, next_citation_id
        sql = str(query).upper()
        if "INSERT INTO SESSION_MEMORY" in sql:
            message = {
                "message_id": next_message_id,
                "session_id": args[0],
                "role": args[1],
                "content": args[2],
                "metadata": args[3],
                "created_at": now(),
            }
            next_message_id += 1
            messages.append(message)
            return create_mock_record(message)
        if "INSERT INTO SESSION_CITATION" in sql:
            citation = {
                "citation_id": next_citation_id,
                "message_id": args[0],
                "source_type": args[1],
                "source_id": args[2],
                "snippet": args[3],
                "metadata": args[4],
                "created_at": now(),
            }
            next_citation_id += 1
            citations.append(citation)
            return create_mock_record(citation)
        if "UPDATE SESSION_MEMORY" in sql:
            message_id = args[0]
            target = next((m for m in messages if m["message_id"] == message_id), None)
            if target is None:
                return None
            assignments_section = sql.split("SET", 1)[1].split("WHERE", 1)[0]
            assignments = [part.strip() for part in assignments_section.split(",") if part.strip()]
            values_iter = iter(args[1:])
            for assignment in assignments:
                column = assignment.split("=", 1)[0].strip().lower()
                try:
                    value = next(values_iter)
                except StopIteration:  # pragma: no cover - defensive
                    break
                target[column] = value
            return create_mock_record(target)
        if "FROM SESSION_MEMORY" in sql and "MESSAGE_ID = $1" in sql:
            message_id = args[0]
            target = next((m for m in messages if m["message_id"] == message_id), None)
            return create_mock_record(target) if target else None
        return None

    async def fetch_side_effect(query, *args, **kwargs):
        sql = str(query).upper()
        if "FROM SESSION_MEMORY" in sql and "SESSION_ID = $1" in sql:
            limit = args[1]
            offset = args[2]
            rows = [m for m in messages if m["session_id"] == args[0]]
            slice_rows = rows[offset : offset + limit]
            return [create_mock_record(row) for row in slice_rows]
        if "FROM SESSION_CITATION" in sql:
            if "ANY($1::BIGINT[])" in sql:
                message_ids = set(args[0])
            else:
                message_ids = {args[0]}
            relevant = [c for c in citations if c["message_id"] in message_ids]
            relevant.sort(key=lambda item: item["citation_id"])
            return [create_mock_record(row) for row in relevant]
        return []

    async def fetchval_side_effect(query, *args, **kwargs):
        sql = str(query).upper()
        if "COUNT(*) FROM SESSION_MEMORY" in sql:
            return sum(1 for m in messages if m["session_id"] == args[0])
        return 0

    async def execute_side_effect(query, *args, **kwargs):
        sql = str(query).upper()
        if "DELETE FROM SESSION_MEMORY WHERE MESSAGE_ID" in sql:
            message_id = args[0]
            before = len(messages)
            messages[:] = [m for m in messages if m["message_id"] != message_id]
            citations[:] = [c for c in citations if c["message_id"] != message_id]
            return f"DELETE {before - len(messages)}"
        if "DELETE FROM SESSION_MEMORY WHERE SESSION_ID" in sql:
            session = args[0]
            removed_ids = {m["message_id"] for m in messages if m["session_id"] == session}
            messages[:] = [m for m in messages if m["session_id"] != session]
            before = len(citations)
            citations[:] = [c for c in citations if c["message_id"] not in removed_ids]
            return f"DELETE {len(removed_ids)}"
        if "DELETE FROM SESSION_CITATION" in sql:
            message_id = args[0]
            before = len(citations)
            citations[:] = [c for c in citations if c["message_id"] != message_id]
            return f"DELETE {before - len(citations)}"
        return "EXECUTE 0"

    mock_pg_conn.fetchrow.side_effect = fetchrow_side_effect
    mock_pg_conn.fetch.side_effect = fetch_side_effect
    mock_pg_conn.fetchval.side_effect = fetchval_side_effect
    mock_pg_conn.execute.side_effect = execute_side_effect

    response = await async_client.post(
        f"/v1/sessions/{session_id}/messages",
        json={
            "role": "user",
            "content": "Hello",
            "metadata": {"tone": "greeting"},
            "citations": [
                {"source_type": "verse", "source_id": "NIV:1:1:1", "snippet": "In the beginning"}
            ],
        },
    )
    assert response.status_code == 201
    created = response.json()
    assert created["role"] == "user"
    assert created["citations"] and created["citations"][0]["source_id"] == "NIV:1:1:1"
    message_id = created["message_id"]

    list_response = await async_client.get(f"/v1/sessions/{session_id}/messages?limit=10")
    assert list_response.status_code == 200
    body = list_response.json()
    assert body["total"] == 1
    assert body["items"][0]["message_id"] == message_id

    update_response = await async_client.patch(
        f"/v1/sessions/{session_id}/messages/{message_id}",
        json={"content": "Updated response", "citations": []},
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["content"] == "Updated response"
    assert updated["citations"] == []

    get_response = await async_client.get(f"/v1/sessions/{session_id}/messages/{message_id}")
    assert get_response.status_code == 200
    assert get_response.json()["content"] == "Updated response"

    delete_response = await async_client.delete(f"/v1/sessions/{session_id}/messages/{message_id}")
    assert delete_response.status_code == 204

    empty_response = await async_client.get(f"/v1/sessions/{session_id}/messages")
    assert empty_response.status_code == 200
    assert empty_response.json()["total"] == 0

    clear_response = await async_client.delete(f"/v1/sessions/{session_id}/messages")
    assert clear_response.status_code == 200
    assert clear_response.json()["deleted"] == 0
