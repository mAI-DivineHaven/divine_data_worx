"""Repository helpers for session memory persistence.

This module provides a focused abstraction for all SQL statements touching the
session memory tables.  Concentrating data access in a single place makes the
logic easier to document, test, and evolve without leaking SQL details into the
service or routing layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg

from ..models import (
    SessionCitation,
    SessionCitationCreate,
    SessionMessage,
)


class SessionMemoryRepository:
    """Encapsulates SQL operations for session memory tables.

    Parameters:
        conn: An active :class:`asyncpg.Connection` used to execute SQL
            statements.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------
    async def insert_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        """Insert a new message row and return the hydrated model."""

        row = await self._conn.fetchrow(
            """
            INSERT INTO session_memory (session_id, role, content, metadata)
            VALUES ($1, $2, $3, $4)
            RETURNING message_id, session_id, role, content, metadata, created_at
            """,
            session_id,
            role,
            content,
            metadata,
        )
        return self._row_to_message(row, [])

    async def update_message(
        self,
        message_id: int,
        *,
        fields: dict[str, Any],
    ) -> SessionMessage | None:
        """Apply partial updates to a message record.

        Args:
            message_id: Primary key of the message to update.
            fields: Mapping of column names to new values.

        Returns:
            The updated message if the row exists, otherwise ``None``.
        """

        if not fields:
            return await self.get_message(message_id)

        assignments = []
        values: list[object] = [message_id]
        for idx, (column, value) in enumerate(fields.items(), start=1):
            assignments.append(f"{column} = ${idx + 1}")
            values.append(value)

        row = await self._conn.fetchrow(
            """
            UPDATE session_memory
               SET {assignments}
             WHERE message_id = $1
         RETURNING message_id, session_id, role, content, metadata, created_at
            """.format(
                assignments=", ".join(assignments)
            ),
            *values,
        )
        return self._row_to_message(row, []) if row else None

    async def get_message(self, message_id: int) -> SessionMessage | None:
        """Return a single message along with its citation trail."""

        row = await self._conn.fetchrow(
            """
            SELECT message_id, session_id, role, content, metadata, created_at
              FROM session_memory
             WHERE message_id = $1
            """,
            message_id,
        )
        if not row:
            return None

        citations = await self.get_citations_for_message(message_id)
        return self._row_to_message(row, citations)

    async def list_messages(
        self,
        session_id: str,
        *,
        limit: int,
        offset: int,
    ) -> list[SessionMessage]:
        """List messages for a session with pagination semantics."""

        rows = await self._conn.fetch(
            """
            SELECT message_id, session_id, role, content, metadata, created_at
              FROM session_memory
             WHERE session_id = $1
             ORDER BY message_id ASC
             LIMIT $2 OFFSET $3
            """,
            session_id,
            limit,
            offset,
        )
        if not rows:
            return []

        message_ids = [row["message_id"] for row in rows]
        citations_map = await self.get_citations_for_messages(message_ids)
        return [self._row_to_message(row, citations_map.get(row["message_id"], [])) for row in rows]

    async def count_messages(self, session_id: str) -> int:
        """Count the number of messages stored for a session."""

        value = await self._conn.fetchval(
            "SELECT COUNT(*) FROM session_memory WHERE session_id = $1",
            session_id,
        )
        return int(value or 0)

    async def delete_message(self, message_id: int) -> int:
        """Delete a message row and return the number of affected rows."""

        result = await self._conn.execute(
            "DELETE FROM session_memory WHERE message_id = $1",
            message_id,
        )
        return self._parse_rowcount(result)

    async def clear_session(self, session_id: str) -> int:
        """Remove every message belonging to the supplied session."""

        result = await self._conn.execute(
            "DELETE FROM session_memory WHERE session_id = $1",
            session_id,
        )
        return self._parse_rowcount(result)

    # ------------------------------------------------------------------
    # Citation helpers
    # ------------------------------------------------------------------
    async def add_citations(
        self,
        message_id: int,
        citations: Sequence[SessionCitationCreate],
    ) -> list[SessionCitation]:
        """Persist a batch of citations for a message."""

        if not citations:
            return []

        results: list[SessionCitation] = []
        for citation in citations:
            row = await self._conn.fetchrow(
                """
                INSERT INTO session_citation (
                    message_id, source_type, source_id, snippet, metadata
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING citation_id, message_id, source_type, source_id,
                          snippet, metadata, created_at
                """,
                message_id,
                citation.source_type,
                citation.source_id,
                citation.snippet,
                citation.metadata,
            )
            results.append(self._row_to_citation(row))
        return results

    async def replace_citations(
        self,
        message_id: int,
        citations: Sequence[SessionCitationCreate],
    ) -> list[SessionCitation]:
        """Replace the citation set for the given message."""

        await self._conn.execute(
            "DELETE FROM session_citation WHERE message_id = $1",
            message_id,
        )
        return await self.add_citations(message_id, citations)

    async def get_citations_for_message(self, message_id: int) -> list[SessionCitation]:
        """Return all citations referencing the specified message."""

        rows = await self._conn.fetch(
            """
            SELECT citation_id, message_id, source_type, source_id,
                   snippet, metadata, created_at
              FROM session_citation
             WHERE message_id = $1
             ORDER BY citation_id ASC
            """,
            message_id,
        )
        return [self._row_to_citation(row) for row in rows]

    async def get_citations_for_messages(
        self,
        message_ids: Sequence[int],
    ) -> dict[int, list[SessionCitation]]:
        """Return a mapping from message id to citations for bulk lookups."""

        if not message_ids:
            return {}

        rows = await self._conn.fetch(
            """
            SELECT citation_id, message_id, source_type, source_id,
                   snippet, metadata, created_at
              FROM session_citation
             WHERE message_id = ANY($1::bigint[])
             ORDER BY citation_id ASC
            """,
            list(message_ids),
        )
        citations: dict[int, list[SessionCitation]] = {}
        for row in rows:
            citation = self._row_to_citation(row)
            citations.setdefault(citation.message_id, []).append(citation)
        return citations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_message(
        row: asyncpg.Record,
        citations: Sequence[SessionCitation],
    ) -> SessionMessage:
        """Convert a message row into a :class:`SessionMessage` instance."""

        metadata = row["metadata"] if row["metadata"] is not None else None
        return SessionMessage(
            message_id=row["message_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            metadata=metadata,
            created_at=row["created_at"],
            citations=list(citations),
        )

    @staticmethod
    def _row_to_citation(row: asyncpg.Record) -> SessionCitation:
        """Transform a citation row into the corresponding model."""

        metadata = row["metadata"] if row["metadata"] is not None else None
        return SessionCitation(
            citation_id=row["citation_id"],
            message_id=row["message_id"],
            source_type=row["source_type"],
            source_id=row["source_id"],
            snippet=row["snippet"],
            metadata=metadata,
            created_at=row["created_at"],
        )

    @staticmethod
    def _parse_rowcount(result: str) -> int:
        """Extract the affected row count from an ``asyncpg`` status string."""

        try:
            return int(result.split(" ")[-1])
        except (IndexError, ValueError):  # pragma: no cover - defensive
            return 0
