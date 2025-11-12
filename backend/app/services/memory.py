"""Service layer for managing conversational session memory.

This module centralises the business logic required to persist, query, and
mutate conversational transcripts and their associated citation trails.  The
service composes the lower-level repository utilities to provide a cohesive
API that agent clients can depend on when storing or retrieving the context of
an interaction.
"""

from __future__ import annotations

import asyncpg

from ..models import (
    SessionCitation,
    SessionContextResponse,
    SessionMessage,
    SessionMessageAppendRequest,
    SessionMessageCreate,
    SessionMessageUpdate,
)
from ..repositories.memory import SessionMemoryRepository


class SessionMemoryService:
    """Coordinates conversational session persistence logic.

    Parameters:
        conn: An :class:`asyncpg.Connection` bound to the current request
            context. The connection is reused across repository calls so that a
            single transactional scope can be established where required.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self._repo = SessionMemoryRepository(conn)

    async def append_message(
        self,
        session_id: str,
        payload: SessionMessageAppendRequest,
    ) -> SessionMessage:
        """Persist a new message with optional citations.

        Args:
            session_id: Identifier for the conversational session to which the
                message belongs.
            payload: Message content plus optional metadata and citation
                descriptors supplied by the client.

        Returns:
            The fully-hydrated :class:`SessionMessage` record, including any
            citations that were persisted within the same transaction.
        """

        create_payload = SessionMessageCreate(
            session_id=session_id,
            role=payload.role,
            content=payload.content,
            metadata=payload.metadata,
            citations=payload.citations,
        )
        async with self._conn.transaction():
            message = await self._repo.insert_message(
                session_id=create_payload.session_id,
                role=create_payload.role,
                content=create_payload.content,
                metadata=create_payload.metadata,
            )
            citations: list[SessionCitation] = await self._repo.add_citations(
                message.message_id,
                create_payload.citations,
            )
        return message.model_copy(update={"citations": citations})

    async def list_session_context(
        self,
        session_id: str,
        *,
        limit: int,
        offset: int,
    ) -> SessionContextResponse:
        """Return a paginated slice of session messages.

        Args:
            session_id: Identifier of the session whose history should be
                retrieved.
            limit: Maximum number of messages to return.
            offset: Number of messages to skip from the beginning of the
                session history.

        Returns:
            A :class:`SessionContextResponse` containing the selected window of
            messages and pagination metadata.
        """

        items = await self._repo.list_messages(
            session_id,
            limit=limit,
            offset=offset,
        )
        total = await self._repo.count_messages(session_id)
        return SessionContextResponse(
            session_id=session_id,
            total=total,
            limit=limit,
            offset=offset,
            items=items,
        )

    async def get_message(self, session_id: str, message_id: int) -> SessionMessage:
        """Fetch a single message ensuring it belongs to the session.

        Args:
            session_id: Identifier of the session that owns the message.
            message_id: Primary key of the message to retrieve.

        Returns:
            The :class:`SessionMessage` if it exists and belongs to the session.

        Raises:
            LookupError: If the message cannot be found or is associated with a
                different session identifier.
        """

        message = await self._repo.get_message(message_id)
        if not message or message.session_id != session_id:
            raise LookupError(f"message {message_id} not found for session {session_id}")
        return message

    async def update_message(
        self,
        session_id: str,
        message_id: int,
        payload: SessionMessageUpdate,
    ) -> SessionMessage:
        """Apply partial updates to a stored message.

        Args:
            session_id: Identifier of the session that owns the message.
            message_id: Primary key of the message to update.
            payload: Partial update payload describing the fields to mutate.

        Returns:
            The updated :class:`SessionMessage` including its citation trail.

        Raises:
            LookupError: If the message does not exist for the specified
                session.
            ValueError: If the payload contains invalid field values as
                enforced by the repository layer or model validators.
        """

        async with self._conn.transaction():
            existing = await self._repo.get_message(message_id)
            if not existing or existing.session_id != session_id:
                raise LookupError(f"message {message_id} not found for session {session_id}")

            fields = payload.model_dump(exclude_unset=True, exclude={"citations"})
            updated: SessionMessage | None
            if fields:
                updated = await self._repo.update_message(message_id, fields=fields)
                if not updated:
                    raise LookupError(f"message {message_id} not found for session {session_id}")
            else:
                updated = existing

            if "citations" in payload.model_fields_set:
                citations = await self._repo.replace_citations(
                    message_id,
                    payload.citations or [],
                )
            else:
                citations = await self._repo.get_citations_for_message(message_id)

        citations_list: list[SessionCitation] = list(citations)
        return updated.model_copy(update={"citations": citations_list})

    async def delete_message(self, session_id: str, message_id: int) -> None:
        """Delete a message and associated citations.

        Args:
            session_id: Identifier of the session that owns the message.
            message_id: Primary key of the message to delete.

        Raises:
            LookupError: If the message does not exist for the provided
                session identifier.
        """

        message = await self._repo.get_message(message_id)
        if not message or message.session_id != session_id:
            raise LookupError(f"message {message_id} not found for session {session_id}")
        deleted = await self._repo.delete_message(message_id)
        if deleted == 0:  # pragma: no cover - defensive
            raise LookupError(f"message {message_id} not found for session {session_id}")

    async def clear_session(self, session_id: str) -> int:
        """Remove all messages for a session.

        Args:
            session_id: Identifier of the session to truncate.

        Returns:
            The number of messages deleted across the session.
        """

        return await self._repo.clear_session(session_id)
