"""Repository helpers for account profiles and AI conversations."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

from ..schemas.users import (
    ConversationCreate,
    MessageCreate,
    ProfileSurvey,
    RegistrationRequest,
    SharePreferenceUpdate,
)
from ..utils.passwords import hash_password


class UserRepository:
    """CRUD helpers for app_user and user_profile tables."""

    @staticmethod
    async def create_user(conn: asyncpg.Connection, payload: RegistrationRequest) -> asyncpg.Record:
        password_hash = hash_password(payload.password)
        row = await conn.fetchrow(
            """
            INSERT INTO app_user (email, display_name, password_hash, role)
            VALUES ($1, $2, $3, $4)
            RETURNING id, email, display_name, role, created_at, updated_at
            """,
            payload.email,
            payload.display_name,
            password_hash,
            payload.role.value,
        )
        return row

    @staticmethod
    async def upsert_profile(
        conn: asyncpg.Connection,
        user_id: UUID,
        profile: ProfileSurvey,
    ) -> asyncpg.Record:
        share_preferences = {
            key: value.model_dump() for key, value in profile.share_preferences.items()
        }

        row = await conn.fetchrow(
            """
            INSERT INTO user_profile (
                user_id,
                bio,
                spiritual_background,
                denominational_identity,
                study_focus_topics,
                study_rhythm,
                guidance_preferences,
                preferred_translations,
                prayer_interests,
                ai_journal_opt_in,
                share_preferences,
                created_at,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now(), now()
            )
            ON CONFLICT (user_id) DO UPDATE SET
                bio = EXCLUDED.bio,
                spiritual_background = EXCLUDED.spiritual_background,
                denominational_identity = EXCLUDED.denominational_identity,
                study_focus_topics = EXCLUDED.study_focus_topics,
                study_rhythm = EXCLUDED.study_rhythm,
                guidance_preferences = EXCLUDED.guidance_preferences,
                preferred_translations = EXCLUDED.preferred_translations,
                prayer_interests = EXCLUDED.prayer_interests,
                ai_journal_opt_in = EXCLUDED.ai_journal_opt_in,
                share_preferences = EXCLUDED.share_preferences,
                updated_at = now()
            RETURNING *
            """,
            user_id,
            profile.bio,
            profile.spiritual_background,
            profile.denominational_identity,
            profile.study_focus_topics,
            profile.study_rhythm,
            profile.guidance_preferences,
            profile.preferred_translations,
            profile.prayer_interests,
            profile.ai_journal_opt_in,
            share_preferences,
        )
        return row

    @staticmethod
    async def update_share_preferences(
        conn: asyncpg.Connection,
        user_id: UUID,
        update: SharePreferenceUpdate,
    ) -> asyncpg.Record:
        # Fetch current share preferences
        existing = await conn.fetchval(
            "SELECT share_preferences FROM user_profile WHERE user_id = $1",
            user_id,
        )

        merged: dict[str, Any] = existing or {}
        for key, value in update.preferences.items():
            merged[key] = value.model_dump()

        row = await conn.fetchrow(
            """
            UPDATE user_profile
            SET share_preferences = $2::jsonb,
                updated_at = now()
            WHERE user_id = $1
            RETURNING *
            """,
            user_id,
            merged,
        )
        return row

    @staticmethod
    async def fetch_profile(conn: asyncpg.Connection, user_id: UUID) -> asyncpg.Record | None:
        row = await conn.fetchrow(
            """
            SELECT
                u.id AS user_id,
                u.email,
                u.display_name,
                u.role,
                u.created_at,
                u.updated_at,
                p.bio,
                p.spiritual_background,
                p.denominational_identity,
                p.study_focus_topics,
                p.study_rhythm,
                p.guidance_preferences,
                p.preferred_translations,
                p.prayer_interests,
                p.ai_journal_opt_in,
                p.share_preferences
            FROM app_user u
            LEFT JOIN user_profile p ON p.user_id = u.id
            WHERE u.id = $1
            """,
            user_id,
        )
        return row


class ConversationRepository:
    """Persistence helpers for AI conversation history."""

    @staticmethod
    async def create_conversation(
        conn: asyncpg.Connection,
        user_id: UUID,
        payload: ConversationCreate,
    ) -> asyncpg.Record:
        row = await conn.fetchrow(
            """
            INSERT INTO ai_conversation (user_id, title, metadata)
            VALUES ($1, $2, $3)
            RETURNING conversation_id, user_id, title, metadata, created_at, updated_at
            """,
            user_id,
            payload.title,
            payload.metadata or {},
        )
        return row

    @staticmethod
    async def list_conversations(conn: asyncpg.Connection, user_id: UUID) -> list[asyncpg.Record]:
        rows = await conn.fetch(
            """
            SELECT conversation_id, user_id, title, metadata, created_at, updated_at
            FROM ai_conversation
            WHERE user_id = $1
            ORDER BY updated_at DESC
            """,
            user_id,
        )
        return rows

    @staticmethod
    async def fetch_conversation(
        conn: asyncpg.Connection, conversation_id: UUID
    ) -> asyncpg.Record | None:
        row = await conn.fetchrow(
            """
            SELECT conversation_id, user_id, title, metadata, created_at, updated_at
            FROM ai_conversation
            WHERE conversation_id = $1
            """,
            conversation_id,
        )
        return row

    @staticmethod
    async def append_message(
        conn: asyncpg.Connection,
        conversation_id: UUID,
        payload: MessageCreate,
    ) -> asyncpg.Record:
        row = await conn.fetchrow(
            """
            WITH inserted AS (
                INSERT INTO ai_message (conversation_id, sender_role, content, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING message_id, conversation_id, sender_role, content, metadata, created_at, sequence
            )
            UPDATE ai_conversation
            SET updated_at = now()
            WHERE conversation_id = $1
            RETURNING (
                SELECT row_to_json(inserted) FROM inserted
            ) AS message
            """,
            conversation_id,
            payload.sender_role,
            payload.content,
            payload.metadata or {},
        )
        message = row["message"]
        return message

    @staticmethod
    async def list_messages(
        conn: asyncpg.Connection, conversation_id: UUID
    ) -> list[asyncpg.Record]:
        rows = await conn.fetch(
            """
            SELECT message_id, conversation_id, sender_role, content, metadata, created_at, sequence
            FROM ai_message
            WHERE conversation_id = $1
            ORDER BY sequence ASC
            """,
            conversation_id,
        )
        return rows


__all__ = ["UserRepository", "ConversationRepository"]
