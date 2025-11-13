"""REST endpoints for account registration, profiles, and AI journaling."""

from __future__ import annotations

from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..db.postgres_async import get_pg
from ..repositories.users import ConversationRepository, UserRepository
from ..schemas.users import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    MessageCreate,
    MessageRecord,
    ProfileResponse,
    ProfileSurvey,
    RegistrationRequest,
    SharePreferenceUpdate,
    UserRole,
    UserSummary,
)
from ..services.profile_privacy import Viewer, filter_profile_fields

router = APIRouter(tags=["users"])


def _record_to_summary(row: asyncpg.Record) -> UserSummary:
    return UserSummary(
        user_id=row["user_id"],
        email=row["email"],
        display_name=row["display_name"],
        role=UserRole(row["role"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_profile(row: asyncpg.Record) -> ProfileSurvey | None:
    if row is None:
        return None
    if row["bio"] is None and row["share_preferences"] is None:
        return None
    share_prefs = row.get("share_preferences") or {}
    return ProfileSurvey(
        display_name=row["display_name"],
        bio=row["bio"],
        spiritual_background=row["spiritual_background"],
        denominational_identity=row["denominational_identity"],
        study_focus_topics=row.get("study_focus_topics") or [],
        study_rhythm=row["study_rhythm"],
        guidance_preferences=row.get("guidance_preferences") or [],
        preferred_translations=row.get("preferred_translations") or [],
        prayer_interests=row.get("prayer_interests") or [],
        ai_journal_opt_in=row.get("ai_journal_opt_in", True),
        share_preferences=share_prefs,
    )


@router.post(
    "/users",
    response_model=ProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    payload: RegistrationRequest,
    pg: asyncpg.Connection = Depends(get_pg),
):
    """Register a user and optionally create a profile in a single request."""

    try:
        user_row = await UserRepository.create_user(pg, payload)
    except asyncpg.UniqueViolationError as exc:
        raise HTTPException(status_code=409, detail="email already registered") from exc

    profile_row = None
    if payload.profile:
        profile_row = await UserRepository.upsert_profile(pg, user_row["id"], payload.profile)

    summary = UserSummary(
        user_id=user_row["id"],
        email=user_row["email"],
        display_name=user_row["display_name"],
        role=UserRole(user_row["role"]),
        created_at=user_row["created_at"],
        updated_at=user_row["updated_at"],
    )

    profile = None
    if profile_row:
        profile = _row_to_profile({**profile_row, "display_name": summary.display_name})

    return ProfileResponse(user=summary, profile=profile)


@router.put("/users/{user_id}/profile", response_model=ProfileResponse)
async def upsert_profile(
    user_id: UUID,
    payload: ProfileSurvey,
    request: Request,
    pg: asyncpg.Connection = Depends(get_pg),
):
    """Create or update onboarding survey responses for a user."""

    viewer = _viewer_from_request(request)
    if not (viewer.is_self(user_id) or viewer.is_admin):
        raise HTTPException(status_code=403, detail="not allowed to modify profile")

    row = await UserRepository.upsert_profile(pg, user_id, payload)
    user_row = await UserRepository.fetch_profile(pg, user_id)
    if not user_row:
        raise HTTPException(status_code=404, detail="user not found")

    profile = _row_to_profile({**row, "display_name": user_row["display_name"]})
    summary = _record_to_summary(user_row)
    return ProfileResponse(user=summary, profile=profile)


@router.patch("/users/{user_id}/profile/share", response_model=ProfileResponse)
async def update_share_settings(
    user_id: UUID,
    payload: SharePreferenceUpdate,
    request: Request,
    pg: asyncpg.Connection = Depends(get_pg),
):
    viewer = _viewer_from_request(request)
    if not (viewer.is_self(user_id) or viewer.is_admin):
        raise HTTPException(status_code=403, detail="not allowed to change visibility")

    row = await UserRepository.update_share_preferences(pg, user_id, payload)
    user_row = await UserRepository.fetch_profile(pg, user_id)
    if not user_row:
        raise HTTPException(status_code=404, detail="user not found")

    profile = _row_to_profile({**row, "display_name": user_row["display_name"]})
    summary = _record_to_summary(user_row)
    return ProfileResponse(user=summary, profile=profile)


@router.get("/users/{user_id}/profile", response_model=ProfileResponse)
async def fetch_profile(
    user_id: UUID,
    request: Request,
    viewer_id: UUID | None = Query(default=None),
    viewer_role: str | None = Query(default=None),
    pg: asyncpg.Connection = Depends(get_pg),
):
    """Retrieve a profile filtered according to visibility preferences."""

    row = await UserRepository.fetch_profile(pg, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="user not found")

    viewer = _viewer_from_request(request, fallback_id=viewer_id, fallback_role=viewer_role)

    profile = _row_to_profile(row)
    share_prefs = profile.share_preferences if profile else {}

    filtered_profile, hidden = filter_profile_fields(
        owner_id=row["user_id"],
        profile=profile,
        share_prefs=share_prefs,
        viewer=viewer,
    )

    summary = _record_to_summary(row)
    return ProfileResponse(user=summary, profile=filtered_profile, hidden_fields=hidden)


@router.post("/users/{user_id}/conversations", response_model=ConversationSummary)
async def create_conversation(
    user_id: UUID,
    payload: ConversationCreate,
    request: Request,
    pg: asyncpg.Connection = Depends(get_pg),
):
    viewer = _viewer_from_request(request)
    if not (viewer.is_self(user_id) or viewer.is_admin):
        raise HTTPException(status_code=403, detail="not allowed to create conversations")

    row = await ConversationRepository.create_conversation(pg, user_id, payload)
    return ConversationSummary(**row)


@router.get("/users/{user_id}/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    user_id: UUID,
    request: Request,
    pg: asyncpg.Connection = Depends(get_pg),
):
    viewer = _viewer_from_request(request)
    if not (viewer.is_self(user_id) or viewer.is_admin):
        raise HTTPException(status_code=403, detail="not allowed to view conversations")

    rows = await ConversationRepository.list_conversations(pg, user_id)
    return [ConversationSummary(**row) for row in rows]


@router.get("/ai/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID,
    request: Request,
    pg: asyncpg.Connection = Depends(get_pg),
):
    convo = await ConversationRepository.fetch_conversation(pg, conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="conversation not found")

    viewer = _viewer_from_request(request)
    owner_id = convo["user_id"]
    if not (viewer.is_self(owner_id) or viewer.is_admin):
        raise HTTPException(status_code=403, detail="not allowed to view conversation")

    messages = await ConversationRepository.list_messages(pg, conversation_id)
    summary = ConversationSummary(**convo)
    message_models = [MessageRecord(**row) for row in messages]
    return ConversationDetail(conversation=summary, messages=message_models)


@router.post(
    "/ai/conversations/{conversation_id}/messages",
    response_model=MessageRecord,
    status_code=status.HTTP_201_CREATED,
)
async def append_message(
    conversation_id: UUID,
    payload: MessageCreate,
    request: Request,
    pg: asyncpg.Connection = Depends(get_pg),
):
    convo = await ConversationRepository.fetch_conversation(pg, conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="conversation not found")

    viewer = _viewer_from_request(request)
    owner_id = convo["user_id"]
    if not (viewer.is_self(owner_id) or viewer.is_admin):
        raise HTTPException(status_code=403, detail="not allowed to modify conversation")

    row = await ConversationRepository.append_message(pg, conversation_id, payload)
    return MessageRecord(**row)


def _viewer_from_request(
    request: Request,
    fallback_id: UUID | None = None,
    fallback_role: str | None = None,
) -> Viewer:
    """Extract viewer context from JWT middleware or query params."""

    user_info = getattr(request.state, "user", None)
    if user_info:
        sub = user_info.get("sub") or user_info.get("user_id")
        role = user_info.get("role")
        try:
            user_uuid = UUID(sub) if sub else None
        except (TypeError, ValueError):
            user_uuid = None
        return Viewer(user_id=user_uuid, role=role)

    return Viewer(user_id=fallback_id, role=fallback_role)


__all__ = ["router"]
