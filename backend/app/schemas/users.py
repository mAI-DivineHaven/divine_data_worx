"""Pydantic models for user registration, profiles, and AI conversations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class UserRole(str, Enum):
    MEMBER = "member"
    ADMIN = "admin"


class ShareScope(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    CUSTOM = "custom"


class SharePreference(BaseModel):
    """Visibility control for a single profile section."""

    scope: ShareScope = ShareScope.PRIVATE
    allowed_user_ids: list[UUID] = Field(default_factory=list)

    @field_validator("allowed_user_ids")
    @classmethod
    def deduplicate(cls, value: list[UUID]) -> list[UUID]:
        seen = set()
        deduped: list[UUID] = []
        for item in value:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped


class ProfileSurvey(BaseModel):
    """Structured responses gathered during onboarding."""

    display_name: str
    bio: str | None = None
    spiritual_background: str | None = None
    denominational_identity: str | None = None
    study_focus_topics: list[str] = Field(default_factory=list)
    study_rhythm: str | None = None
    guidance_preferences: list[str] = Field(default_factory=list)
    preferred_translations: list[str] = Field(default_factory=list)
    prayer_interests: list[str] = Field(default_factory=list)
    ai_journal_opt_in: bool = True
    share_preferences: dict[str, SharePreference] = Field(default_factory=dict)


class RegistrationRequest(BaseModel):
    """Payload used to register a new account and optional profile."""

    email: str
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=2)
    role: UserRole = UserRole.MEMBER
    profile: ProfileSurvey | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("email must include '@' and local part")
        return value


class UserSummary(BaseModel):
    user_id: UUID
    email: str
    display_name: str
    role: UserRole
    created_at: datetime
    updated_at: datetime


class ProfileResponse(BaseModel):
    """Profile data filtered according to visibility rules."""

    user: UserSummary
    profile: ProfileSurvey | None = None
    hidden_fields: list[str] = Field(default_factory=list)


class SharePreferenceUpdate(BaseModel):
    """Partial update to one or more share settings."""

    preferences: dict[str, SharePreference]


class ConversationCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    metadata: dict | None = None


class ConversationSummary(BaseModel):
    conversation_id: UUID
    user_id: UUID
    title: str
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    sender_role: str = Field(pattern="^(user|assistant|system)$")
    content: str = Field(min_length=1)
    metadata: dict | None = None


class MessageRecord(BaseModel):
    message_id: UUID
    conversation_id: UUID
    sender_role: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    sequence: int


class ConversationDetail(BaseModel):
    conversation: ConversationSummary
    messages: list[MessageRecord]


__all__ = [
    "ConversationCreate",
    "ConversationDetail",
    "ConversationSummary",
    "MessageCreate",
    "MessageRecord",
    "ProfileResponse",
    "ProfileSurvey",
    "RegistrationRequest",
    "SharePreference",
    "SharePreferenceUpdate",
    "ShareScope",
    "UserRole",
    "UserSummary",
]
