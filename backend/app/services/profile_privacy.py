"""Helpers to enforce profile visibility rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from ..schemas.users import ProfileSurvey, SharePreference, ShareScope

PROFILE_FIELDS: dict[str, dict[str, object]] = {
    "bio": {"basic": True, "empty": None},
    "spiritual_background": {"basic": True, "empty": None},
    "denominational_identity": {"basic": True, "empty": None},
    "study_focus_topics": {"basic": False, "empty": []},
    "study_rhythm": {"basic": False, "empty": None},
    "guidance_preferences": {"basic": False, "empty": []},
    "preferred_translations": {"basic": True, "empty": []},
    "prayer_interests": {"basic": False, "empty": []},
    "ai_journal_opt_in": {"basic": False, "empty": False},
}


@dataclass
class Viewer:
    user_id: UUID | None
    role: str | None

    @property
    def is_admin(self) -> bool:
        return (self.role or "").lower() == "admin"

    def is_self(self, owner_id: UUID) -> bool:
        return self.user_id is not None and self.user_id == owner_id


def merge_share_preferences(
    stored: dict[str, SharePreference] | None,
) -> dict[str, SharePreference]:
    """Ensure every known profile field has a share preference."""

    stored = stored or {}
    merged: dict[str, SharePreference] = {}
    for field in PROFILE_FIELDS.keys():
        merged[field] = stored.get(field, SharePreference())
    return merged


def filter_profile_fields(
    owner_id: UUID,
    profile: ProfileSurvey | None,
    share_prefs: dict[str, SharePreference] | None,
    viewer: Viewer,
) -> tuple[ProfileSurvey | None, list[str]]:
    """Filter profile data based on share preferences and viewer context."""

    if profile is None:
        return None, []

    prefs = merge_share_preferences(share_prefs)
    hidden: list[str] = []

    profile_data = profile.model_dump()

    for field_name, options in PROFILE_FIELDS.items():
        if field_name not in profile_data:
            continue

        share_pref = prefs[field_name]

        if viewer.is_self(owner_id) or viewer.is_admin:
            continue

        if share_pref.scope == ShareScope.PUBLIC:
            continue

        if share_pref.scope == ShareScope.CUSTOM:
            allowed: Iterable[UUID] = share_pref.allowed_user_ids
            if viewer.user_id and viewer.user_id in allowed:
                continue
            profile_data[field_name] = options.get("empty")
            hidden.append(field_name)
            continue

        # Private by default
        profile_data[field_name] = options.get("empty")
        hidden.append(field_name)

    filtered = ProfileSurvey(**profile_data)
    return filtered, hidden


__all__ = ["Viewer", "filter_profile_fields", "merge_share_preferences", "PROFILE_FIELDS"]
