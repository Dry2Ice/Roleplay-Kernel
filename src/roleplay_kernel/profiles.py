from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers import ChatProvider


class Stage(StrEnum):
    PLAN = "plan"
    RENDER = "render"
    EXTRACT = "extract"
    CRITIC = "critic"
    REPAIR = "repair"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    provider: ChatProvider
    max_output_tokens: int | None = None
    temperature: float | None = None


class ProfileRouter:
    def __init__(
        self,
        default: ChatProvider,
        profiles: Mapping[str, ModelProfile] | None = None,
        stage_map: Mapping[str, str] | None = None,
    ) -> None:
        self._default = default
        self._profiles: dict[str, ModelProfile] = dict(profiles or {})
        self._stage_map: dict[str, str] = dict(stage_map or {})
        self._override: ChatProvider | None = None

    def override(self, provider: ChatProvider) -> None:
        self._override = provider

    def clear_override(self) -> None:
        self._override = None

    def provider_for(self, stage: Stage | str) -> ChatProvider:
        if self._override is not None:
            return self._override
        key = stage.value if isinstance(stage, Stage) else stage
        profile_name = self._stage_map.get(key)
        if profile_name is None:
            return self._default
        profile = self._profiles.get(profile_name)
        return profile.provider if profile is not None else self._default

    def profile_for(self, stage: Stage | str) -> ModelProfile | None:
        key = stage.value if isinstance(stage, Stage) else stage
        profile_name = self._stage_map.get(key)
        if profile_name is None:
            return None
        return self._profiles.get(profile_name)

    def describe(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for stage in Stage:
            profile = self.profile_for(stage)
            result[stage.value] = profile.name if profile else "default"
        return result


class SingleProfileRouter(ProfileRouter):
    def __init__(self, provider: ChatProvider) -> None:
        super().__init__(provider)


@dataclass(frozen=True, slots=True)
class RouterConfig:
    default_provider: ChatProvider
    profiles: dict[str, ModelProfile] = field(default_factory=dict)
    stage_map: dict[str, str] = field(default_factory=dict)

    def to_router(self) -> ProfileRouter:
        return ProfileRouter(self.default_provider, self.profiles, self.stage_map)
