from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .models import RoleplayState


class ModuleDefinitionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModuleDefinition:
    id: str
    version: str
    category: str
    instructions: tuple[str, ...]
    hard_rules: tuple[str, ...] = ()
    avoid_terms: tuple[str, ...] = ()
    trigger_terms: tuple[str, ...] = ()
    state_tags: tuple[str, ...] = ()
    active_by_default: bool = False
    requires: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.version.strip():
            raise ModuleDefinitionError("module id and version must be non-empty")
        if not self.instructions:
            raise ModuleDefinitionError(f"module {self.id!r} must contain instructions")
        if self.id in self.requires:
            raise ModuleDefinitionError(f"module {self.id!r} cannot require itself")
        if self.id in self.conflicts_with:
            raise ModuleDefinitionError(f"module {self.id!r} cannot conflict with itself")


@dataclass(frozen=True, slots=True)
class ModuleActivation:
    definition: ModuleDefinition
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.definition.id,
            "version": self.definition.version,
            "category": self.definition.category,
            "reason": self.reason,
            "priority": str(self.definition.priority),
        }


class ModuleRegistry:
    def __init__(self, modules: Iterable[ModuleDefinition] = ()) -> None:
        self._modules: dict[str, ModuleDefinition] = {}
        for module in modules:
            self.register(module)

    @classmethod
    def default(cls) -> ModuleRegistry:
        return cls(_default_modules())

    def register(self, module: ModuleDefinition) -> None:
        normalized = module.id.strip()
        if normalized != module.id:
            raise ModuleDefinitionError("module id must not contain surrounding whitespace")
        if module.id in self._modules:
            raise ModuleDefinitionError(f"module {module.id!r} is already registered")
        self._modules[module.id] = module

    def get(self, module_id: str) -> ModuleDefinition:
        try:
            return self._modules[module_id]
        except KeyError as error:
            raise ModuleDefinitionError(f"unknown module {module_id!r}") from error

    def definitions(self) -> tuple[ModuleDefinition, ...]:
        return tuple(self._modules[module_id] for module_id in sorted(self._modules))

    def _validate_relationships(self) -> None:
        known = self._modules.keys()
        for module in self.definitions():
            unknown_requirements = sorted(set(module.requires) - set(known))
            unknown_conflicts = sorted(set(module.conflicts_with) - set(known))
            if unknown_requirements:
                raise ModuleDefinitionError(
                    f"module {module.id!r} has unknown requirements: "
                    f"{', '.join(unknown_requirements)}"
                )
            if unknown_conflicts:
                raise ModuleDefinitionError(
                    f"module {module.id!r} has unknown conflicts: {', '.join(unknown_conflicts)}"
                )
            for conflict in module.conflicts_with:
                if module.id not in self._modules[conflict].conflicts_with:
                    raise ModuleDefinitionError(
                        f"conflict between {module.id!r} and {conflict!r} must be symmetric"
                    )

    def activate(
        self,
        *,
        state: RoleplayState,
        user_input: str,
        forced: Iterable[str] = (),
        disabled: Iterable[str] = (),
    ) -> tuple[ModuleActivation, ...]:
        self._validate_relationships()
        forced_ids = set(forced)
        disabled_ids = set(disabled)
        unknown = (forced_ids | disabled_ids) - self._modules.keys()
        if unknown:
            raise ModuleDefinitionError(f"unknown modules requested: {', '.join(sorted(unknown))}")
        if forced_ids & disabled_ids:
            raise ModuleDefinitionError("a module cannot be forced and disabled at the same time")

        state_tags = {tag.casefold() for tag in state.scene_tags}
        base_reasons: dict[str, str] = {}

        for module in self.definitions():
            if module.id in disabled_ids:
                continue
            if module.id in forced_ids:
                base_reasons[module.id] = "forced"
            elif module.active_by_default:
                base_reasons[module.id] = "default"
            else:
                matched_term = next(
                    (term for term in module.trigger_terms if _contains_term(user_input, term)),
                    None,
                )
                matched_tag = next(
                    (tag for tag in module.state_tags if tag.casefold() in state_tags),
                    None,
                )
                if matched_term is not None:
                    base_reasons[module.id] = f"input:{matched_term}"
                elif matched_tag is not None:
                    base_reasons[module.id] = f"state:{matched_tag}"

        while True:
            reasons = dict(base_reasons)
            pending = list(reasons)
            while pending:
                module_id = pending.pop()
                for dependency in self.get(module_id).requires:
                    if dependency in disabled_ids:
                        raise ModuleDefinitionError(
                            f"module {module_id!r} requires disabled module {dependency!r}"
                        )
                    if dependency not in reasons:
                        reasons[dependency] = f"required_by:{module_id}"
                        pending.append(dependency)

            active_ids = set(reasons)
            for module in sorted(
                (self._modules[module_id] for module_id in active_ids),
                key=lambda item: (-item.priority, item.id),
            ):
                if module.id not in active_ids:
                    continue
                for conflict in sorted(active_ids.intersection(module.conflicts_with)):
                    if conflict not in active_ids:
                        continue
                    if conflict in forced_ids:
                        raise ModuleDefinitionError(
                            f"forced module {conflict!r} conflicts with active module {module.id!r}"
                        )
                    if (
                        module.id not in forced_ids
                        and module.priority == self._modules[conflict].priority
                    ):
                        raise ModuleDefinitionError(
                            f"modules {module.id!r} and {conflict!r} have an unresolved conflict"
                        )
                    active_ids.remove(conflict)
                    reasons.pop(conflict, None)

            removed_invalid = False
            while True:
                missing_requirements = {
                    module_id: set(self._modules[module_id].requires) - active_ids
                    for module_id in active_ids
                    if set(self._modules[module_id].requires) - active_ids
                }
                if not missing_requirements:
                    break
                for module_id, missing in missing_requirements.items():
                    if module_id in forced_ids:
                        raise ModuleDefinitionError(
                            f"forced module {module_id!r} lost required modules: "
                            f"{', '.join(sorted(missing))}"
                        )
                    active_ids.remove(module_id)
                    reasons.pop(module_id, None)
                    removed_invalid = True

            next_base = {
                module_id: reason
                for module_id, reason in base_reasons.items()
                if module_id in active_ids
            }
            if not removed_invalid or next_base == base_reasons:
                break
            base_reasons = next_base

        return tuple(
            ModuleActivation(definition=self._modules[module_id], reason=reasons[module_id])
            for module_id in sorted(
                active_ids,
                key=lambda module_id: (-self._modules[module_id].priority, module_id),
            )
        )


def _contains_term(text: str, term: str) -> bool:
    normalized = term.strip()
    if not normalized:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text, re.IGNORECASE) is not None


def _default_modules() -> tuple[ModuleDefinition, ...]:
    return (
        ModuleDefinition(
            id="core_agency",
            version="1.0.0",
            category="core",
            active_by_default=True,
            priority=100,
            instructions=(
                "Render only the world's response to the player's stated attempt.",
                "Do not decide the player's unspoken thoughts, dialogue, or final actions.",
            ),
            hard_rules=(
                "An attempted action may succeed, fail, or remain unresolved "
                "according to the scene.",
            ),
        ),
        ModuleDefinition(
            id="continuity",
            version="1.0.0",
            category="core",
            active_by_default=True,
            priority=95,
            instructions=(
                (
                    "Treat committed state as authoritative and preserve established names, "
                    "objects, injuries, and locations."
                ),
                "Introduce no contradiction merely to make the current moment more dramatic.",
            ),
            hard_rules=(
                (
                    "Every new causal development must be compatible with the supplied state "
                    "and recent turns."
                ),
            ),
        ),
        ModuleDefinition(
            id="knowledge_fences",
            version="1.0.0",
            category="core",
            active_by_default=True,
            priority=90,
            instructions=(
                "Separate objective facts, character beliefs, rumors, and hypotheses.",
                (
                    "A character may act only on information available through their "
                    "established perspective."
                ),
            ),
            hard_rules=(
                "Never present one character's private belief as objective truth without evidence.",
                "Do not grant NPCs knowledge that has no diegetic source.",
            ),
        ),
        ModuleDefinition(
            id="dialogue",
            version="1.0.0",
            category="core",
            active_by_default=True,
            priority=80,
            instructions=(
                (
                    "Let dialogue arise from the speaker's goal, relationship history, "
                    "and immediate subtext."
                ),
                "Do not mirror the player's phrasing or explain the scene through a recap.",
            ),
        ),
        ModuleDefinition(
            id="anti_repetition",
            version="1.0.0",
            category="quality",
            active_by_default=True,
            priority=70,
            instructions=(
                "Avoid reusing recent sentence patterns, gestures, metaphors, and emotional beats.",
                "If a motif returns, give it a changed function or consequence.",
            ),
        ),
        ModuleDefinition(
            id="combat",
            version="1.0.0",
            category="genre",
            trigger_terms=(
                "бой",
                "атака",
                "удар",
                "меч",
                "огонь",
                "fight",
                "attack",
                "sword",
            ),
            instructions=(
                (
                    "Track distance, positioning, fatigue, weapons, injuries, "
                    "and environmental constraints."
                ),
                "Make outcomes causally legible without revealing hidden enemy intentions.",
            ),
        ),
        ModuleDefinition(
            id="investigation",
            version="1.0.0",
            category="genre",
            trigger_terms=(
                "расследование",
                "след",
                "улика",
                "детектив",
                "mystery",
                "clue",
                "investigate",
            ),
            instructions=(
                (
                    "Preserve evidence provenance and keep hypotheses distinct from "
                    "verified conclusions."
                ),
                "Reward observation with causal information rather than immediate total answers.",
            ),
        ),
        ModuleDefinition(
            id="intimacy",
            version="1.0.0",
            category="genre",
            trigger_terms=(
                "роман",
                "любовь",
                "поцелуй",
                "объятие",
                "romance",
                "love",
                "kiss",
                "embrace",
            ),
            instructions=(
                "Base intimacy on consent, established trust, vulnerability, and consequences.",
                "Use subtext and reciprocal agency rather than automatic escalation.",
            ),
        ),
        ModuleDefinition(
            id="survival",
            version="1.0.0",
            category="genre",
            trigger_terms=(
                "выживание",
                "голод",
                "холод",
                "вода",
                "лагерь",
                "survival",
                "hunger",
                "shelter",
            ),
            instructions=(
                "Track time, weather, supplies, fatigue, and the practical cost of each action.",
                (
                    "Let the environment constrain plans without taking control of "
                    "the player character."
                ),
            ),
        ),
    )
