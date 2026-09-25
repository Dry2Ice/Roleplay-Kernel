from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, cast

from .models import (
    Impact,
    JsonValue,
    RoleplayState,
    StateDelta,
    StateOperation,
)
from .modules import ModuleActivation

_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_LABEL_PATTERN = re.compile(r"^\s*(?:gm|npc|pc|player|assistant|user)\s*:", re.IGNORECASE)
_RUNTIME_LEAK_PATTERNS = (
    re.compile(r"\bstate\s+delta\b", re.IGNORECASE),
    re.compile(r"\bapproved\s+plan\b", re.IGNORECASE),
    re.compile(r"\bactive\s+modules?\b", re.IGNORECASE),
    re.compile(r"^\s*[\[{].*[\]}]\s*$", re.DOTALL),
)

_INHERENT_IMPACT: dict[str, Impact] = {
    "set_time": "low",
    "set_location": "low",
    "set_summary": "medium",
    "set_scene_tag": "low",
    "upsert_fact": "medium",
    "upsert_belief": "medium",
    "set_relationship": "high",
    "set_resource": "medium",
    "set_injury": "high",
    "open_thread": "medium",
    "resolve_thread": "high",
    "record_event": "low",
}
_OBJECTIVE_OPERATIONS = {
    "set_time",
    "set_location",
    "upsert_fact",
    "set_relationship",
    "set_resource",
    "set_injury",
    "open_thread",
    "resolve_thread",
}


@dataclass(frozen=True, slots=True)
class Finding:
    severity: str
    code: str
    message: str
    evidence: str = ""
    rule: str = ""
    confidence: float = 1.0
    source: str = "deterministic"
    operation_index: int | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "evidence": self.evidence,
            "rule": self.rule,
            "confidence": self.confidence,
            "source": self.source,
            "operation_index": self.operation_index,
        }


@dataclass(frozen=True, slots=True)
class PlanResult:
    plan: dict[str, JsonValue]
    findings: tuple[Finding, ...]


@dataclass(frozen=True, slots=True)
class DeltaResult:
    delta: StateDelta
    findings: tuple[Finding, ...]


def fallback_plan(state: RoleplayState, user_input: str) -> dict[str, JsonValue]:
    return {
        "goal": user_input.strip(),
        "pov": state.pov,
        "must_fact_ids": [],
        "must_events": [],
        "information_release": ["Only information observable by the current viewpoint"],
        "allowed_inventions": ["Bounded environmental details and autonomous NPC reactions"],
        "beats": [
            {
                "action": user_input.strip(),
                "reaction": "The world responds causally without deciding the player's inner state",
                "causality": "Direct response to the player's stated attempt",
                "sensory_focus": "One detail that clarifies the immediate consequence",
                "state_effect": "Only changes directly supported by the response",
            }
        ],
        "prohibited_moves": [
            "Do not decide the player's unspoken thoughts, dialogue, or final actions",
            "Do not contradict committed state",
        ],
        "style_mode": "Concrete, restrained, consequence-focused prose",
        "novelty_requirement": "Do not reuse the previous post's sentence pattern",
        "target_state_change": [],
        "uncertainty": ["Outcomes not established by the scene remain unresolved"],
    }


def normalize_plan(
    raw: dict[str, Any],
    *,
    state: RoleplayState,
    user_input: str,
) -> PlanResult:
    findings: list[Finding] = []
    if not isinstance(raw.get("beats"), list) or not raw["beats"]:
        return PlanResult(
            plan=fallback_plan(state, user_input),
            findings=(
                Finding(
                    severity="hard",
                    code="planner_fallback",
                    message="Planner returned no usable beats; deterministic fallback was used",
                    confidence=1.0,
                ),
            ),
        )

    plan = fallback_plan(state, user_input)
    fallback_goal = user_input.strip() or "Respond to the player's attempt"
    plan["goal"] = _clean_string(raw.get("goal"), fallback_goal)
    plan["pov"] = _clean_string(raw.get("pov"), state.pov)
    for key in (
        "must_events",
        "information_release",
        "allowed_inventions",
        "prohibited_moves",
        "target_state_change",
        "uncertainty",
    ):
        plan[key] = cast(JsonValue, _string_list(raw.get(key)))
    plan["style_mode"] = _clean_string(
        raw.get("style_mode"), "Concrete, restrained, consequence-focused prose"
    )
    plan["novelty_requirement"] = _clean_string(
        raw.get("novelty_requirement"), "Do not reuse the previous post's sentence pattern"
    )

    requested_fact_ids = _string_list(raw.get("must_fact_ids"))
    valid_fact_ids = [fact_id for fact_id in requested_fact_ids if fact_id in state.facts]
    unknown_fact_ids = sorted(set(requested_fact_ids) - set(valid_fact_ids))
    plan["must_fact_ids"] = cast(JsonValue, valid_fact_ids)
    if unknown_fact_ids:
        findings.append(
            Finding(
                severity="warning",
                code="unknown_fact_reference",
                message=f"Planner referenced unknown fact ids: {', '.join(unknown_fact_ids)}",
                confidence=1.0,
            )
        )

    beats: list[JsonValue] = []
    for raw_beat in raw["beats"][:2]:
        if not isinstance(raw_beat, dict):
            continue
        beats.append(
            {
                "action": _clean_string(raw_beat.get("action"), "Respond to the player's attempt"),
                "reaction": _clean_string(
                    raw_beat.get("reaction"), "Produce an observable causal consequence"
                ),
                "causality": _clean_string(raw_beat.get("causality"), "Direct causal response"),
                "sensory_focus": _clean_string(
                    raw_beat.get("sensory_focus"), "One relevant detail"
                ),
                "state_effect": _clean_string(
                    raw_beat.get("state_effect"), "No unsupported state change"
                ),
            }
        )
    if not beats:
        fallback_beats = fallback_plan(state, user_input)["beats"]
        beats = fallback_beats if isinstance(fallback_beats, list) else []
        findings.append(
            Finding(
                severity="hard",
                code="planner_fallback",
                message="Planner beats were malformed; deterministic fallback was used",
                confidence=1.0,
            )
        )
    plan["beats"] = beats
    return PlanResult(plan=plan, findings=tuple(findings))


def parse_delta(raw: dict[str, Any], *, candidate: str) -> DeltaResult:
    if "operations" not in raw:
        return DeltaResult(
            delta=StateDelta(),
            findings=(
                Finding(
                    severity="hard",
                    code="invalid_delta",
                    message="State extractor omitted the required operations field",
                    confidence=1.0,
                ),
            ),
        )
    operations_value = raw.get("operations")
    if not isinstance(operations_value, list):
        return DeltaResult(
            delta=StateDelta(),
            findings=(
                Finding(
                    severity="hard",
                    code="invalid_delta",
                    message="State extractor returned a non-array operations field",
                    confidence=1.0,
                ),
            ),
        )

    operations: list[StateOperation] = []
    findings: list[Finding] = []
    for index, raw_operation in enumerate(operations_value):
        if not isinstance(raw_operation, dict):
            findings.append(
                Finding(
                    severity="hard",
                    code="invalid_operation",
                    message=f"State operation {index} is not an object",
                    confidence=1.0,
                    operation_index=index,
                )
            )
            continue
        try:
            operation = StateOperation.from_dict(
                {key: value for key, value in raw_operation.items() if _is_json_value(value)}
            )
        except ValueError as error:
            findings.append(
                Finding(
                    severity="hard",
                    code="rejected_operation",
                    message=f"State operation {index} was rejected: {error}",
                    confidence=1.0,
                    operation_index=index,
                )
            )
            continue
        evidence = operation.evidence.strip()
        if not evidence or evidence.casefold() not in candidate.casefold():
            findings.append(
                Finding(
                    severity="hard",
                    code="unsupported_operation",
                    message=f"State operation {index} had no exact supporting evidence",
                    evidence=evidence,
                    confidence=1.0,
                    operation_index=index,
                )
            )
            continue
        normalized = _normalize_operation(operation)
        if normalized is None:
            findings.append(
                Finding(
                    severity="hard",
                    code="rejected_operation",
                    message=f"State operation {index} is structurally invalid",
                    confidence=1.0,
                    operation_index=index,
                )
            )
            continue
        if operation.kind in _OBJECTIVE_OPERATIONS and operation.certainty < 0.8:
            normalized = replace(normalized, impact="high")
            findings.append(
                Finding(
                    severity="warning",
                    code="low_certainty_state_change",
                    message=f"State operation {index} requires explicit confirmation",
                    evidence=evidence,
                    confidence=operation.certainty,
                    operation_index=index,
                )
            )
        operations.append(normalized)
    return DeltaResult(delta=StateDelta(tuple(operations)), findings=tuple(findings))


def defer_critic_flagged_operations(
    result: DeltaResult,
    findings: tuple[Finding, ...],
) -> DeltaResult:
    flagged = {
        finding.operation_index
        for finding in findings
        if finding.code == "delta_semantics"
        and finding.operation_index is not None
    }
    if not flagged:
        return result
    operations = tuple(
        replace(operation, impact="high")
        if index in flagged
        else operation
        for index, operation in enumerate(result.delta.operations)
    )
    return DeltaResult(delta=StateDelta(operations), findings=result.findings)


def defer_unverified_operations(
    result: DeltaResult,
    *,
    reason: str,
) -> DeltaResult:
    if not result.delta.operations:
        return result
    operations = tuple(replace(operation, impact="high") for operation in result.delta.operations)
    finding = Finding(
        severity="warning",
        code="unverified_state_change",
        message=reason,
        confidence=1.0,
    )
    return DeltaResult(
        delta=StateDelta(operations),
        findings=merge_findings(result.findings, (finding,)),
    )


def parse_critic_findings(
    raw: dict[str, Any],
    *,
    candidate: str,
    operation_count: int = 0,
) -> tuple[Finding, ...]:
    if operation_count < 0:
        raise ValueError("operation_count cannot be negative")
    if "findings" not in raw:
        return (
            Finding(
                severity="hard",
                code="invalid_critic",
                message="Critic omitted the required findings field",
                confidence=1.0,
            ),
        )
    values = raw.get("findings")
    if not isinstance(values, list):
        return (
            Finding(
                severity="hard",
                code="invalid_critic",
                message="Critic returned a non-array findings field",
                confidence=1.0,
            ),
        )
    findings: list[Finding] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            findings.append(
                Finding(
                    severity="hard",
                    code="invalid_critic",
                    message=f"Critic finding {index} is not an object",
                    confidence=1.0,
                )
            )
            continue
        evidence = _clean_string(value.get("evidence"), "")
        if not evidence or evidence.casefold() not in candidate.casefold():
            findings.append(
                Finding(
                    severity="hard",
                    code="invalid_critic",
                    message=f"Critic finding {index} has no exact candidate evidence",
                    evidence=evidence,
                    confidence=1.0,
                )
            )
            continue
        severity = "hard" if value.get("severity") == "hard" else "warning"
        confidence = _bounded_confidence(value.get("confidence"))
        operation_index = _optional_index(value.get("operation_index"))
        code = _clean_string(value.get("code"), "critic_finding")
        if code == "delta_semantics" and (
            operation_index is None or operation_index >= operation_count
        ):
            findings.append(
                Finding(
                    severity="hard",
                    code="invalid_critic_operation_reference",
                    message="Critic delta_semantics finding has no valid operation_index",
                    evidence=evidence,
                    rule=code,
                    confidence=confidence,
                    source="model_critic",
                    operation_index=operation_index,
                )
            )
            continue
        if severity == "hard" and confidence < 0.8:
            severity = "warning"
        findings.append(
            Finding(
                severity=severity,
                code=code,
                message=_clean_string(value.get("message"), "Potential quality issue"),
                evidence=evidence,
                rule=_clean_string(value.get("rule"), ""),
                confidence=confidence,
                source="model_critic",
                operation_index=operation_index,
            )
        )
    return tuple(findings)


def validate_candidate(
    *,
    candidate: str,
    previous_assistant_turns: Iterable[str],
    activations: tuple[ModuleActivation, ...],
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    text = candidate.strip()
    if not text:
        return (
            Finding(
                severity="hard",
                code="empty_candidate",
                message="Renderer returned an empty post",
                confidence=1.0,
            ),
        )

    label_match = _LABEL_PATTERN.search(text)
    if label_match is not None:
        findings.append(
            Finding(
                severity="hard",
                code="speaker_label_leak",
                message="Generated post contains a technical speaker label",
                evidence=label_match.group(0),
                rule="Output only diegetic story content",
                confidence=1.0,
            )
        )

    for pattern in _RUNTIME_LEAK_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            findings.append(
                Finding(
                    severity="hard",
                    code="runtime_metadata_leak",
                    message="Generated post exposes runtime metadata",
                    evidence=match.group(0)[:200],
                    rule="Do not expose plans, modules, state, or structured output",
                    confidence=0.95,
                )
            )

    avoid_terms = {
        term.casefold()
        for activation in activations
        for term in activation.definition.avoid_terms
        if term.strip()
    }
    for term in sorted(avoid_terms):
        index = text.casefold().find(term)
        if index >= 0:
            findings.append(
                Finding(
                    severity="hard",
                    code="forbidden_term",
                    message=f"Generated post contains forbidden expression {term!r}",
                    evidence=text[index : index + len(term)],
                    rule="Active module lexical restriction",
                    confidence=1.0,
                )
            )

    repeated = _repeated_ngrams(text, tuple(previous_assistant_turns), size=5)
    if repeated:
        findings.append(
            Finding(
                severity="warning",
                code="repetitive_ngrams",
                message=f"Candidate repeats {len(repeated)} recent five-word sequence(s)",
                evidence=" ".join(repeated[:3]),
                rule="Avoid automatic lexical recurrence",
                confidence=min(1.0, 0.6 + len(repeated) * 0.1),
            )
        )
    return tuple(findings)


def merge_findings(*groups: Iterable[Finding]) -> tuple[Finding, ...]:
    seen: set[tuple[str, str, str, int | None]] = set()
    result: list[Finding] = []
    for group in groups:
        for finding in group:
            key = (
                finding.severity,
                finding.code,
                finding.evidence.casefold(),
                finding.operation_index,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(finding)
    return tuple(result)


def apply_delta(
    state: RoleplayState,
    delta: StateDelta,
    *,
    allow_high_impact: bool = False,
) -> tuple[StateDelta, StateDelta]:
    applied: list[StateOperation] = []
    pending: list[StateOperation] = []
    for operation in delta.operations:
        normalized = _normalize_operation(operation)
        if normalized is None:
            continue
        if normalized.impact == "high" and not allow_high_impact:
            pending.append(normalized)
            continue
        _apply_operation(state, normalized)
        applied.append(normalized)
    if applied:
        state.version += 1
    return StateDelta(tuple(applied)), StateDelta(tuple(pending))


def _apply_operation(state: RoleplayState, operation: StateOperation) -> None:
    if operation.kind == "set_time":
        state.time = operation.value
    elif operation.kind == "set_location":
        state.location = operation.value
    elif operation.kind == "set_summary":
        state.summary = operation.value
    elif operation.kind == "set_scene_tag":
        state.set_scene_tag(operation)
    elif operation.kind == "upsert_fact":
        state.facts[operation.target] = operation.value
    elif operation.kind == "upsert_belief":
        state.upsert_belief(operation)
    elif operation.kind == "set_relationship":
        state.relationships[operation.target] = operation.value
    elif operation.kind == "set_resource":
        state.resources[operation.target] = operation.value
    elif operation.kind == "set_injury":
        state.injuries[operation.target] = operation.value
    elif operation.kind == "open_thread":
        state.open_threads[operation.target] = operation.value
    elif operation.kind == "resolve_thread":
        if operation.target not in state.resolved_threads:
            state.resolved_threads.append(operation.target)
        state.open_threads.pop(operation.target, None)
    elif operation.kind == "record_event" and operation.value not in state.events:
        state.events.append(operation.value)


def _normalize_operation(operation: StateOperation) -> StateOperation | None:
    if not _is_valid_operation(operation):
        return None
    if operation.kind not in _INHERENT_IMPACT:
        return None
    if not math.isfinite(operation.certainty) or not 0.0 <= operation.certainty <= 1.0:
        return None
    impact = _max_impact(_INHERENT_IMPACT[operation.kind], operation.impact)
    return replace(
        operation,
        target=operation.target.strip(),
        value=operation.value.strip(),
        impact=impact,
        evidence=operation.evidence.strip(),
    )


def _is_valid_operation(operation: StateOperation) -> bool:
    if not operation.value:
        return False
    if operation.kind in {
        "upsert_fact",
        "upsert_belief",
        "set_relationship",
        "set_resource",
        "set_injury",
        "open_thread",
        "resolve_thread",
    }:
        return bool(operation.target)
    return True


def _repeated_ngrams(candidate: str, previous: tuple[str, ...], *, size: int) -> list[str]:
    candidate_ngrams = _ngrams(candidate, size)
    if not candidate_ngrams:
        return []
    previous_ngrams = {ngram for text in previous for ngram in _ngrams(text, size)}
    return sorted(candidate_ngrams.intersection(previous_ngrams))


def _ngrams(text: str, size: int) -> set[str]:
    words = [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _clean_string(value: object, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _bounded_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.5
    result = float(value)
    if not math.isfinite(result):
        return 0.0
    return min(1.0, max(0.0, result))


def _optional_index(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _impact_rank(impact: Impact) -> int:
    return {"low": 0, "medium": 1, "high": 2}[impact]


def _max_impact(left: Impact, right: Impact) -> Impact:
    return left if _impact_rank(left) >= _impact_rank(right) else right


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False
