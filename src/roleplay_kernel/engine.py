from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from .compiler import ContextCompiler, PromptPack
from .models import (
    ChatMessage,
    Completion,
    JsonValue,
    Session,
    StateDelta,
    new_id,
)
from .modules import ModuleActivation, ModuleRegistry
from .providers import ChatProvider
from .utils import ProviderError, parse_json_object
from .validators import (
    DeltaResult,
    Finding,
    PlanResult,
    apply_delta,
    defer_critic_flagged_operations,
    defer_unverified_operations,
    fallback_plan,
    merge_findings,
    normalize_plan,
    parse_critic_findings,
    parse_delta,
    validate_candidate,
)

EngineMode = Literal["fast", "balanced", "strict"]
TurnStatus = Literal["ok", "repaired", "needs_confirmation", "needs_attention"]
_ALLOWED_FINISH_REASONS = {None, "stop", "end_turn", "eos"}


class StaleTurnResultError(RuntimeError):
    pass


class UnknownPendingCommitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EngineConfig:
    mode: EngineMode = "balanced"
    use_json_mode: bool = True
    auto_commit: bool = True
    max_repairs: int = 1
    context_window: int = 32768
    max_output_tokens: int = 900
    max_internal_tokens: int = 900
    plan_temperature: float = 0.2
    render_temperature: float = 0.8
    critic_temperature: float = 0.1
    extract_temperature: float = 0.0
    repair_temperature: float = 0.3

    def __post_init__(self) -> None:
        if self.mode not in {"fast", "balanced", "strict"}:
            raise ValueError("unsupported engine mode")
        if self.max_repairs < 0:
            raise ValueError("max_repairs cannot be negative")
        if self.context_window < 1:
            raise ValueError("context_window must be positive")
        if self.max_output_tokens < 1 or self.max_internal_tokens < 1:
            raise ValueError("token limits must be positive")
        for name, value in (
            ("plan_temperature", self.plan_temperature),
            ("render_temperature", self.render_temperature),
            ("critic_temperature", self.critic_temperature),
            ("extract_temperature", self.extract_temperature),
            ("repair_temperature", self.repair_temperature),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value) or not 0.0 <= value <= 2.0:
                raise ValueError(f"{name} must be between 0 and 2")


@dataclass(frozen=True, slots=True)
class PendingCommit:
    request_id: str
    assistant_turn_id: str
    state_version: int
    operations: StateDelta
    digest: str


@dataclass(frozen=True, slots=True)
class TurnResult:
    request_id: str
    session_id: str
    assistant_turn_id: str
    base_state_version: int
    committed_state_version: int
    text: str
    plan: dict[str, JsonValue]
    active_modules: tuple[str, ...]
    module_versions: dict[str, str]
    state_delta: StateDelta
    applied_operations: StateDelta
    pending_operations: StateDelta
    findings: tuple[Finding, ...]
    repairs: int
    provider_calls: int
    usage: dict[str, int]
    status: TurnStatus
    pending_digest: str = ""

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "assistant_turn_id": self.assistant_turn_id,
            "base_state_version": self.base_state_version,
            "committed_state_version": self.committed_state_version,
            "text": self.text,
            "plan": self.plan,
            "active_modules": list(self.active_modules),
            "module_versions": dict(self.module_versions),
            "state_delta": self.state_delta.to_dict(),
            "applied_operations": self.applied_operations.to_dict(),
            "pending_operations": self.pending_operations.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
            "repairs": self.repairs,
            "provider_calls": self.provider_calls,
            "usage": dict(self.usage),
            "status": self.status,
            "pending_digest": self.pending_digest,
        }


class Engine:
    def __init__(
        self,
        provider: ChatProvider,
        *,
        registry: ModuleRegistry | None = None,
        compiler: ContextCompiler | None = None,
        config: EngineConfig | None = None,
        integrity_key: bytes | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry or ModuleRegistry.default()
        self.compiler = compiler or ContextCompiler()
        self.config = config or EngineConfig()
        if integrity_key is not None and len(integrity_key) < 32:
            raise ValueError("integrity_key must contain at least 32 bytes")
        self._integrity_key = integrity_key or secrets.token_bytes(32)
        output_reserve = max(self.config.max_output_tokens, self.config.max_internal_tokens)
        if self.compiler.token_budget + output_reserve > self.config.context_window:
            raise ValueError("compiler token budget plus output reserve exceeds context window")
        self._pending_commits: dict[tuple[str, str], PendingCommit] = {}

    def clear_pending_for_session(self, session_id: str) -> None:
        for key in tuple(self._pending_commits):
            if key[0] == session_id:
                self._pending_commits.pop(key, None)

    def restore_pending_from_result(
        self,
        session: Session,
        result: dict[str, JsonValue],
    ) -> None:
        with session._lock:
            result_session_id = result.get("session_id")
            request_id = result.get("request_id")
            assistant_turn_id = result.get("assistant_turn_id")
            pending_value = result.get("pending_operations")
            state_version = result.get("committed_state_version")
            if (
                result_session_id != session.id
                or not isinstance(request_id, str)
                or not isinstance(assistant_turn_id, str)
                or not isinstance(pending_value, dict)
                or not isinstance(state_version, int)
                or isinstance(state_version, bool)
            ):
                raise ValueError("persisted pending result is invalid")
            operations = StateDelta.from_dict(pending_value)
            if not operations.operations:
                return
            if state_version != session._state.version:
                raise StaleTurnResultError("persisted pending result is stale")
            digest = self._pending_digest(
                session.id,
                request_id,
                assistant_turn_id,
                operations,
            )
            stored_digest = result.get("pending_digest", "")
            if stored_digest and stored_digest != digest:
                raise ValueError("persisted pending result integrity check failed")
            self._pending_commits[(session.id, assistant_turn_id)] = PendingCommit(
                request_id=request_id,
                assistant_turn_id=assistant_turn_id,
                state_version=state_version,
                operations=operations,
                digest=digest,
            )

    def new_session(
        self,
        *,
        language: str = "ru",
        pov: str = "third_person_limited",
        tense: str = "past",
        setting: str = "",
    ) -> Session:
        return Session.create(language=language, pov=pov, tense=tense, setting=setting)

    def advance(
        self,
        session: Session,
        user_input: str,
        *,
        forced_modules: Iterable[str] = (),
        disabled_modules: Iterable[str] = (),
        external_context: str = "",
        sampling: Mapping[str, JsonValue] | None = None,
    ) -> TurnResult:
        with session._lock:
            return self._advance_locked(
                session,
                user_input,
                forced_modules=forced_modules,
                disabled_modules=disabled_modules,
                external_context=external_context,
                sampling=sampling,
            )

    def _advance_locked(
        self,
        session: Session,
        user_input: str,
        *,
        forced_modules: Iterable[str],
        disabled_modules: Iterable[str],
        external_context: str,
        sampling: Mapping[str, JsonValue] | None,
    ) -> TurnResult:
        content = user_input.strip()
        if not content:
            raise ValueError("user_input must be non-empty")

        request_id = new_id("request")
        base_state_version = session._state.version
        activations = self.registry.activate(
            state=session._state,
            user_input=content,
            forced=forced_modules,
            disabled=disabled_modules,
        )
        completions: list[Completion] = []
        plan, plan_findings = self._plan(
            session,
            content,
            activations,
            completions,
            external_context,
        )

        render_pack = self.compiler.compile_render(
            state=session._state,
            turns=session._turns,
            user_input=content,
            plan=plan,
            activations=activations,
            external_context=external_context,
        )
        candidate = self._complete(
            render_pack,
            temperature=self.config.render_temperature,
            max_tokens=self.config.max_output_tokens,
            json_mode=False,
            completions=completions,
            sampling=sampling,
        ).content.strip()
        if not candidate:
            raise RuntimeError("renderer returned an empty post")

        candidate_findings, delta_result = self._evaluate_candidate(
            session,
            plan,
            candidate,
            activations,
            completions,
            external_context,
        )
        repairs = 0
        if self.config.mode == "strict":
            for _ in range(self.config.max_repairs):
                if not self._repairable(candidate_findings):
                    break
                repaired = self._repair(
                    session,
                    plan,
                    candidate,
                    candidate_findings,
                    activations,
                    completions,
                    external_context,
                    sampling,
                ).strip()
                if not repaired:
                    candidate_findings = merge_findings(
                        candidate_findings,
                        (
                            Finding(
                                severity="hard",
                                code="repair_empty",
                                message="Repair pass returned an empty post",
                                confidence=1.0,
                            ),
                        ),
                    )
                    break
                candidate = repaired
                repairs += 1
                candidate_findings, delta_result = self._evaluate_candidate(
                    session,
                    plan,
                    candidate,
                    activations,
                    completions,
                    external_context,
                )

        all_findings = merge_findings(plan_findings, candidate_findings)
        if session._state.version != base_state_version:
            raise StaleTurnResultError("session state changed while the turn was being generated")

        delta_blocked = any(finding.severity == "hard" for finding in all_findings)
        if delta_blocked:
            applied_delta = StateDelta()
            pending_delta = StateDelta()
        elif self.config.auto_commit:
            applied_delta, pending_delta = apply_delta(session._state, delta_result.delta)
        else:
            applied_delta = StateDelta()
            pending_delta = delta_result.delta

        if pending_delta.operations:
            all_findings = merge_findings(
                all_findings,
                (
                    Finding(
                        severity="warning",
                        code="state_confirmation_required",
                        message="High-impact, low-certainty, or deferred changes need approval",
                        confidence=1.0,
                    ),
                ),
            )

        user_turn = session.append_turn("user", content)
        assistant_turn = session.append_turn("assistant", candidate)
        pending_key = (session.id, assistant_turn.id)
        pending_digest = (
            self._pending_digest(
                session.id,
                request_id,
                assistant_turn.id,
                pending_delta,
            )
            if pending_delta.operations
            else ""
        )
        if pending_delta.operations:
            self._pending_commits[pending_key] = PendingCommit(
                request_id=request_id,
                assistant_turn_id=assistant_turn.id,
                state_version=session._state.version,
                operations=pending_delta,
                digest=pending_digest,
            )
        else:
            self._pending_commits.pop(pending_key, None)

        session.append_ledger_event(
            kind="turn_committed",
            turn_id=assistant_turn.id,
            payload={
                "request_id": request_id,
                "user_turn_id": user_turn.id,
                "from_state_version": base_state_version,
                "to_state_version": session._state.version,
                "active_modules": cast(
                    JsonValue,
                    [activation.to_dict() for activation in activations],
                ),
                "plan": plan,
                "state_delta": delta_result.delta.to_dict(),
                "applied_operations": applied_delta.to_dict(),
                "pending_operations": pending_delta.to_dict(),
                "pending_digest": pending_digest,
                "findings": [finding.to_dict() for finding in all_findings],
                "repairs": repairs,
            },
        )

        status: TurnStatus
        if any(finding.severity == "hard" for finding in all_findings):
            status = "needs_attention"
        elif pending_delta.operations:
            status = "needs_confirmation"
        elif repairs:
            status = "repaired"
        else:
            status = "ok"

        return TurnResult(
            request_id=request_id,
            session_id=session.id,
            assistant_turn_id=assistant_turn.id,
            base_state_version=base_state_version,
            committed_state_version=session._state.version,
            text=candidate,
            plan=plan,
            active_modules=tuple(activation.definition.id for activation in activations),
            module_versions={
                activation.definition.id: activation.definition.version
                for activation in activations
            },
            state_delta=delta_result.delta,
            applied_operations=applied_delta,
            pending_operations=pending_delta,
            findings=all_findings,
            repairs=repairs,
            provider_calls=len(completions),
            usage=_aggregate_usage(completions),
            status=status,
            pending_digest=pending_digest,
        )

    def commit_pending(self, session: Session, result: TurnResult) -> int:
        with session._lock:
            if result.session_id != session.id:
                raise ValueError("turn result belongs to another session")
            if result.committed_state_version != session._state.version:
                raise StaleTurnResultError("session state changed after this turn was generated")
            return self._commit_pending_by_id_locked(
                session,
                request_id=result.request_id,
                assistant_turn_id=result.assistant_turn_id,
            )

    def commit_pending_by_id(
        self,
        session: Session,
        *,
        request_id: str,
        assistant_turn_id: str,
    ) -> int:
        with session._lock:
            return self._commit_pending_by_id_locked(
                session,
                request_id=request_id,
                assistant_turn_id=assistant_turn_id,
            )

    def discard_pending_by_id(
        self,
        session: Session,
        *,
        request_id: str,
        assistant_turn_id: str,
    ) -> int:
        with session._lock:
            key = (session.id, assistant_turn_id)
            pending = self._pending_commits.get(key)
            if pending is None:
                pending = self._restore_pending(
                    session,
                    request_id=request_id,
                    assistant_turn_id=assistant_turn_id,
                )
            if pending is None:
                raise UnknownPendingCommitError(
                    "no trusted pending commit exists for this identity"
                )
            if pending.request_id != request_id:
                raise UnknownPendingCommitError("turn request identity does not match")
            if pending.state_version != session._state.version:
                raise StaleTurnResultError("pending state commit is stale")
            expected_digest = self._pending_digest(
                session.id,
                request_id,
                assistant_turn_id,
                pending.operations,
            )
            if not hmac.compare_digest(pending.digest, expected_digest):
                raise UnknownPendingCommitError(
                    "pending state delta failed integrity validation"
                )
            self._pending_commits.pop(key, None)
            session.append_ledger_event(
                kind="state_delta_rejected",
                turn_id=assistant_turn_id,
                payload={
                    "request_id": request_id,
                    "from_state_version": session._state.version,
                    "to_state_version": session._state.version,
                    "pending_digest": pending.digest,
                },
            )
            return session._state.version

    def _commit_pending_by_id_locked(
        self,
        session: Session,
        *,
        request_id: str,
        assistant_turn_id: str,
    ) -> int:
        key = (session.id, assistant_turn_id)
        pending = self._pending_commits.get(key)
        if pending is None:
            pending = self._restore_pending(
                session,
                request_id=request_id,
                assistant_turn_id=assistant_turn_id,
            )
        if pending is None:
            raise UnknownPendingCommitError("no trusted pending commit exists for this identity")
        if pending.request_id != request_id:
            raise UnknownPendingCommitError("turn request identity does not match")
        if pending.state_version != session._state.version:
            raise StaleTurnResultError("pending state commit is stale")
        if not any(turn.id == assistant_turn_id for turn in session._turns):
            raise UnknownPendingCommitError("pending turn is not present in the session")
        expected_digest = self._pending_digest(
            session.id,
            request_id,
            assistant_turn_id,
            pending.operations,
        )
        if not hmac.compare_digest(pending.digest, expected_digest):
            raise UnknownPendingCommitError("pending state delta failed integrity validation")

        from_version = session._state.version
        applied, unexpected_pending = apply_delta(
            session._state,
            pending.operations,
            allow_high_impact=True,
        )
        if unexpected_pending.operations:
            raise RuntimeError("explicit state commit left operations pending")
        self._pending_commits.pop(key, None)
        session.append_ledger_event(
            kind="state_delta_confirmed",
            turn_id=assistant_turn_id,
            payload={
                "request_id": request_id,
                "from_state_version": from_version,
                "to_state_version": session._state.version,
                "operations": applied.to_dict(),
            },
        )
        return session._state.version

    def _restore_pending(
        self,
        session: Session,
        *,
        request_id: str,
        assistant_turn_id: str,
    ) -> PendingCommit | None:
        for event in reversed(session._ledger):
            if event.kind != "turn_committed" or event.turn_id != assistant_turn_id:
                continue
            payload_request = event.payload.get("request_id")
            if payload_request != request_id:
                continue
            to_version = event.payload.get("to_state_version")
            if (
                isinstance(to_version, bool)
                or not isinstance(to_version, int)
                or to_version != session._state.version
            ):
                raise StaleTurnResultError("persisted pending state commit is stale")
            raw_pending = event.payload.get("pending_operations")
            if not isinstance(raw_pending, dict):
                raise UnknownPendingCommitError("persisted pending delta is malformed")
            try:
                operations = StateDelta.from_dict(raw_pending)
            except ValueError as error:
                raise UnknownPendingCommitError(
                    f"persisted pending delta is invalid: {error}"
                ) from error
            if not operations.operations:
                return None
            raw_digest = event.payload.get("pending_digest")
            if not isinstance(raw_digest, str) or not raw_digest:
                raise UnknownPendingCommitError("persisted pending delta has no integrity digest")
            expected_digest = self._pending_digest(
                session.id,
                request_id,
                assistant_turn_id,
                operations,
            )
            if not hmac.compare_digest(raw_digest, expected_digest):
                raise UnknownPendingCommitError(
                    "persisted pending delta failed integrity validation"
                )
            pending = PendingCommit(
                request_id=request_id,
                assistant_turn_id=assistant_turn_id,
                state_version=session._state.version,
                operations=operations,
                digest=raw_digest,
            )
            self._pending_commits[(session.id, assistant_turn_id)] = pending
            return pending
        return None

    def _evaluate_candidate(
        self,
        session: Session,
        plan: dict[str, JsonValue],
        candidate: str,
        activations: tuple[ModuleActivation, ...],
        completions: list[Completion],
        external_context: str,
    ) -> tuple[tuple[Finding, ...], DeltaResult]:
        deterministic = validate_candidate(
            candidate=candidate,
            previous_assistant_turns=(
                turn.content for turn in session._turns if turn.role == "assistant"
            ),
            activations=activations,
        )
        delta_result = self._extract(
            session,
            plan,
            candidate,
            completions,
            external_context,
        )
        if self.config.mode == "fast":
            delta_result = defer_unverified_operations(
                delta_result,
                reason="Fast mode has no independent semantic state verifier",
            )
        model_findings: tuple[Finding, ...] = ()
        if self.config.mode in {"balanced", "strict"}:
            model_findings = self._critic(
                session,
                plan,
                candidate,
                delta_result.delta,
                deterministic,
                activations,
                completions,
                external_context,
            )
            delta_result = defer_critic_flagged_operations(delta_result, model_findings)
        return (
            merge_findings(deterministic, delta_result.findings, model_findings),
            delta_result,
        )

    def _plan(
        self,
        session: Session,
        user_input: str,
        activations: tuple[ModuleActivation, ...],
        completions: list[Completion],
        external_context: str,
    ) -> tuple[dict[str, JsonValue], tuple[Finding, ...]]:
        if self.config.mode == "fast":
            return fallback_plan(session._state, user_input), ()
        prompt = self.compiler.compile_plan(
            state=session._state,
            turns=session._turns,
            user_input=user_input,
            activations=activations,
            external_context=external_context,
        )
        completion = self._complete(
            prompt,
            temperature=self.config.plan_temperature,
            max_tokens=self.config.max_internal_tokens,
            json_mode=True,
            completions=completions,
        )
        try:
            raw = parse_json_object(completion.content)
        except ValueError as error:
            return fallback_plan(session._state, user_input), (
                Finding(
                    severity="hard",
                    code="planner_parse_error",
                    message=f"Planner output was invalid; fallback was used: {error}",
                    confidence=1.0,
                ),
            )
        result: PlanResult = normalize_plan(raw, state=session._state, user_input=user_input)
        return result.plan, result.findings

    def _extract(
        self,
        session: Session,
        plan: dict[str, JsonValue],
        candidate: str,
        completions: list[Completion],
        external_context: str,
    ) -> DeltaResult:
        prompt = self.compiler.compile_extract(
            state=session._state,
            plan=plan,
            candidate=candidate,
            external_context=external_context,
        )
        completion = self._complete(
            prompt,
            temperature=self.config.extract_temperature,
            max_tokens=self.config.max_internal_tokens,
            json_mode=True,
            completions=completions,
        )
        try:
            raw = parse_json_object(completion.content)
        except ValueError as error:
            return DeltaResult(
                delta=StateDelta(),
                findings=(
                    Finding(
                        severity="hard",
                        code="delta_parse_error",
                        message=f"State extraction failed: {error}",
                        confidence=1.0,
                    ),
                ),
            )
        return parse_delta(raw, candidate=candidate)

    def _critic(
        self,
        session: Session,
        plan: dict[str, JsonValue],
        candidate: str,
        delta: StateDelta,
        deterministic: tuple[Finding, ...],
        activations: tuple[ModuleActivation, ...],
        completions: list[Completion],
        external_context: str,
    ) -> tuple[Finding, ...]:
        prompt = self.compiler.compile_critic(
            state=session._state,
            plan=plan,
            candidate=candidate,
            delta=delta,
            deterministic_codes=tuple(finding.code for finding in deterministic),
            activations=activations,
            external_context=external_context,
        )
        completion = self._complete(
            prompt,
            temperature=self.config.critic_temperature,
            max_tokens=self.config.max_internal_tokens,
            json_mode=True,
            completions=completions,
        )
        try:
            raw = parse_json_object(completion.content)
        except ValueError as error:
            return (
                Finding(
                    severity="hard",
                    code="critic_parse_error",
                    message=f"Critic output was invalid: {error}",
                    confidence=1.0,
                ),
            )
        return parse_critic_findings(
            raw,
            candidate=candidate,
            operation_count=len(delta.operations),
        )

    def _repair(
        self,
        session: Session,
        plan: dict[str, JsonValue],
        candidate: str,
        findings: tuple[Finding, ...],
        activations: tuple[ModuleActivation, ...],
        completions: list[Completion],
        external_context: str,
        sampling: Mapping[str, JsonValue] | None,
    ) -> str:
        prompt = self.compiler.compile_repair(
            state=session._state,
            plan=plan,
            candidate=candidate,
            findings=[finding.to_dict() for finding in findings],
            activations=activations,
            external_context=external_context,
        )
        return self._complete(
            prompt,
            temperature=self.config.repair_temperature,
            max_tokens=self.config.max_output_tokens,
            json_mode=False,
            completions=completions,
            sampling=sampling,
        ).content

    def _complete(
        self,
        prompt: PromptPack,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        completions: list[Completion],
        sampling: Mapping[str, JsonValue] | None = None,
    ) -> Completion:
        completion = self.provider.complete(
            (
                ChatMessage(role="system", content=prompt.system),
                ChatMessage(role="user", content=prompt.user),
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode and self.config.use_json_mode,
            sampling=sampling,
        )
        if completion.finish_reason not in _ALLOWED_FINISH_REASONS:
            raise ProviderError(
                f"provider stopped with unsupported finish_reason={completion.finish_reason}"
            )
        completions.append(completion)
        return completion

    @staticmethod
    def _repairable(findings: tuple[Finding, ...]) -> bool:
        return any(
            finding.severity == "hard"
            and finding.confidence >= 0.8
            and finding.code
            not in {
                "invalid_delta",
                "invalid_operation",
                "rejected_operation",
                "unsupported_operation",
                "invalid_critic",
                "delta_parse_error",
                "critic_parse_error",
                "planner_parse_error",
                "planner_fallback",
                "repair_empty",
            }
            for finding in findings
        )
    def _pending_digest(
        self,
        session_id: str,
        request_id: str,
        assistant_turn_id: str,
        operations: StateDelta,
    ) -> str:
        payload = {
            "session_id": session_id,
            "request_id": request_id,
            "assistant_turn_id": assistant_turn_id,
            "operations": operations.to_dict(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(self._integrity_key, encoded, hashlib.sha256).hexdigest()


def _aggregate_usage(completions: Iterable[Completion]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for completion in completions:
        for key, value in completion.usage.items():
            usage[key] = usage.get(key, 0) + value
    return usage
