from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Literal, TypeAlias, cast
from uuid import uuid4

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]
MessageRole: TypeAlias = Literal["system", "developer", "user", "assistant"]
Impact: TypeAlias = Literal["low", "medium", "high"]
OperationKind: TypeAlias = Literal[
    "set_time",
    "set_location",
    "set_summary",
    "set_scene_tag",
    "upsert_fact",
    "upsert_belief",
    "set_relationship",
    "set_resource",
    "set_injury",
    "open_thread",
    "resolve_thread",
    "record_event",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Belief:
    holder: str
    proposition: str
    stance: str = "believes"
    certainty: float = 0.5
    source: str = ""

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> Belief:
        return cls(
            holder=_required_str(data, "holder"),
            proposition=_required_str(data, "proposition"),
            stance=_optional_str(data, "stance", "believes"),
            certainty=_bounded_float(data.get("certainty"), 0.5),
            source=_optional_str(data, "source", ""),
        )


@dataclass(frozen=True, slots=True)
class StateOperation:
    kind: OperationKind
    value: str
    target: str = ""
    impact: Impact = "low"
    evidence: str = ""
    certainty: float = 1.0

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> StateOperation:
        required = {"kind", "value", "target", "impact", "evidence", "certainty"}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"missing operation fields: {', '.join(missing)}")
        return cls(
            kind=_operation_kind(data.get("kind")),
            value=_required_str(data, "value"),
            target=_required_field(data, "target", allow_empty=True),
            impact=_impact(data.get("impact")),
            evidence=_required_str(data, "evidence"),
            certainty=_required_bounded_float(data, "certainty"),
        )


@dataclass(frozen=True, slots=True)
class StateDelta:
    operations: tuple[StateOperation, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        return {"operations": [operation.to_dict() for operation in self.operations]}

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> StateDelta:
        if "operations" not in data:
            raise ValueError("operations is required")
        raw_operations = data.get("operations")
        if not isinstance(raw_operations, list):
            raise ValueError("operations must be an array")
        operations = tuple(
            StateOperation.from_dict(_object(operation, "operation"))
            for operation in raw_operations
        )
        return cls(operations=operations)


@dataclass(slots=True)
class RoleplayState:
    version: int = 0
    language: str = "ru"
    pov: str = "third_person_limited"
    tense: str = "past"
    time: str = "unspecified"
    location: str = "unspecified"
    summary: str = ""
    facts: dict[str, str] = field(default_factory=lambda: {})
    beliefs: list[Belief] = field(default_factory=lambda: [])
    relationships: dict[str, str] = field(default_factory=lambda: {})
    resources: dict[str, str] = field(default_factory=lambda: {})
    injuries: dict[str, str] = field(default_factory=lambda: {})
    open_threads: dict[str, str] = field(default_factory=lambda: {})
    resolved_threads: list[str] = field(default_factory=lambda: [])
    scene_tags: list[str] = field(default_factory=lambda: [])
    events: list[str] = field(default_factory=lambda: [])
    provenance: dict[str, str] = field(default_factory=lambda: {})
    """Tracks which assistant turn last wrote each state entry.

    Keys are `category:name` (for example `injuries:knee`). When a chat is
    rewritten and the originating turn is superseded, the audit uses this map
    to report entries that no longer rest on visible text.
    """
    elapsed_hint: int = 0
    """Seconds elapsed since the previous turn, refreshed before each prompt."""

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "language": self.language,
            "pov": self.pov,
            "tense": self.tense,
            "time": self.time,
            "location": self.location,
            "summary": self.summary,
            "facts": dict(sorted(self.facts.items())),
            "beliefs": [belief.to_dict() for belief in self.beliefs],
            "relationships": dict(sorted(self.relationships.items())),
            "resources": dict(sorted(self.resources.items())),
            "injuries": dict(sorted(self.injuries.items())),
            "open_threads": dict(sorted(self.open_threads.items())),
            "resolved_threads": list(self.resolved_threads),
            "scene_tags": list(self.scene_tags),
            "events": list(self.events),
            "provenance": dict(sorted(self.provenance.items())),
            "elapsed_hint": self.elapsed_hint,
        }

    def to_prompt_dict(self, *, max_items: int = 80) -> dict[str, JsonValue]:
        limit = max(1, max_items)
        data = self.to_dict()
        # Provenance and elapsed time are internal bookkeeping, not story state.
        data.pop("provenance", None)
        data.pop("elapsed_hint", None)
        data["summary"] = _limit_prompt_text(self.summary, 4000)
        facts_prompt = _tail_dict(self.facts, limit, 1000)
        beliefs_prompt = cast(
            list[JsonValue],
            [_belief_prompt(belief) for belief in self.beliefs[-limit:]],
        )
        resolved_prompt = cast(
            list[JsonValue],
            [_limit_prompt_text(value, 1000) for value in self.resolved_threads[-limit:]],
        )
        events_prompt = cast(list[JsonValue], [value[-1000:] for value in self.events[-limit:]])
        data["facts"] = cast(JsonValue, facts_prompt)
        data["beliefs"] = beliefs_prompt
        data["relationships"] = cast(JsonValue, _tail_dict(self.relationships, limit, 1000))
        data["resources"] = cast(JsonValue, _tail_dict(self.resources, limit, 1000))
        data["injuries"] = cast(JsonValue, _tail_dict(self.injuries, limit, 1000))
        data["open_threads"] = cast(JsonValue, _tail_dict(self.open_threads, limit, 1000))
        data["resolved_threads"] = resolved_prompt
        data["scene_tags"] = cast(list[JsonValue], self.scene_tags[-limit:])
        data["events"] = events_prompt
        data["prompt_compaction"] = {
            "facts_total": len(self.facts),
            "facts_included": len(facts_prompt),
            "beliefs_total": len(self.beliefs),
            "beliefs_included": len(beliefs_prompt),
            "events_total": len(self.events),
            "events_included": len(events_prompt),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> RoleplayState:
        return cls(
            version=_required_int(data, "version", minimum=0),
            language=_optional_str(data, "language", "ru"),
            pov=_optional_str(data, "pov", "third_person_limited"),
            tense=_optional_str(data, "tense", "past"),
            time=_optional_str(data, "time", "unspecified"),
            location=_optional_str(data, "location", "unspecified"),
            summary=_optional_str(data, "summary", ""),
            facts=_string_map(data.get("facts")),
            beliefs=[
                Belief.from_dict(_object(item, "belief"))
                for item in _array(data.get("beliefs"), "beliefs")
            ],
            relationships=_string_map(data.get("relationships")),
            resources=_string_map(data.get("resources")),
            injuries=_string_map(data.get("injuries")),
            open_threads=_string_map(data.get("open_threads")),
            resolved_threads=_string_list(
                _array(data.get("resolved_threads"), "resolved_threads"),
                "resolved_threads",
            ),
            scene_tags=_string_list(
                _array(data.get("scene_tags"), "scene_tags"),
                "scene_tags",
            ),
            events=_string_list(_array(data.get("events"), "events"), "events"),
            provenance=_string_map(data.get("provenance")),
            elapsed_hint=_optional_int(data, "elapsed_hint", 0),
        )

    def upsert_belief(self, operation: StateOperation) -> None:
        self.beliefs = [
            belief
            for belief in self.beliefs
            if not (
                belief.holder.casefold() == operation.target.casefold()
                and belief.proposition.casefold() == operation.value.casefold()
            )
        ]
        self.beliefs.append(
            Belief(
                holder=operation.target,
                proposition=operation.value,
                certainty=operation.certainty,
                source=operation.evidence,
            )
        )

    def set_scene_tag(self, operation: StateOperation) -> None:
        tags = {tag.casefold(): tag for tag in self.scene_tags}
        if operation.value.casefold() in {"", "none", "null"}:
            self.scene_tags = []
            return
        tags[operation.value.casefold()] = operation.value
        self.scene_tags = sorted(tags.values(), key=str.casefold)

    def copy(self) -> RoleplayState:
        return RoleplayState.from_dict(self.to_dict())


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    superseded: bool = False
    """True when the chat was rewritten outside the kernel.

    A superseded turn stays in the ledger because ledger events reference it,
    but it must never reach a prompt: the authoritative SillyTavern transcript
    already replaced it.
    """

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> ConversationTurn:
        role = _required_str(data, "role")
        if role == "user":
            turn_role = "user"
        elif role == "assistant":
            turn_role = "assistant"
        else:
            raise ValueError("role must be user or assistant")
        superseded = data.get("superseded", False)
        if not isinstance(superseded, bool):
            raise ValueError("superseded must be a boolean")
        return cls(
            id=_required_str(data, "id"),
            role=cast(Literal["user", "assistant"], turn_role),
            content=_required_str(data, "content"),
            created_at=_required_str(data, "created_at"),
            superseded=superseded,
        )


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    id: str
    kind: str
    turn_id: str
    created_at: str
    payload: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> LedgerEvent:
        return cls(
            id=_required_str(data, "id"),
            kind=_required_str(data, "kind"),
            turn_id=_required_str(data, "turn_id"),
            created_at=_required_str(data, "created_at"),
            payload=_object(data.get("payload"), "payload"),
        )


@dataclass(slots=True)
class Session:
    id: str
    _state: RoleplayState
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    _turns: list[ConversationTurn] = field(default_factory=list)
    _ledger: list[LedgerEvent] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        language: str = "ru",
        pov: str = "third_person_limited",
        tense: str = "past",
        setting: str = "",
    ) -> Session:
        now = utc_now()
        state = RoleplayState(
            language=language,
            pov=pov,
            tense=tense,
            summary=setting,
        )
        return cls(id=new_id("session"), _state=state, created_at=now, updated_at=now)

    @property
    def state(self) -> RoleplayState:
        with self._lock:
            return deepcopy(self._state)

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        with self._lock:
            return tuple(self._turns)

    @property
    def ledger(self) -> tuple[LedgerEvent, ...]:
        with self._lock:
            return tuple(deepcopy(self._ledger))

    def edit_state(
        self,
        expected_version: int,
        mutator: Callable[[RoleplayState], None],
    ) -> int:
        with self._lock:
            if self._state.version != expected_version:
                raise ValueError("state version changed before edit")
            if any(
                event.kind
                in {"turn_committed", "state_delta_confirmed", "state_delta_rejected"}
                for event in self._ledger
            ):
                raise RuntimeError("state cannot be edited after transaction history begins")
            working = deepcopy(self._state)
            mutator(working)
            working.version = expected_version + 1
            self._state = working
            self.updated_at = utc_now()
            return working.version

    def elapsed_since_last_turn(self) -> float:
        """Seconds elapsed since the previous turn, or 0 for the first one."""
        with self._lock:
            if not self._turns:
                return 0.0
            previous = self._turns[-1].created_at
        try:
            stamp = datetime.fromisoformat(previous)
        except ValueError:
            return 0.0
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - stamp).total_seconds())

    def active_turns(self) -> tuple[ConversationTurn, ...]:
        """Turns that still match the authoritative chat transcript."""
        with self._lock:
            return tuple(turn for turn in self._turns if not turn.superseded)

    def supersede_turns(self, keep_identities: Sequence[tuple[str, str]]) -> int:
        """Mark turns that the external transcript replaced as superseded.

        `keep_identities` is the authoritative history as (role, content) pairs.
        The longest matching suffix is retained and everything older is marked,
        so prompts keep only turns the user can still see. Ledger events are
        untouched: they are an audit log and may still reference these turns.
        """
        with self._lock:
            if not keep_identities:
                return 0
            suffix_length = min(len(keep_identities), len(self._turns))
            matched = 0
            while (
                matched < suffix_length
                and self._turns[len(self._turns) - 1 - matched].role
                == keep_identities[len(keep_identities) - 1 - matched][0]
                and _identity(self._turns[len(self._turns) - 1 - matched].content)
                == _identity(keep_identities[len(keep_identities) - 1 - matched][1])
            ):
                matched += 1
            cut = len(self._turns) - matched
            changed = 0
            for turn in self._turns[:cut]:
                if not turn.superseded:
                    object.__setattr__(turn, "superseded", True)
                    changed += 1
            if changed:
                self.updated_at = utc_now()
            return changed

    def apply_output_settings(
        self,
        *,
        language: str,
        pov: str,
        tense: str,
    ) -> bool:
        """Align output settings with the current client configuration.

        These fields only steer rendering; they are not narrative facts, so they
        may change at any time without touching the state version or the ledger.
        """
        with self._lock:
            changed = (
                self._state.language != language
                or self._state.pov != pov
                or self._state.tense != tense
            )
            if not changed:
                return False
            self._state.language = language
            self._state.pov = pov
            self._state.tense = tense
            self.updated_at = utc_now()
            return True

    def to_dict(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "schema_version": self.schema_version,
                "id": self.id,
                "state": self._state.to_dict(),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "turns": [turn.to_dict() for turn in self._turns],
                "ledger": [event.to_dict() for event in self._ledger],
            }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> Session:
        schema_version = _required_int(data, "schema_version", minimum=1)
        if schema_version != 1:
            raise ValueError(f"unsupported session schema version {schema_version}")
        session = cls(
            id=_required_str(data, "id"),
            _state=RoleplayState.from_dict(_object(data.get("state"), "state")),
            schema_version=schema_version,
            created_at=_required_str(data, "created_at"),
            updated_at=_required_str(data, "updated_at"),
            _turns=[
                ConversationTurn.from_dict(_object(turn, "turn"))
                for turn in _array(data.get("turns"), "turns")
            ],
            _ledger=[
                LedgerEvent.from_dict(_object(event, "event"))
                for event in _array(data.get("ledger"), "ledger")
            ],
        )
        session._validate_integrity()
        return session

    def append_turn(self, role: Literal["user", "assistant"], content: str) -> ConversationTurn:
        with self._lock:
            turn = ConversationTurn(
                id=new_id("turn"),
                role=role,
                content=content,
                created_at=utc_now(),
            )
            self._turns.append(turn)
            self.updated_at = turn.created_at
            return turn

    def append_ledger_event(
        self,
        *,
        kind: str,
        turn_id: str,
        payload: dict[str, JsonValue],
    ) -> LedgerEvent:
        with self._lock:
            event = LedgerEvent(
                id=new_id("event"),
                kind=kind,
                turn_id=turn_id,
                created_at=utc_now(),
                payload=deepcopy(payload),
            )
            self._ledger.append(event)
            self.updated_at = event.created_at
            return event

    def _validate_integrity(self) -> None:
        turn_ids = [turn.id for turn in self._turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("turn ids must be unique")
        turns_by_id = {turn.id: turn for turn in self._turns}
        event_ids = [event.id for event in self._ledger]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("ledger event ids must be unique")

        turn_id_set = set(turn_ids)
        request_ids: set[str] = set()
        previous_version: int | None = None
        for event in self._ledger:
            if event.turn_id not in turn_id_set:
                raise ValueError("ledger event references an unknown turn")
            if event.kind not in {
                "turn_committed",
                "state_delta_confirmed",
                "state_delta_rejected",
            }:
                continue
            request_id = _required_str(event.payload, "request_id")
            if event.kind == "turn_committed":
                if request_id in request_ids:
                    raise ValueError("ledger request ids must be unique")
                request_ids.add(request_id)
            elif request_id not in request_ids:
                raise ValueError("state resolution references an unknown request")
            elif turns_by_id[event.turn_id].role != "assistant":
                raise ValueError("state resolution must reference an assistant turn")
            from_version = _required_int(event.payload, "from_state_version", minimum=0)
            to_version = _required_int(event.payload, "to_state_version", minimum=0)
            if to_version < from_version:
                raise ValueError("state version cannot decrease")
            if event.kind == "state_delta_rejected" and to_version != from_version:
                raise ValueError("rejected state delta must not change the version")
            if previous_version is not None and from_version != previous_version:
                raise ValueError("ledger state versions must be contiguous")
            previous_version = to_version

            if event.kind == "state_delta_rejected":
                _required_str(event.payload, "pending_digest")

            if event.kind == "turn_committed":
                user_turn_id = _required_str(event.payload, "user_turn_id")
                if user_turn_id not in turn_id_set:
                    raise ValueError("turn commit references an unknown user turn")
                if turns_by_id[event.turn_id].role != "assistant":
                    raise ValueError("turn commit must reference an assistant turn")
                if turns_by_id[user_turn_id].role != "user":
                    raise ValueError("turn commit must reference a user turn")

        if previous_version is not None and previous_version != self._state.version:
            raise ValueError("state version does not match the ledger")


def _limit_prompt_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…"


def _belief_prompt(belief: Belief) -> dict[str, JsonValue]:
    data = belief.to_dict()
    data["holder"] = _limit_prompt_text(belief.holder, 200)
    data["proposition"] = _limit_prompt_text(belief.proposition, 1000)
    data["source"] = _limit_prompt_text(belief.source, 500)
    return data


def _tail_dict(values: dict[str, str], limit: int, value_limit: int) -> dict[str, str]:
    return {
        key: _limit_prompt_text(value, value_limit)
        for key, value in list(values.items())[-limit:]
    }


def _object(value: JsonValue | None, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _array(value: JsonValue | None, name: str) -> list[JsonValue]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _identity(value: str) -> str:
    """Normalised comparison key for transcript matching."""
    normalized = unicodedata.normalize("NFKC", value).replace("\u200b", "")
    normalized = re.sub(r"[*_`~]", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _required_str(data: dict[str, JsonValue], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_field(
    data: dict[str, JsonValue],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{key} must be a string")
    return value.strip()


def _optional_str(data: dict[str, JsonValue], key: str, default: str) -> str:
    value = data.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _string_map(value: JsonValue | None) -> dict[str, str]:
    # A missing key means "no entries", not corruption: sessions written by an
    # older version must stay loadable after an upgrade.
    if value is None:
        return {}
    data = _object(value, "object")
    result: dict[str, str] = {}
    for key, item in data.items():
        if not isinstance(item, str):
            raise ValueError(f"{key} must contain only string values")
        result[key] = item
    return result


def _string_list(values: list[JsonValue], name: str) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{name} must contain only strings")
        result.append(value)
    return result


def _bounded_float(value: JsonValue | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("expected a number between 0 and 1")
    return result


def _required_bounded_float(data: dict[str, JsonValue], key: str) -> float:
    if key not in data:
        raise ValueError(f"{key} is required")
    return _bounded_float(data.get(key), 1.0)


def _required_int(data: dict[str, JsonValue], key: str, *, minimum: int = 0) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def _optional_int(data: dict[str, JsonValue], key: str, default: int) -> int:
    value = data.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _impact(value: JsonValue | None) -> Impact:
    if value == "low":
        return "low"
    if value == "medium":
        return "medium"
    if value == "high":
        return "high"
    raise ValueError("impact must be low, medium, or high")


def _operation_kind(value: JsonValue | None) -> OperationKind:
    allowed: set[OperationKind] = {
        "set_time",
        "set_location",
        "set_summary",
        "set_scene_tag",
        "upsert_fact",
        "upsert_belief",
        "set_relationship",
        "set_resource",
        "set_injury",
        "open_thread",
        "resolve_thread",
        "record_event",
    }
    if value not in allowed:
        raise ValueError("unsupported operation kind")
    return value
