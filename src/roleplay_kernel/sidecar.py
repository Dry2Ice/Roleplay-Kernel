from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal, Protocol, cast
from urllib.parse import urlsplit

from .compiler import ContextBudgetError, ContextCompiler
from .engine import (
    ClientGoneError,
    Engine,
    EngineConfig,
    EngineMode,
    StaleTurnResultError,
    UnknownPendingCommitError,
)
from .models import ChatMessage, JsonValue, Session, new_id, utc_now
from .profiles import ProfileRouter
from .providers import (
    ChatProvider,
    DeltaSink,
    OpenAICompatibleProvider,
    STConnectionProfileProvider,
    TokenParameter,
)
from .summarizer import SummaryConfig, TranscriptSummarizer
from .utils import ProviderError

SIDECAR_VERSION = "0.10.0"
PROTOCOL_VERSION = 1
ENVELOPE_PREFIX = "[ROLEPLAY_KERNEL_ENVELOPE_V1]"
CONTROL_PREFIX = "[ROLEPLAY_KERNEL_CONTROL_V1]"
CONTROL_MODEL_PREFIX = "roleplay-kernel-control/"
MAX_REQUEST_BYTES = 2_097_152
MAX_CONTEXT_CHARS = 16_000
SUPPORTED_GENERATION_TYPES = {"normal", "regenerate", "swipe"}
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CONFIG_KEYS = {
    "host",
    "port",
    "upstream_base_url",
    "upstream_model",
    "upstream_api_key_env",
    "upstream_token_parameter",
    "integration_key",
    "state_dir",
    "mode",
    "context_window",
    "token_budget",
    "max_output_tokens",
    "max_internal_tokens",
    "upstream_timeout_seconds",
    "max_repairs",
    "max_context_chars",
    "allow_insecure_http",
    "turn_budget_seconds",
    "post_render_grace_seconds",
    "require_upstream_profile",
}


class SidecarError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class KernelServiceProtocol(Protocol):
    def generate(
        self,
        envelope: GenerationEnvelope,
        messages: list[IncomingMessage],
        context_prompt: str,
        *,
        on_render_delta: DeltaSink | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> TurnPayload: ...

    def control(self, action: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]: ...

    def health(self) -> dict[str, JsonValue]: ...

    def diagnostics(self) -> dict[str, JsonValue]: ...

    def self_test(self) -> dict[str, JsonValue]: ...


@dataclass(frozen=True, slots=True)
class SidecarConfig:
    host: str
    port: int
    upstream_base_url: str
    upstream_model: str
    upstream_api_key_env: str
    upstream_token_parameter: TokenParameter
    integration_key: str
    state_dir: Path
    mode: EngineMode
    context_window: int
    token_budget: int
    max_output_tokens: int
    max_internal_tokens: int
    upstream_timeout_seconds: int
    max_repairs: int
    max_context_chars: int
    allow_insecure_http: bool
    turn_budget_seconds: float = 300.0
    post_render_grace_seconds: float = 90.0
    require_upstream_profile: bool = False
    stage_profiles: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> SidecarConfig:
        data = _load_config_file(path)
        unknown = sorted(set(data) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        env = os.environ
        host = _env_string(env, "RPK_SIDECAR_HOST", _config_string(data.get("host"), "127.0.0.1"))
        port = _env_int(env, "RPK_SIDECAR_PORT", _config_int(data.get("port"), 8787), minimum=0)
        base_url = _env_string(
            env,
            "RPK_UPSTREAM_BASE_URL",
            _config_string(data.get("upstream_base_url"), ""),
        )
        model = _env_string(
            env,
            "RPK_UPSTREAM_MODEL",
            _config_string(data.get("upstream_model"), ""),
        )
        api_key_env = _env_string(
            env,
            "RPK_UPSTREAM_API_KEY_ENV",
            _config_string(data.get("upstream_api_key_env"), "OPENAI_API_KEY"),
        )
        token_parameter = _token_parameter(
            _env_string(
                env,
                "RPK_UPSTREAM_TOKEN_PARAMETER",
                _config_string(data.get("upstream_token_parameter"), "max_tokens"),
            )
        )
        integration_key = _env_string(
            env,
            "RPK_INTEGRATION_KEY",
            _config_string(data.get("integration_key"), ""),
        )
        state_dir = Path(
            _env_string(
                env,
                "RPK_STATE_DIR",
                _config_string(
                    data.get("state_dir"),
                    str(Path.home() / ".roleplay-kernel"),
                ),
            )
        ).expanduser()
        mode = _engine_mode(
            _env_string(env, "RPK_MODE", _config_string(data.get("mode"), "balanced"))
        )
        context_window = _env_int(
            env,
            "RPK_CONTEXT_WINDOW",
            _config_int(data.get("context_window"), 32768),
            minimum=2048,
        )
        token_budget = _env_int(
            env,
            "RPK_TOKEN_BUDGET",
            _config_int(data.get("token_budget"), 18000),
            minimum=512,
        )
        max_output_tokens = _env_int(
            env,
            "RPK_MAX_OUTPUT_TOKENS",
            _config_int(data.get("max_output_tokens"), 8192),
            minimum=1,
        )
        max_internal_tokens = _env_int(
            env,
            "RPK_MAX_INTERNAL_TOKENS",
            _config_int(data.get("max_internal_tokens"), 4096),
            minimum=1,
        )
        upstream_timeout_seconds = _env_int(
            env,
            "RPK_UPSTREAM_TIMEOUT_SECONDS",
            _config_int(data.get("upstream_timeout_seconds"), 600),
            minimum=30,
        )
        max_repairs = _env_int(
            env,
            "RPK_MAX_REPAIRS",
            _config_int(data.get("max_repairs"), 1),
            minimum=0,
        )
        max_context_chars = _env_int(
            env,
            "RPK_MAX_CONTEXT_CHARS",
            _config_int(data.get("max_context_chars"), MAX_CONTEXT_CHARS),
            minimum=1000,
        )
        allow_insecure_http = _env_bool(
            env,
            "RPK_ALLOW_INSECURE_HTTP",
            _config_bool(data.get("allow_insecure_http"), False),
        )
        turn_budget_seconds = _env_float(
            env,
            "RPK_TURN_BUDGET_SECONDS",
            _config_float(data.get("turn_budget_seconds"), 300.0),
            minimum=30.0,
            maximum=3600.0,
        )
        post_render_grace_seconds = _env_float(
            env,
            "RPK_POST_RENDER_GRACE_SECONDS",
            _config_float(data.get("post_render_grace_seconds"), 90.0),
            minimum=0.0,
            maximum=600.0,
        )
        require_upstream_profile = _env_bool(
            env,
            "RPK_REQUIRE_UPSTREAM_PROFILE",
            _config_bool(data.get("require_upstream_profile"), False),
        )
        stage_profiles: dict[str, str] = {}
        raw_stage_profiles = data.get("stage_profiles", {})
        if isinstance(raw_stage_profiles, dict):
            for key, value in raw_stage_profiles.items():
                if isinstance(value, str):
                    stage_profiles[str(key)] = value
        config = cls(
            host=host,
            port=port,
            upstream_base_url=base_url,
            upstream_model=model,
            upstream_api_key_env=api_key_env,
            upstream_token_parameter=token_parameter,
            integration_key=integration_key,
            state_dir=state_dir,
            mode=mode,
            context_window=context_window,
            token_budget=token_budget,
            max_output_tokens=max_output_tokens,
            max_internal_tokens=max_internal_tokens,
            upstream_timeout_seconds=upstream_timeout_seconds,
            max_repairs=max_repairs,
            max_context_chars=max_context_chars,
            allow_insecure_http=allow_insecure_http,
            turn_budget_seconds=turn_budget_seconds,
            post_render_grace_seconds=post_render_grace_seconds,
            require_upstream_profile=require_upstream_profile,
            stage_profiles=stage_profiles,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if (
            isinstance(self.turn_budget_seconds, bool)
            or not math.isfinite(self.turn_budget_seconds)
            or not 30.0 <= self.turn_budget_seconds <= 3600.0
        ):
            raise ValueError("turn_budget_seconds must be between 30 and 3600")
        if (
            isinstance(self.post_render_grace_seconds, bool)
            or not math.isfinite(self.post_render_grace_seconds)
            or not 0.0 <= self.post_render_grace_seconds <= 600.0
        ):
            raise ValueError("post_render_grace_seconds must be between 0 and 600")
        if not self.upstream_base_url:
            raise ValueError("upstream_base_url is required")
        if not self.upstream_model:
            raise ValueError("upstream_model is required")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if not _is_loopback(self.host):
            raise ValueError(
                "sidecar host must be loopback; use a TLS reverse proxy for remote access"
            )
        if self.max_context_chars > self.token_budget:
            raise ValueError("max_context_chars must not exceed the conservative token budget")
        output_reserve = max(self.max_output_tokens, self.max_internal_tokens)
        if self.token_budget + output_reserve > self.context_window:
            raise ValueError("token budget plus output reserve exceeds context window")
        if len(self.integration_key) < 32:
            raise ValueError("integration_key must contain at least 32 characters")
        if any(
            not 0x21 <= ord(character) <= 0x7E
            for character in self.integration_key
        ):
            raise ValueError("integration_key must contain printable ASCII characters")
        if self.integration_key in {
            "replace-with-a-long-random-value",
            "replace-with-at-least-32-random-characters",
        }:
            raise ValueError("integration_key must be replaced before starting the sidecar")
        OpenAICompatibleProvider(
            model=self.upstream_model,
            base_url=self.upstream_base_url,
            api_key_env=self.upstream_api_key_env,
            token_parameter=self.upstream_token_parameter,
            allow_insecure_http=self.allow_insecure_http,
        )


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    index: int
    role: str
    content: str
    name: str = ""
    swipe_id: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "index": self.index,
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "swipe_id": self.swipe_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> TranscriptItem:
        role = _required_string(data, "role")
        if role not in {"user", "assistant"}:
            raise SidecarError("invalid_transcript", "transcript role must be user or assistant")
        content = _required_string(data, "content")
        if len(content) > 64_000:
            raise SidecarError("invalid_transcript", "transcript message is too large", 413)
        name = data.get("name", "")
        swipe_id = data.get("swipe_id")
        if name is not None and (not isinstance(name, str) or len(name) > 200):
            raise SidecarError("invalid_transcript", "transcript name is invalid")
        if swipe_id is not None and (not isinstance(swipe_id, str) or len(swipe_id) > 128):
            raise SidecarError("invalid_transcript", "transcript swipe_id is invalid")
        return cls(
            index=_required_int(data, "index"),
            role=role,
            content=content,
            name=name if isinstance(name, str) else "",
            swipe_id=swipe_id,
        )


@dataclass(frozen=True, slots=True)
class STProfileConfig:
    profile_id: str
    st_base_url: str
    source: str
    api_url: str
    model: str
    secret_id: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "profile_id": self.profile_id,
            "st_base_url": self.st_base_url,
            "source": self.source,
            "api_url": self.api_url,
            "model": self.model,
            "secret_id": self.secret_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> STProfileConfig:
        profile_id = _required_string(data, "profile_id")
        st_base_url = _required_string(data, "st_base_url")
        source = _required_string(data, "source")
        # An ST profile without a custom endpoint sends empty strings here, and
        # the sidecar resolves the URL itself, so empty is a valid value.
        api_url = _config_string(data.get("api_url"), "", allow_empty=True)
        model = _required_string(data, "model")
        secret_id = _config_string(data.get("secret_id"), "", allow_empty=True)
        if len(profile_id) > 128 or len(source) > 64 or len(model) > 256:
            raise SidecarError("invalid_profile", "connection profile fields are too long")
        if len(api_url) > 2048 or len(secret_id) > 256:
            raise SidecarError("invalid_profile", "connection profile fields are too long")
        return cls(
            profile_id=profile_id,
            st_base_url=st_base_url,
            source=source,
            api_url=api_url,
            model=model,
            secret_id=secret_id,
        )


@dataclass(frozen=True, slots=True)
class GenerationEnvelope:
    protocol: int
    session_id: str
    request_key: str
    transcript: tuple[TranscriptItem, ...]
    generation_type: str
    language: str
    pov: str
    tense: str
    operation: str = "generate"
    mode: EngineMode = "balanced"
    request_delay_seconds: float = 0.0
    upstream_profile: STProfileConfig | None = None
    sampling: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "protocol": self.protocol,
            "operation": self.operation,
            "session_id": self.session_id,
            "request_key": self.request_key,
            "transcript": [item.to_dict() for item in self.transcript],
            "generation_type": self.generation_type,
            "language": self.language,
            "pov": self.pov,
            "tense": self.tense,
            "mode": self.mode,
            "request_delay_seconds": self.request_delay_seconds,
            "sampling": dict(self.sampling),
        }
        if self.upstream_profile is not None:
            data["upstream_profile"] = self.upstream_profile.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> GenerationEnvelope:
        protocol = _required_int(data, "protocol")
        if protocol != PROTOCOL_VERSION:
            raise SidecarError("unsupported_protocol", "unsupported extension protocol")
        session_id = _required_string(data, "session_id")
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise SidecarError("invalid_session", "session_id contains invalid characters")
        request_key = _required_string(data, "request_key")
        if not re.fullmatch(r"[a-f0-9]{8,128}", request_key):
            raise SidecarError("invalid_request_key", "request_key format is invalid")
        transcript_values = _array(data.get("transcript"), "transcript")
        if len(transcript_values) > 200:
            raise SidecarError("invalid_transcript", "transcript is too long", 413)
        transcript_items: list[TranscriptItem] = []
        for index, value in enumerate(transcript_values):
            item = TranscriptItem.from_dict(_object(value, "transcript item"))
            if item.index != index or item.role not in {"user", "assistant"}:
                raise SidecarError("invalid_transcript", "transcript item is invalid")
            transcript_items.append(item)
        transcript = tuple(transcript_items)
        generation_type = _required_string(data, "generation_type")
        if generation_type not in SUPPORTED_GENERATION_TYPES:
            raise SidecarError(
                "unsupported_generation",
                f"generation type {generation_type!r} is not supported",
            )
        operation = _optional_string(data, "operation", "generate")
        if operation != "generate":
            raise SidecarError("unsupported_operation", "operation is not supported")
        profile_value = data.get("upstream_profile")
        profile = (
            STProfileConfig.from_dict(_object(profile_value, "upstream_profile"))
            if profile_value is not None
            else None
        )
        sampling_value = data.get("sampling")
        sampling = _object(sampling_value, "sampling") if sampling_value is not None else {}
        mode = cast(
            EngineMode,
            _choice(data, "mode", "balanced", {"lite", "fast", "balanced", "strict"}),
        )
        delay_value = data.get("request_delay_seconds", 0.0)
        if (
            isinstance(delay_value, bool)
            or not isinstance(delay_value, (int, float))
            or not math.isfinite(float(delay_value))
            or not 0.0 <= float(delay_value) <= 600.0
        ):
            raise SidecarError("invalid_envelope", "request_delay_seconds is invalid")
        request_delay_seconds = float(delay_value)
        return cls(
            protocol=protocol,
            session_id=session_id,
            request_key=request_key,
            transcript=transcript,
            generation_type=generation_type,
            language=_choice(data, "language", "ru", {"ru", "en"}),
            pov=_choice(
                data,
                "pov",
                "third_person_limited",
                {"first_person", "second_person", "third_person_limited"},
            ),
            tense=_choice(data, "tense", "past", {"past", "present", "future"}),
            operation=operation,
            mode=mode,
            request_delay_seconds=request_delay_seconds,
            upstream_profile=profile,
            sampling=sampling,
        )


@dataclass(frozen=True, slots=True)
class TurnPayload:
    text: str
    model: str
    usage: dict[str, int]
    session_id: str
    state_version: int
    status: str
    pending_request_id: str | None


@dataclass(slots=True)
class SessionRecord:
    session: Session
    storage_id: str
    checkpoint: dict[str, JsonValue] | None = None
    last_result: dict[str, JsonValue] | None = None
    last_request_key: str | None = None
    last_request_fingerprint: str | None = None
    pending_request_id: str | None = None
    last_control_action: str | None = None
    last_control_request_id: str | None = None
    context_hash: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "session": self.session.to_dict(),
            "storage_id": self.storage_id,
            "checkpoint": self.checkpoint,
            "last_result": self.last_result,
            "last_request_key": self.last_request_key,
            "last_request_fingerprint": self.last_request_fingerprint,
            "pending_request_id": self.pending_request_id,
            "last_control_action": self.last_control_action,
            "last_control_request_id": self.last_control_request_id,
            "context_hash": self.context_hash,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> SessionRecord:
        checkpoint = data.get("checkpoint")
        if checkpoint is not None and not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be an object")
        last_result = data.get("last_result")
        if last_result is not None and not isinstance(last_result, dict):
            raise ValueError("last_result must be an object")
        last_request_key = data.get("last_request_key")
        if last_request_key is not None and not isinstance(last_request_key, str):
            raise ValueError("last_request_key must be a string")
        last_fingerprint = data.get("last_request_fingerprint")
        if last_fingerprint is not None and not isinstance(last_fingerprint, str):
            raise ValueError("last_request_fingerprint must be a string")
        pending = data.get("pending_request_id")
        if pending is not None and not isinstance(pending, str):
            raise ValueError("pending_request_id must be a string")
        last_action = data.get("last_control_action")
        if last_action is not None and not isinstance(last_action, str):
            raise ValueError("last_control_action must be a string")
        last_request = data.get("last_control_request_id")
        if last_request is not None and not isinstance(last_request, str):
            raise ValueError("last_control_request_id must be a string")
        return cls(
            session=Session.from_dict(_object(data.get("session"), "session")),
            storage_id=_required_string(data, "storage_id"),
            checkpoint=checkpoint,
            last_result=last_result,
            last_request_key=last_request_key,
            last_request_fingerprint=last_fingerprint,
            pending_request_id=pending,
            last_control_action=last_action,
            last_control_request_id=last_request,
            context_hash=_optional_string(data, "context_hash", ""),
            updated_at=_optional_string(data, "updated_at", ""),
        )


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()
        self._integrity_key = self._load_or_create_integrity_key()

    def lock(self, session_id: str) -> threading.RLock:
        _validate_session_id(session_id)
        with self._guard:
            return self._locks.setdefault(session_id, threading.RLock())

    @property
    def integrity_key(self) -> bytes:
        return self._integrity_key

    def session_count(self) -> int:
        try:
            return sum(1 for item in self.root.glob("*.json") if item.stem != "integrity")
        except OSError:
            return 0

    def load(self, session_id: str) -> SessionRecord | None:
        path = self._record_path(session_id)
        if not path.exists():
            return None
        wrapper = _read_json_object(path)
        record_data = _object(wrapper.get("record"), "record")
        digest = _required_string(wrapper, "digest")
        if not hmac.compare_digest(digest, self._record_digest(record_data)):
            raise ValueError("session record integrity check failed")
        record = SessionRecord.from_dict(record_data)
        if record.storage_id != session_id:
            raise ValueError("session storage identity mismatch")
        return record

    def save(self, record: SessionRecord) -> None:
        session_id = record.storage_id
        path = self._record_path(session_id)
        record.updated_at = str(record.session.updated_at)
        record_data = record.to_dict()
        _atomic_write_json(
            path,
            {
                "record": record_data,
                "digest": self._record_digest(record_data),
            },
        )

    def delete(self, session_id: str) -> None:
        self._record_path(session_id).unlink(missing_ok=True)

    def _record_digest(self, record: dict[str, JsonValue]) -> str:
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(self._integrity_key, encoded, hashlib.sha256).hexdigest()

    def _record_path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self.root / f"{session_id}.json"

    def _load_or_create_integrity_key(self) -> bytes:
        path = self.root / "integrity.key"
        if path.exists():
            return self._read_integrity_key(path)
        key = secrets.token_bytes(32)
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except FileExistsError:
            return self._read_integrity_key(path)
        try:
            os.write(descriptor, key)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _restrict_permissions(path)
        return key

    @staticmethod
    def _read_integrity_key(path: Path) -> bytes:
        key = path.read_bytes()
        if len(key) != 32:
            raise ValueError("integrity key must contain exactly 32 bytes")
        return key


class SessionService:
    def __init__(
        self,
        config: SidecarConfig,
        *,
        provider: ChatProvider | None = None,
        store: SessionStore | None = None,
        router: ProfileRouter | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.store = store or SessionStore(config.state_dir)
        self.provider = provider or OpenAICompatibleProvider(
            model=config.upstream_model,
            base_url=config.upstream_base_url,
            api_key=os.getenv("RPK_UPSTREAM_API_KEY"),
            api_key_env=config.upstream_api_key_env,
            token_parameter=config.upstream_token_parameter,
            timeout=float(config.upstream_timeout_seconds),
            allow_insecure_http=config.allow_insecure_http,
        )
        self._profile_provider: STConnectionProfileProvider | None = None
        self._profile_signature: tuple[str, str, str, str, str, str, str] | None = None
        self._generation_lock = threading.RLock()
        self._active_lock = threading.Lock()
        self._active_session_id: str | None = None
        self._reconciled_sessions: set[str] = set()
        self._aborted_sessions: set[str] = set()
        self._economy_mode_active = False
        self._router = router or ProfileRouter(self.provider)
        summarizer_config = SummaryConfig(
            enabled=False,
            keep_recent_pairs=6,
            summarize_every_pairs=4,
            hide_old_after_pairs=24,
        )
        summarizer = TranscriptSummarizer(summarizer_config)
        self.engine = Engine(
            self.provider,
            compiler=ContextCompiler(token_budget=config.token_budget),
            config=EngineConfig(
                mode=config.mode,
                context_window=config.context_window,
                max_output_tokens=config.max_output_tokens,
                max_internal_tokens=config.max_internal_tokens,
                max_repairs=config.max_repairs,
                turn_budget_seconds=config.turn_budget_seconds,
                post_render_grace_seconds=config.post_render_grace_seconds,
                stage_profiles=config.stage_profiles,
            ),
            integrity_key=self.store.integrity_key,
            router=self._router,
            summarizer=summarizer,
        )

    def health(self) -> dict[str, JsonValue]:
        return {
            "status": "ok",
            "service": "roleplay-kernel-sidecar",
            "version": SIDECAR_VERSION,
            "protocol": PROTOCOL_VERSION,
            "upstream_profile_id": (
                self._profile_signature[0] if self._profile_signature is not None else None
            ),
            "metrics": self.engine.metrics.to_dict(),
        }

    def diagnostics(self) -> dict[str, JsonValue]:
        """Collect everything needed to explain a failure in one message.

        Secrets never appear here: the integration key, upstream keys and
        cookies are reduced to presence flags, not values.
        """
        config = self.config
        provider = self.provider
        return {
            "generated_at": utc_now(),
            "sidecar": {
                "version": SIDECAR_VERSION,
                "protocol": PROTOCOL_VERSION,
                "host": config.host,
                "port": config.port,
                "python": sys.version.split()[0],
                "pid": os.getpid(),
            },
            "config": {
                "mode": config.mode,
                "upstream_base_url": config.upstream_base_url,
                "upstream_model": config.upstream_model,
                "upstream_api_key_present": bool(config.upstream_api_key_env),
                "token_parameter": config.upstream_token_parameter,
                "context_window": config.context_window,
                "token_budget": config.token_budget,
                "max_output_tokens": config.max_output_tokens,
                "max_internal_tokens": config.max_internal_tokens,
                "upstream_timeout_seconds": config.upstream_timeout_seconds,
                "max_repairs": config.max_repairs,
                "turn_budget_seconds": config.turn_budget_seconds,
                "post_render_grace_seconds": config.post_render_grace_seconds,
                "allow_insecure_http": config.allow_insecure_http,
                "require_upstream_profile": config.require_upstream_profile,
                "state_dir": str(config.state_dir),
            },
            "upstream": {
                "provider": type(provider).__name__,
                "profile_id": (
                    self._profile_signature[0] if self._profile_signature is not None else None
                ),
                "rate_limited": bool(getattr(provider, "rate_limited", False)),
                "model": getattr(provider, "model", None),
            },
            "runtime": {
                "engine_mode": self.engine.config.mode,
                "economy_mode": self._economy_mode_active,
                "metrics": self.engine.metrics.to_dict(),
                "sessions": self.store.session_count(),
                "state_dir_exists": config.state_dir.exists(),
            },
            "limits": {
                "max_request_bytes": MAX_REQUEST_BYTES,
                "max_context_chars": MAX_CONTEXT_CHARS,
            },
        }

    def self_test(self) -> dict[str, JsonValue]:
        """Verify every layer the extension depends on, cheapest check first.

        The upstream probe is a single token request so the test stays inside a
        provider's rate limit. Failures are reported instead of raised so the
        panel can show every broken layer at once.
        """
        checks: list[JsonValue] = []

        def record(name: str, ok: bool, detail: str = "") -> None:
            checks.append({"name": name, "ok": ok, "detail": detail})

        record(
            "state_dir",
            self.config.state_dir.exists() and os.access(self.config.state_dir, os.W_OK),
            str(self.config.state_dir),
        )
        record(
            "integration_key",
            len(self.config.integration_key) >= 32,
            f"{len(self.config.integration_key)} chars",
        )
        record("upstream_profile", self._profile_provider is not None, type(self.provider).__name__)

        upstream_detail = "not probed"
        if self._profile_provider is not None:
            try:
                probe = self.provider.complete(
                    (ChatMessage(role="user", content="Reply with the single word: ok"),),
                    temperature=0.0,
                    max_tokens=16,
                )
                upstream_ok = bool(probe.content.strip())
                upstream_detail = probe.content.strip()[:40] or "empty response"
            except Exception as error:  # reported, never raised: the test lists all layers
                upstream_ok = False
                upstream_detail = str(error)[:200]
            record("upstream_reachable", upstream_ok, upstream_detail)

        return {
            "ok": all(
                bool(check.get("ok"))
                for check in checks
                if isinstance(check, dict)
            ),
            "checks": checks,
            "version": SIDECAR_VERSION,
            "upstream_profile_id": (
                self._profile_signature[0] if self._profile_signature is not None else None
            ),
        }

    def _apply_mode(self, mode: EngineMode) -> None:
        if self._profile_provider is not None and self._profile_provider.rate_limited:
            # The upstream refused the previous turn. Retrying a four-call
            # pipeline would only fail again, so the turn degrades to a single
            # render call instead of surfacing an error the user cannot fix.
            if mode != "lite":
                mode = "lite"
                with self._active_lock:
                    self._economy_mode_active = True
        elif mode != "lite":
            with self._active_lock:
                self._economy_mode_active = False
        if self.engine.config.mode != mode:
            self.engine.config = replace(self.engine.config, mode=mode)

    def _require_generation_profile(self, envelope: GenerationEnvelope) -> None:
        """Refuse to send a turn to the fallback upstream when launched by ST.

        The direct provider is a legitimate standalone configuration, so this
        only applies when the server plugin marks the runtime as ST-managed.
        Without a profile the turn would otherwise leave this machine for
        whatever host the config happens to name.
        """
        if not self.config.require_upstream_profile:
            return
        if envelope.upstream_profile is not None:
            return
        if self._profile_provider is not None:
            return
        raise SidecarError(
            "upstream_profile_required",
            "no ST connection profile is selected, so the turn was not sent",
            409,
        )

    def _select_profile(
        self, payload: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Apply an ST connection profile without waiting for a generation.

        Without this the service keeps the direct fallback provider until the
        first turn, which makes the self-test report a false negative and lets
        a request that arrives without an envelope profile reach the fallback
        upstream.
        """
        profile_value = payload.get("upstream_profile")
        profile = (
            STProfileConfig.from_dict(_object(profile_value, "upstream_profile"))
            if profile_value is not None
            else None
        )
        request_delay_seconds = 0.0
        delay_value = payload.get("request_delay_seconds")
        if delay_value is not None:
            request_delay_seconds = _config_float(delay_value, 0.0)
            if not 0.0 <= request_delay_seconds <= 600.0:
                raise SidecarError(
                    "invalid_request",
                    "request_delay_seconds must be between 0 and 600",
                    400,
                )
        with self._generation_lock:
            self._apply_upstream_profile(profile, request_delay_seconds)
        return {
            "selected": profile is not None,
            "upstream_profile_id": (
                profile.profile_id if profile is not None else None
            ),
            "upstream_model": (
                profile.model if profile is not None else self.config.upstream_model
            ),
            "provider": type(self.provider).__name__,
        }

    def _apply_upstream_profile(
        self,
        profile: STProfileConfig | None,
        request_delay_seconds: float = 0.0,
    ) -> None:
        if profile is None:
            if self._profile_signature is not None:
                self._router.clear_override()
                self._profile_provider = None
                self._profile_signature = None
            return
        signature = (
            profile.profile_id,
            profile.st_base_url,
            profile.source,
            profile.api_url,
            profile.model,
            profile.secret_id,
            str(request_delay_seconds),
        )
        if signature == self._profile_signature and self._profile_provider is not None:
            return
        try:
            provider = STConnectionProfileProvider(
                st_base_url=profile.st_base_url,
                source=profile.source,
                api_url=profile.api_url,
                model=profile.model,
                secret_id=profile.secret_id,
                token_parameter=self.config.upstream_token_parameter,
                timeout=float(self.config.upstream_timeout_seconds),
                inter_request_delay_seconds=request_delay_seconds,
            )
        except ValueError as error:
            raise SidecarError("invalid_profile", str(error), 400) from error
        self._profile_provider = provider
        self._profile_signature = signature
        self.provider = provider
        self.engine._router.override(provider)

    def generate(
        self,
        envelope: GenerationEnvelope,
        messages: list[IncomingMessage],
        context_prompt: str,
        *,
        on_render_delta: DeltaSink | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> TurnPayload:
        with self.store.lock(envelope.session_id):
            record = self.store.load(envelope.session_id)
            if record is None:
                session = self.engine.new_session(
                    language=envelope.language,
                    pov=envelope.pov,
                    tense=envelope.tense,
                )
                record = SessionRecord(
                    session=session,
                    storage_id=envelope.session_id,
                )
            request_fingerprint = _request_fingerprint(envelope, context_prompt)
            if record.pending_request_id is not None:
                self._restore_pending(record)
            if record.last_request_key == envelope.request_key and record.last_result is not None:
                if (
                    record.last_request_fingerprint is not None
                    and record.last_request_fingerprint != request_fingerprint
                ):
                    raise SidecarError(
                        "request_key_reused",
                        "request key was already used for a different prompt",
                        409,
                    )
                return self._cached_turn(record)
            if (
                record.last_request_fingerprint == request_fingerprint
                and record.last_result is not None
            ):
                return self._cached_turn(record)
            if (
                envelope.generation_type in {"regenerate", "swipe"}
                and record.checkpoint is not None
            ):
                self._validate_checkpoint(record.session, record.checkpoint)
                record.session = Session.from_dict(record.checkpoint)
                record.pending_request_id = None
                self.engine.clear_pending_for_session(record.session.id)
            if record.pending_request_id is not None:
                raise SidecarError(
                    "pending_confirmation_required",
                    "the current state delta must be approved or rejected before continuing",
                    409,
                )
            conversation, user_input = _transcript_exchange(envelope.transcript)
            self._reconciled_sessions.discard(envelope.session_id)
            if self._sync_transcript(record.session, conversation):
                self._reconciled_sessions.add(envelope.session_id)
            record.checkpoint = record.session.to_dict()
            with self._generation_lock:
                self._apply_mode(envelope.mode)
                record.session.apply_output_settings(
                    language=envelope.language,
                    pov=envelope.pov,
                    tense=envelope.tense,
                )
                self._require_generation_profile(envelope)
                self._apply_upstream_profile(
                    envelope.upstream_profile,
                    envelope.request_delay_seconds,
                )
                with self._active_lock:
                    self._active_session_id = envelope.session_id
                    self._aborted_sessions.discard(envelope.session_id)
                guarded_delta: DeltaSink | None = None
                if on_render_delta is not None:
                    guarded_delta = (
                        on_render_delta
                        if should_abort is None
                        else _aborting_sink(
                            on_render_delta,
                            should_abort,
                            self._aborted_sessions,
                            envelope.session_id,
                        )
                    )
                try:
                    result = self.engine.advance(
                        record.session,
                        user_input,
                        external_context=context_prompt,
                        sampling=_render_sampling(
                            envelope.sampling,
                            self.config.max_output_tokens,
                        ),
                        on_render_delta=guarded_delta,
                        should_abort=should_abort,
                    )
                finally:
                    with self._active_lock:
                        self._active_session_id = None
            record.last_result = result.to_dict()
            record.last_request_key = envelope.request_key
            record.last_request_fingerprint = request_fingerprint
            record.pending_request_id = (
                result.request_id if result.pending_operations.operations else None
            )
            record.context_hash = _digest(context_prompt)
            self.store.save(record)
            return TurnPayload(
                text=result.text,
                model=(
                    envelope.upstream_profile.model
                    if envelope.upstream_profile is not None
                    else self.config.upstream_model
                ),
                usage=result.usage,
                session_id=record.session.id,
                state_version=record.session.state.version,
                status=result.status,
                pending_request_id=record.pending_request_id,
            )

    def control(self, action: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if action == "health":
            return self.health()
        if action == "diagnostics":
            return self.diagnostics()
        if action == "self_test":
            return self.self_test()
        if action == "select_profile":
            return self._select_profile(payload)
        session_id = _required_string(payload, "session_id")
        if action == "status":
            with self._active_lock:
                active_session_id = self._active_session_id
            if active_session_id == session_id:
                return self._live_status(session_id)
        with self.store.lock(session_id):
            record = self.store.load(session_id)
            if action == "status":
                status = self._status(record, session_id)
                transcript_value = payload.get("transcript")
                if record is not None and transcript_value is not None:
                    status["transcript_matches"] = self._transcript_matches(
                        record,
                        transcript_value,
                    )
                return status
            if record is None:
                if action == "reset":
                    return {
                        "session_id": session_id,
                        "reset": True,
                        "state_version": 0,
                        "pending_request_id": None,
                    }
                raise SidecarError("session_not_found", "session does not exist", 404)
            if record.pending_request_id is not None:
                self._restore_pending(record)
            if action == "state":
                return {
                    "session_id": session_id,
                    "state": record.session.state.to_dict(),
                    "pending_request_id": record.pending_request_id,
                    "last_result": record.last_result or {},
                }
            if action in {"commit", "reject"}:
                request_id = (
                    _required_string(record.last_result, "request_id")
                    if record.last_result is not None
                    else ""
                )
                if record.pending_request_id is None or record.last_result is None:
                    if (
                        record.last_control_action == action
                        and record.last_control_request_id == request_id
                    ):
                        return self._status(record, session_id)
                    raise SidecarError("no_pending_delta", "session has no pending delta", 404)
                assistant_turn_id = _required_string(record.last_result, "assistant_turn_id")
                if action == "commit":
                    self.engine.commit_pending_by_id(
                        record.session,
                        request_id=request_id,
                        assistant_turn_id=assistant_turn_id,
                    )
                else:
                    self.engine.discard_pending_by_id(
                        record.session,
                        request_id=request_id,
                        assistant_turn_id=assistant_turn_id,
                    )
                record.pending_request_id = None
                record.last_control_action = action
                record.last_control_request_id = request_id
                self.store.save(record)
                return self._status(record, session_id)
            if action == "reset":
                self.store.delete(session_id)
                self.engine.clear_pending_for_session(session_id)
                return {
                    "session_id": session_id,
                    "reset": True,
                    "state_version": 0,
                    "pending_request_id": None,
                }
            raise SidecarError("unknown_control", f"unknown control action {action!r}")

    def _restore_pending(self, record: SessionRecord) -> None:
        if record.last_result is None:
            raise SidecarError("pending_result_missing", "pending result is missing", 409)
        self.engine.restore_pending_from_result(record.session, record.last_result)

    def _cached_turn(self, record: SessionRecord) -> TurnPayload:
        result = record.last_result or {}
        status = _optional_string(result, "status", "ok")
        if record.pending_request_id is None and status == "needs_confirmation":
            status = "ok"
        return TurnPayload(
            text=_required_string(result, "text"),
            model=(
                self._profile_provider.model
                if self._profile_provider is not None
                else self.config.upstream_model
            ),
            usage={},
            session_id=record.session.id,
            state_version=record.session.state.version,
            status=status,
            pending_request_id=record.pending_request_id,
        )

    @staticmethod
    def _validate_checkpoint(
        session: Session,
        checkpoint: dict[str, JsonValue],
    ) -> None:
        checkpoint_turns = _array(checkpoint.get("turns"), "checkpoint turns")
        current_turns = _array(session.to_dict().get("turns"), "current turns")
        checkpoint_turn_ids = [
            _required_string(_object(value, "checkpoint turn"), "id")
            for value in checkpoint_turns
        ]
        current_turn_ids = [
            _required_string(_object(value, "current turn"), "id")
            for value in current_turns
        ]
        if current_turn_ids[: len(checkpoint_turn_ids)] != checkpoint_turn_ids:
            raise SidecarError("invalid_checkpoint", "session checkpoint is not a prefix", 409)
        checkpoint_ledger = _array(checkpoint.get("ledger"), "checkpoint ledger")
        current_ledger = _array(session.to_dict().get("ledger"), "current ledger")
        checkpoint_event_ids = [
            _required_string(_object(value, "checkpoint event"), "id")
            for value in checkpoint_ledger
        ]
        current_event_ids = [
            _required_string(_object(value, "current event"), "id")
            for value in current_ledger
        ]
        if current_event_ids[: len(checkpoint_event_ids)] != checkpoint_event_ids:
            raise SidecarError("invalid_checkpoint", "ledger checkpoint is not a prefix", 409)

    @staticmethod
    def _transcript_matches(
        record: SessionRecord,
        transcript_value: JsonValue,
    ) -> bool:
        try:
            values = _array(transcript_value, "transcript")
            transcript = tuple(
                TranscriptItem.from_dict(_object(value, "transcript item"))
                for value in values
            )
            actual = _complete_transcript_pairs(transcript)
        except (SidecarError, ValueError):
            return False
        expected = [(turn.role, turn.content) for turn in record.session.turns]
        if len(actual) > len(expected):
            return False
        if expected and not actual:
            return False
        expected_tail = expected[-len(actual) :] if actual else []
        return all(
            left_role == right_role
            and _message_identity(left_content) == _message_identity(right_content)
            for (left_role, left_content), (right_role, right_content)
            in zip(actual, expected_tail, strict=True)
        )

    def _live_status(self, session_id: str) -> dict[str, JsonValue]:
        return {
            "session_id": session_id,
            "version": SIDECAR_VERSION,
            "exists": True,
            "state_version": 0,
            "pending_request_id": None,
            "pending_count": 0,
            "status": "running",
            "progress": self.engine.progress,
            "active_modules": [],
            "findings": [],
            "context_hash": None,
            "transcript_reconciled": False,
        }

    def _status(
        self,
        record: SessionRecord | None,
        session_id: str,
    ) -> dict[str, JsonValue]:
        if record is None:
            return {
                "session_id": session_id,
                "version": SIDECAR_VERSION,
                "exists": False,
                "state_version": 0,
                "pending_request_id": None,
                "last_result": {},
                "progress": self.engine.progress,
                "transcript_reconciled": session_id in self._reconciled_sessions,
                "economy_mode": self._economy_mode_active,
            }
        last_result = record.last_result or {}
        pending_value = last_result.get("pending_operations")
        pending = pending_value if isinstance(pending_value, dict) else {}
        pending_count = (
            len(_array(pending.get("operations"), "pending_operations"))
            if record.pending_request_id is not None
            else 0
        )
        status = _optional_string(last_result, "status", "idle")
        if record.pending_request_id is None and status == "needs_confirmation":
            status = "ok"
        return {
            "session_id": session_id,
            "version": SIDECAR_VERSION,
            "exists": True,
            "state_version": record.session.state.version,
            "pending_request_id": record.pending_request_id,
            "pending_count": pending_count,
            "status": status,
            "progress": self.engine.progress,
            "active_modules": last_result.get("active_modules", []),
            "findings": last_result.get("findings", []),
            "context_hash": record.context_hash,
            "transcript_reconciled": session_id in self._reconciled_sessions,
            "metrics": self.engine.metrics.to_dict(),
            "economy_mode": self._economy_mode_active,
            "insights": _turn_insights(last_result),
        }

    def _sync_transcript(
        self,
        session: Session,
        incoming: list[tuple[str, str]],
    ) -> bool:
        """Append any missing authoritative turns from the ST transcript.

        Returns True when the kernel history was stale and new turns had to be
        appended. Turns are never removed: the ledger references them, so the
        only safe reconciliation is additive.
        """
        existing = [(turn.role, turn.content) for turn in session.turns]
        existing_keys = [(role, _message_identity(content)) for role, content in existing]
        incoming_keys = [(role, _message_identity(content)) for role, content in incoming]
        if incoming_keys == existing_keys:
            return False
        shorter = len(incoming)
        if shorter and existing_keys[-shorter:] == incoming_keys:
            return False
        if len(incoming) > len(existing) and incoming_keys[: len(existing)] == existing_keys:
            suffix = incoming[len(existing) :]
        else:
            # The chat diverged (edit, deletion, swipe, or history produced
            # outside the kernel). Mark the replaced turns and append the
            # authoritative history verbatim.
            session.supersede_turns(
                [
                    (role, _message_identity(content))
                    for role, content in existing
                ]
            )
            suffix = incoming
        self._validate_pairs(suffix)
        for role, content in suffix:
            session.append_turn(cast(Literal["user", "assistant"], role), content)
        return True

    @staticmethod
    def _validate_pairs(items: list[tuple[str, str]]) -> None:
        if len(items) % 2:
            raise SidecarError("transcript_mismatch", "incomplete transcript exchange", 409)
        for index in range(0, len(items), 2):
            if items[index][0] != "user" or items[index + 1][0] != "assistant":
                raise SidecarError("transcript_mismatch", "invalid transcript ordering", 409)


_OPERATION_LABELS: dict[str, str] = {
    "set_time": "Time",
    "set_location": "Location",
    "set_summary": "Scene summary",
    "set_scene_tag": "Scene tag",
    "upsert_fact": "Fact",
    "upsert_belief": "Belief",
    "set_relationship": "Relationship",
    "set_resource": "Resource",
    "set_injury": "Injury",
    "open_thread": "Open thread",
    "resolve_thread": "Resolved thread",
    "record_event": "Event",
}


def _describe_operation(operation: Mapping[str, JsonValue]) -> str:
    """Render one state operation as a sentence a player can judge."""
    kind = str(operation.get("kind", ""))
    target = str(operation.get("target", "")).strip()
    value = str(operation.get("value", "")).strip()
    label = _OPERATION_LABELS.get(kind, kind or "Change")
    match kind:
        case "set_time" | "set_location":
            return f"{label}: {value}"
        case "set_summary":
            summary = value if len(value) <= 160 else f"{value[:157]}..."
            return f"{label} rewritten: {summary}"
        case "set_scene_tag":
            return f"{label} +{value}"
        case "upsert_fact":
            return f"{label} {target} = {value}"
        case "upsert_belief":
            return f"{label} {target} now holds: {value}"
        case "set_relationship":
            return f"{label} with {target}: {value}"
        case "set_injury":
            return f"{label} {target} — {value}"
        case "open_thread":
            return f"{label}: {target}"
        case "resolve_thread":
            return f"{label}: {target}"
        case "record_event":
            event = value if len(value) <= 160 else f"{value[:157]}..."
            return f"{label}: {event}"
    return f"{label} {target}: {value}" if target else f"{label}: {value}"


def _delta_preview(delta: Mapping[str, JsonValue]) -> list[JsonValue]:
    raw = delta.get("operations")
    operations = raw if isinstance(raw, list) else []
    preview: list[JsonValue] = []
    for item in operations:
        if not isinstance(item, dict):
            continue
        preview.append(
            {
                "kind": str(item.get("kind", "")),
                "label": _describe_operation(item),
                "impact": str(item.get("impact", "low")),
                "certainty": item.get("certainty", 0.0),
                "evidence": str(item.get("evidence", ""))[:200],
            }
        )
    return preview


def _turn_insights(last_result: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Everything the user needs to judge what the kernel decided this turn."""
    plan = last_result.get("plan")
    plan_goal = ""
    if isinstance(plan, dict):
        raw_goal = plan.get("goal")
        plan_goal = str(raw_goal) if isinstance(raw_goal, str) else ""
    reasons = last_result.get("module_reasons")
    module_map = reasons if isinstance(reasons, dict) else {}
    raw_modules = last_result.get("active_modules")
    module_names = raw_modules if isinstance(raw_modules, list) else []
    modules: list[JsonValue] = []
    for name in module_names:
        if not isinstance(name, str):
            continue
        modules.append({"id": name, "reason": str(module_map.get(name, ""))})
    raw_findings = last_result.get("findings")
    finding_items = raw_findings if isinstance(raw_findings, list) else []
    findings: list[JsonValue] = []
    for item in finding_items:
        if not isinstance(item, dict):
            continue
        findings.append(
            {
                "severity": str(item.get("severity", "warning")),
                "code": str(item.get("code", "")),
                "message": str(item.get("message", ""))[:240],
            }
        )
    applied = last_result.get("applied_operations")
    pending = last_result.get("pending_operations")
    return {
        "plan_goal": plan_goal,
        "modules": modules,
        "findings": findings,
        "repairs": last_result.get("repairs", 0),
        "applied": _delta_preview(applied) if isinstance(applied, dict) else [],
        "pending": _delta_preview(pending) if isinstance(pending, dict) else [],
    }


class AuthThrottle:
    """Blocks an address after repeated failed authentications.

    The sidecar is a loopback service protected only by an integration key, so
    unlimited guessing from another local process must not be possible.
    """

    def __init__(self, *, limit: int = 10, window_seconds: float = 60.0) -> None:
        self._limit = limit
        self._window = window_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def record_failure(self, address: str, now: float) -> int:
        with self._lock:
            bucket = self._failures.setdefault(address, [])
            cutoff = now - self._window
            bucket[:] = [stamp for stamp in bucket if stamp > cutoff]
            bucket.append(now)
            if len(bucket) > self._limit:
                # Keep the bucket bounded; the address is blocked anyway.
                del bucket[: len(bucket) - self._limit - 1]
            return len(bucket)

    def blocked(self, address: str, now: float) -> bool:
        with self._lock:
            bucket = self._failures.get(address)
            if not bucket:
                return False
            cutoff = now - self._window
            return len([stamp for stamp in bucket if stamp > cutoff]) > self._limit

    def reset(self, address: str) -> None:
        with self._lock:
            self._failures.pop(address, None)


class SidecarApplication:
    def __init__(self, config: SidecarConfig, service: KernelServiceProtocol) -> None:
        self.config = config
        self.service = service
        self.throttle = AuthThrottle()

    def chat_completion(
        self,
        payload: dict[str, JsonValue],
        *,
        on_render_delta: DeltaSink | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, JsonValue], bool]:
        model = _required_string(payload, "model")
        messages = _parse_messages(payload.get("messages"))
        stream = bool(payload.get("stream", False))
        if model.startswith(CONTROL_MODEL_PREFIX):
            stream = False
            control_payload = _extract_control_payload(messages)
            action = model.removeprefix(CONTROL_MODEL_PREFIX)
            control_result = self.service.control(action, control_payload)
            content = json.dumps(control_result, ensure_ascii=False, separators=(",", ":"))
            return (
                _completion_response(
                    content,
                    model,
                    {"prompt_tokens": 0, "completion_tokens": 0},
                ),
                stream,
            )
        envelope = _extract_generation_envelope(messages)
        context_prompt = _limit_text(
            "\n\n".join(
                message.content
                for message in messages
                if message.role == "system"
                and not message.content.startswith(ENVELOPE_PREFIX)
            ),
            self.config.max_context_chars,
        )
        turn = self.service.generate(
            envelope,
            messages,
            context_prompt,
            on_render_delta=on_render_delta,
            should_abort=should_abort,
        )
        return (
            _completion_response(turn.text, model, turn.usage),
            stream,
        )

    def authenticated(self, authorization: str | None) -> bool:
        if not authorization:
            return False
        scheme, separator, token = authorization.strip().partition(" ")
        if not separator or scheme.casefold() != "bearer":
            return False
        return hmac.compare_digest(token.strip(), self.config.integration_key)

    def host_header_allowed(self, host: str | None) -> bool:
        """Reject requests whose Host header is not a loopback authority.

        Without this check a browser page on any origin can reach the loopback
        sidecar through DNS rebinding, because the request would carry an
        attacker-controlled Host while still targeting 127.0.0.1.
        """
        if not host:
            return False
        value = host.strip()
        if not value:
            return False
        if value.startswith("["):
            closing = value.find("]")
            if closing < 0:
                return False
            hostname = value[1:closing]
            remainder = value[closing + 1 :]
            if remainder and not remainder.startswith(":"):
                return False
        else:
            hostname, separator, port = value.rpartition(":")
            if not separator or not port.isdigit():
                hostname = value
        if not hostname:
            return False
        return _is_loopback(hostname)


def _aborting_sink(
    sink: DeltaSink,
    should_abort: Callable[[], bool],
    aborted: set[str],
    session_id: str,
) -> DeltaSink:
    """Wrap a delta sink so a lost client aborts generation instead of retrying."""

    def guarded(text: str) -> None:
        if should_abort():
            aborted.add(session_id)
            raise ClientGoneError("client disconnected during generation")
        sink(text)

    return guarded


class _SseWriter:
    """Writes OpenAI-compatible SSE chunks for one response.

    The writer switches to a failed state after the first write error so a
    disconnected client (a Stop press in SillyTavern) is detected once and the
    generation can be abandoned instead of burning provider quota.
    """

    def __init__(self, handler: SidecarRequestHandler) -> None:
        self._handler = handler
        self._base: dict[str, JsonValue] = {}
        self._started = False
        self._failed = False
        self._sent_role = False

    def start(self, response: dict[str, JsonValue]) -> None:
        self._base = {
            "id": _optional_string(response, "id", new_id("chatcmpl")),
            "object": "chat.completion.chunk",
            "created": _optional_int(response, "created", int(time.time())),
            "model": _optional_string(response, "model", "roleplay-kernel"),
        }
        self._handler.send_response(200)
        self._handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self._handler.send_header("Cache-Control", "no-cache")
        self._handler.send_header("X-Accel-Buffering", "no")
        self._handler._send_cors_headers()
        self._handler.end_headers()
        self._started = True

    @property
    def failed(self) -> bool:
        return self._failed

    def write_role(self) -> None:
        if self._sent_role or self._failed:
            return
        self._write({"role": "assistant"})

    def write_delta(self, text: str) -> None:
        if not text or self._failed:
            return
        self.write_role()
        self._write({"content": text})

    def write_final(self, response: dict[str, JsonValue] | None = None) -> None:
        if self._failed:
            return
        self.write_role()
        if response is not None:
            usage = response.get("usage")
            if isinstance(usage, dict) and usage:
                chunk = dict(self._base)
                chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                chunk["usage"] = cast(JsonValue, usage)
                self._send(chunk)
                self._write({})
                return
        self._write({}, finish_reason="stop")
        self._finish()

    def write_error(self, code: str, message: str) -> None:
        if self._failed:
            return
        self._send(
            {
                "error": {"message": message, "type": "roleplay_kernel_error", "code": code}
            }
        )
        self._finish()

    def _write(
        self,
        delta: dict[str, JsonValue],
        *,
        finish_reason: str | None = None,
    ) -> None:
        chunk = dict(self._base)
        chunk["choices"] = [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ]
        self._send(chunk)

    def _send(self, payload: dict[str, JsonValue]) -> None:
        if self._failed:
            return
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            self._handler.wfile.write(f"data: {data}\n\n".encode())
            self._handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            self._failed = True

    def _finish(self) -> None:
        if self._failed:
            return
        try:
            self._handler.wfile.write(b"data: [DONE]\n\n")
            self._handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            self._failed = True


class SidecarRequestHandler(BaseHTTPRequestHandler):
    server_version = f"RoleplayKernel/{SIDECAR_VERSION}"
    application: SidecarApplication

    def _cors_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        if not origin:
            return None
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        if not _is_loopback(parsed.hostname):
            return None
        return origin

    def _send_cors_headers(self) -> None:
        origin = self._cors_origin()
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Max-Age", "600")

    def _client_address(self) -> str:
        address = self.client_address[0] if self.client_address else "unknown"
        return str(address)

    def _guard_request(self) -> bool:
        if not self.application.host_header_allowed(self.headers.get("Host")):
            self._send_error_json(421, "invalid_host", "Host header is not a loopback authority")
            return False
        if self.application.throttle.blocked(self._client_address(), time.monotonic()):
            self._send_error_json(
                429,
                "too_many_attempts",
                "too many failed authentication attempts",
            )
            return False
        return True

    def _require_auth(self) -> bool:
        if not self.application.authenticated(self.headers.get("Authorization")):
            self.application.throttle.record_failure(self._client_address(), time.monotonic())
            self._send_error_json(401, "unauthorized", "invalid integration key")
            return False
        self.application.throttle.reset(self._client_address())
        return True

    def do_OPTIONS(self) -> None:
        if not self._guard_request():
            return
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if not self._guard_request():
            return
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(200, self.application.service.health())
            return
        if path == "/v1/models":
            if not self._require_auth():
                return
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "roleplay-kernel",
                            "object": "model",
                            "owned_by": "roleplay-kernel",
                        }
                    ],
                },
            )
            return
        self._send_error_json(404, "not_found", "route not found")

    def do_POST(self) -> None:
        if not self._guard_request():
            return
        if urlsplit(self.path).path != "/v1/chat/completions":
            self._send_error_json(404, "not_found", "route not found")
            return
        if not self._require_auth():
            return
        try:
            payload = self._read_payload()
        except SidecarError as error:
            self._send_error_json(error.status, error.code, error.message)
            return
        if bool(payload.get("stream", False)) and not _required_string(
            payload, "model"
        ).startswith(CONTROL_MODEL_PREFIX):
            self._stream_generation(payload)
            return
        try:
            response, stream = self.application.chat_completion(payload)
        except SidecarError as error:
            self._send_error_json(error.status, error.code, error.message)
            return
        except ClientGoneError:
            return
        except ContextBudgetError:
            self._send_error_json(
                413,
                "context_budget_exceeded",
                "compiled prompt exceeds the configured context budget",
            )
            return
        except (StaleTurnResultError, UnknownPendingCommitError):
            self._send_error_json(409, "stale_session", "session state changed; refresh and retry")
            return
        except ProviderError as error:
            detail = str(error)[:500] or "upstream provider request failed"
            self._send_error_json(502, "upstream_error", detail)
            return
        except OSError:
            self._send_error_json(503, "storage_error", "session storage is unavailable")
            return
        except ValueError:
            # A generic 400 with no server-side detail is a dead end for
            # whoever has to fix it, so keep the original failure visible.
            traceback.print_exc()
            self._send_error_json(400, "invalid_request", "request data is invalid")
            return
        except RuntimeError:
            self._send_error_json(500, "internal_error", "sidecar operation failed")
            return
        except Exception:
            self._send_error_json(500, "internal_error", "unexpected sidecar failure")
            return
        if stream:
            self._send_sse(response)
        else:
            self._send_json(200, response)

    def _stream_generation(self, payload: dict[str, JsonValue]) -> None:
        """Stream the render stage to ST, then run the remaining stages.

        The text is delivered while the model writes it, so extraction and the
        critic no longer add to the time the user waits for the first token. If
        the client disconnects the turn is abandoned instead of spending the
        remaining provider calls.
        """
        writer = _SseWriter(self)
        writer.start({"model": _required_string(payload, "model")})
        try:
            response, _stream = self.application.chat_completion(
                payload,
                on_render_delta=writer.write_delta,
                should_abort=lambda: writer.failed,
            )
        except ClientGoneError:
            return
        except SidecarError as error:
            writer.write_error(error.code, error.message)
            return
        except ContextBudgetError:
            writer.write_error(
                "context_budget_exceeded",
                "compiled prompt exceeds the configured context budget",
            )
            return
        except (StaleTurnResultError, UnknownPendingCommitError):
            writer.write_error("stale_session", "session state changed; refresh and retry")
            return
        except ProviderError as error:
            writer.write_error("upstream_error", str(error)[:500] or "upstream request failed")
            return
        except OSError:
            writer.write_error("storage_error", "session storage is unavailable")
            return
        except ValueError:
            writer.write_error("invalid_request", "request data is invalid")
            return
        except Exception:
            writer.write_error("internal_error", "unexpected sidecar failure")
            return
        if writer.failed:
            return
        writer.write_final(response)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_payload(self) -> dict[str, JsonValue]:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise SidecarError(
                "unsupported_media_type",
                "Content-Type must be application/json",
                415,
            )
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise SidecarError("length_required", "Content-Length is required", 411)
        try:
            length = int(raw_length)
        except ValueError as error:
            raise SidecarError("invalid_length", "Content-Length is invalid") from error
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise SidecarError("request_too_large", "request body is too large", 413)
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SidecarError("invalid_json", "request body is invalid JSON") from error
        return _object(data, "request")

    def _send_json(self, status: int, payload: dict[str, JsonValue]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, code: str, message: str) -> None:
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": "roleplay_kernel_error",
                    "code": code,
                }
            },
        )

    def _send_sse(self, response: dict[str, JsonValue]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._send_cors_headers()
        self.end_headers()
        choices = _array(response.get("choices"), "choices")
        if not choices:
            return
        first = _object(choices[0], "choice")
        message = _object(first.get("message"), "message")
        text = _optional_string(message, "content", "")
        base: dict[str, JsonValue] = {
            "id": _optional_string(response, "id", new_id("chatcmpl")),
            "object": "chat.completion.chunk",
            "created": _optional_int(response, "created", int(time.time())),
            "model": _optional_string(response, "model", "roleplay-kernel"),
        }
        chunks = _stream_chunks(text)
        first_chunk = dict(base)
        first_chunk["choices"] = [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ]
        self._write_sse(first_chunk)
        for chunk in chunks:
            item = dict(base)
            item["choices"] = [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
            self._write_sse(item)
        final_chunk = dict(base)
        final_chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        self._write_sse(final_chunk)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _write_sse(self, payload: dict[str, JsonValue]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()


class SidecarHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


def create_server(
    config: SidecarConfig,
    service: KernelServiceProtocol | None = None,
) -> SidecarHTTPServer:
    application = SidecarApplication(config, service or SessionService(config))
    handler = type(
        "BoundSidecarRequestHandler",
        (SidecarRequestHandler,),
        {"application": application},
    )
    return SidecarHTTPServer((config.host, config.port), handler)


def run(config: SidecarConfig) -> None:
    server = create_server(config)
    address = server.server_address
    host_value = address[0]
    host = host_value.decode("utf-8") if isinstance(host_value, bytes) else host_value
    port = int(address[1])
    print(
        f"Roleplay Kernel sidecar {SIDECAR_VERSION} listening on http://{host}:{port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _load_config_file(path: Path | None) -> dict[str, JsonValue]:
    if path is None:
        return {}
    if not path.exists():
        raise ValueError(f"config file does not exist: {path}")
    return _read_json_object(path)


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error
    return _object(value, "root")


def _atomic_write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _atomic_write_bytes(path, data.encode("utf-8"))
    _restrict_permissions(path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _restrict_permissions(temporary)
    os.replace(temporary, path)


def _restrict_permissions(path: Path) -> None:
    if os.name == "nt":
        username = getpass.getuser()
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(F)"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            raise OSError(f"cannot restrict permissions for {path}")
        return
    path.chmod(0o600)


def _parse_messages(value: JsonValue | None) -> list[IncomingMessage]:
    messages: list[IncomingMessage] = []
    for index, raw_message in enumerate(_array(value, "messages")):
        data = _object(raw_message, f"messages[{index}]")
        role = _required_string(data, "role")
        content = _message_content(data.get("content"))
        messages.append(IncomingMessage(role=role, content=content))
    if not messages:
        raise SidecarError("missing_messages", "messages must not be empty")
    return messages


def _message_content(value: JsonValue | None) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _extract_generation_envelope(messages: list[IncomingMessage]) -> GenerationEnvelope:
    for message in reversed(messages):
        if message.role != "system" or not message.content.startswith(ENVELOPE_PREFIX):
            continue
        payload = _decode_prefixed_json(message.content, ENVELOPE_PREFIX, "envelope")
        return GenerationEnvelope.from_dict(payload)
    raise SidecarError("missing_envelope", "Roleplay Kernel envelope is missing")


def _extract_control_payload(messages: list[IncomingMessage]) -> dict[str, JsonValue]:
    for message in reversed(messages):
        if message.role == "system" and message.content.startswith(CONTROL_PREFIX):
            return _decode_prefixed_json(message.content, CONTROL_PREFIX, "control")
    raise SidecarError("missing_control_payload", "Roleplay Kernel control payload is missing")


def _decode_prefixed_json(
    content: str,
    prefix: str,
    name: str,
) -> dict[str, JsonValue]:
    encoded = content[len(prefix) :].strip()
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise SidecarError(f"invalid_{name}", f"{name} is invalid JSON") from error
    return _object(value, name)


def _transcript_exchange(
    transcript: tuple[TranscriptItem, ...],
) -> tuple[list[tuple[str, str]], str]:
    last_user_index = -1
    for index in range(len(transcript) - 1, -1, -1):
        if transcript[index].role == "user":
            last_user_index = index
            break
    if last_user_index < 0:
        raise SidecarError("missing_user_message", "transcript has no user message")
    pairs: list[tuple[str, str]] = []
    pending_user: tuple[str, str] | None = None
    for item in transcript[:last_user_index]:
        if item.role == "user":
            if pending_user is not None:
                raise SidecarError("invalid_transcript", "multiple user messages in an exchange")
            pending_user = (item.role, item.content)
        elif pending_user is not None:
            pairs.extend((pending_user, (item.role, item.content)))
            pending_user = None
    if pending_user is not None:
        raise SidecarError("invalid_transcript", "transcript ends with an unanswered user message")
    return pairs, transcript[last_user_index].content


def _complete_transcript_pairs(
    transcript: tuple[TranscriptItem, ...],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending_user: tuple[str, str] | None = None
    seen_user = False
    for item in transcript:
        if item.role == "user":
            seen_user = True
            if pending_user is not None:
                raise SidecarError("invalid_transcript", "multiple user messages in an exchange")
            pending_user = (item.role, item.content)
        elif not seen_user:
            continue
        elif pending_user is not None:
            pairs.extend((pending_user, (item.role, item.content)))
            pending_user = None
        else:
            raise SidecarError("invalid_transcript", "assistant message has no user predecessor")
    if pending_user is not None:
        raise SidecarError("invalid_transcript", "transcript ends with an unanswered user message")
    return pairs


def _limit_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n\n[Roleplay Kernel truncated external context]\n\n"
    available = max(0, limit - len(marker))
    if available == 0:
        return marker[:limit]
    head = int(available * 0.65)
    tail = available - head
    suffix = text[-tail:] if tail > 0 else ""
    return text[:head] + marker + suffix


def _completion_response(
    content: str,
    model: str,
    usage: dict[str, int],
) -> dict[str, JsonValue]:
    return {
        "id": new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": dict(usage),
    }


def _stream_chunks(text: str, size: int = 96) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def _completion_content(response: dict[str, JsonValue]) -> str:
    choices = _array(response.get("choices"), "choices")
    if not choices:
        return ""
    message = _object(_object(choices[0], "choice").get("message"), "message")
    return _optional_string(message, "content", "")


def _completion_usage(response: dict[str, JsonValue]) -> dict[str, int]:
    usage = _object(response.get("usage"), "usage")
    result: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
    return result


def _choice(
    data: dict[str, JsonValue],
    key: str,
    default: str,
    allowed: set[str],
) -> str:
    if key not in data:
        value = default
    else:
        raw_value = data[key]
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise SidecarError("invalid_envelope", f"{key} is not supported")
        value = raw_value.strip()
    if value not in allowed:
        raise SidecarError("invalid_envelope", f"{key} is not supported")
    return value


def _required_string(data: dict[str, JsonValue], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SidecarError("invalid_request", f"{key} must be a non-empty string")
    return value.strip()


def _config_string(
    value: JsonValue | None,
    default: str,
    *,
    allow_empty: bool = False,
) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("config string values must be strings")
    text = value.strip()
    if not text and not allow_empty:
        raise ValueError("config string values must be non-empty strings")
    return text


def _config_int(value: JsonValue | None, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("config integer values must be integers")
    return value


def _config_float(value: JsonValue | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("config numeric values must be numbers")
    if not math.isfinite(float(value)):
        raise ValueError("config numeric values must be finite")
    return float(value)


def _config_bool(value: JsonValue | None, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("config boolean values must be booleans")
    return value


def _optional_string(data: dict[str, JsonValue], key: str, default: str) -> str:
    value = data.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _required_int(data: dict[str, JsonValue], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SidecarError("invalid_request", f"{key} must be an integer")
    return value


def _optional_int(data: dict[str, JsonValue], key: str, default: int) -> int:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _optional_bool(data: dict[str, JsonValue], key: str, default: bool) -> bool:
    value = data.get(key)
    return value if isinstance(value, bool) else default


def _object(value: JsonValue | None, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise SidecarError("invalid_request", f"{name} must be an object")
    return value


def _array(value: JsonValue | None, name: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise SidecarError("invalid_request", f"{name} must be an array")
    return value


def _engine_mode(value: str) -> EngineMode:
    if value not in {"lite", "fast", "balanced", "strict"}:
        raise ValueError("mode must be lite, fast, balanced, or strict")
    return cast(EngineMode, value)


def _token_parameter(value: str) -> TokenParameter:
    if value not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("upstream_token_parameter is invalid")
    return cast(TokenParameter, value)


def _env_string(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    return value.strip() if value and value.strip() else default


def _env_int(env: Mapping[str, str], key: str, default: int, *, minimum: int) -> int:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error
    if parsed < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return parsed


def _env_float(
    env: Mapping[str, str],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{key} must be a number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite")
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return parsed


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean")


def _is_loopback(host: str) -> bool:
    normalized = host.casefold().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise SidecarError("invalid_session", "session_id contains invalid characters")


def _message_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\u200b", "")
    normalized = re.sub(r"[*_`~]", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _render_sampling(
    sampling: dict[str, JsonValue],
    minimum_tokens: int,
) -> dict[str, JsonValue]:
    result = dict(sampling)
    value = result.get("max_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        requested = min(max(value, minimum_tokens), 16384)
    else:
        requested = minimum_tokens
    result["max_tokens"] = requested
    return result


def _request_fingerprint(
    envelope: GenerationEnvelope,
    context_prompt: str,
) -> str:
    transcript = [
        [
            item.role,
            _message_identity(item.content),
            item.name,
            item.swipe_id,
        ]
        for item in envelope.transcript
    ]
    profile = envelope.upstream_profile.to_dict() if envelope.upstream_profile is not None else None
    payload = json.dumps(
        [
            envelope.generation_type,
            envelope.mode,
            envelope.request_delay_seconds,
            transcript,
            context_prompt.strip(),
            profile,
            envelope.sampling,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _digest(payload)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _main() -> None:
    parser = argparse.ArgumentParser(prog="roleplay-kernel-sidecar")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    run(SidecarConfig.load(args.config))


if __name__ == "__main__":
    _main()
