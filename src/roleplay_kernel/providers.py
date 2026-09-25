from __future__ import annotations

import ipaddress
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from email.message import Message
from typing import IO, Literal, Protocol

from .models import ChatMessage, Completion
from .utils import ProviderError

TokenParameter = Literal["max_tokens", "max_completion_tokens"]
MAX_RESPONSE_BYTES = 1_048_576
MAX_ERROR_RESPONSE_BYTES = 65_536
_PROTECTED_BODY_KEYS = frozenset({"model", "messages", "stream", "response_format"})
_ALLOWED_FINISH_REASONS = frozenset({"stop", "end_turn", "eos"})


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class ChatProvider(Protocol):
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        ...


@dataclass(frozen=True, slots=True)
class OpenAICompatibleProvider:
    model: str
    base_url: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = field(default="OPENAI_API_KEY", repr=False)
    timeout: float = 120.0
    token_parameter: TokenParameter = "max_tokens"
    default_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    extra_body: Mapping[str, object] = field(default_factory=dict, repr=False)
    endpoint_query: Mapping[str, str] = field(default_factory=dict, repr=False)
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be non-empty")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not isinstance(self.allow_insecure_http, bool):
            raise ValueError("allow_insecure_http must be a boolean")
        _validate_base_url(self.base_url, self.allow_insecure_http)
        _validate_timeout(self.timeout)
        if not isinstance(self.token_parameter, str) or self.token_parameter not in {
            "max_tokens",
            "max_completion_tokens",
        }:
            raise ValueError("unsupported token parameter")
        _validate_headers(self.default_headers)
        _validate_extra_body(self.extra_body)
        _validate_query(self.endpoint_query)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        if not messages:
            raise ValueError("messages must not be empty")
        if not isinstance(json_mode, bool):
            raise ValueError("json_mode must be a boolean")
        _validate_temperature(temperature)
        _validate_max_tokens(max_tokens)
        _validate_extra_body(self.extra_body)
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload[self.token_parameter] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)

        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        headers.update(self.default_headers)
        api_key = self.api_key or os.getenv(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            self._completion_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        opener = urllib.request.build_opener(_NoRedirectHandler())
        try:
            with opener.open(request, timeout=self.timeout) as response:
                status = response.status if hasattr(response, "status") else None
                if isinstance(status, int) and 300 <= status < 400:
                    raise ProviderError("provider redirect response rejected")
                raw_bytes = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw_bytes) > MAX_RESPONSE_BYTES:
                    raise ProviderError("provider response exceeds the size limit")
        except urllib.error.HTTPError as error:
            try:
                detail_bytes = error.read(MAX_ERROR_RESPONSE_BYTES + 1)
            finally:
                error.close()
            if 300 <= error.code < 400:
                detail = "redirect response rejected"
            elif len(detail_bytes) > MAX_ERROR_RESPONSE_BYTES:
                detail = "response exceeds the size limit"
            else:
                detail = detail_bytes.decode("utf-8", errors="replace")[:1000]
            raise ProviderError(f"provider returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProviderError(f"provider request failed: {error.reason}") from error
        except TimeoutError as error:
            raise ProviderError("provider request timed out") from error

        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProviderError("provider returned invalid UTF-8") from error
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProviderError("provider returned invalid JSON") from error
        if not isinstance(data, dict):
            raise ProviderError("provider response must be a JSON object")
        return self._parse_completion(data)

    def _completion_url(self) -> str:
        parsed = _validate_base_url(self.base_url, self.allow_insecure_http)
        path = parsed.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path = f"{path}/chat/completions" if path else "/chat/completions"
        query = urllib.parse.urlencode(self.endpoint_query)
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, ""))

    def _parse_completion(self, data: dict[str, object]) -> Completion:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("provider response does not contain choices")
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and finish_reason not in _ALLOWED_FINISH_REASONS:
                raise ProviderError(f"provider returned unsupported finish_reason={finish_reason}")
        first = choices[0]
        if not isinstance(first, dict):
            raise ProviderError("provider choice must be an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ProviderError("provider choice does not contain a message")
        content = _content_text(message.get("content"))
        if not content:
            raise ProviderError("provider returned empty content")
        usage = _usage(data.get("usage"))
        model = data.get("model")
        finish_reason = first.get("finish_reason")
        return Completion(
            content=content,
            model=model if isinstance(model, str) and model else self.model,
            usage=usage,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )


def _validate_base_url(base_url: str, allow_insecure_http: bool) -> urllib.parse.SplitResult:
    if not isinstance(base_url, str) or not base_url or base_url != base_url.strip():
        raise ValueError("base_url must be a non-empty URL")
    if any(character.isspace() for character in base_url):
        raise ValueError("base_url must not contain whitespace")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in base_url):
        raise ValueError("base_url must not contain control characters")
    if "\\" in base_url:
        raise ValueError("base_url must not contain backslashes")
    try:
        parsed = urllib.parse.urlsplit(base_url)
        hostname = parsed.hostname
        port = parsed.port
        username = parsed.username
        password = parsed.password
    except ValueError as error:
        raise ValueError("base_url is invalid") from error
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("base_url must use http or https")
    if (
        not parsed.netloc
        or not hostname
        or (port is not None and not 0 <= port <= 65535)
    ):
        raise ValueError("base_url must include a host")
    if username is not None or password is not None or "@" in parsed.netloc:
        raise ValueError("base_url must not contain userinfo")
    if parsed.netloc.endswith(":"):
        raise ValueError("base_url must include a valid port")
    if parsed.query or parsed.fragment or "?" in base_url or "#" in base_url:
        raise ValueError("base_url must not contain a query or fragment")
    if scheme == "http" and not _is_loopback_hostname(hostname) and not allow_insecure_http:
        raise ValueError("insecure HTTP requires loopback or allow_insecure_http=True")
    return urllib.parse.SplitResult(scheme, parsed.netloc, parsed.path, "", "")


def _is_loopback_hostname(hostname: str) -> bool:
    normalized = hostname.rstrip(".").casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_timeout(timeout: object) -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    try:
        finite = math.isfinite(timeout)
    except OverflowError as error:
        raise ValueError("timeout must be a positive finite number") from error
    if not finite or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")


def _validate_temperature(temperature: object) -> None:
    if temperature is None:
        return
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite number")
    try:
        finite = math.isfinite(temperature)
    except OverflowError as error:
        raise ValueError("temperature must be a finite number") from error
    if not finite or not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")


def _validate_max_tokens(max_tokens: object) -> None:
    if max_tokens is None:
        return
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")


def _validate_headers(headers: Mapping[str, str]) -> None:
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or "\r" in key
        or "\n" in key
        or "\r" in value
        or "\n" in value
        for key, value in headers.items()
    ):
        raise ValueError("default_headers contains an invalid header")


def _validate_query(query: Mapping[str, str]) -> None:
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or "\r" in key
        or "\n" in key
        or "\r" in value
        or "\n" in value
        for key, value in query.items()
    ):
        raise ValueError("endpoint_query contains an invalid key or value")


def _validate_extra_body(extra_body: Mapping[str, object]) -> None:
    if any(key in _PROTECTED_BODY_KEYS for key in extra_body):
        raise ValueError("extra_body cannot override model, messages, stream, or response_format")
    try:
        json.dumps(dict(extra_body), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("extra_body must be JSON serializable") from error


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _usage(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, int) and not isinstance(item, bool):
            result[key] = item
    return result
