from __future__ import annotations

import json
from typing import Any

JsonObject = dict[str, Any]


class ProviderError(RuntimeError):
    pass


def parse_json_object(content: str) -> JsonObject:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _scan_object(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def _scan_object(text: str) -> object:
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise ValueError("model response does not contain a JSON object")
    try:
        value, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise ValueError("model response contains a truncated or invalid JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("model response does not contain a JSON object")
    return value
