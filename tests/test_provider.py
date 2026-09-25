from __future__ import annotations

import io
import json
import math
import unittest
import urllib.error
import urllib.request
from email.message import Message
from typing import cast
from unittest.mock import patch

from roleplay_kernel.models import ChatMessage
from roleplay_kernel.providers import (
    MAX_ERROR_RESPONSE_BYTES,
    MAX_RESPONSE_BYTES,
    OpenAICompatibleProvider,
    STConnectionProfileProvider,
)
from roleplay_kernel.utils import ProviderError


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            return self.body
        return self.body[:size]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class FakeOpener:
    def __init__(self, result: FakeResponse | urllib.error.HTTPError) -> None:
        self.result = result
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float] = []

    def open(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if isinstance(self.result, urllib.error.HTTPError):
            raise self.result
        return self.result


class SequenceOpener:
    def __init__(self, results: list[FakeResponse]) -> None:
        self.results = list(results)
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("unexpected provider call")
        return self.results.pop(0)


def _response(content: str = "hello", finish_reason: object = "stop") -> bytes:
    value = {
        "model": "server-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "ignored": "x"},
    }
    return json.dumps(value).encode("utf-8")


class ProviderRequestTests(unittest.TestCase):
    def test_complete_builds_request_and_parses_response(self) -> None:
        response = FakeResponse(_response())
        opener = FakeOpener(response)
        provider = OpenAICompatibleProvider(
            model="client-model",
            base_url="https://api.example.test/v1/",
            api_key="api-secret",
            default_headers={"X-Test": "header-value"},
            timeout=3.5,
        )

        with patch("urllib.request.build_opener", return_value=opener):
            result = provider.complete(
                (ChatMessage(role="user", content="hello"),),
                temperature=0.25,
                max_tokens=17,
                json_mode=True,
                sampling={"top_p": 0.9, "frequency_penalty": 0.2},
            )

        self.assertEqual(result.content, "hello")
        self.assertEqual(result.model, "server-model")
        self.assertEqual(result.usage, {"prompt_tokens": 2, "completion_tokens": 3})
        self.assertEqual(opener.timeouts, [3.5])
        self.assertEqual(len(opener.requests), 1)
        request = opener.requests[0]
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer api-secret")
        self.assertEqual(request.get_header("X-test"), "header-value")
        body = request.data
        self.assertIsNotNone(body)
        if not isinstance(body, bytes):
            self.fail("request data is not bytes")
        payload = json.loads(body)
        self.assertEqual(payload["model"], "client-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(payload["max_tokens"], 17)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["frequency_penalty"], 0.2)
        self.assertIs(payload["stream"], False)

    def test_length_finish_reason_is_accepted(self) -> None:
        provider = OpenAICompatibleProvider(model="test", base_url="https://api.example.test/v1")
        opener = FakeOpener(FakeResponse(_response(finish_reason="length")))
        with patch("urllib.request.build_opener", return_value=opener):
            result = provider.complete((ChatMessage(role="user", content="hello"),))
        self.assertEqual(result.finish_reason, "length")

    def test_successful_response_read_is_bounded(self) -> None:
        response = FakeResponse(b"x" * (MAX_RESPONSE_BYTES + 1))
        opener = FakeOpener(response)
        provider = OpenAICompatibleProvider(model="test", base_url="https://api.example.test/v1")

        with (
            patch("urllib.request.build_opener", return_value=opener),
            self.assertRaisesRegex(ProviderError, "size limit"),
        ):
            provider.complete((ChatMessage(role="user", content="hello"),))

        self.assertEqual(response.read_sizes, [MAX_RESPONSE_BYTES + 1])

    def test_redirect_response_is_rejected_without_reading(self) -> None:
        response = FakeResponse(_response(), status=302)
        opener = FakeOpener(response)
        provider = OpenAICompatibleProvider(model="test", base_url="https://api.example.test/v1")

        with (
            patch("urllib.request.build_opener", return_value=opener),
            self.assertRaisesRegex(ProviderError, "redirect"),
        ):
            provider.complete((ChatMessage(role="user", content="hello"),))

        self.assertEqual(response.read_sizes, [])

    def test_http_error_read_is_bounded(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.example.test/v1/chat/completions",
            500,
            "Server Error",
            Message(),
            io.BytesIO(b"x" * (MAX_ERROR_RESPONSE_BYTES + 1)),
        )
        opener = FakeOpener(error)
        provider = OpenAICompatibleProvider(model="test", base_url="https://api.example.test/v1")

        with (
            patch("urllib.request.build_opener", return_value=opener),
            self.assertRaisesRegex(ProviderError, "size limit"),
        ):
            provider.complete((ChatMessage(role="user", content="hello"),))

    def test_invalid_json_and_non_object_are_rejected(self) -> None:
        provider = OpenAICompatibleProvider(model="test", base_url="https://api.example.test/v1")
        for body, message in (
            (b"not-json", "invalid JSON"),
            (b"[]", "JSON object"),
        ):
            with self.subTest(body=body):
                opener = FakeOpener(FakeResponse(body))
                with (
                    patch("urllib.request.build_opener", return_value=opener),
                    self.assertRaisesRegex(ProviderError, message),
                ):
                    provider.complete((ChatMessage(role="user", content="hello"),))


class STProfileProviderTests(unittest.TestCase):
    def test_profile_request_uses_st_csrf_and_secret_id(self) -> None:
        opener = SequenceOpener(
            [
                FakeResponse(json.dumps({"token": "csrf-token"}).encode()),
                FakeResponse(_response("profiled")),
            ]
        )
        with patch("urllib.request.build_opener", return_value=opener):
            provider = STConnectionProfileProvider(
                st_base_url="http://127.0.0.1:8000",
                source="custom",
                api_url="http://127.0.0.1:9000/v1",
                model="profile-model",
                secret_id="secret-uuid",
            )
            result = provider.complete(
                (ChatMessage(role="user", content="hello"),),
                max_tokens=32,
                json_mode=True,
            )

        self.assertEqual(result.content, "profiled")
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(opener.requests[0].full_url, "http://127.0.0.1:8000/csrf-token")
        self.assertEqual(
            opener.requests[1].full_url,
            "http://127.0.0.1:8000/api/backends/chat-completions/generate",
        )
        self.assertEqual(opener.requests[1].get_header("X-csrf-token"), "csrf-token")
        body = opener.requests[1].data
        self.assertIsInstance(body, bytes)
        if not isinstance(body, bytes):
            self.fail("request data is not bytes")
        payload = json.loads(body)
        self.assertEqual(payload["chat_completion_source"], "custom")
        self.assertEqual(payload["custom_url"], "http://127.0.0.1:9000/v1")
        self.assertEqual(payload["secret_id"], "secret-uuid")
        self.assertNotIn("reverse_proxy", payload)
        self.assertNotIn("proxy_password", payload)


class ProviderValidationTests(unittest.TestCase):
    def test_repr_hides_secret_fields(self) -> None:
        provider = OpenAICompatibleProvider(
            model="test",
            base_url="https://api.example.test/v1",
            api_key="api-secret",
            default_headers={"X-Api-Key": "header-secret"},
            extra_body={"private": "body-secret"},
            endpoint_query={"access_token": "query-secret"},
        )

        rendered = repr(provider)

        self.assertNotIn("api-secret", rendered)
        self.assertNotIn("header-secret", rendered)
        self.assertNotIn("body-secret", rendered)
        self.assertNotIn("query-secret", rendered)
        self.assertNotIn("api_key", rendered)
        self.assertNotIn("default_headers=", rendered)
        self.assertNotIn("extra_body=", rendered)

    def test_reserved_extra_body_fields_are_rejected(self) -> None:
        for key in ("model", "messages", "stream", "response_format"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                OpenAICompatibleProvider(
                    model="test",
                    base_url="https://api.example.test/v1",
                    extra_body={key: "override"},
                )

    def test_request_parameters_are_validated(self) -> None:
        provider = OpenAICompatibleProvider(model="test", base_url="https://api.example.test/v1")
        message = (ChatMessage(role="user", content="hello"),)

        with self.assertRaises(ValueError):
            provider.complete(())
        for temperature in (math.nan, math.inf, -0.1, 2.1, True):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                provider.complete(message, temperature=temperature)
        for max_tokens in (0, -1, 1.5, True):
            with self.subTest(max_tokens=max_tokens), self.assertRaises(ValueError):
                provider.complete(message, max_tokens=cast(int | None, max_tokens))

    def test_headers_and_extra_body_must_be_serializable(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider(
                model="test",
                base_url="https://api.example.test/v1",
                default_headers={"X-Test": "value\ninjected"},
            )
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider(
                model="test",
                base_url="https://api.example.test/v1",
                extra_body={"value": object()},
            )
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider(
                model="test",
                base_url="https://api.example.test/v1",
                extra_body={"value": math.nan},
            )

    def test_timeout_must_be_positive_and_finite(self) -> None:
        for timeout in (0.0, -1.0, math.nan, math.inf, True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                OpenAICompatibleProvider(
                    model="test",
                    base_url="https://api.example.test/v1",
                    timeout=timeout,
                )

    def test_finish_reasons_are_allowlisted(self) -> None:
        provider = OpenAICompatibleProvider(model="test", base_url="https://api.example.test/v1")
        for finish_reason in ("content_filter", "error", "tool_calls", 7):
            with self.subTest(finish_reason=finish_reason):
                opener = FakeOpener(FakeResponse(_response(finish_reason=finish_reason)))
                with (
                    patch("urllib.request.build_opener", return_value=opener),
                    self.assertRaisesRegex(ProviderError, "finish_reason"),
                ):
                    provider.complete((ChatMessage(role="user", content="hello"),))


class ProviderURLTests(unittest.TestCase):
    def test_completion_endpoint_is_built_without_duplicate_suffix(self) -> None:
        cases = (
            ("https://api.example.test", "https://api.example.test/chat/completions"),
            ("https://api.example.test/", "https://api.example.test/chat/completions"),
            ("https://api.example.test/v1", "https://api.example.test/v1/chat/completions"),
            (
                "https://api.example.test/v1/",
                "https://api.example.test/v1/chat/completions",
            ),
            (
                "https://api.example.test/v1/chat/completions",
                "https://api.example.test/v1/chat/completions",
            ),
        )
        for base_url, expected in cases:
            with self.subTest(base_url=base_url):
                provider = OpenAICompatibleProvider(model="test", base_url=base_url)
                self.assertEqual(provider._completion_url(), expected)

    def test_endpoint_query_is_preserved_and_validated(self) -> None:
        provider = OpenAICompatibleProvider(
            model="test",
            base_url="https://api.example.test/v1",
            endpoint_query={"api-version": "2026-01-01", "region": "west europe"},
        )

        self.assertEqual(
            provider._completion_url(),
            "https://api.example.test/v1/chat/completions"
            "?api-version=2026-01-01&region=west+europe",
        )
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider(
                model="test",
                base_url="https://api.example.test/v1",
                endpoint_query={"bad": "value\ninjected"},
            )

    def test_url_rejects_unsafe_forms(self) -> None:
        for base_url in (
            "ftp://api.example.test/v1",
            "file:///tmp/provider",
            "https://user:password@api.example.test/v1",
            "https://api.example.test/v1?key=value",
            "https://api.example.test/v1#fragment",
            "https://api.example.test:invalid/v1",
            "https://api.example.test /v1",
        ):
            with self.subTest(base_url=base_url), self.assertRaises(ValueError):
                OpenAICompatibleProvider(model="test", base_url=base_url)

    def test_http_is_limited_to_loopback_without_explicit_opt_in(self) -> None:
        for base_url in (
            "http://localhost:8080/v1",
            "http://127.0.0.1/v1",
            "http://[::1]/v1",
        ):
            with self.subTest(base_url=base_url):
                provider = OpenAICompatibleProvider(model="test", base_url=base_url)
                self.assertTrue(provider.base_url.startswith("http://"))

        with self.assertRaises(ValueError):
            OpenAICompatibleProvider(model="test", base_url="http://api.example.test/v1")

        provider = OpenAICompatibleProvider(
            model="test",
            base_url="http://api.example.test/v1",
            allow_insecure_http=True,
        )
        self.assertEqual(
            provider._completion_url(),
            "http://api.example.test/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
