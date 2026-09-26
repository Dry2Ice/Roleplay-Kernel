from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from roleplay_kernel import Completion
from roleplay_kernel.engine import ClientGoneError, EngineMode
from roleplay_kernel.models import ChatMessage, JsonValue
from roleplay_kernel.providers import STConnectionProfileProvider
from roleplay_kernel.sidecar import (
    CONTROL_PREFIX,
    ENVELOPE_PREFIX,
    PROTOCOL_VERSION,
    GenerationEnvelope,
    IncomingMessage,
    SessionService,
    SidecarConfig,
    SidecarError,
    STProfileConfig,
    TranscriptItem,
    create_server,
)

TEST_INTEGRATION_KEY = "integration-key-for-tests-32chars-long"


class ScriptedProvider:
    def __init__(self, responses: list[Completion]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[ChatMessage, ...]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        sampling: Mapping[str, object] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Completion:
        self.calls.append(tuple(messages))
        if not self.responses:
            raise AssertionError("unexpected provider call")
        response = self.responses.pop(0)
        if on_delta is not None and not json_mode:
            midpoint = max(1, len(response.content) // 2)
            for part in (response.content[:midpoint], response.content[midpoint:]):
                if part:
                    on_delta(part)
        return response


class SidecarTests(unittest.TestCase):
    def test_generation_persists_pending_commit_and_restores_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            provider = _pending_provider()
            service = SessionService(config, provider=provider)
            messages = _messages(session_id="chat_one", user_text="Я вхожу в комнату")

            turn = service.generate(
                _envelope("chat_one", "Я вхожу в комнату"),
                messages,
                "Character: Aria",
            )

            self.assertEqual(turn.status, "needs_confirmation")
            self.assertIsNotNone(turn.pending_request_id)
            self.assertIn("EXTERNAL_CONTEXT_DATA", provider.calls[0][1].content)
            status = service.control("status", {"session_id": "chat_one"})
            self.assertEqual(status["pending_count"], 1)

            committed = service.control("commit", {"session_id": "chat_one"})

            self.assertEqual(committed["pending_request_id"], None)
            self.assertEqual(committed["state_version"], 1)
            repeated_commit = service.control("commit", {"session_id": "chat_one"})
            self.assertEqual(repeated_commit["pending_request_id"], None)
            state = service.control("state", {"session_id": "chat_one"})
            state_data = state["state"]
            self.assertIsInstance(state_data, dict)
            if not isinstance(state_data, dict):
                self.fail("state must be an object")
            injuries = state_data["injuries"]
            self.assertIsInstance(injuries, dict)
            if not isinstance(injuries, dict):
                self.fail("injuries must be an object")
            self.assertEqual(injuries["player.knee"], "горящая рана")

            resumed = SessionService(config, provider=ScriptedProvider([]))
            restored = resumed.control("status", {"session_id": "chat_one"})

            self.assertTrue(restored["exists"])
            self.assertEqual(restored["pending_count"], 0)
            self.assertEqual(restored["state_version"], 1)

    def test_pending_delta_survives_sidecar_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            first_service = SessionService(config, provider=_pending_provider())
            self.assertEqual(len((Path(temporary) / "integrity.key").read_bytes()), 32)
            first_service.generate(
                _envelope("chat_restart", "Я вхожу в комнату"),
                _messages("chat_restart", "Я вхожу в комнату"),
                "Character: Aria",
            )

            resumed = SessionService(config, provider=ScriptedProvider([]))
            status = resumed.control("status", {"session_id": "chat_restart"})
            self.assertEqual(status["pending_count"], 1)
            committed = resumed.control("commit", {"session_id": "chat_restart"})

            self.assertEqual(committed["pending_request_id"], None)
            self.assertEqual(committed["state_version"], 1)

    def test_pending_delta_can_be_rejected_before_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            provider = _pending_provider()
            provider.responses.extend(_empty_turn_responses())
            service = SessionService(config, provider=provider)

            service.generate(
                _envelope("chat_two", "Я вхожу в комнату"),
                _messages("chat_two", "Я вхожу в комнату"),
                "Character: Aria",
            )
            rejected = service.control("reject", {"session_id": "chat_two"})

            self.assertEqual(rejected["pending_request_id"], None)
            state = service.control("state", {"session_id": "chat_two"})
            state_data = state["state"]
            self.assertIsInstance(state_data, dict)
            if not isinstance(state_data, dict):
                self.fail("state must be an object")
            self.assertEqual(state_data["injuries"], {})

            next_messages = _messages(
                "chat_two",
                "Я продолжаю идти",
                include_previous=True,
            )
            next_turn = service.generate(
                _envelope(
                    "chat_two",
                    "Я продолжаю идти",
                    include_previous=True,
                    include_rejected_previous=True,
                ),
                next_messages,
                "Character: Aria",
            )

            self.assertEqual(next_turn.pending_request_id, None)
            self.assertEqual(next_turn.status, "ok")

    def test_pending_delta_blocks_next_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=_pending_provider(),
            )
            service.generate(
                _envelope("chat_three", "Я вхожу в комнату"),
                _messages("chat_three", "Я вхожу в комнату"),
                "Character: Aria",
            )

            with self.assertRaisesRegex(SidecarError, "approved or rejected"):
                service.generate(
                    _envelope("chat_three", "Я продолжаю идти", include_previous=True),
                    _messages(
                        "chat_three",
                        "Я продолжаю идти",
                        include_previous=True,
                    ),
                    "Character: Aria",
                )

    def test_profile_envelope_round_trip_keeps_secret_reference(self) -> None:
        envelope = _envelope("chat_profile", "Я вхожу в комнату")
        profile = STProfileConfig(
            profile_id="profile-1",
            st_base_url="http://127.0.0.1:8000",
            source="custom",
            api_url="http://127.0.0.1:9000/v1",
            model="profile-model",
            secret_id="secret-uuid",
        )
        data = envelope.to_dict()
        profile_data = profile.to_dict()
        data["upstream_profile"] = profile_data

        parsed = GenerationEnvelope.from_dict(data)

        self.assertEqual(parsed.upstream_profile, profile)
        self.assertNotIn("api_key", profile_data)

    def test_profile_provider_switches_without_reading_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scripted = ScriptedProvider([])
            service = SessionService(_config(Path(temporary)), provider=scripted)
            profile = STProfileConfig(
                profile_id="profile-1",
                st_base_url="http://127.0.0.1:8000",
                source="openai",
                api_url="",
                model="profile-model",
                secret_id="secret-uuid",
            )

            service._apply_upstream_profile(profile)
            self.assertIsInstance(service.engine.provider, STConnectionProfileProvider)
            service._apply_upstream_profile(None)
            self.assertIs(service.engine.provider, scripted)

    def test_status_detects_transcript_drift_and_ignores_post_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=ScriptedProvider(_empty_turn_responses()),
            )
            service.generate(
                _envelope("chat_status", "Я вхожу в комнату"),
                _messages("chat_status", "Я вхожу в комнату"),
                "Character: Aria",
            )
            transcript = [
                {"index": 0, "role": "assistant", "content": "Добро пожаловать."},
                {"index": 1, "role": "user", "content": "Я вхожу в комнату"},
                {"index": 2, "role": "assistant", "content": "**Вокруг** была пустая станция."},
            ]
            status = service.control(
                "status",
                {"session_id": "chat_status", "transcript": cast(JsonValue, transcript)},
            )
            self.assertTrue(status["transcript_matches"])

            transcript[1]["content"] = "Другая реплика"
            drifted = service.control(
                "status",
                {"session_id": "chat_status", "transcript": cast(JsonValue, transcript)},
            )
            self.assertFalse(drifted["transcript_matches"])

    def test_transcript_drift_resyncs_instead_of_failing_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=ScriptedProvider(_empty_turn_responses() * 3),
            )
            first = _envelope_with_history(
                "chat_drift",
                [("user", "Первый ход"), ("assistant", "Колено горело.")],
            )
            service.generate(first, _messages_for_envelope(first), "Character: Aria")
            second = _envelope_with_history(
                "chat_drift",
                [
                    ("user", "Первый ход"),
                    ("assistant", "Колено горело."),
                    ("user", "Второй ход"),
                    ("assistant", "Вокруг была пустая станция."),
                ],
            )
            service.generate(second, _messages_for_envelope(second), "Character: Aria")
            drifted = _envelope_with_history(
                "chat_drift",
                [
                    ("user", "Первый ход"),
                    ("assistant", "Совершенно другой ответ"),
                    ("user", "Второй ход"),
                    ("assistant", "Вокруг была пустая станция."),
                ],
            )
            turn = service.generate(
                drifted,
                _messages_for_envelope(drifted),
                "Character: Aria",
            )

            self.assertEqual(turn.status, "ok")
            status = service.control("status", {"session_id": "chat_drift"})
            self.assertTrue(status["transcript_reconciled"])

    def test_transcript_truncation_resyncs_to_shorter_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=ScriptedProvider(_empty_turn_responses() * 3),
            )
            first = _envelope_with_history(
                "chat_trunc",
                [("user", "Первый ход"), ("assistant", "Колено горело.")],
            )
            service.generate(first, _messages_for_envelope(first), "Character: Aria")
            second = _envelope_with_history(
                "chat_trunc",
                [
                    ("user", "Первый ход"),
                    ("assistant", "Колено горело."),
                    ("user", "Второй ход"),
                    ("assistant", "Вокруг была пустая станция."),
                ],
            )
            service.generate(second, _messages_for_envelope(second), "Character: Aria")
            truncated = _envelope_with_history(
                "chat_trunc",
                [("user", "Первый ход"), ("assistant", "Колено горело.")],
            )
            turn = service.generate(
                truncated,
                _messages_for_envelope(truncated),
                "Character: Aria",
            )
            self.assertEqual(turn.status, "ok")
            self.assertTrue(
                service.control("status", {"session_id": "chat_trunc"})[
                    "transcript_reconciled"
                ]
            )

    def test_completely_unrelated_transcript_is_reconciled_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=ScriptedProvider(_empty_turn_responses() * 2),
            )
            first = _envelope_with_history(
                "chat_alien",
                [("user", "Первый ход"), ("assistant", "Колено горело.")],
            )
            service.generate(first, _messages_for_envelope(first), "Character: Aria")
            alien = _envelope_with_history(
                "chat_alien",
                [("user", "Совсем другой вопрос"), ("assistant", "Иной ответ")],
            )
            turn = service.generate(
                alien,
                _messages_for_envelope(alien),
                "Character: Aria",
            )
            self.assertEqual(turn.status, "ok")
            status = service.control("status", {"session_id": "chat_alien"})
            self.assertTrue(status["transcript_reconciled"])

    def test_reconciled_history_persists_and_stays_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            provider = ScriptedProvider(_empty_turn_responses() * 3)
            service = SessionService(config, provider=provider)
            first = _envelope_with_history(
                "chat_persist",
                [("user", "Первый ход"), ("assistant", "Колено горело.")],
            )
            service.generate(first, _messages_for_envelope(first), "Character: Aria")
            alien = _envelope_with_history(
                "chat_persist",
                [
                    ("user", "Совсем другой вопрос"),
                    ("assistant", "Вокруг была пустая станция."),
                    ("user", "Новый ход"),
                    ("assistant", "Вокруг была пустая станция."),
                ],
            )
            service.generate(alien, _messages_for_envelope(alien), "Character: Aria")

            restarted = SessionService(config, provider=ScriptedProvider(_empty_turn_responses()))
            status = restarted.control("status", {"session_id": "chat_persist"})
            self.assertTrue(status["exists"])
            self.assertTrue(
                restarted.control(
                    "status",
                    {
                        "session_id": "chat_persist",
                        "transcript": cast(
                            JsonValue,
                            [
                                {"index": index, "role": role, "content": content}
                                for index, (role, content) in enumerate(
                                    [
                                        ("user", "Совсем другой вопрос"),
                                        ("assistant", "Вокруг была пустая станция."),
                                        ("user", "Новый ход"),
                                        ("assistant", "Вокруг была пустая станция."),
                                    ]
                                )
                            ],
                        ),
                    },
                )["transcript_matches"]
            )

    def test_language_setting_updates_an_existing_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provider = ScriptedProvider(_empty_turn_responses() * 2)
            service = SessionService(_config(Path(temporary)), provider=provider)
            russian = _envelope_with_history(
                "chat_lang",
                [("user", "Привет"), ("assistant", "Колено горело.")],
            )
            service.generate(russian, _messages_for_envelope(russian), "Character: Aria")
            english = _envelope_with_history(
                "chat_lang",
                [
                    ("user", "Привет"),
                    ("assistant", "Колено горело."),
                    ("user", "Hello there"),
                    ("assistant", "Колено горело."),
                ],
                language="en",
            )
            service.generate(english, _messages_for_envelope(english), "Character: Aria")
            state = service.control("state", {"session_id": "chat_lang"})
            self.assertIsInstance(state["state"], dict)
            if not isinstance(state["state"], dict):
                self.fail("state must be an object")
            self.assertEqual(state["state"]["language"], "en")
            rendered = "\n".join(
                message.content
                for call in provider.calls
                for message in call
            )
            self.assertIn("Write the whole post in English now.", rendered)

    def test_normal_request_is_idempotent_while_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provider = _pending_provider()
            service = SessionService(_config(Path(temporary)), provider=provider)
            envelope = _envelope("chat_retry", "Я вхожу в комнату")

            first = service.generate(
                envelope,
                _messages("chat_retry", "Я вхожу в комнату"),
                "Character: Aria",
            )
            call_count = len(provider.calls)
            second = service.generate(
                envelope,
                _messages("chat_retry", "Другой raw prompt"),
                "Character: Aria",
            )

            self.assertEqual(second.text, first.text)
            self.assertEqual(second.pending_request_id, first.pending_request_id)
            self.assertEqual(len(provider.calls), call_count)
            recovered = service.generate(
                _envelope("chat_retry", "Я вхожу в комнату"),
                _messages("chat_retry", "Я вхожу в комнату"),
                "Character: Aria",
            )
            self.assertEqual(recovered.text, first.text)
            self.assertEqual(len(provider.calls), call_count)
            with self.assertRaisesRegex(SidecarError, "different prompt"):
                service.generate(
                    _envelope(
                        "chat_retry",
                        "Другой user input",
                        request_key=envelope.request_key,
                    ),
                    _messages("chat_retry", "Другой user input"),
                    "Character: Aria",
                )

    def test_regenerate_rewinds_pending_branch_before_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provider = _pending_provider()
            provider.responses.extend(_empty_turn_responses())
            service = SessionService(_config(Path(temporary)), provider=provider)
            service.generate(
                _envelope("chat_regenerate", "Первый вариант"),
                _messages("chat_regenerate", "Первый вариант"),
                "Character: Aria",
            )

            turn = service.generate(
                _envelope(
                    "chat_regenerate",
                    "Первый вариант",
                    generation_type="regenerate",
                ),
                _messages("chat_regenerate", "Другой raw prompt"),
                "Character: Aria",
            )

            self.assertEqual(turn.pending_request_id, None)
            self.assertEqual(turn.status, "ok")

    def test_client_disconnect_stops_the_remaining_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provider = _AbortingProvider()
            service = SessionService(_config(Path(temporary)), provider=provider)
            envelope = _envelope_with_history(
                "chat_gone",
                [("user", "Я вхожу в комнату"), ("assistant", "Колено горело.")],
                mode="lite",
            )
            calls: list[str] = []

            def abort_after_first_piece() -> bool:
                return len(calls) >= 1

            with self.assertRaisesRegex(ClientGoneError, "disconnected"):
                service.generate(
                    envelope,
                    _messages_for_envelope(envelope),
                    "Character: Aria",
                    on_render_delta=calls.append,
                    should_abort=abort_after_first_piece,
                )
            self.assertEqual(provider.calls, 1)
            self.assertIn("chat_gone", service._aborted_sessions)

    def test_rate_limited_profile_drops_the_turn_into_lite_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=ScriptedProvider(_empty_turn_responses()),
            )
            provider = STConnectionProfileProvider(
                st_base_url="http://127.0.0.1:8000",
                source="custom",
                api_url="https://api.example.test/v1",
                model="test-model",
                secret_id="secret",
            )
            service._profile_provider = provider

            object.__setattr__(provider, "rate_limited", True)
            service._apply_mode("balanced")
            self.assertEqual(service.engine.config.mode, "lite")
            self.assertTrue(service._economy_mode_active)
            status = service.control("status", {"session_id": "chat_economy"})
            self.assertTrue(status["economy_mode"])

            object.__setattr__(provider, "rate_limited", False)
            service._apply_mode("balanced")
            self.assertEqual(service.engine.config.mode, "balanced")
            self.assertFalse(service._economy_mode_active)

    def test_operations_are_described_in_plain_language(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=ScriptedProvider(_empty_turn_responses()),
            )
            service.generate(
                _envelope("chat_plain", "Первый ход"),
                _messages("chat_plain", "Первый ход"),
                "Character: Aria",
            )
            status = service.control("status", {"session_id": "chat_plain"})
            insights = status["insights"]
            self.assertIsInstance(insights, dict)
            if not isinstance(insights, dict):
                self.fail("insights must be an object")
            self.assertIn("plan_goal", insights)
            self.assertIsInstance(insights["modules"], list)
            self.assertIsInstance(insights["pending"], list)
            for item in cast(list[JsonValue], insights["pending"]):
                if not isinstance(item, dict):
                    continue
                self.assertIsInstance(item.get("label"), str)
                self.assertIn("impact", item)

    def test_pending_operations_read_as_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=_pending_provider(),
            )
            service.generate(
                _envelope("chat_pending_words", "Первый ход"),
                _messages("chat_pending_words", "Первый ход"),
                "Character: Aria",
            )
            status = service.control("status", {"session_id": "chat_pending_words"})
            insights = cast(dict[str, JsonValue], status["insights"])
            pending = cast(list[JsonValue], insights["pending"])
            self.assertTrue(pending, "the fixture produces a pending delta")
            labels = [
                str(item["label"])
                for item in pending
                if isinstance(item, dict)
            ]
            self.assertTrue(labels)
            self.assertTrue(
                all(label and label[0].isupper() for label in labels),
                f"labels must read as sentences: {labels}",
            )

    def test_streamed_turn_reports_metrics_in_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            provider = ScriptedProvider(_empty_turn_responses())
            service = SessionService(config, provider=provider)
            server = create_server(config, service)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host_value, port_value = server.server_address[:2]
            host = host_value.decode("utf-8") if isinstance(host_value, bytes) else host_value
            port = int(port_value)
            try:
                stream_messages = _message_dicts(_messages("chat_metrics", "Я осматриваюсь"))
                body = _request_text(
                    f"http://{host}:{port}/v1/chat/completions",
                    TEST_INTEGRATION_KEY,
                    {
                        "model": "roleplay-kernel",
                        "messages": cast(JsonValue, stream_messages),
                        "stream": True,
                    },
                )
                self.assertIn("data: [DONE]", body)
                self.assertEqual(
                    "".join(_sse_text_deltas(body)),
                    "Вокруг была пустая станция.",
                )
                status = service.control("status", {"session_id": "chat_metrics"})
                metrics = status["metrics"]
                self.assertIsInstance(metrics, dict)
                if not isinstance(metrics, dict):
                    self.fail("metrics must be an object")
                self.assertEqual(metrics["provider_calls"], 4)
                self.assertTrue(metrics["streamed"])
                self.assertIsNotNone(metrics["first_token_seconds"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_diagnostics_reports_config_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=ScriptedProvider(_empty_turn_responses()),
            )
            report = service.control("diagnostics", {})

            sidecar = report["sidecar"]
            self.assertIsInstance(sidecar, dict)
            if not isinstance(sidecar, dict):
                self.fail("sidecar block must be an object")
            self.assertEqual(sidecar["version"], "0.2.0")
            self.assertIn("python", sidecar)
            config = report["config"]
            self.assertIsInstance(config, dict)
            if not isinstance(config, dict):
                self.fail("config must be an object")
            self.assertEqual(config["turn_budget_seconds"], 300.0)
            self.assertNotIn(TEST_INTEGRATION_KEY, json.dumps(report, default=str))

    def test_self_test_lists_checks_and_skips_upstream_without_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = SessionService(
                _config(Path(temporary)),
                provider=ScriptedProvider(_empty_turn_responses()),
            )
            report = service.control("self_test", {})

            names = {
                check["name"]
                for check in cast(list[JsonValue], report["checks"])
                if isinstance(check, dict)
            }
            self.assertIn("state_dir", names)
            self.assertIn("integration_key", names)
            self.assertNotIn("upstream_reachable", names)
            self.assertFalse(report["ok"])

    def test_non_loopback_host_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            server = create_server(config, SessionService(config, provider=_pending_provider()))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host_value, port_value = server.server_address[:2]
            host = host_value.decode("utf-8") if isinstance(host_value, bytes) else host_value
            port = int(port_value)
            try:
                request = urllib.request.Request(
                    f"http://{host}:{port}/v1/chat/completions",
                    data=b"{}",
                    method="POST",
                    headers={"Host": "evil.example.com", "Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(request, timeout=2)
                self.assertEqual(context.exception.code, 421)
                context.exception.close()

                allowed = urllib.request.Request(
                    f"http://{host}:{port}/health",
                    headers={"Host": f"127.0.0.1:{port}"},
                )
                with urllib.request.urlopen(allowed, timeout=2) as response:
                    self.assertEqual(json.loads(response.read())["status"], "ok")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_repeated_auth_failures_are_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            server = create_server(config, SessionService(config, provider=_pending_provider()))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host_value, port_value = server.server_address[:2]
            host = host_value.decode("utf-8") if isinstance(host_value, bytes) else host_value
            port = int(port_value)
            base_url = f"http://{host}:{port}"
            try:
                codes: list[int] = []
                for _ in range(14):
                    request = urllib.request.Request(
                        f"{base_url}/v1/chat/completions",
                        data=b"{}",
                        method="POST",
                    )
                    try:
                        urllib.request.urlopen(request, timeout=2)
                        codes.append(200)
                    except urllib.error.HTTPError as error:
                        codes.append(error.code)
                        error.close()
                self.assertIn(401, codes)
                self.assertEqual(codes[-1], 429)
                # A valid key must not bypass the throttle, but it must reset it.
                self.assertTrue(
                    all(code == 401 for code in codes[:11]),
                    f"expected only 401 before the limit, got {codes[:11]}",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_openai_endpoint_supports_auth_and_control_tunnel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary), integration_key=TEST_INTEGRATION_KEY)
            provider = _pending_provider()
            service = SessionService(config, provider=provider)
            server = create_server(config, service)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host_value, port_value = server.server_address[:2]
            host = host_value.decode("utf-8") if isinstance(host_value, bytes) else host_value
            port = int(port_value)
            base_url = f"http://{host}:{port}"
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                    health = json.loads(response.read())
                self.assertEqual(health["status"], "ok")
                cors_request = urllib.request.Request(
                    f"{base_url}/v1/chat/completions",
                    method="OPTIONS",
                    headers={
                        "Origin": "http://127.0.0.1:8000",
                        "Access-Control-Request-Method": "POST",
                    },
                )
                with urllib.request.urlopen(cors_request, timeout=2) as response:
                    self.assertEqual(response.status, 204)
                    self.assertEqual(
                        response.headers.get("Access-Control-Allow-Origin"),
                        "http://127.0.0.1:8000",
                    )

                unauthorized = urllib.request.Request(
                    f"{base_url}/v1/chat/completions",
                    data=b"{}",
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(unauthorized, timeout=2)
                self.assertEqual(context.exception.code, 401)
                context.exception.close()

                generation_messages: list[JsonValue] = [
                    {"role": "system", "content": "Character: Aria"}
                ]
                generation_messages.extend(
                    cast(JsonValue, item)
                    for item in _message_dicts(
                        _messages("chat_four", "Я вхожу в комнату")[1:]
                    )
                )
                generation = _request(
                    f"{base_url}/v1/chat/completions",
                    TEST_INTEGRATION_KEY,
                    {
                        "model": "roleplay-kernel",
                        "messages": generation_messages,
                        "stream": False,
                    },
                )
                self.assertEqual(_completion_content(generation), "Колено горело.")

                control = _request(
                    f"{base_url}/v1/chat/completions",
                    TEST_INTEGRATION_KEY,
                    {
                        "model": "roleplay-kernel-control/status",
                        "messages": [
                            {
                                "role": "system",
                                "content": CONTROL_PREFIX
                                + json.dumps({"session_id": "chat_four"}),
                            }
                        ],
                        "stream": False,
                    },
                )
                control_data = json.loads(_completion_content(control))
                self.assertEqual(control_data["pending_count"], 1)

                provider.responses.extend(_empty_turn_responses())
                stream_messages = _message_dicts(
                    _messages("chat_stream", "Я осматриваюсь")
                )
                stream_body = _request_text(
                    f"{base_url}/v1/chat/completions",
                    TEST_INTEGRATION_KEY,
                    {
                        "model": "roleplay-kernel",
                        "messages": cast(JsonValue, stream_messages),
                        "stream": True,
                    },
                )
                self.assertIn("data: [DONE]", stream_body)
                deltas = _sse_text_deltas(stream_body)
                self.assertEqual("".join(deltas), "Вокруг была пустая станция.")
                self.assertGreater(len(deltas), 1, "render must be delivered in pieces")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_raw_final_messages_do_not_override_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provider = _pending_provider()
            provider.responses.extend(_empty_turn_responses())
            service = SessionService(_config(Path(temporary)), provider=provider)
            service.generate(
                _envelope("chat_five", "Я вхожу в комнату"),
                _messages("chat_five", "Я вхожу в комнату"),
                "Character: Aria",
            )
            service.control("reject", {"session_id": "chat_five"})

            raw_messages = _messages(
                "chat_five",
                "Новый ход",
                include_previous=True,
            )
            raw_messages[-1] = IncomingMessage("user", "Подмененный сырой prompt")

            turn = service.generate(
                _envelope("chat_five", "Новый ход", include_previous=True),
                raw_messages,
                "Character: Aria",
            )

            self.assertEqual(turn.status, "ok")
            rendered_prompts = [
                message.content
                for call in provider.calls
                for message in call
            ]
            self.assertTrue(any("Новый ход" in prompt for prompt in rendered_prompts))
            self.assertFalse(any("Подмененный" in prompt for prompt in rendered_prompts))

    def test_status_reports_live_progress_while_generation_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provider = _BlockingProvider()
            service = SessionService(_config(Path(temporary)), provider=provider)
            errors: list[BaseException] = []

            def run_generation() -> None:
                try:
                    service.generate(
                        _envelope("chat_live", "Я вхожу в комнату"),
                        _messages("chat_live", "Я вхожу в комнату"),
                        "Character: Aria",
                    )
                except BaseException as error:  # pragma: no cover - surfaced via errors
                    errors.append(error)

            thread = threading.Thread(target=run_generation)
            thread.start()
            self.assertTrue(provider.started.wait(timeout=2))
            try:
                status = service.control("status", {"session_id": "chat_live"})
            finally:
                provider.release.set()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(status["status"], "running")
            progress = status["progress"]
            self.assertIsInstance(progress, dict)
            if not isinstance(progress, dict):
                self.fail("progress must be an object")
            self.assertTrue(progress["active"])
            self.assertEqual(progress["phase"], "plan")
            self.assertEqual(progress["completed"], 0)
            self.assertEqual(progress["total"], 4)
            idle = service.control("status", {"session_id": "chat_live"})
            idle_progress = idle["progress"]
            self.assertIsInstance(idle_progress, dict)
            if not isinstance(idle_progress, dict):
                self.fail("progress must be an object")
            self.assertFalse(idle_progress["active"])
            self.assertEqual(idle_progress["completed"], 4)

    def test_status_does_not_deadlock_when_a_different_session_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provider = _BlockingProvider()
            service = SessionService(_config(Path(temporary)), provider=provider)
            errors: list[BaseException] = []

            def run_generation() -> None:
                try:
                    service.generate(
                        _envelope("chat_busy", "Я вхожу в комнату"),
                        _messages("chat_busy", "Я вхожу в комнату"),
                        "Character: Aria",
                    )
                except BaseException as error:  # pragma: no cover - surfaced via errors
                    errors.append(error)

            thread = threading.Thread(target=run_generation)
            thread.start()
            self.assertTrue(provider.started.wait(timeout=2))
            try:
                other = service.control("status", {"session_id": "chat_other"})
            finally:
                provider.release.set()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(other["exists"], False)


def _config(root: Path, *, integration_key: str = TEST_INTEGRATION_KEY) -> SidecarConfig:
    return SidecarConfig(
        host="127.0.0.1",
        port=0,
        upstream_base_url="https://api.example.test/v1",
        upstream_model="test-model",
        upstream_api_key_env="TEST_API_KEY",
        upstream_token_parameter="max_tokens",
        integration_key=integration_key,
        state_dir=root,
        mode="balanced",
        context_window=32768,
        token_budget=18000,
        max_output_tokens=1200,
        max_internal_tokens=1200,
        upstream_timeout_seconds=600,
        max_repairs=1,
        max_context_chars=16000,
        allow_insecure_http=False,
    )


_request_counter = 0


def _new_request_key() -> str:
    global _request_counter
    _request_counter += 1
    return f"{_request_counter:016x}"


def _envelope(
    session_id: str,
    user_text: str = "Я вхожу в помещение.",
    *,
    include_previous: bool = False,
    include_rejected_previous: bool = False,
    generation_type: str = "normal",
    request_key: str | None = None,
) -> GenerationEnvelope:
    transcript: list[TranscriptItem] = []
    if include_previous:
        transcript.extend(
            (
                TranscriptItem(0, "user", "Я вхожу в комнату"),
                TranscriptItem(1, "assistant", "Колено горело."),
            )
        )
        if include_rejected_previous:
            transcript.extend(
                (
                    TranscriptItem(2, "user", "Я осматриваюсь"),
                    TranscriptItem(3, "assistant", "Колено горело."),
                )
            )
    transcript.append(TranscriptItem(len(transcript), "user", user_text))
    return GenerationEnvelope(
        protocol=PROTOCOL_VERSION,
        session_id=session_id,
        generation_type=generation_type,
        request_key=request_key or _new_request_key(),
        language="ru",
        pov="third_person_limited",
        tense="past",
        transcript=tuple(transcript),
    )


def _messages(
    session_id: str,
    user_text: str,
    *,
    include_previous: bool = False,
) -> list[IncomingMessage]:
    envelope = _envelope(session_id, user_text, include_previous=include_previous)
    envelope_data = envelope.to_dict()
    envelope_data["operation"] = "generate"
    messages = [IncomingMessage("system", "Character: Aria. Setting: an abandoned station.")]
    if include_previous:
        messages.extend(
            (
                IncomingMessage("user", "Первый ход"),
                IncomingMessage("assistant", "Колено горело."),
            )
        )
    messages.extend(
        (
            IncomingMessage(
                "system",
                ENVELOPE_PREFIX + json.dumps(envelope_data),
            ),
            IncomingMessage("user", user_text),
        )
    )
    return messages


def _messages_for_envelope(envelope: GenerationEnvelope) -> list[IncomingMessage]:
    envelope_data = envelope.to_dict()
    envelope_data["operation"] = "generate"
    return [
        IncomingMessage("system", "Character: Aria. Setting: an abandoned station."),
        IncomingMessage("system", ENVELOPE_PREFIX + json.dumps(envelope_data)),
        IncomingMessage("user", "текущий ход"),
    ]


def _envelope_with_history(
    session_id: str,
    history: Sequence[tuple[str, str]],
    *,
    language: str = "ru",
    mode: str = "balanced",
) -> GenerationEnvelope:
    transcript = [
        TranscriptItem(index, role, content)
        for index, (role, content) in enumerate(history)
    ]
    return GenerationEnvelope(
        protocol=PROTOCOL_VERSION,
        session_id=session_id,
        generation_type="normal",
        request_key=_new_request_key(),
        language=language,
        pov="third_person_limited",
        tense="past",
        transcript=tuple(transcript),
        mode=cast(EngineMode, mode),
    )


def _message_dicts(messages: list[IncomingMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


class _BudgetProvider:
    """Lite-mode provider that simulates an exhausted post-render budget."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        sampling: Mapping[str, object] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Completion:
        self.calls += 1
        return Completion("The station was silent.", "test-model")


class _AbortingProvider:
    """Lite-mode provider that reports a lost client during the render stage."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        sampling: Mapping[str, object] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Completion:
        self.calls += 1
        if on_delta is not None:
            on_delta("Колено ")
            # The client disconnects right after the first piece arrives.
            on_delta("горело.")
        return Completion("Колено горело.", "test-model")


def _pending_provider() -> ScriptedProvider:
    return ScriptedProvider(
        [
            Completion(_plan_json(), "test-model"),
            Completion("Колено горело.", "test-model"),
            Completion(_pending_delta_json(), "test-model"),
            Completion('{"findings": []}', "test-model"),
        ]
    )


def _empty_turn_responses() -> list[Completion]:
    return [
        Completion(_plan_json(), "test-model"),
        Completion("Вокруг была пустая станция.", "test-model"),
        Completion('{"operations": []}', "test-model"),
        Completion('{"findings": []}', "test-model"),
    ]


class _BlockingProvider:
    """Provider that blocks inside the planner call so progress can be observed."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[ChatMessage, ...]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        sampling: Mapping[str, object] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Completion:
        self.calls.append(tuple(messages))
        if len(self.calls) == 1:
            self.started.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("planner call was not released")
            return Completion(_plan_json(), "test-model")
        if len(self.calls) == 2:
            return Completion("Вокруг была пустая станция.", "test-model")
        if len(self.calls) == 3:
            return Completion('{"operations": []}', "test-model")
        return Completion('{"findings": []}', "test-model")


def _request(
    url: str,
    integration_key: str,
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {integration_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise AssertionError("response must be an object")
    return cast(dict[str, JsonValue], value)


def _sse_text_deltas(body: str) -> list[str]:
    """Collect the streamed text pieces from an OpenAI-compatible SSE body."""
    deltas: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if not isinstance(event, dict):
            continue
        for choice in cast(list[JsonValue], event.get("choices") or []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    deltas.append(content)
    return deltas


def _request_text(
    url: str,
    integration_key: str,
    payload: dict[str, JsonValue],
) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {integration_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        raw = response.read()
    if not isinstance(raw, bytes):
        raise AssertionError("response body must be bytes")
    return raw.decode("utf-8")


def _completion_content(response: dict[str, JsonValue]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError("completion choices are missing")
    first = choices[0]
    if not isinstance(first, dict):
        raise AssertionError("completion choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise AssertionError("completion message must be an object")
    content = message.get("content")
    if not isinstance(content, str):
        raise AssertionError("completion content must be a string")
    return content


def _plan_json() -> str:
    return json.dumps(
        {
            "goal": "Ответить на ход игрока",
            "pov": "third_person_limited",
            "must_fact_ids": [],
            "must_events": [],
            "information_release": [],
            "allowed_inventions": [],
            "beats": [
                {
                    "action": "Игрок входит",
                    "reaction": "Колено горит",
                    "causality": "Предыдущая травма",
                    "sensory_focus": "Боль",
                    "state_effect": "Травма подтверждается",
                }
            ],
            "prohibited_moves": [],
            "style_mode": "restrained",
            "novelty_requirement": "concrete",
            "target_state_change": ["injury"],
            "uncertainty": [],
        },
        ensure_ascii=False,
    )


def _pending_delta_json() -> str:
    return json.dumps(
        {
            "operations": [
                {
                    "kind": "set_injury",
                    "target": "player.knee",
                    "value": "горящая рана",
                    "impact": "high",
                    "evidence": "Колено горело.",
                    "certainty": 1.0,
                }
            ]
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    unittest.main()
