from __future__ import annotations

import threading
import unittest
from collections.abc import Sequence
from dataclasses import replace

from roleplay_kernel import (
    Belief,
    ChatMessage,
    Completion,
    ContextBudgetError,
    ContextCompiler,
    Engine,
    EngineConfig,
    JsonValue,
    ModuleDefinition,
    ModuleDefinitionError,
    ModuleRegistry,
    ProviderError,
    RoleplayState,
    Session,
    StateDelta,
    StateOperation,
    UnknownPendingCommitError,
)


class ScriptedProvider:
    def __init__(self, responses: list[Completion]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, float | None, int | None, bool]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self.calls.append((messages[-1].content, temperature, max_tokens, json_mode))
        if not self.responses:
            raise AssertionError("scripted provider received an unexpected request")
        return self.responses.pop(0)


class BlockingProvider:
    def __init__(self, responses: list[Completion]) -> None:
        self.delegate = ScriptedProvider(responses)
        self.started = threading.Event()
        self.release = threading.Event()

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self.started.set()
        if not self.release.wait(timeout=3):
            raise TimeoutError("test provider was not released")
        return self.delegate.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )


class EngineTests(unittest.TestCase):
    def test_balanced_turn_commits_safe_state_and_defers_high_impact(self) -> None:
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model", {"prompt_tokens": 10}),
                Completion(
                    "Дверь открылась в тёмный коридор. На засове осталась свежая царапина. "
                    "Колено горело, когда он переступил порог.",
                    "test-model",
                    {"completion_tokens": 20},
                ),
                Completion(
                    _delta_json(),
                    "test-model",
                    {"completion_tokens": 30},
                ),
                Completion('{"findings": []}', "test-model", {"completion_tokens": 5}),
            ]
        )
        engine = Engine(provider)
        session = engine.new_session(setting="Игрок стоит перед дверью")

        result = engine.advance(session, "Я открываю дверь и вхожу")

        self.assertEqual(
            result.text,
            "Дверь открылась в тёмный коридор. На засове осталась свежая царапина. "
            "Колено горело, когда он переступил порог.",
        )
        self.assertEqual(result.provider_calls, 4)
        self.assertEqual(result.status, "needs_confirmation")
        self.assertEqual(session.state.location, "тёмный коридор")
        self.assertEqual(session.state.facts["door.lock"], "true")
        self.assertEqual(len(result.applied_operations.operations), 2)
        self.assertEqual(len(result.pending_operations.operations), 1)
        self.assertEqual(session.state.version, 1)
        self.assertEqual(len(session.turns), 2)
        self.assertEqual(len(session.ledger), 1)

        forged = replace(
            result,
            pending_operations=StateDelta(
                (
                    StateOperation(
                        kind="set_resource",
                        target="coins",
                        value="999",
                        impact="high",
                        evidence="forged",
                    ),
                )
            ),
        )
        committed_version = engine.commit_pending(session, forged)

        self.assertEqual(committed_version, 2)
        self.assertEqual(session.state.injuries["player.knee"], "горящая рана")
        self.assertNotIn("coins", session.state.resources)
        self.assertEqual(len(session.ledger), 2)

    def test_pending_commit_survives_session_serialization(self) -> None:
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion(
                    "Дверь открылась в тёмный коридор. На засове осталась свежая царапина. "
                    "Колено горело, когда он переступил порог.",
                    "test-model",
                ),
                Completion(_delta_json(), "test-model"),
                Completion('{"findings": []}', "test-model"),
            ]
        )
        integrity_key = b"k" * 32
        engine = Engine(provider, integrity_key=integrity_key)
        session = engine.new_session()
        result = engine.advance(session, "Я открываю дверь")

        restored = Session.from_dict(session.to_dict())
        resumed_engine = Engine(ScriptedProvider([]), integrity_key=integrity_key)

        version = resumed_engine.commit_pending_by_id(
            restored,
            request_id=result.request_id,
            assistant_turn_id=result.assistant_turn_id,
        )

        self.assertEqual(version, 2)
        self.assertEqual(restored.state.injuries["player.knee"], "горящая рана")
        self.assertEqual(Session.from_dict(restored.to_dict()).state.version, 2)

    def test_persisted_pending_delta_is_hmac_protected(self) -> None:
        integrity_key = b"k" * 32
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion(
                    "Дверь открылась в тёмный коридор. На засове осталась свежая царапина. "
                    "Колено горело, когда он переступил порог.",
                    "test-model",
                ),
                Completion(_delta_json(), "test-model"),
                Completion('{"findings": []}', "test-model"),
            ]
        )
        engine = Engine(provider, integrity_key=integrity_key)
        session = engine.new_session()
        result = engine.advance(session, "Я открываю дверь")
        data = session.to_dict()
        ledger = data["ledger"]
        self.assertIsInstance(ledger, list)
        if not isinstance(ledger, list) or not isinstance(ledger[0], dict):
            self.fail("ledger event must be an object")
        payload = ledger[0].get("payload")
        self.assertIsInstance(payload, dict)
        if not isinstance(payload, dict):
            self.fail("ledger payload must be an object")
        pending = payload.get("pending_operations")
        self.assertIsInstance(pending, dict)
        if not isinstance(pending, dict):
            self.fail("pending delta must be an object")
        operations = pending.get("operations")
        self.assertIsInstance(operations, list)
        if not isinstance(operations, list) or not isinstance(operations[0], dict):
            self.fail("pending operation must be an object")
        operations[0]["value"] = "attacker-room"

        restored = Session.from_dict(data)
        resumed = Engine(ScriptedProvider([]), integrity_key=integrity_key)

        with self.assertRaises(UnknownPendingCommitError):
            resumed.commit_pending_by_id(
                restored,
                request_id=result.request_id,
                assistant_turn_id=result.assistant_turn_id,
            )

    def test_fast_mode_skips_planner_and_critic(self) -> None:
        provider = ScriptedProvider(
            [
                Completion("Скригнула дверь, и в комнату проник холодный воздух.", "test-model"),
                Completion('{"operations": []}', "test-model"),
            ]
        )
        engine = Engine(provider, config=EngineConfig(mode="fast"))

        result = engine.advance(engine.new_session(), "Я открываю дверь")

        self.assertEqual(result.provider_calls, 2)
        self.assertEqual(result.status, "ok")
        self.assertEqual([call[3] for call in provider.calls], [False, True])

    def test_manual_commit_mode_defers_every_state_operation(self) -> None:
        delta = """{
          "operations": [{
            "kind": "set_location",
            "target": "",
            "value": "corridor",
            "impact": "low",
            "evidence": "The door opened into a corridor.",
            "certainty": 1.0
          }]
        }"""
        provider = ScriptedProvider(
            [
                Completion("The door opened into a corridor.", "test-model"),
                Completion(delta, "test-model"),
            ]
        )
        engine = Engine(
            provider,
            config=EngineConfig(mode="fast", auto_commit=False),
        )
        session = engine.new_session()

        result = engine.advance(session, "I open the door")

        self.assertEqual(result.status, "needs_confirmation")
        self.assertEqual(session.state.version, 0)
        self.assertEqual(engine.commit_pending(session, result), 1)
        self.assertEqual(session.state.location, "corridor")

    def test_strict_mode_repairs_a_hard_deterministic_violation(self) -> None:
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion("NPC: Дверь открылась.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion('{"findings": []}', "test-model"),
                Completion("Дверь открылась с тихим скрипом.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion('{"findings": []}', "test-model"),
            ]
        )
        engine = Engine(provider, config=EngineConfig(mode="strict"))

        result = engine.advance(engine.new_session(), "Я открываю дверь")

        self.assertEqual(result.repairs, 1)
        self.assertEqual(result.text, "Дверь открылась с тихим скрипом.")
        self.assertEqual(result.status, "repaired")
        self.assertFalse(any(finding.severity == "hard" for finding in result.findings))

    def test_strict_respects_zero_repair_limit(self) -> None:
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion("NPC: Дверь открылась.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion('{"findings": []}', "test-model"),
            ]
        )
        engine = Engine(
            provider,
            config=EngineConfig(mode="strict", max_repairs=0),
        )

        result = engine.advance(engine.new_session(), "Я открываю дверь")

        self.assertEqual(result.repairs, 0)
        self.assertEqual(result.status, "needs_attention")

    def test_strict_can_use_multiple_repairs(self) -> None:
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion("NPC: Дверь открылась.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion('{"findings": []}', "test-model"),
                Completion("User: Дверь открылась.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion('{"findings": []}', "test-model"),
                Completion("Дверь открылась с тихим скрипом.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion('{"findings": []}', "test-model"),
            ]
        )
        engine = Engine(
            provider,
            config=EngineConfig(mode="strict", max_repairs=2),
        )

        result = engine.advance(engine.new_session(), "Я открываю дверь")

        self.assertEqual(result.repairs, 2)
        self.assertEqual(result.text, "Дверь открылась с тихим скрипом.")
        self.assertEqual(result.status, "repaired")

    def test_strict_keeps_unresolved_critic_finding_after_repair(self) -> None:
        critic = """{
          "findings": [{
            "severity": "hard",
            "code": "continuity",
            "message": "Contradicts committed state",
            "evidence": "The door opened.",
            "rule": "The door is locked",
            "confidence": 0.99
          }]
        }"""
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion("The door opened.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion(critic, "test-model"),
                Completion("The door opened.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion(critic, "test-model"),
            ]
        )
        engine = Engine(provider, config=EngineConfig(mode="strict"))

        result = engine.advance(engine.new_session(), "Я открываю дверь")

        self.assertEqual(result.repairs, 1)
        self.assertEqual(result.status, "needs_attention")
        self.assertTrue(any(finding.code == "continuity" for finding in result.findings))

    def test_missing_delta_field_is_fail_closed(self) -> None:
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion("Дверь открылась в тёмный коридор.", "test-model"),
                Completion("{}", "test-model"),
                Completion('{"findings": []}', "test-model"),
            ]
        )
        engine = Engine(provider)
        session = engine.new_session()

        result = engine.advance(session, "Я открываю дверь")

        self.assertEqual(result.status, "needs_attention")
        self.assertIn("invalid_delta", {finding.code for finding in result.findings})
        self.assertEqual(session.state.version, 0)

    def test_invalid_critic_blocks_valid_delta_autocommit(self) -> None:
        delta = """{
          "operations": [{
            "kind": "set_location",
            "target": "",
            "value": "corridor",
            "impact": "low",
            "evidence": "The door opened into a corridor.",
            "certainty": 1.0
          }]
        }"""
        provider = ScriptedProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion("The door opened into a corridor.", "test-model"),
                Completion(delta, "test-model"),
                Completion("{}", "test-model"),
            ]
        )
        engine = Engine(provider)
        session = engine.new_session()

        result = engine.advance(session, "I open the door")

        self.assertEqual(result.status, "needs_attention")
        self.assertIn("invalid_critic", {finding.code for finding in result.findings})
        self.assertEqual(session.state.location, "unspecified")
        self.assertEqual(session.state.version, 0)
        self.assertEqual(result.pending_operations.operations, ())

    def test_fast_mode_defers_unverified_state_changes(self) -> None:
        delta = """{
          "operations": [{
            "kind": "set_location",
            "target": "",
            "value": "open",
            "impact": "low",
            "evidence": "The door is not open",
            "certainty": 1.0
          }]
        }"""
        provider = ScriptedProvider(
            [
                Completion("The door is not open.", "test-model"),
                Completion(delta, "test-model"),
            ]
        )
        engine = Engine(provider, config=EngineConfig(mode="fast"))
        session = engine.new_session()

        result = engine.advance(session, "I inspect the door")

        self.assertEqual(result.status, "needs_confirmation")
        self.assertEqual(session.state.location, "unspecified")
        self.assertEqual(result.pending_operations.operations[0].impact, "high")

    def test_truncated_completion_is_rejected(self) -> None:
        provider = ScriptedProvider(
            [Completion("The door opened", "test-model", finish_reason="length")]
        )
        engine = Engine(provider, config=EngineConfig(mode="fast"))
        session = engine.new_session()

        with self.assertRaisesRegex(ProviderError, "length"):
            engine.advance(session, "I open the door")

        self.assertEqual(session.state.version, 0)
        self.assertEqual(session.turns, ())

    def test_session_snapshot_waits_for_the_turn_transaction(self) -> None:
        provider = BlockingProvider(
            [
                Completion(_plan_json(), "test-model"),
                Completion("Cold air moved through the open doorway.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion('{"findings": []}', "test-model"),
            ]
        )
        engine = Engine(provider)
        session = engine.new_session()
        advance_errors: list[BaseException] = []

        def advance() -> None:
            try:
                engine.advance(session, "I open the door")
            except BaseException as error:
                advance_errors.append(error)

        advance_thread = threading.Thread(target=advance)
        advance_thread.start()
        self.assertTrue(provider.started.wait(timeout=1))

        snapshot: dict[str, JsonValue] = {}
        snapshot_started = threading.Event()
        snapshot_done = threading.Event()

        def take_snapshot() -> None:
            snapshot_started.set()
            snapshot.update(session.to_dict())
            snapshot_done.set()

        snapshot_thread = threading.Thread(target=take_snapshot)
        snapshot_thread.start()
        self.assertTrue(snapshot_started.wait(timeout=1))
        self.assertFalse(snapshot_done.wait(timeout=0.05))

        provider.release.set()
        advance_thread.join(timeout=3)
        snapshot_thread.join(timeout=3)

        self.assertFalse(advance_thread.is_alive())
        self.assertFalse(snapshot_thread.is_alive())
        self.assertEqual(advance_errors, [])
        snapshot_turns = snapshot["turns"]
        snapshot_ledger = snapshot["ledger"]
        self.assertIsInstance(snapshot_turns, list)
        self.assertIsInstance(snapshot_ledger, list)
        if not isinstance(snapshot_turns, list) or not isinstance(snapshot_ledger, list):
            self.fail("snapshot collections must be lists")
        self.assertEqual(len(snapshot_turns), 2)
        self.assertEqual(len(snapshot_ledger), 1)

    def test_context_budget_reserves_output_space(self) -> None:
        provider = ScriptedProvider([])
        compiler = ContextCompiler(token_budget=1024)

        with self.assertRaises(ValueError):
            Engine(
                provider,
                compiler=compiler,
                config=EngineConfig(
                    context_window=1200,
                    max_output_tokens=300,
                    max_internal_tokens=300,
                ),
            )

    def test_session_round_trip_preserves_state_and_ledger(self) -> None:
        session = Session.create(language="ru", setting="Зимний город")

        def initialize(state: RoleplayState) -> None:
            state.facts["key.location"] = "архив"
            state.beliefs.append(
                Belief(
                    holder="Ира",
                    proposition="дверь заперта",
                    certainty=0.7,
                    source="проверила ручку",
                )
            )

        session.edit_state(0, initialize)
        turn = session.append_turn("user", "Я иду к архиву")

        restored = Session.from_dict(session.to_dict())

        self.assertEqual(restored.id, session.id)
        self.assertEqual(restored.state.facts, session.state.facts)
        self.assertEqual(restored.turns[0].id, turn.id)
        self.assertEqual(restored.state.beliefs[0].holder, "Ира")

    def test_session_exposes_read_only_state_and_history_snapshots(self) -> None:
        session = Session.create()
        state = session.state
        state.location = "mutated-copy"
        session.edit_state(0, lambda working: working.facts.__setitem__("key", "value"))

        self.assertEqual(session.state.location, "unspecified")
        self.assertEqual(session.state.facts["key"], "value")
        self.assertIsInstance(session.turns, tuple)
        self.assertIsInstance(session.ledger, tuple)

    def test_session_rejects_broken_integrity_invariants(self) -> None:
        duplicate = Session.create()
        user_turn = duplicate.append_turn("user", "test")
        duplicate._turns.append(user_turn)
        with self.assertRaisesRegex(ValueError, "unique"):
            Session.from_dict(duplicate.to_dict())

        dangling = Session.create()
        dangling.append_ledger_event(
            kind="custom",
            turn_id="missing",
            payload={},
        )
        with self.assertRaisesRegex(ValueError, "unknown turn"):
            Session.from_dict(dangling.to_dict())

        mismatched = Session.create()
        user = mismatched.append_turn("user", "test")
        assistant = mismatched.append_turn("assistant", "answer")
        mismatched.edit_state(0, lambda state: None)
        mismatched.append_ledger_event(
            kind="turn_committed",
            turn_id=assistant.id,
            payload={
                "request_id": "request",
                "user_turn_id": user.id,
                "from_state_version": 0,
                "to_state_version": 1,
            },
        )
        mismatched._state.version = 2
        with self.assertRaisesRegex(ValueError, "does not match"):
            Session.from_dict(mismatched.to_dict())

    def test_invalid_planner_output_uses_deterministic_fallback(self) -> None:
        provider = ScriptedProvider(
            [
                Completion("not json", "test-model"),
                Completion("Пол шатнул под ногами, и пыль поднялась облаком.", "test-model"),
                Completion('{"operations": []}', "test-model"),
                Completion('{"findings": []}', "test-model"),
            ]
        )
        engine = Engine(provider)

        result = engine.advance(engine.new_session(), "Я делаю шаг")

        self.assertIn("planner_parse_error", {finding.code for finding in result.findings})
        self.assertTrue(result.plan["beats"])


class ModuleTests(unittest.TestCase):
    def test_triggered_module_is_activated(self) -> None:
        activations = ModuleRegistry.default().activate(
            state=RoleplayState(),
            user_input="Я вступаю в бой",
        )

        self.assertIn("combat", {activation.definition.id for activation in activations})
        self.assertIn("core_agency", {activation.definition.id for activation in activations})

    def test_higher_priority_conflict_wins(self) -> None:
        low = ModuleDefinition(
            id="low",
            version="1",
            category="test",
            instructions=("low",),
            active_by_default=True,
            conflicts_with=("high",),
            priority=1,
        )
        high = ModuleDefinition(
            id="high",
            version="1",
            category="test",
            instructions=("high",),
            active_by_default=True,
            conflicts_with=("low",),
            priority=2,
        )

        activations = ModuleRegistry((low, high)).activate(
            state=RoleplayState(),
            user_input="test",
        )

        self.assertEqual([activation.definition.id for activation in activations], ["high"])

    def test_missing_required_module_is_rejected(self) -> None:
        dependent = ModuleDefinition(
            id="dependent",
            version="1",
            category="test",
            instructions=("dependent",),
            requires=("missing",),
        )

        with self.assertRaises(ModuleDefinitionError):
            ModuleRegistry((dependent,)).activate(
                state=RoleplayState(),
                user_input="test",
                forced=("dependent",),
            )
    def test_trigger_matching_respects_word_boundaries(self) -> None:
        registry = ModuleRegistry.default()

        gloves = registry.activate(state=RoleplayState(), user_input="I put on my gloves")
        love = registry.activate(state=RoleplayState(), user_input="I love mystery")

        self.assertNotIn("intimacy", {item.definition.id for item in gloves})
        self.assertIn("intimacy", {item.definition.id for item in love})

    def test_asymmetric_conflict_is_rejected(self) -> None:
        first = ModuleDefinition(
            id="first",
            version="1",
            category="test",
            instructions=("first",),
            conflicts_with=("second",),
        )
        second = ModuleDefinition(
            id="second",
            version="1",
            category="test",
            instructions=("second",),
        )

        with self.assertRaises(ModuleDefinitionError):
            ModuleRegistry((first, second)).activate(
                state=RoleplayState(),
                user_input="test",
            )

    def test_forced_conflict_requires_resolution(self) -> None:
        first = ModuleDefinition(
            id="first",
            version="1",
            category="test",
            instructions=("first",),
            active_by_default=True,
            conflicts_with=("second",),
            priority=2,
        )
        second = ModuleDefinition(
            id="second",
            version="1",
            category="test",
            instructions=("second",),
            active_by_default=True,
            conflicts_with=("first",),
            priority=1,
        )

        with self.assertRaises(ModuleDefinitionError):
            ModuleRegistry((first, second)).activate(
                state=RoleplayState(),
                user_input="test",
                forced=("first", "second"),
            )

    def test_conflict_resolution_prunes_invalid_dependents(self) -> None:
        low = ModuleDefinition(
            id="low",
            version="1",
            category="test",
            instructions=("low",),
            active_by_default=True,
            conflicts_with=("high",),
            priority=1,
        )
        high = ModuleDefinition(
            id="high",
            version="1",
            category="test",
            instructions=("high",),
            active_by_default=True,
            conflicts_with=("low",),
            priority=2,
        )
        dependent = ModuleDefinition(
            id="dependent",
            version="1",
            category="test",
            instructions=("dependent",),
            active_by_default=True,
            requires=("low",),
            priority=3,
        )

        activations = ModuleRegistry((low, high, dependent)).activate(
            state=RoleplayState(),
            user_input="test",
        )

        self.assertEqual([item.definition.id for item in activations], ["high"])
    def test_dependency_pruning_reaches_a_fixed_point(self) -> None:
        root = ModuleDefinition(
            id="root",
            version="1",
            category="test",
            instructions=("root",),
            active_by_default=True,
            requires=("left", "right"),
            priority=3,
        )
        left = ModuleDefinition(
            id="left",
            version="1",
            category="test",
            instructions=("left",),
            conflicts_with=("right",),
            priority=2,
        )
        right = ModuleDefinition(
            id="right",
            version="1",
            category="test",
            instructions=("right",),
            conflicts_with=("left",),
            priority=1,
        )

        activations = ModuleRegistry((root, left, right)).activate(
            state=RoleplayState(),
            user_input="test",
        )

        self.assertEqual(activations, ())
    def test_transitive_dependency_order_does_not_leave_invalid_modules(self) -> None:
        winner = ModuleDefinition(
            id="winner",
            version="1",
            category="test",
            instructions=("winner",),
            active_by_default=True,
            conflicts_with=("zdep",),
            priority=2,
        )
        zdep = ModuleDefinition(
            id="zdep",
            version="1",
            category="test",
            instructions=("zdep",),
            conflicts_with=("winner",),
            priority=1,
        )
        aroot = ModuleDefinition(
            id="aroot",
            version="1",
            category="test",
            instructions=("aroot",),
            active_by_default=True,
            requires=("zdep",),
            priority=1,
        )

        activations = ModuleRegistry((winner, zdep, aroot)).activate(
            state=RoleplayState(),
            user_input="test",
        )

        self.assertEqual([item.definition.id for item in activations], ["winner"])


class CompilerTests(unittest.TestCase):
    def test_prompt_pack_respects_budget_and_manifests_modules(self) -> None:
        compiler = ContextCompiler(token_budget=2048, max_recent_turns=4)
        state = RoleplayState(summary="Комната")
        session = Session.create()
        activations = ModuleRegistry.default().activate(
            state=state,
            user_input="Обычное действие",
        )

        pack = compiler.compile_plan(
            state=state,
            turns=session.turns,
            user_input="Я осматриваюсь",
            activations=activations,
        )

        self.assertLessEqual(pack.estimated_tokens, 2048)
        self.assertIn("core_agency", pack.modules)
        self.assertIn("PLAYER_INPUT_DATA", pack.user)
        self.assertIn("MODULE core_agency", pack.system)

    def test_recent_context_contains_only_complete_exchanges(self) -> None:
        compiler = ContextCompiler(token_budget=2048, max_recent_turns=2)
        session = Session.create()
        first_user = session.append_turn("user", "first")
        first_assistant = session.append_turn("assistant", "answer")
        second_user = session.append_turn("user", "second")
        second_assistant = session.append_turn("assistant", "answer")
        session.append_turn("user", "unpaired")
        activations = ModuleRegistry.default().activate(
            state=session.state,
            user_input="next",
        )

        pack = compiler.compile_plan(
            state=session.state,
            turns=session.turns,
            user_input="next",
            activations=activations,
        )

        recent_ids = pack.metadata["recent_turn_ids"]
        self.assertIsInstance(recent_ids, list)
        if not isinstance(recent_ids, list):
            self.fail("recent_turn_ids must be a list")
        self.assertEqual(recent_ids, [second_user.id, second_assistant.id])
        self.assertNotIn(first_user.id, recent_ids)
        self.assertNotIn(first_assistant.id, recent_ids)

    def test_oversized_state_is_rejected_instead_of_silently_truncated(self) -> None:
        compiler = ContextCompiler(token_budget=512)
        state = RoleplayState(summary="x" * 5000)

        with self.assertRaises(ContextBudgetError):
            compiler.compile_plan(
                state=state,
                turns=(),
                user_input="test",
                activations=(),
            )


def _plan_json() -> str:
    return """{
      "goal": "Открыть дверь",
      "pov": "third_person_limited",
      "must_fact_ids": [],
      "must_events": ["The player's attempt receives a bounded consequence"],
      "information_release": ["Only the corridor becomes visible"],
      "allowed_inventions": ["A corridor and an observed physical response"],
      "beats": [{
        "action": "The player opens the door",
        "reaction": "The door opens into a dark corridor",
        "causality": "The latch releases",
        "sensory_focus": "cold air",
        "state_effect": "The player enters the corridor"
      }],
      "prohibited_moves": ["Do not decide the player's final destination"],
      "style_mode": "restrained",
      "novelty_requirement": "Use a concrete consequence",
      "target_state_change": ["location"],
      "uncertainty": ["Who is in the corridor"]
    }"""


def _delta_json() -> str:
    return """{
      "operations": [
        {
          "kind": "set_location",
          "target": "",
          "value": "тёмный коридор",
          "impact": "low",
          "evidence": "Дверь открылась в тёмный коридор.",
          "certainty": 1.0
        },
        {
          "kind": "upsert_fact",
          "target": "door.lock",
          "value": "true",
          "impact": "low",
          "evidence": "На засове осталась свежая царапина.",
          "certainty": 1.0
        },
        {
          "kind": "set_injury",
          "target": "player.knee",
          "value": "горящая рана",
          "impact": "high",
          "evidence": "Колено горело, когда он переступил порог.",
          "certainty": 0.9
        }
      ]
    }"""


if __name__ == "__main__":
    unittest.main()
