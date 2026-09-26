from __future__ import annotations

import unittest

from roleplay_kernel import (
    ModuleDefinition,
    ModuleRegistry,
    RoleplayState,
    StateDelta,
    StateOperation,
    parse_json_object,
    validate_candidate,
)
from roleplay_kernel.validators import (
    apply_delta,
    defer_critic_flagged_operations,
    parse_critic_findings,
    parse_delta,
)


class LanguageValidationTests(unittest.TestCase):
    def test_russian_post_is_flagged_when_english_requested(self) -> None:
        candidate = (
            "Колено горело, и он медленно опустился на стул, "
            "пытаясь перевести дыхание после долгого бега."
        )
        findings = validate_candidate(
            candidate=candidate,
            previous_assistant_turns=(),
            activations=(),
            expected_language="en",
        )
        codes = {finding.code for finding in findings}
        self.assertIn("response_language_mismatch", codes)
        hard = [f for f in findings if f.code == "response_language_mismatch"]
        self.assertEqual(hard[0].severity, "hard")

    def test_english_post_is_accepted(self) -> None:
        candidate = (
            "His knee was burning, and he lowered himself onto the bench, "
            "trying to catch his breath after the long run."
        )
        findings = validate_candidate(
            candidate=candidate,
            previous_assistant_turns=(),
            activations=(),
            expected_language="en",
        )
        self.assertNotIn(
            "response_language_mismatch",
            {finding.code for finding in findings},
        )

    def test_short_post_is_not_judged(self) -> None:
        findings = validate_candidate(
            candidate="Он молчит.",
            previous_assistant_turns=(),
            activations=(),
            expected_language="en",
        )
        self.assertNotIn(
            "response_language_mismatch",
            {finding.code for finding in findings},
        )

    def test_russian_target_is_not_enforced(self) -> None:
        candidate = (
            "His knee was burning, and he lowered himself onto the bench, "
            "trying to catch his breath after the long run."
        )
        findings = validate_candidate(
            candidate=candidate,
            previous_assistant_turns=(),
            activations=(),
            expected_language="ru",
        )
        self.assertNotIn(
            "response_language_mismatch",
            {finding.code for finding in findings},
        )


class ValidationTests(unittest.TestCase):
    def test_speaker_label_and_runtime_leak_are_hard_findings(self) -> None:
        findings = validate_candidate(
            candidate='NPC: The approved plan is here.',
            previous_assistant_turns=(),
            activations=(),
        )

        codes = {finding.code for finding in findings}
        self.assertIn("speaker_label_leak", codes)
        self.assertIn("runtime_metadata_leak", codes)
        self.assertTrue(all(finding.severity == "hard" for finding in findings))

    def test_repeated_ngrams_produce_a_warning(self) -> None:
        previous = "The cold air moved across the floor and the metal latch answered."

        findings = validate_candidate(
            candidate="The cold air moved across the floor and the metal latch answered again.",
            previous_assistant_turns=(previous,),
            activations=(),
        )

        self.assertIn("repetitive_ngrams", {finding.code for finding in findings})

    def test_forbidden_term_scope_is_deterministic(self) -> None:
        module = ModuleDefinition(
            id="lexical",
            version="1",
            category="test",
            instructions=("Avoid the term",),
            avoid_terms=("smirk",),
        )
        activations = ModuleRegistry((module,)).activate(
            state=RoleplayState(),
            user_input="test",
            forced=("lexical",),
        )

        findings = validate_candidate(
            candidate="A quick smirk crossed his face.",
            previous_assistant_turns=(),
            activations=activations,
        )

        forbidden = [finding for finding in findings if finding.code == "forbidden_term"]
        self.assertEqual(len(forbidden), 1)
        self.assertEqual(forbidden[0].evidence, "smirk")

    def test_delta_requires_exact_evidence_and_cannot_downgrade_impact(self) -> None:
        candidate = "The door opened into a corridor."

        result = parse_delta(
            {
                "operations": [
                    {
                        "kind": "set_location",
                        "target": "",
                        "value": "corridor",
                        "impact": "low",
                        "evidence": "The door opened",
                        "certainty": 1.0,
                    },
                    {
                        "kind": "set_injury",
                        "target": "hand",
                        "value": "cut",
                        "impact": "low",
                        "evidence": "not present",
                        "certainty": 0.8,
                    },
                ]
            },
            candidate=candidate,
        )

        self.assertEqual(len(result.delta.operations), 1)
        self.assertIn("unsupported_operation", {finding.code for finding in result.findings})

    def test_explicit_commit_applies_high_impact_operation(self) -> None:
        state = RoleplayState()
        delta = StateDelta(
            (
                StateOperation(
                    kind="set_injury",
                    target="hand",
                    value="cut",
                    impact="high",
                    evidence="A red line opened across his palm.",
                ),
            )
        )

        applied, pending = apply_delta(state, delta)
        confirmed, still_pending = apply_delta(state, pending, allow_high_impact=True)

        self.assertFalse(applied.operations)
        self.assertEqual(len(pending.operations), 1)
        self.assertEqual(len(confirmed.operations), 1)
        self.assertFalse(still_pending.operations)
        self.assertEqual(state.injuries["hand"], "cut")

    def test_apply_boundary_cannot_downgrade_inherent_impact(self) -> None:
        state = RoleplayState()
        delta = StateDelta(
            (
                StateOperation(
                    kind="set_injury",
                    target="hand",
                    value="cut",
                    impact="low",
                    evidence="A red line opened across his palm.",
                ),
            )
        )

        applied, pending = apply_delta(state, delta)

        self.assertFalse(applied.operations)
        self.assertEqual(pending.operations[0].impact, "high")
        self.assertEqual(state.version, 0)
        self.assertEqual(state.injuries, {})

    def test_event_and_scene_tag_operations_are_applied(self) -> None:
        state = RoleplayState(scene_tags=["combat"])
        delta = StateDelta(
            (
                StateOperation(
                    kind="record_event",
                    value="The gate opened.",
                    impact="low",
                ),
                StateOperation(
                    kind="set_scene_tag",
                    value="none",
                    impact="low",
                ),
            )
        )

        applied, pending = apply_delta(state, delta)

        self.assertEqual(len(applied.operations), 2)
        self.assertFalse(pending.operations)
        self.assertEqual(state.events, ["The gate opened."])
        self.assertEqual(state.scene_tags, [])

    def test_low_certainty_objective_operation_requires_confirmation(self) -> None:
        result = parse_delta(
            {
                "operations": [
                    {
                        "kind": "set_location",
                        "target": "",
                        "value": "corridor",
                        "impact": "low",
                        "evidence": "The door opened",
                        "certainty": 0.5,
                    }
                ]
            },
            candidate="The door opened into a corridor.",
        )

        self.assertEqual(result.delta.operations[0].impact, "high")
        self.assertIn("low_certainty_state_change", {item.code for item in result.findings})

    def test_critic_can_defer_a_lexically_supported_operation(self) -> None:
        delta_result = parse_delta(
            {
                "operations": [
                    {
                        "kind": "set_location",
                        "target": "",
                        "value": "open",
                        "impact": "low",
                        "evidence": "The door is not open",
                        "certainty": 1.0,
                    }
                ]
            },
            candidate="The door is not open.",
        )
        findings = parse_critic_findings(
            {
                "findings": [
                    {
                        "severity": "hard",
                        "code": "delta_semantics",
                        "message": "Evidence negates the proposed state",
                        "evidence": "The door is not open",
                        "rule": "Objective operations require semantic support",
                        "confidence": 0.99,
                        "operation_index": 0,
                    }
                ]
            },
            candidate="The door is not open.",
            operation_count=1,
        )

        deferred = defer_critic_flagged_operations(delta_result, findings)

        self.assertEqual(deferred.delta.operations[0].impact, "high")
        self.assertEqual(findings[0].operation_index, 0)

    def test_low_confidence_semantic_finding_still_defers_operation(self) -> None:
        delta_result = parse_delta(
            {
                "operations": [
                    {
                        "kind": "set_location",
                        "target": "",
                        "value": "open",
                        "impact": "low",
                        "evidence": "The door is not open",
                        "certainty": 1.0,
                    }
                ]
            },
            candidate="The door is not open.",
        )
        findings = parse_critic_findings(
            {
                "findings": [
                    {
                        "severity": "hard",
                        "code": "delta_semantics",
                        "message": "Likely unsupported mutation",
                        "evidence": "The door is not open",
                        "rule": "State",
                        "confidence": 0.2,
                        "operation_index": 0,
                    }
                ]
            },
            candidate="The door is not open.",
            operation_count=1,
        )

        deferred = defer_critic_flagged_operations(delta_result, findings)

        self.assertEqual(findings[0].severity, "warning")
        self.assertEqual(deferred.delta.operations[0].impact, "high")

    def test_missing_required_structures_fail_closed(self) -> None:
        delta = parse_delta({}, candidate="Text")
        critic = parse_critic_findings({}, candidate="Text")

        self.assertEqual(delta.findings[0].severity, "hard")
        self.assertEqual(delta.findings[0].code, "invalid_delta")
        self.assertEqual(critic[0].severity, "hard")
        self.assertEqual(critic[0].code, "invalid_critic")

    def test_extractor_operation_schema_is_strict(self) -> None:
        result = parse_delta(
            {
                "operations": [
                    {
                        "kind": "set_location",
                        "target": "",
                        "value": "corridor",
                        "evidence": "The door opened into a corridor.",
                    }
                ]
            },
            candidate="The door opened into a corridor.",
        )

        self.assertEqual(result.delta.operations, ())
        self.assertIn("rejected_operation", {item.code for item in result.findings})

    def test_semantic_critic_finding_requires_a_valid_operation_index(self) -> None:
        findings = parse_critic_findings(
            {
                "findings": [
                    {
                        "severity": "hard",
                        "code": "delta_semantics",
                        "message": "Unsupported mutation",
                        "evidence": "The door opened",
                        "rule": "State",
                        "confidence": 1.0,
                    }
                ]
            },
            candidate="The door opened.",
            operation_count=1,
        )

        self.assertEqual(findings[0].code, "invalid_critic_operation_reference")
        self.assertEqual(findings[0].severity, "hard")

    def test_critic_evidence_must_exist_in_candidate(self) -> None:
        candidate = "The red lantern marked the locked gate."

        findings = parse_critic_findings(
            {
                "findings": [
                    {
                        "severity": "hard",
                        "code": "continuity",
                        "message": "Contradiction",
                        "evidence": "The blue lantern marked the gate.",
                        "rule": "State",
                        "confidence": 0.99,
                    }
                ]
            },
            candidate=candidate,
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "invalid_critic")
        self.assertEqual(findings[0].severity, "hard")


class JsonTests(unittest.TestCase):
    def test_parser_accepts_fenced_and_surrounded_objects(self) -> None:
        value = parse_json_object('prefix\n```json\n{"ok": true}\n```\nsuffix')

        self.assertEqual(value, {"ok": True})

    def test_parser_does_not_accept_a_nested_object_from_truncated_output(self) -> None:
        truncated = '{"operations": [{"kind": "set_time", "value": "dawn"}'

        with self.assertRaisesRegex(ValueError, "truncated"):
            parse_json_object(truncated)


if __name__ == "__main__":
    unittest.main()
