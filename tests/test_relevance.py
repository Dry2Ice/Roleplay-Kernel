from __future__ import annotations

import unittest

from roleplay_kernel.relevance import RelevanceWeights, score_entry, tokenize


class RelevanceTests(unittest.TestCase):
    def test_tokenize_removes_stopwords(self) -> None:
        tokens = tokenize("The quick brown fox jumps")
        self.assertIn("quick", tokens)
        self.assertIn("fox", tokens)
        self.assertNotIn("the", tokens)

    def test_score_entry_with_high_overlap(self) -> None:
        score = score_entry(
            "Aria drew her sword",
            recency=0.8,
            candidate_tokens=frozenset({"aria", "sword", "drew"}),
            plan_tokens=frozenset({"fight", "sword"}),
        )
        self.assertGreater(score, 0.5)

    def test_score_entry_with_no_overlap(self) -> None:
        score = score_entry(
            "The weather was calm",
            recency=0.0,
            candidate_tokens=frozenset({"aria", "sword", "fight"}),
            plan_tokens=frozenset({"battle"}),
        )
        self.assertLess(score, 0.2)

    def test_emotional_marker_boosts_score(self) -> None:
        base = score_entry(
            "She stood quietly",
            recency=0.2,
            candidate_tokens=frozenset({"stood"}),
            plan_tokens=frozenset(),
        )
        emotional = score_entry(
            "She stood in rage",
            recency=0.2,
            candidate_tokens=frozenset({"stood"}),
            plan_tokens=frozenset(),
        )
        self.assertGreater(emotional, base)

    def test_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            RelevanceWeights(
                recency=0.5,
                entity_overlap=0.3,
                causal_link=0.1,
                emotional_salience=0.0,
            )

    def test_recent_event_scores_higher_than_old(self) -> None:
        old = score_entry(
            "Aria left the tavern",
            recency=0.1,
            candidate_tokens=frozenset({"aria"}),
            plan_tokens=frozenset(),
        )
        recent = score_entry(
            "Aria left the tavern",
            recency=0.9,
            candidate_tokens=frozenset({"aria"}),
            plan_tokens=frozenset(),
        )
        self.assertGreater(recent, old)

    def test_plan_causality_boosts_score(self) -> None:
        no_plan = score_entry(
            "The door creaked",
            recency=0.3,
            candidate_tokens=frozenset(),
            plan_tokens=frozenset(),
        )
        with_plan = score_entry(
            "The door creaked",
            recency=0.3,
            candidate_tokens=frozenset(),
            plan_tokens=frozenset({"door", "creaked"}),
        )
        self.assertGreater(with_plan, no_plan)


if __name__ == "__main__":
    unittest.main()
