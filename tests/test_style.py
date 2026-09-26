from __future__ import annotations

import unittest

from roleplay_kernel.style import StyleTracker


class StyleTrackerTests(unittest.TestCase):
    def test_records_opening(self) -> None:
        tracker = StyleTracker()
        tracker.record("The rain fell softly on the window.")
        self.assertIn("The rain fell softly", tracker.recent_openings)

    def test_records_semantic_keys(self) -> None:
        tracker = StyleTracker()
        tracker.record("She smiled and nodded in silence.")
        self.assertIn("smiled", tracker.used_semantics)
        self.assertIn("nodded", tracker.used_semantics)
        self.assertIn("silence", tracker.used_semantics)

    def test_records_sentence_pattern(self) -> None:
        tracker = StyleTracker()
        tracker.record("He ran. He fell. He got up.")
        self.assertIn("short_sentences", tracker.used_patterns)

    def test_constraints_returns_all_lists(self) -> None:
        tracker = StyleTracker()
        tracker.record("She looked at him with cold eyes.")
        constraints = tracker.constraints()
        self.assertIn("avoid_openings", constraints)
        self.assertIn("avoid_semantics", constraints)
        self.assertIn("avoid_patterns", constraints)
        self.assertGreater(len(constraints["avoid_openings"]), 0)
        self.assertGreater(len(constraints["avoid_semantics"]), 0)

    def test_reset_clears_all(self) -> None:
        tracker = StyleTracker()
        tracker.record("The door opened slowly.")
        tracker.reset()
        self.assertEqual(len(tracker.recent_openings), 0)
        self.assertEqual(len(tracker.used_semantics), 0)
        self.assertEqual(len(tracker.used_patterns), 0)

    def test_max_openings_respected(self) -> None:
        tracker = StyleTracker(max_openings=2)
        tracker.record("First opening here.")
        tracker.record("Second opening here.")
        tracker.record("Third opening here.")
        self.assertEqual(len(tracker.recent_openings), 2)
        self.assertEqual(tracker.recent_openings[0], "Second opening here.")


if __name__ == "__main__":
    unittest.main()
