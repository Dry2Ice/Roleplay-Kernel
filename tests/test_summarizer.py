from __future__ import annotations

import unittest

from roleplay_kernel.models import ConversationTurn
from roleplay_kernel.summarizer import SummaryConfig, TranscriptSummarizer


class TranscriptSummarizerTests(unittest.TestCase):
    def _turns(self, count: int) -> list[ConversationTurn]:
        return [
            ConversationTurn(
                id=f"turn-{index}",
                role="user" if index % 2 == 0 else "assistant",
                content=f"Message {index}",
                created_at="2026-01-01T00:00:00Z",
            )
            for index in range(count)
        ]

    def test_disabled_returns_all_turns(self) -> None:
        config = SummaryConfig(enabled=False)
        summarizer = TranscriptSummarizer(config)
        turns = self._turns(10)
        visible, summary = summarizer.compile(turns, lambda old: "summary")
        self.assertEqual(len(visible), 10)
        self.assertEqual(summary, "")

    def test_short_transcript_returns_all_turns(self) -> None:
        config = SummaryConfig(enabled=True, keep_recent_pairs=6)
        summarizer = TranscriptSummarizer(config)
        turns = self._turns(6)
        visible, summary = summarizer.compile(turns, lambda old: "summary")
        self.assertEqual(len(visible), 6)
        self.assertEqual(summary, "")

    def test_long_transcript_triggers_summary(self) -> None:
        config = SummaryConfig(enabled=True, keep_recent_pairs=2, summarize_every_pairs=2)
        summarizer = TranscriptSummarizer(config)
        turns = self._turns(8)
        calls: list[int] = []

        def summarize(old: list[ConversationTurn]) -> str:
            calls.append(len(old))
            return "compressed history"

        visible, summary = summarizer.compile(turns, summarize)
        self.assertEqual(summary, "compressed history")
        self.assertEqual(len(visible), 4)
        self.assertGreater(len(calls), 0)

    def test_cache_avoids_reshummarize(self) -> None:
        config = SummaryConfig(enabled=True, keep_recent_pairs=2, summarize_every_pairs=4)
        summarizer = TranscriptSummarizer(config)
        calls: list[int] = []

        def summarize(old: list[ConversationTurn]) -> str:
            calls.append(len(old))
            return "compressed"

        summarizer.compile(self._turns(8), summarize)
        summarizer.compile(self._turns(9), summarize)
        self.assertEqual(len(calls), 1)

    def test_hide_old_after_threshold(self) -> None:
        config = SummaryConfig(
            enabled=True,
            keep_recent_pairs=2,
            summarize_every_pairs=2,
            hide_old_after_pairs=6,
        )
        summarizer = TranscriptSummarizer(config)
        turns = self._turns(16)
        visible, summary = summarizer.compile(turns, lambda old: "compressed")
        self.assertEqual(len(visible), 4)
        self.assertEqual(summary, "")

    def test_invalidate_clears_cache(self) -> None:
        config = SummaryConfig(enabled=True, keep_recent_pairs=2, summarize_every_pairs=2)
        summarizer = TranscriptSummarizer(config)
        calls: list[int] = []

        def summarize(old: list[ConversationTurn]) -> str:
            calls.append(len(old))
            return "compressed"

        summarizer.compile(self._turns(8), summarize)
        summarizer.invalidate()
        summarizer.compile(self._turns(8), summarize)
        self.assertEqual(len(calls), 2)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            SummaryConfig(keep_recent_pairs=1)
        with self.assertRaises(ValueError):
            SummaryConfig(hide_old_after_pairs=2, keep_recent_pairs=6)


if __name__ == "__main__":
    unittest.main()
