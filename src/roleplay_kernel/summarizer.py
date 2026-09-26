from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .models import ConversationTurn


@dataclass(frozen=True, slots=True)
class SummaryConfig:
    enabled: bool = False
    keep_recent_pairs: int = 6
    summarize_every_pairs: int = 4
    hide_old_after_pairs: int = 24

    def __post_init__(self) -> None:
        if self.keep_recent_pairs < 2:
            raise ValueError("keep_recent_pairs must be at least 2")
        if self.summarize_every_pairs < 1:
            raise ValueError("summarize_every_pairs must be at least 1")
        if self.hide_old_after_pairs < self.keep_recent_pairs:
            raise ValueError("hide_old_after_pairs must not be below keep_recent_pairs")


@dataclass(slots=True)
class TranscriptSummarizer:
    config: SummaryConfig
    _cache_index: int = 0
    _cache_text: str = ""

    def compile(
        self,
        turns: Sequence[ConversationTurn],
        summarize: Callable[[list[ConversationTurn]], str],
    ) -> tuple[list[ConversationTurn], str]:
        usable = [turn for turn in turns if not turn.superseded]
        if not self.config.enabled or len(usable) <= self.config.keep_recent_pairs * 2:
            return list(usable), ""
        keep = self.config.keep_recent_pairs * 2
        old = usable[:-keep]
        recent = usable[-keep:]
        if len(usable) > self.config.hide_old_after_pairs * 2:
            self._cache_index = 0
            self._cache_text = ""
            return list(recent), ""
        if self._should_resummarize(len(usable)):
            self._cache_text = summarize(old)
            self._cache_index = len(usable)
        if not self._cache_text:
            return list(recent), ""
        return list(recent), self._cache_text

    def _should_resummarize(self, current_length: int) -> bool:
        if not self._cache_text:
            return True
        return current_length - self._cache_index >= self.config.summarize_every_pairs * 2

    def invalidate(self) -> None:
        self._cache_index = 0
        self._cache_text = ""
