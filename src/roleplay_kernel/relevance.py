from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я'\-]*")  # noqa: RUF001
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
        "for", "with", "was", "were", "is", "are", "be", "been", "he", "she",
        "they", "it", "his", "her", "their", "him", "them", "i", "you", "we",
    }
)
_EMOTION_MARKERS = frozenset(
    {
        "afraid", "angry", "anxious", "ashamed", "calm", "curious", "determined",
        "disgusted", "embarrassed", "excited", "furious", "glad", "guilty",
        "hopeful", "jealous", "lonely", "nervous", "proud", "relieved", "sad",
        "shocked", "surprised", "tired", "worried", "fear", "hate", "love",
        "rage", "tears", "smile", "cry",
    }
)


@dataclass(frozen=True, slots=True)
class RelevanceWeights:
    recency: float = 0.4
    entity_overlap: float = 0.3
    causal_link: float = 0.2
    emotional_salience: float = 0.1

    def __post_init__(self) -> None:
        total = self.recency + self.entity_overlap + self.causal_link + self.emotional_salience
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError("relevance weights must sum to 1.0")


def score_entry(
    text: str,
    *,
    recency: float,
    candidate_tokens: frozenset[str],
    plan_tokens: frozenset[str],
    weights: RelevanceWeights | None = None,
) -> float:
    effective = weights or RelevanceWeights()
    entry_tokens = {token.casefold() for token in _WORD_RE.findall(text)}
    entry_tokens -= _STOPWORDS
    overlap = len(entry_tokens & candidate_tokens) / max(1, len(entry_tokens))
    causal = len(entry_tokens & plan_tokens) / max(1, len(entry_tokens))
    emotional = 1.0 if _EMOTION_MARKERS & entry_tokens else 0.0
    return (
        effective.recency * recency
        + effective.entity_overlap * overlap
        + effective.causal_link * causal
        + effective.emotional_salience * emotional
    )


def tokenize(text: str) -> frozenset[str]:
    tokens = {token.casefold() for token in _WORD_RE.findall(text)}
    tokens -= _STOPWORDS
    return frozenset(tokens)
