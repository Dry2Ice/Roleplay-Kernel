from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StyleTracker:
    max_openings: int = 8
    max_semantics: int = 32
    recent_openings: list[str] = field(default_factory=list)
    used_semantics: list[str] = field(default_factory=list)
    used_patterns: list[str] = field(default_factory=list)

    def record(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        opening = stripped[:20]
        if opening not in self.recent_openings:
            self.recent_openings.append(opening)
        if len(self.recent_openings) > self.max_openings:
            self.recent_openings.pop(0)
        semantics = _extract_semantic_keys(stripped)
        for key in semantics:
            if key not in self.used_semantics:
                self.used_semantics.append(key)
        if len(self.used_semantics) > self.max_semantics:
            self.used_semantics.pop(0)
        pattern = _sentence_pattern(stripped)
        if pattern and pattern not in self.used_patterns:
            self.used_patterns.append(pattern)
        if len(self.used_patterns) > self.max_semantics:
            self.used_patterns.pop(0)

    def constraints(self) -> dict[str, list[str]]:
        return {
            "avoid_openings": list(self.recent_openings),
            "avoid_semantics": list(self.used_semantics[-16:]),
            "avoid_patterns": list(self.used_patterns[-16:]),
        }

    def reset(self) -> None:
        self.recent_openings.clear()
        self.used_semantics.clear()
        self.used_patterns.clear()


def _extract_semantic_keys(text: str) -> list[str]:
    lowered = text.casefold()
    keys: list[str] = []
    markers = (
        "looked", "gaze", "eyes", "smiled", "frowned", "sighed",
        "nodded", "shook", "laughed", "whispered", "shouted",
        "silence", "quiet", "dark", "light", "cold", "warm",
        "door", "window", "room", "street", "rain", "snow",
    )
    for marker in markers:
        if marker in lowered:
            keys.append(marker)
    return keys


def _sentence_pattern(text: str) -> str:
    normalized = text.replace("!", ".").replace("?", ".")
    sentences = [part.strip() for part in normalized.split(".") if part.strip()]
    if not sentences:
        return ""
    lengths = [len(sentence.split()) for sentence in sentences]
    avg = sum(lengths) / len(lengths)
    if avg < 6:
        return "short_sentences"
    if avg > 20:
        return "long_sentences"
    return "medium_sentences"
