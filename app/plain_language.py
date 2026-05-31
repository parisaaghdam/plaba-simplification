from __future__ import annotations

import re
from typing import Any

# Criteria aligned with PLAIN (plainlanguage.gov) and NIH/CDC health literacy guidance.
PLAIN_LANGUAGE_CRITERIA: list[dict[str, str]] = [
    {
        "id": "common_words",
        "name": "Common words",
        "description": "Prefer everyday words; replace jargon with glossary plain forms.",
    },
    {
        "id": "short_sentences",
        "name": "Short sentences",
        "description": "Aim for ≤20 words per sentence; split long sentences.",
    },
    {
        "id": "one_idea",
        "name": "One idea per sentence",
        "description": "Each sentence should express a single clear point.",
    },
    {
        "id": "active_voice",
        "name": "Active voice",
        "description": "Prefer active voice over passive constructions when possible.",
    },
    {
        "id": "define_acronyms",
        "name": "Define acronyms",
        "description": "Spell out acronyms on first use, then use the short form.",
    },
    {
        "id": "explain_terms",
        "name": "Explain medical terms",
        "description": "Use glossary plain-language forms; add brief parenthetical explanations when needed.",
    },
    {
        "id": "no_double_negatives",
        "name": "Avoid double negatives",
        "description": "Use positive wording (e.g. 'often' not 'not uncommon').",
    },
    {
        "id": "concrete_language",
        "name": "Concrete language",
        "description": "Prefer specific, concrete wording over vague abstractions.",
    },
    {
        "id": "logical_flow",
        "name": "Logical flow",
        "description": "Keep cause-effect and lists in a clear order for lay readers.",
    },
]

CRITERIA_PROMPT_BLOCK = "\n".join(
    f"- [{c['id']}] {c['name']}: {c['description']}" for c in PLAIN_LANGUAGE_CRITERIA
)

_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|been|being)\s+(?:\w+ed|shown|found|seen|used|given|made)\b",
    re.IGNORECASE,
)
_DOUBLE_NEGATIVE_RE = re.compile(
    r"\b(?:not|no|never|without)\s+\w+\s+(?:un\w+|in\w+|non\w+|no\b)",
    re.IGNORECASE,
)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_PAREN_EXPANSION_RE = re.compile(r"\(([A-Za-z][\w\s\-]{2,})\)")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _word_count(sentence: str) -> int:
    return len(re.findall(r"\b\w+\b", sentence))


def _long_words(text: str, min_len: int = 12) -> int:
    return sum(1 for w in re.findall(r"\b[A-Za-z]+\b", text) if len(w) >= min_len)


def _undefined_acronyms(text: str) -> list[str]:
    undefined: list[str] = []
    for match in _ACRONYM_RE.finditer(text):
        acr = match.group()
        start = max(0, match.start() - 80)
        window = text[start : match.end() + 80]
        if _PAREN_EXPANSION_RE.search(window):
            continue
        if acr not in undefined:
            undefined.append(acr)
    return undefined


def compute_plain_language_metrics(text: str) -> dict[str, Any]:
    sentences = _sentences(text)
    lengths = [_word_count(s) for s in sentences] if sentences else [0]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    max_len = max(lengths) if lengths else 0
    long_sentence_count = sum(1 for n in lengths if n > 20)
    passive_hits = len(_PASSIVE_RE.findall(text))
    double_negative_hits = len(_DOUBLE_NEGATIVE_RE.findall(text))
    undefined_acronyms = _undefined_acronyms(text)
    long_word_count = _long_words(text)

    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_len, 2),
        "max_sentence_length": max_len,
        "long_sentence_count": long_sentence_count,
        "long_word_count": long_word_count,
        "passive_voice_hits": passive_hits,
        "double_negative_hits": double_negative_hits,
        "undefined_acronyms": undefined_acronyms,
    }


def plain_language_passes(
    metrics: dict[str, Any],
    *,
    max_avg_sentence_length: float = 20.0,
    max_sentence_length: int = 30,
    max_long_sentences: int = 2,
    max_passive_hits: int = 3,
    max_double_negatives: int = 0,
    max_undefined_acronyms: int = 0,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if metrics["avg_sentence_length"] > max_avg_sentence_length:
        failures.append(
            f"Average sentence length {metrics['avg_sentence_length']:.1f} "
            f"exceeds {max_avg_sentence_length}."
        )
    if metrics["max_sentence_length"] > max_sentence_length:
        failures.append(
            f"Longest sentence has {metrics['max_sentence_length']} words "
            f"(max {max_sentence_length})."
        )
    if metrics["long_sentence_count"] > max_long_sentences:
        failures.append(
            f"{metrics['long_sentence_count']} sentences exceed 20 words "
            f"(max {max_long_sentences} allowed)."
        )
    if metrics["passive_voice_hits"] > max_passive_hits:
        failures.append(
            f"Passive voice hits ({metrics['passive_voice_hits']}) exceed {max_passive_hits}."
        )
    if metrics["double_negative_hits"] > max_double_negatives:
        failures.append("Double negatives detected; rephrase positively.")
    if len(metrics["undefined_acronyms"]) > max_undefined_acronyms:
        acrs = ", ".join(metrics["undefined_acronyms"])
        failures.append(f"Undefined acronyms: {acrs}. Spell out on first use.")

    return len(failures) == 0, failures
