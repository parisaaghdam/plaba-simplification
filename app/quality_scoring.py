"""Weighted quality-gate scoring (replaces strict AND acceptance).

Each signal contributes a weighted partial score (0–weight). Acceptance triggers
when total score >= threshold (default 70/100), making the refine loop functional
even when one minor criterion fails.
"""

from __future__ import annotations

import os
from typing import Any

from app.models import PlainLanguageCriterionResult, QualityGateOutput, ReadabilityScores

# Weights sum to 100.
CRITERION_WEIGHTS: dict[str, float] = {
    "fidelity": 25.0,
    "no_omissions": 15.0,
    "no_hallucinations": 10.0,
    "glossary_preserved": 5.0,
    "plain_language_llm": 10.0,
    "plain_language_checklist": 10.0,
    "readability_metrics": 10.0,
    "plain_language_metrics": 5.0,
    "sari_metrics": 5.0,
    "readability_llm": 5.0,
}

DEFAULT_ACCEPT_THRESHOLD = 70.0


def acceptance_threshold() -> float:
    raw = os.getenv("QUALITY_ACCEPT_THRESHOLD", str(DEFAULT_ACCEPT_THRESHOLD))
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_ACCEPT_THRESHOLD


def _checklist_fraction(criteria: list[PlainLanguageCriterionResult]) -> float:
    if not criteria:
        return 1.0
    passed = sum(1 for c in criteria if c.passed)
    return passed / len(criteria)


def compute_quality_score(
    llm: QualityGateOutput,
    *,
    metrics_readability_ok: bool,
    metrics_plain_language_ok: bool,
    metrics_sari_ok: bool,
) -> tuple[float, dict[str, float]]:
    """Return (total_score, per-criterion breakdown)."""
    breakdown: dict[str, float] = {}

    breakdown["fidelity"] = CRITERION_WEIGHTS["fidelity"] if llm.fidelity_ok else 0.0
    breakdown["no_omissions"] = (
        CRITERION_WEIGHTS["no_omissions"] if not llm.missing_information else 0.0
    )
    breakdown["no_hallucinations"] = (
        CRITERION_WEIGHTS["no_hallucinations"] if not llm.unsupported_additions else 0.0
    )
    breakdown["glossary_preserved"] = (
        CRITERION_WEIGHTS["glossary_preserved"] if llm.glossary_terms_preserved else 0.0
    )
    breakdown["plain_language_llm"] = (
        CRITERION_WEIGHTS["plain_language_llm"] if llm.plain_language_ok else 0.0
    )
    breakdown["plain_language_checklist"] = (
        CRITERION_WEIGHTS["plain_language_checklist"] * _checklist_fraction(llm.plain_language_criteria)
    )
    breakdown["readability_metrics"] = (
        CRITERION_WEIGHTS["readability_metrics"] if metrics_readability_ok else 0.0
    )
    breakdown["plain_language_metrics"] = (
        CRITERION_WEIGHTS["plain_language_metrics"] if metrics_plain_language_ok else 0.0
    )
    breakdown["sari_metrics"] = CRITERION_WEIGHTS["sari_metrics"] if metrics_sari_ok else 0.0
    breakdown["readability_llm"] = (
        CRITERION_WEIGHTS["readability_llm"] if llm.readability_ok else 0.0
    )

    total = round(sum(breakdown.values()), 2)
    return total, breakdown


def apply_weighted_acceptance(
    llm: QualityGateOutput,
    readability: ReadabilityScores,
    sari: float | None,
    plain_failures: list[str],
    metrics_plain_language_ok: bool,
    *,
    metrics_readability_ok: bool | None = None,
    metrics_sari_ok: bool | None = None,
) -> QualityGateOutput:
    """Merge LLM verdict with metrics and apply weighted acceptance."""
    from app.metrics import readability_passes, sari_passes

    if metrics_readability_ok is None:
        metrics_readability_ok = readability_passes(readability)
    if metrics_sari_ok is None:
        metrics_sari_ok = sari_passes(sari)

    threshold = acceptance_threshold()
    score, breakdown = compute_quality_score(
        llm,
        metrics_readability_ok=metrics_readability_ok,
        metrics_plain_language_ok=metrics_plain_language_ok,
        metrics_sari_ok=metrics_sari_ok,
    )
    accepted = score >= threshold

    notes = llm.revision_notes.strip()
    if not accepted:
        weak = [k for k, v in breakdown.items() if v < CRITERION_WEIGHTS[k] * 0.5]
        notes = (
            notes
            + f"\nQuality score {score:.1f}/{threshold:.0f} below threshold. "
            f"Weak areas: {', '.join(weak) or 'multiple criteria'}."
        ).strip()
    if not metrics_readability_ok:
        from app.metrics import MAX_FK_GRADE, MIN_FLESCH_EASE

        notes = (
            notes
            + f"\nReadability metrics: FK grade {readability.flesch_kincaid_grade:.1f} "
            f"(target <={MAX_FK_GRADE:.0f}), Flesch ease {readability.flesch_reading_ease:.1f} "
            f"(target >={MIN_FLESCH_EASE:.0f})."
        ).strip()
    if not metrics_plain_language_ok and plain_failures:
        notes = (notes + "\nPlain-language metrics: " + "; ".join(plain_failures)).strip()
    if not metrics_sari_ok and sari is not None:
        notes = (notes + f"\nSARI {sari:.1f} below threshold; align closer to reference style.").strip()

    violations = list(llm.plain_language_violations)
    if not metrics_plain_language_ok:
        violations = list(dict.fromkeys(violations + plain_failures))

    return QualityGateOutput(
        fidelity_ok=llm.fidelity_ok,
        missing_information=llm.missing_information,
        unsupported_additions=llm.unsupported_additions,
        glossary_terms_preserved=llm.glossary_terms_preserved,
        plain_language_ok=llm.plain_language_ok,
        plain_language_criteria=llm.plain_language_criteria,
        plain_language_violations=violations,
        readability_scores=readability,
        readability_ok=llm.readability_ok,
        metrics_readability_ok=metrics_readability_ok,
        metrics_plain_language_ok=metrics_plain_language_ok,
        metrics_sari_ok=metrics_sari_ok,
        sari=sari,
        quality_score=score,
        acceptance_threshold=threshold,
        score_breakdown=breakdown,
        revision_notes=notes,
        accepted=accepted,
    )


def score_summary(breakdown: dict[str, Any]) -> str:
    parts = [f"{k}={v:.1f}" for k, v in sorted(breakdown.items())]
    return ", ".join(parts)
