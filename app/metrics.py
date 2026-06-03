from __future__ import annotations

from typing import Any

import textstat

from app.models import ReadabilityScores, SimplificationMetrics
from app.plain_language import compute_plain_language_metrics, plain_language_passes

# Readability thresholds calibrated to the PLABA gold adaptations on val.csv
# (FK grade: median 12.4, p75 13.97; Flesch ease: median 42, p25 33.6).
# Targets are set so that output at least as readable as the harder human
# references is accepted, rather than an unrealistic grade-8/9 target that the
# gold simplifications themselves do not meet.
MAX_FK_GRADE = 14.0
MIN_FLESCH_EASE = 33.0


def compute_readability_scores(text: str) -> ReadabilityScores:
    return ReadabilityScores(
        flesch_kincaid_grade=textstat.flesch_kincaid_grade(text),
        flesch_reading_ease=textstat.flesch_reading_ease(text),
        smog_index=textstat.smog_index(text),
        gunning_fog=textstat.gunning_fog(text),
        avg_sentence_length=textstat.avg_sentence_length(text),
    )


def compute_sari(
    source: str,
    prediction: str,
    references: list[str],
) -> float | None:
    if not references:
        return None
    import evaluate

    sari_metric = evaluate.load("sari")
    result = sari_metric.compute(
        sources=[source],
        predictions=[prediction],
        references=[references],
    )
    return float(result["sari"])


def readability_passes(
    scores: ReadabilityScores,
    *,
    max_fk_grade: float = MAX_FK_GRADE,
    min_flesch_ease: float = MIN_FLESCH_EASE,
) -> bool:
    return scores.flesch_kincaid_grade <= max_fk_grade and scores.flesch_reading_ease >= min_flesch_ease


def sari_passes(sari: float | None, *, min_sari: float = 35.0) -> bool:
    if sari is None:
        return True
    return sari >= min_sari


def build_metric_snapshot(
    source: str,
    prediction: str,
    references: list[str] | None,
) -> dict[str, Any]:
    readability = compute_readability_scores(prediction)
    sari = compute_sari(source, prediction, references or [])
    plain = compute_plain_language_metrics(prediction)
    plain_ok, plain_failures = plain_language_passes(plain)
    return {
        "readability": readability.model_dump(),
        "readability_passed": readability_passes(readability),
        "plain_language": plain,
        "plain_language_passed": plain_ok,
        "plain_language_failures": plain_failures,
        "sari": sari,
        "sari_passed": sari_passes(sari),
    }


def to_simplification_metrics(sari: float | None) -> SimplificationMetrics:
    return SimplificationMetrics(sari=sari)
