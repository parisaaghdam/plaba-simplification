from __future__ import annotations

import unittest

from app.models import PlainLanguageCriterionResult, QualityGateOutput, ReadabilityScores
from app.quality_scoring import apply_weighted_acceptance, compute_quality_score


def _base_llm(**overrides) -> QualityGateOutput:
    defaults = dict(
        fidelity_ok=True,
        missing_information=[],
        unsupported_additions=[],
        glossary_terms_preserved=True,
        plain_language_ok=True,
        plain_language_criteria=[
            PlainLanguageCriterionResult(criterion_id="short_sentences", passed=True),
            PlainLanguageCriterionResult(criterion_id="active_voice", passed=True),
        ],
        plain_language_violations=[],
        readability_scores=ReadabilityScores(
            flesch_kincaid_grade=10.0,
            flesch_reading_ease=50.0,
            smog_index=10.0,
            gunning_fog=10.0,
            avg_sentence_length=14.0,
        ),
        readability_ok=True,
        metrics_readability_ok=True,
        metrics_plain_language_ok=True,
        metrics_sari_ok=True,
        sari=42.0,
    )
    defaults.update(overrides)
    return QualityGateOutput(**defaults)


class QualityScoringTests(unittest.TestCase):
    def test_perfect_score_accepts(self) -> None:
        llm = _base_llm()
        readability = llm.readability_scores
        out = apply_weighted_acceptance(
            llm, readability, sari=42.0, plain_failures=[], metrics_plain_language_ok=True
        )
        self.assertGreaterEqual(out.quality_score, 70.0)
        self.assertTrue(out.accepted)

    def test_major_fidelity_failure_rejects(self) -> None:
        llm = _base_llm(fidelity_ok=False, missing_information=["hyperkalemia"])
        readability = llm.readability_scores
        out = apply_weighted_acceptance(
            llm, readability, sari=42.0, plain_failures=[], metrics_plain_language_ok=True
        )
        self.assertLess(out.quality_score, 70.0)
        self.assertFalse(out.accepted)

    def test_partial_checklist_reduces_score(self) -> None:
        llm = _base_llm(
            plain_language_criteria=[
                PlainLanguageCriterionResult(criterion_id="a", passed=True),
                PlainLanguageCriterionResult(criterion_id="b", passed=False),
            ]
        )
        score, breakdown = compute_quality_score(
            llm,
            metrics_readability_ok=True,
            metrics_plain_language_ok=True,
            metrics_sari_ok=True,
        )
        self.assertLess(breakdown["plain_language_checklist"], 10.0)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
