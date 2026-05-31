from __future__ import annotations

import unittest

from app.metrics import readability_passes, sari_passes
from app.models import ReadabilityScores


class QualityGateMetricTests(unittest.TestCase):
    def test_readability_passes_within_targets(self) -> None:
        scores = ReadabilityScores(
            flesch_kincaid_grade=8.0,
            flesch_reading_ease=65.0,
            smog_index=9.0,
            gunning_fog=10.0,
            avg_sentence_length=14.0,
        )
        self.assertTrue(readability_passes(scores))

    def test_readability_fails_high_grade(self) -> None:
        scores = ReadabilityScores(
            flesch_kincaid_grade=14.0,
            flesch_reading_ease=65.0,
            smog_index=9.0,
            gunning_fog=10.0,
            avg_sentence_length=14.0,
        )
        self.assertFalse(readability_passes(scores))

    def test_sari_skipped_when_no_reference(self) -> None:
        self.assertTrue(sari_passes(None))

    def test_sari_threshold(self) -> None:
        self.assertTrue(sari_passes(40.0))
        self.assertFalse(sari_passes(20.0))


if __name__ == "__main__":
    unittest.main()
