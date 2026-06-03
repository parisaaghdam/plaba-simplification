from __future__ import annotations

import unittest

from app.plain_language import compute_plain_language_metrics, plain_language_passes


class PlainLanguageMetricTests(unittest.TestCase):
    def test_short_plain_text_passes(self) -> None:
        text = "Muscle cramps hurt. They can happen during sports. Drink water to help."
        metrics = compute_plain_language_metrics(text)
        ok, failures = plain_language_passes(metrics)
        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_long_sentence_fails(self) -> None:
        text = " ".join(["word"] * 60) + "."
        metrics = compute_plain_language_metrics(text)
        ok, failures = plain_language_passes(metrics)
        self.assertFalse(ok)
        self.assertTrue(any("Longest sentence" in f for f in failures))

    def test_undefined_acronym_fails(self) -> None:
        # Four distinct, unexpanded, non-whitelisted acronyms exceed the budget.
        text = "ABCX, DEFX, GHIX, and JKLX are studied terms."
        metrics = compute_plain_language_metrics(text)
        ok, failures = plain_language_passes(metrics)
        self.assertFalse(ok)
        self.assertTrue(any("acronym" in f.lower() for f in failures))

    def test_safe_acronyms_not_flagged(self) -> None:
        text = "DNA and RNA carry genetic information in the body."
        metrics = compute_plain_language_metrics(text)
        self.assertEqual(metrics["undefined_acronyms"], [])

    def test_acronym_with_expansion_passes(self) -> None:
        text = "Exercise-associated muscle cramps (EAMC) are common."
        metrics = compute_plain_language_metrics(text)
        self.assertNotIn("EAMC", metrics["undefined_acronyms"])


if __name__ == "__main__":
    unittest.main()
