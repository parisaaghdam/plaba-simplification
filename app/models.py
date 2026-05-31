from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator


# --- Analysis (complex words + concepts + glossary) ---


class ComplexWord(BaseModel):
    word: str = Field(min_length=1)
    reason: str = Field(
        description="Why the word is complex, e.g. Latin origin, technical jargon, rare term."
    )
    suggested_plain: str | None = None


class MedicalConcept(BaseModel):
    name: str = Field(min_length=1)
    category: str = Field(
        description="Concept type, e.g. disease, drug, procedure, anatomy, lab_test."
    )
    plain_explanation: str = Field(min_length=1)
    must_preserve: bool = True


class GlossaryEntry(BaseModel):
    term: str = Field(min_length=1)
    plain_language: str = Field(min_length=1)
    short_definition: str | None = None


class PlainLanguageIssue(BaseModel):
    criterion_id: str = Field(
        description="Plain-language criterion id, e.g. short_sentences, define_acronyms."
    )
    span_or_pattern: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)


class AnalysisOutput(BaseModel):
    complex_words: List[ComplexWord] = Field(default_factory=list)
    medical_concepts: List[MedicalConcept] = Field(default_factory=list)
    glossary: List[GlossaryEntry] = Field(default_factory=list)
    medical_terms: List[str] = Field(
        default_factory=list,
        description="Flat list of salient medical terms that downstream agents must preserve.",
    )
    plain_language_issues: List[PlainLanguageIssue] = Field(
        default_factory=list,
        description="Source-text violations of plain-language criteria with rewrite hints.",
    )


class AnalyzerInput(BaseModel):
    source_text: str = Field(min_length=1)


# --- Simplification ---


class SimplifierInput(BaseModel):
    source_text: str = Field(min_length=1)
    analysis: AnalysisOutput | None = None
    revision_notes: str | None = None


class SimplifierOutput(BaseModel):
    simplified_text: str = Field(min_length=1)


# --- Unified quality gate (faithfulness + readability metrics + optional SARI) ---


class ReadabilityScores(BaseModel):
    flesch_kincaid_grade: float
    flesch_reading_ease: float
    smog_index: float
    gunning_fog: float
    avg_sentence_length: float


class SimplificationMetrics(BaseModel):
    sari: float | None = None


class QualityGateInput(BaseModel):
    source_text: str = Field(min_length=1)
    simplified_text: str = Field(min_length=1)
    analysis: AnalysisOutput | None = None
    references: List[str] = Field(default_factory=list)
    metric_snapshot: dict = Field(default_factory=dict)


class PlainLanguageCriterionResult(BaseModel):
    criterion_id: str
    passed: bool
    note: str = ""


class QualityGateOutput(BaseModel):
    fidelity_ok: bool
    missing_information: List[str] = Field(default_factory=list)
    unsupported_additions: List[str] = Field(default_factory=list)
    glossary_terms_preserved: bool = True
    plain_language_ok: bool = True
    plain_language_criteria: List[PlainLanguageCriterionResult] = Field(default_factory=list)
    plain_language_violations: List[str] = Field(default_factory=list)
    readability_scores: ReadabilityScores
    readability_ok: bool
    metrics_readability_ok: bool
    metrics_plain_language_ok: bool = True
    metrics_sari_ok: bool
    sari: float | None = None
    revision_notes: str = ""
    accepted: bool

    @model_validator(mode="after")
    def accepted_consistent(self) -> "QualityGateOutput":
        criteria_ok = all(c.passed for c in self.plain_language_criteria) if self.plain_language_criteria else True
        should_accept = (
            self.fidelity_ok
            and self.readability_ok
            and self.plain_language_ok
            and criteria_ok
            and self.metrics_readability_ok
            and self.metrics_plain_language_ok
            and self.metrics_sari_ok
            and self.glossary_terms_preserved
            and not self.missing_information
            and not self.unsupported_additions
            and not self.plain_language_violations
        )
        if self.accepted != should_accept:
            self.accepted = should_accept
        return self


class GraphState(BaseModel):
    source_text: str
    references: List[str] = Field(default_factory=list)
    analysis: AnalysisOutput | None = None
    simplification: str = ""
    quality_feedback: QualityGateOutput | None = None
    accepted: bool = False
    iteration: int = 0
    max_iterations: int = 4


# Backward-compatible aliases for older code paths
CriticInput = QualityGateInput
CriticOutput = QualityGateOutput
