from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.metrics import (
    build_metric_snapshot,
    compute_readability_scores,
    readability_passes,
    sari_passes,
)
from app.models import (
    AnalysisOutput,
    AnalyzerInput,
    QualityGateInput,
    QualityGateOutput,
    ReadabilityScores,
    SimplifierInput,
    SimplifierOutput,
)
from app.plain_language import CRITERIA_PROMPT_BLOCK, plain_language_passes


def _format_glossary_for_prompt(analysis: AnalysisOutput | None) -> str:
    if not analysis or not analysis.glossary:
        return "No glossary provided."
    lines = []
    for entry in analysis.glossary:
        definition = f" — {entry.short_definition}" if entry.short_definition else ""
        lines.append(f"- {entry.term} → {entry.plain_language}{definition}")
    return "\n".join(lines)


def _format_plain_language_issues(analysis: AnalysisOutput | None) -> str:
    if not analysis or not analysis.plain_language_issues:
        return "None flagged in source."
    lines = []
    for issue in analysis.plain_language_issues:
        lines.append(
            f"- [{issue.criterion_id}] {issue.span_or_pattern}: {issue.issue} → {issue.suggestion}"
        )
    return "\n".join(lines)


def run_analyzer(model: BaseChatModel, payload: AnalyzerInput) -> AnalysisOutput:
    prompt = (
        "You are a medical text analysis agent preparing text for lay-language simplification.\n"
        "Produce:\n"
        "1) complex_words: jargon/Latin/rare/long terms with reason and suggested_plain;\n"
        "2) medical_concepts: clinical concepts with category and plain_explanation;\n"
        "3) glossary: term → plain_language (+ optional short_definition);\n"
        "4) medical_terms: salient terms that must stay medically faithful when simplified;\n"
        "5) plain_language_issues: where the SOURCE violates plain-language criteria "
        "(use criterion_id from the list below) with span, issue, and suggestion.\n\n"
        "Plain-language criteria to apply when analyzing the source:\n"
        f"{CRITERIA_PROMPT_BLOCK}\n\n"
        "Do not rewrite the full text—only analyze and build glossary plus issue list."
    )
    structured = model.with_structured_output(AnalysisOutput)
    result = structured.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Source text:\n{payload.source_text}"),
        ]
    )
    return AnalysisOutput.model_validate(result)


def run_simplifier(model: BaseChatModel, payload: SimplifierInput) -> SimplifierOutput:
    prompt = (
        "You are a medical text simplifier writing for a general (lay) audience.\n"
        "Rewrite the source into plain language while preserving all medical meaning.\n\n"
        "Plain-language writing criteria (all must be followed):\n"
        f"{CRITERIA_PROMPT_BLOCK}\n\n"
        "Additional rules:\n"
        "- Use the glossary for difficult terms; define acronyms on first use.\n"
        "- Do not omit facts listed in medical_terms.\n"
        "- Do not add unsupported facts.\n"
        "- Target ~8th-grade reading level: short sentences (≤20 words when possible), "
        "active voice, one idea per sentence, no double negatives.\n"
    )
    if payload.analysis:
        prompt += (
            f"\nMedical terms to preserve:\n"
            f"{', '.join(payload.analysis.medical_terms) or 'none listed'}\n"
            f"\nGlossary:\n{_format_glossary_for_prompt(payload.analysis)}\n"
            f"\nSource plain-language issues to fix in your rewrite:\n"
            f"{_format_plain_language_issues(payload.analysis)}\n"
        )
    if payload.revision_notes:
        prompt += f"\nQuality gate revision notes you must address:\n{payload.revision_notes}\n"

    structured = model.with_structured_output(SimplifierOutput)
    result = structured.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Source text:\n{payload.source_text}"),
        ]
    )
    return SimplifierOutput.model_validate(result)


def _merge_metric_and_llm_verdict(
    llm: QualityGateOutput,
    readability: ReadabilityScores,
    sari: float | None,
    plain_failures: list[str],
    metrics_plain_language_ok: bool,
) -> QualityGateOutput:
    metrics_readability_ok = readability_passes(readability)
    metrics_sari_ok = sari_passes(sari)
    criteria_ok = all(c.passed for c in llm.plain_language_criteria) if llm.plain_language_criteria else True
    accepted = (
        llm.fidelity_ok
        and llm.readability_ok
        and llm.plain_language_ok
        and criteria_ok
        and llm.glossary_terms_preserved
        and metrics_readability_ok
        and metrics_plain_language_ok
        and metrics_sari_ok
        and not llm.missing_information
        and not llm.unsupported_additions
        and not llm.plain_language_violations
    )
    notes = llm.revision_notes.strip()
    if not metrics_readability_ok:
        notes = (
            notes
            + f"\nReadability metrics: FK grade {readability.flesch_kincaid_grade:.1f} "
            f"(target ≤9), Flesch ease {readability.flesch_reading_ease:.1f} (target ≥60)."
        ).strip()
    if not metrics_plain_language_ok and plain_failures:
        notes = (notes + "\nPlain-language metrics: " + "; ".join(plain_failures)).strip()
    if not metrics_sari_ok and sari is not None:
        notes = (notes + f"\nSARI {sari:.1f} below threshold; align closer to reference style.").strip()

    return QualityGateOutput(
        fidelity_ok=llm.fidelity_ok,
        missing_information=llm.missing_information,
        unsupported_additions=llm.unsupported_additions,
        glossary_terms_preserved=llm.glossary_terms_preserved,
        plain_language_ok=llm.plain_language_ok,
        plain_language_criteria=llm.plain_language_criteria,
        plain_language_violations=llm.plain_language_violations,
        readability_scores=readability,
        readability_ok=llm.readability_ok,
        metrics_readability_ok=metrics_readability_ok,
        metrics_plain_language_ok=metrics_plain_language_ok,
        metrics_sari_ok=metrics_sari_ok,
        sari=sari,
        revision_notes=notes,
        accepted=accepted,
    )


def run_quality_gate(model: BaseChatModel, payload: QualityGateInput) -> QualityGateOutput:
    readability = compute_readability_scores(payload.simplified_text)
    snapshot = payload.metric_snapshot or build_metric_snapshot(
        payload.source_text,
        payload.simplified_text,
        payload.references,
    )
    sari = snapshot.get("sari")
    plain_metrics = snapshot.get("plain_language", {})
    metrics_plain_ok, plain_failures = plain_language_passes(plain_metrics)

    glossary_block = _format_glossary_for_prompt(payload.analysis)
    medical_terms = (
        ", ".join(payload.analysis.medical_terms)
        if payload.analysis and payload.analysis.medical_terms
        else "none"
    )

    prompt = (
        "You are a unified quality gate for biomedical text simplification.\n"
        "Use automatic metrics AND your judgment. Evaluate medical fidelity AND plain-language quality.\n\n"
        "Plain-language criteria (assess each; populate plain_language_criteria with passed/note):\n"
        f"{CRITERIA_PROMPT_BLOCK}\n\n"
        "Also check:\n"
        "- fidelity to source meaning\n"
        "- missing information (especially medical_terms)\n"
        "- unsupported additions / hallucinations\n"
        "- glossary terms preserved in plain language\n"
        "- Populate plain_language_violations with specific failed criteria\n"
        "Set plain_language_ok only if lay plain-language criteria are met.\n"
        "If automatic plain_language_failures are listed, address them in revision_notes."
    )

    structured = model.with_structured_output(QualityGateOutput)
    llm_result = structured.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=(
                    f"Source text:\n{payload.source_text}\n\n"
                    f"Simplified text:\n{payload.simplified_text}\n\n"
                    f"Medical terms to preserve:\n{medical_terms}\n\n"
                    f"Glossary:\n{glossary_block}\n\n"
                    f"Automatic metrics:\n{json.dumps(snapshot, indent=2)}"
                )
            ),
        ]
    )
    llm = QualityGateOutput.model_validate(llm_result)
    llm.readability_scores = readability
    llm.sari = sari
    if not metrics_plain_ok:
        llm.plain_language_violations = list(
            dict.fromkeys(llm.plain_language_violations + plain_failures)
        )
    return _merge_metric_and_llm_verdict(
        llm, readability, sari, plain_failures, metrics_plain_ok
    )


run_critic = run_quality_gate
