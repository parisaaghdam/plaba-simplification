"""Experiment conditions for PLABA baselines and ablations.

Conditions (research matrix):
  single-mini              — single-pass GPT-4o-mini, no pipeline
  single-finetuned         — single-pass fine-tuned model, no pipeline
  pipeline-mini            — full pipeline, GPT-4o-mini simplifier
  pipeline-finetuned       — full pipeline, fine-tuned simplifier (full system)
  ablation-no-analyzer     — pipeline without analyzer, fine-tuned simplifier
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

_SIMPLIFIER_ENV_KEYS = (
    # Analyzer
    "USE_OLLAMA_ANALYZER",
    "OLLAMA_ANALYZER_MODEL",
    "USE_HF_ANALYZER",
    "HF_ANALYZER_PATH",
    # Simplifier
    "USE_OLLAMA_SIMPLIFIER",
    "OLLAMA_SIMPLIFIER_MODEL",
    "USE_HF_SIMPLIFIER",
    "HF_SIMPLIFIER_PATH",
    # Quality gate
    "USE_OLLAMA_QUALITY_GATE",
    "OLLAMA_QUALITY_GATE_MODEL",
    "USE_HF_QUALITY_GATE",
    "HF_QUALITY_GATE_PATH",
    "QUALITY_GATE_MODEL",
)


@dataclass(frozen=True)
class ExperimentCondition:
    name: str
    description: str
    single_pass: bool
    skip_analyzer: bool
    simplifier: str  # "api" | "finetuned"
    max_iterations: int | None = None  # overrides the global --max-iterations when set


CONDITIONS: dict[str, ExperimentCondition] = {
    "single-mini": ExperimentCondition(
        name="single-mini",
        description="Single-pass GPT-4o-mini (no pipeline)",
        single_pass=True,
        skip_analyzer=True,
        simplifier="api",
    ),
    "single-finetuned": ExperimentCondition(
        name="single-finetuned",
        description="Single-pass fine-tuned model (no pipeline)",
        single_pass=True,
        skip_analyzer=True,
        simplifier="finetuned",
    ),
    "pipeline-mini": ExperimentCondition(
        name="pipeline-mini",
        description="Pipeline with GPT-4o-mini simplifier",
        single_pass=False,
        skip_analyzer=False,
        simplifier="api",
    ),
    "pipeline-finetuned": ExperimentCondition(
        name="pipeline-finetuned",
        description="Full pipeline with fine-tuned simplifier",
        single_pass=False,
        skip_analyzer=False,
        simplifier="finetuned",
    ),
    "ablation-no-analyzer": ExperimentCondition(
        name="ablation-no-analyzer",
        description="Pipeline minus analyzer (fine-tuned simplifier)",
        single_pass=False,
        skip_analyzer=True,
        simplifier="finetuned",
    ),
    "ablation-single-iter": ExperimentCondition(
        name="ablation-single-iter",
        description="Full pipeline, fine-tuned simplifier, max 1 refinement iteration",
        single_pass=False,
        skip_analyzer=False,
        simplifier="finetuned",
        max_iterations=1,
    ),
}

SUITE_ORDER = [
    "single-mini",
    "single-finetuned",
    "pipeline-mini",
    "pipeline-finetuned",
    "ablation-no-analyzer",
    "ablation-single-iter",
]


def _snapshot_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _SIMPLIFIER_ENV_KEYS}


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _clear_simplifier_env() -> None:
    for key in _SIMPLIFIER_ENV_KEYS:
        os.environ.pop(key, None)


def _enable_finetuned_simplifier() -> None:
    """Prefer HF GPU path on HPC; fall back to Ollama on a local PC."""
    merged = os.getenv("HF_SIMPLIFIER_PATH", "outputs/plaba-merged")
    if os.path.isfile(os.path.join(merged, "config.json")):
        os.environ["USE_HF_SIMPLIFIER"] = "1"
        os.environ["HF_SIMPLIFIER_PATH"] = merged
        return
    os.environ["USE_OLLAMA_SIMPLIFIER"] = "1"
    os.environ.setdefault("OLLAMA_SIMPLIFIER_MODEL", "plaba-simplifier-chat")


def apply_condition(condition_name: str) -> ExperimentCondition:
    if condition_name not in CONDITIONS:
        valid = ", ".join(CONDITIONS)
        raise ValueError(f"Unknown condition '{condition_name}'. Choose from: {valid}")
    cond = CONDITIONS[condition_name]
    _clear_simplifier_env()
    if cond.simplifier == "finetuned":
        _enable_finetuned_simplifier()
    return cond


@contextmanager
def experiment_context(condition_name: str) -> Iterator[ExperimentCondition]:
    snapshot = _snapshot_env()
    cond = apply_condition(condition_name)
    try:
        yield cond
    finally:
        _restore_env(snapshot)
