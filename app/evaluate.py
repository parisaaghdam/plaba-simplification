from __future__ import annotations

import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from app.experiments import CONDITIONS, SUITE_ORDER, ExperimentCondition, experiment_context
from app.graph import simplify_with_refinement
from app.metrics import compute_bertscore_f1, compute_readability_scores, compute_sari
from app.plaba_data import pick_random_samples


def bootstrap_ci(
    scores: list[float],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI for the mean.

    Returns (lower, upper) rounded to 4 decimal places, or None if fewer than
    2 observations are available (CI is undefined).
    """
    if len(scores) < 2:
        return None
    rng = random.Random(seed)
    n = len(scores)
    boot_means = sorted(mean(rng.choices(scores, k=n)) for _ in range(n_boot))
    alpha = (1 - confidence) / 2
    lo = boot_means[int(alpha * n_boot)]
    hi = boot_means[min(int((1 - alpha) * n_boot), n_boot - 1)]
    return round(lo, 4), round(hi, 4)


_EVAL_FIELDNAMES = [
    "index",
    "condition",
    "source",
    "prediction",
    "reference_count",
    "sari",
    "bertscore_f1",
    "fk_grade",
    "quality_score",
    "accepted",
    "iterations",
    "metrics_readability_ok",
    "metrics_plain_language_ok",
    "plain_language_ok",
    "metrics_sari_ok",
]


def select_samples(samples: list, k: int, seed: int) -> list:
    """k <= 0 means use all samples (full test set)."""
    if k <= 0 or k >= len(samples):
        return list(samples)
    return pick_random_samples(samples, k=k, seed=seed)


def _append_partial_row(row: dict[str, Any], partial_csv: Path) -> None:
    partial_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not partial_csv.exists()
    with partial_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EVAL_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in _EVAL_FIELDNAMES})


def evaluate_on_samples(
    samples,
    k: int = 0,
    seed: int = 42,
    *,
    condition: ExperimentCondition | None = None,
    verbose: bool = True,
    max_iterations: int = 4,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    chosen = select_samples(samples, k=k, seed=seed)
    total = len(chosen)
    cond_name = condition.name if condition else "default"
    single_pass = condition.single_pass if condition else False
    skip_analyzer = condition.skip_analyzer if condition else False
    # Condition-level override (e.g. ablation-single-iter forces max_iterations=1)
    if condition is not None and condition.max_iterations is not None:
        max_iterations = condition.max_iterations

    if verbose:
        os.environ["EVAL_VERBOSE"] = "1"
        print(f"Condition: {cond_name}", flush=True)
        if condition:
            print(f"  {condition.description}", flush=True)
        print(f"Evaluating {total} samples (seed={seed})...", flush=True)
        if os.getenv("USE_HF_SIMPLIFIER", "").strip() in {"1", "true", "True"}:
            print("Note: using fine-tuned HF model on GPU.", flush=True)
        elif os.getenv("USE_OLLAMA_SIMPLIFIER", "").strip() in {"1", "true", "True"}:
            print(
                "Note: local Ollama on CPU is slow (~2-15 min per sample).",
                flush=True,
            )

    partial_csv: Path | None = None
    if output_dir is not None:
        partial_csv = Path(output_dir) / f"eval-inprogress-{cond_name}.csv"
        if partial_csv.exists():
            partial_csv.unlink()
        if verbose:
            print(f"Partial results: {partial_csv}", flush=True)

    sari_scores: list[float] = []
    bertscore_scores: list[float] = []
    readability_grades: list[float] = []
    quality_scores: list[float] = []
    accepted_flags: list[bool] = []
    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(chosen):
        if verbose:
            preview = item.source_text[:60].replace("\n", " ")
            if len(item.source_text) > 60:
                preview += "..."
            print(f"\n[{idx + 1}/{total}] {preview}", flush=True)

        t0 = time.perf_counter()
        if verbose:
            print("  pipeline: running...", flush=True)
        result = simplify_with_refinement(
            item.source_text,
            references=item.references,
            max_iterations=max_iterations,
            skip_analyzer=skip_analyzer,
            single_pass=single_pass,
        )
        if verbose:
            print(f"  pipeline: finished in {time.perf_counter() - t0:.0f}s", flush=True)

        output = result.simplification
        if verbose:
            print("  computing SARI, BERTScore, readability...", flush=True)
        sari = compute_sari(item.source_text, output, item.references)
        bertscore = compute_bertscore_f1(output, item.references)
        if sari is not None:
            sari_scores.append(sari)
        if bertscore is not None:
            bertscore_scores.append(bertscore)
        fk_grade = compute_readability_scores(output).flesch_kincaid_grade
        readability_grades.append(fk_grade)
        accepted = bool(result.accepted)
        accepted_flags.append(accepted)
        qf = result.quality_feedback
        q_score = qf.quality_score if qf else None
        if q_score is not None:
            quality_scores.append(q_score)

        if verbose:
            sari_str = f"{sari:.1f}" if sari is not None else "n/a"
            bert_str = f"{bertscore:.3f}" if bertscore is not None else "n/a"
            q_str = f"{q_score:.1f}" if q_score is not None else "n/a"
            status = "accepted" if accepted else "rejected"
            print(
                f"  -> {status} | iters={result.iteration} | "
                f"SARI={sari_str} | BERTScore={bert_str} | "
                f"Q={q_str} | FK={fk_grade:.1f}",
                flush=True,
            )

        row = {
            "index": idx,
            "condition": cond_name,
            "source": item.source_text,
            "prediction": output,
            "reference_count": len(item.references),
            "sari": sari,
            "bertscore_f1": bertscore,
            "fk_grade": fk_grade,
            "quality_score": q_score,
            "accepted": accepted,
            "iterations": result.iteration,
            "metrics_readability_ok": qf.metrics_readability_ok if qf else None,
            "metrics_plain_language_ok": qf.metrics_plain_language_ok if qf else None,
            "plain_language_ok": qf.plain_language_ok if qf else None,
            "metrics_sari_ok": qf.metrics_sari_ok if qf else None,
        }
        rows.append(row)
        if partial_csv is not None:
            _append_partial_row(row, partial_csv)

    # --- Automatic metrics (fully reproducible, model-independent) ---
    metrics: dict[str, Any] = {
        "condition": cond_name,
        "description": condition.description if condition else "default pipeline",
        "n_samples": len(chosen),
        "seed": seed,
    }
    if sari_scores:
        metrics["avg_sari"] = round(mean(sari_scores), 4)
        ci = bootstrap_ci(sari_scores, seed=seed)
        if ci:
            metrics["avg_sari_ci95"] = list(ci)
    if bertscore_scores:
        metrics["avg_bertscore_f1"] = round(mean(bertscore_scores), 4)
        ci = bootstrap_ci(bertscore_scores, seed=seed)
        if ci:
            metrics["avg_bertscore_f1_ci95"] = list(ci)
    if readability_grades:
        metrics["avg_fk_grade"] = round(mean(readability_grades), 4)
        ci = bootstrap_ci(readability_grades, seed=seed)
        if ci:
            metrics["avg_fk_grade_ci95"] = list(ci)

    # --- LLM-judge metrics (depend on the quality gate model — not independent) ---
    # These are useful as pipeline-internal signals but should NOT be used as
    # the primary comparison metric in publications, because the same LLM that
    # runs the quality gate also influences the acceptance decision.
    metrics["_llm_judge_note"] = (
        "quality_score and acceptance_rate are produced by the quality gate LLM. "
        "They are not independent of the pipeline. Use avg_sari / avg_bertscore_f1 "
        "/ avg_fk_grade as primary reported metrics."
    )
    metrics["accepted_count"] = sum(accepted_flags)
    metrics["acceptance_rate"] = round(sum(accepted_flags) / len(chosen), 4) if chosen else 0.0
    if quality_scores:
        metrics["avg_quality_score"] = round(mean(quality_scores), 4)

    if verbose:
        print("\n--- Evaluation complete ---", flush=True)
        print(f"  Condition : {cond_name}", flush=True)
        print(f"  n_samples : {total}", flush=True)
        if metrics.get("avg_sari") is not None:
            ci = metrics.get("avg_sari_ci95", [])
            ci_str = f"  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
            print(f"  SARI      : {metrics['avg_sari']:.2f}{ci_str}", flush=True)
        if metrics.get("avg_bertscore_f1") is not None:
            ci = metrics.get("avg_bertscore_f1_ci95", [])
            ci_str = f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            print(f"  BERTScore : {metrics['avg_bertscore_f1']:.3f}{ci_str}", flush=True)
        if metrics.get("avg_fk_grade") is not None:
            ci = metrics.get("avg_fk_grade_ci95", [])
            ci_str = f"  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
            print(f"  FK grade  : {metrics['avg_fk_grade']:.2f}{ci_str}", flush=True)
        print(
            f"  Accepted  : {metrics['accepted_count']}/{total} "
            f"(LLM-judge — not primary metric)",
            flush=True,
        )

    return {"metrics": metrics, "rows": rows}


def run_experiment_suite(
    samples,
    *,
    k: int = 0,
    seed: int = 42,
    max_iterations: int = 2,
    output_dir: str | Path = "outputs/experiments",
    conditions: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all baseline/ablation conditions and return a comparison table."""
    names = conditions or SUITE_ORDER
    all_results: dict[str, Any] = {"conditions": {}, "comparison": []}

    for name in names:
        if name not in CONDITIONS:
            raise ValueError(f"Unknown condition: {name}")
        if verbose:
            print(f"\n{'=' * 60}\nRunning condition: {name}\n{'=' * 60}", flush=True)
        with experiment_context(name) as cond:
            result = evaluate_on_samples(
                samples,
                k=k,
                seed=seed,
                condition=cond,
                verbose=verbose,
                max_iterations=max_iterations,
                output_dir=output_dir,
            )
            paths = save_experiment_run(
                metrics=result["metrics"],
                rows=result["rows"],
                output_dir=output_dir,
                condition=name,
            )
            all_results["conditions"][name] = {
                "metrics": result["metrics"],
                "paths": paths,
            }
            all_results["comparison"].append(result["metrics"])

    summary_path = Path(output_dir) / f"suite-summary-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"\nSuite summary: {summary_path}", flush=True)
        print("\n--- Comparison ---", flush=True)
        for row in all_results["comparison"]:
            sari = row.get("avg_sari")
            bert = row.get("avg_bertscore_f1")
            sari_str = f"{sari:.2f}" if sari is not None else "n/a"
            bert_str = f"{bert:.3f}" if bert is not None else "n/a"
            print(
                f"  {row['condition']}: SARI={sari_str} BERTScore={bert_str} "
                f"accept={row.get('accepted_count')}/{row.get('n_samples')}",
                flush=True,
            )
    return all_results


def save_experiment_run(
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: str | Path = "outputs/experiments",
    *,
    condition: str | None = None,
) -> dict[str, str]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"eval-{condition}-" if condition else "eval-"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{prefix}{run_id}.json"
    csv_path = out / f"{prefix}{run_id}.csv"

    payload = {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "condition": condition or metrics.get("condition"),
        "metrics": metrics,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EVAL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in _EVAL_FIELDNAMES})

    return {"json": str(json_path), "csv": str(csv_path)}
