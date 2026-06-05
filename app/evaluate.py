from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from app.graph import simplify_with_refinement
from app.metrics import compute_readability_scores, compute_sari
from app.plaba_data import pick_random_samples

_EVAL_FIELDNAMES = [
    "index",
    "source",
    "prediction",
    "reference_count",
    "sari",
    "fk_grade",
    "accepted",
    "metrics_readability_ok",
    "metrics_plain_language_ok",
    "plain_language_ok",
    "metrics_sari_ok",
]


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
    k: int = 20,
    seed: int = 42,
    *,
    verbose: bool = True,
    max_iterations: int = 4,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    chosen = pick_random_samples(samples, k=k, seed=seed)
    total = len(chosen)

    if verbose:
        os.environ["EVAL_VERBOSE"] = "1"
        print(f"Evaluating {total} samples (seed={seed})...", flush=True)
        if os.getenv("USE_HF_SIMPLIFIER", "").strip() in {"1", "true", "True"}:
            print(
                "Note: using fine-tuned HF model on GPU (fast on HPC).",
                flush=True,
            )
        elif os.getenv("USE_OLLAMA_SIMPLIFIER", "").strip() in {"1", "true", "True"}:
            print(
                "Note: local Ollama on CPU is slow (~2-15 min per sample). "
                "Use EVAL_VERBOSE=1 to see each agent step.",
                flush=True,
            )

    partial_csv: Path | None = None
    if output_dir is not None:
        partial_csv = Path(output_dir) / "eval-inprogress.csv"
        if partial_csv.exists():
            partial_csv.unlink()
        if verbose:
            print(f"Partial results will be saved to: {partial_csv}", flush=True)

    sari_scores: list[float] = []
    readability_grades: list[float] = []
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
            print("  pipeline: running (see agent steps below)...", flush=True)
        result = simplify_with_refinement(
            item.source_text,
            references=item.references,
            max_iterations=max_iterations,
        )
        if verbose:
            print(f"  pipeline: finished in {time.perf_counter() - t0:.0f}s", flush=True)

        output = result.simplification
        if verbose:
            print("  computing SARI and readability...", flush=True)
        sari = compute_sari(item.source_text, output, item.references)
        if sari is not None:
            sari_scores.append(sari)
        fk_grade = compute_readability_scores(output).flesch_kincaid_grade
        readability_grades.append(fk_grade)
        accepted = bool(result.accepted)
        accepted_flags.append(accepted)
        qf = result.quality_feedback

        if verbose:
            sari_str = f"{sari:.1f}" if sari is not None else "n/a"
            status = "accepted" if accepted else "rejected"
            iters = result.iteration
            print(
                f"  -> {status} | iterations={iters} | SARI={sari_str} | FK grade={fk_grade:.1f}",
                flush=True,
            )

        row = {
            "index": idx,
            "source": item.source_text,
            "prediction": output,
            "reference_count": len(item.references),
            "sari": sari,
            "fk_grade": fk_grade,
            "accepted": accepted,
            "metrics_readability_ok": qf.metrics_readability_ok if qf else None,
            "metrics_plain_language_ok": qf.metrics_plain_language_ok if qf else None,
            "plain_language_ok": qf.plain_language_ok if qf else None,
            "metrics_sari_ok": qf.metrics_sari_ok if qf else None,
        }
        rows.append(row)
        if partial_csv is not None:
            _append_partial_row(row, partial_csv)
            if verbose:
                print(f"  saved partial row -> {partial_csv}", flush=True)

    metrics: dict[str, Any] = {
        "n_samples": len(chosen),
        "avg_fk_grade": mean(readability_grades) if readability_grades else None,
        "accepted_count": sum(accepted_flags),
    }
    if sari_scores:
        metrics["avg_sari"] = mean(sari_scores)

    if verbose:
        print("\n--- Evaluation complete ---", flush=True)
        print(f"  Accepted: {metrics['accepted_count']}/{total}", flush=True)
        if metrics.get("avg_sari") is not None:
            print(f"  Avg SARI: {metrics['avg_sari']:.2f}", flush=True)
        if metrics.get("avg_fk_grade") is not None:
            print(f"  Avg FK grade: {metrics['avg_fk_grade']:.2f}", flush=True)

    return {"metrics": metrics, "rows": rows}


def save_experiment_run(
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: str | Path = "outputs/experiments",
) -> dict[str, str]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"eval-{run_id}.json"
    csv_path = out / f"eval-{run_id}.csv"

    payload = {"run_id": run_id, "timestamp_utc": run_id, "metrics": metrics, "rows": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EVAL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in _EVAL_FIELDNAMES})

    return {"json": str(json_path), "csv": str(csv_path)}
