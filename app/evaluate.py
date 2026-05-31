from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from app.graph import simplify_with_refinement
from app.metrics import compute_readability_scores, compute_sari
from app.plaba_data import pick_random_samples


def evaluate_on_samples(samples, k: int = 20, seed: int = 42) -> dict[str, Any]:
    chosen = pick_random_samples(samples, k=k, seed=seed)

    sari_scores: list[float] = []
    readability_grades: list[float] = []
    accepted_flags: list[bool] = []
    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(chosen):
        result = simplify_with_refinement(item.source_text, references=item.references)
        output = result.simplification
        sari = compute_sari(item.source_text, output, item.references)
        if sari is not None:
            sari_scores.append(sari)
        fk_grade = compute_readability_scores(output).flesch_kincaid_grade
        readability_grades.append(fk_grade)
        accepted = bool(result.accepted)
        accepted_flags.append(accepted)
        qf = result.quality_feedback
        rows.append(
            {
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
        )

    metrics: dict[str, Any] = {
        "n_samples": len(chosen),
        "avg_fk_grade": mean(readability_grades) if readability_grades else None,
        "accepted_count": sum(accepted_flags),
    }
    if sari_scores:
        metrics["avg_sari"] = mean(sari_scores)
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

    fieldnames = [
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
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return {"json": str(json_path), "csv": str(csv_path)}
