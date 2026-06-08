from __future__ import annotations

import argparse
import os
import sys

# Ensure non-ASCII output (e.g. the "<=" symbol in metric notes) prints on
# Windows consoles that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.evaluate import evaluate_on_samples, run_experiment_suite, save_experiment_run
from app.experiments import CONDITIONS, experiment_context
from app.graph import simplify_with_refinement
from app.plaba_data import load_sentence_level_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PLABA multi-agent medical text simplification"
    )
    parser.add_argument("--text", type=str, help="Single medical text to simplify")
    parser.add_argument("--eval", action="store_true", help="Run a batch evaluation")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run all baseline/ablation conditions (single-mini, pipeline-finetuned, etc.)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=list(CONDITIONS.keys()),
        help="Experiment condition for --eval (see app/experiments.py)",
    )
    parser.add_argument("--data", type=str, default="data/plaba/val.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--k", type=int, default=20, help="Number of samples (--eval). Use 0 with --all.")
    parser.add_argument("--all", action="store_true", help="Evaluate full dataset (ignores --k)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (--eval)")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Max simplifier retries per sample (--eval)",
    )
    args = parser.parse_args()

    if args.text:
        result = simplify_with_refinement(args.text)
        print("=== Analysis (glossary) ===")
        if result.analysis:
            print(result.analysis.model_dump_json(indent=2))
        print("\n=== Simplification ===")
        print(result.simplification)
        print("\n=== Quality gate ===")
        if result.quality_feedback:
            print(result.quality_feedback.model_dump_json(indent=2))
        return

    sample_k = 0 if args.all else args.k

    if args.suite:
        print("Loading dataset...", flush=True)
        samples = load_sentence_level_samples(args.data)
        print(f"Loaded {len(samples)} sentence groups from {args.data}", flush=True)
        run_experiment_suite(
            samples,
            k=sample_k,
            seed=args.seed,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir,
        )
        return

    if args.eval:
        print("Loading dataset...", flush=True)
        samples = load_sentence_level_samples(args.data)
        print(f"Loaded {len(samples)} sentence groups from {args.data}", flush=True)
        n = sample_k if sample_k > 0 else len(samples)
        if os.getenv("USE_HF_SIMPLIFIER", "").strip() in {"1", "true", "True"}:
            print(
                f"Using fine-tuned HF simplifier on GPU (~{n * 2}-{n * 5} min for {n} samples).",
                flush=True,
            )
        elif os.getenv("USE_OLLAMA_SIMPLIFIER", "").strip() in {"1", "true", "True"}:
            print(
                f"Using Ollama simplifier on CPU (~{n * 5}-{n * 15} min for {n} samples).",
                flush=True,
            )

        if args.condition:
            with experiment_context(args.condition) as cond:
                result = evaluate_on_samples(
                    samples,
                    k=sample_k,
                    seed=args.seed,
                    condition=cond,
                    max_iterations=args.max_iterations,
                    output_dir=args.output_dir,
                )
        else:
            result = evaluate_on_samples(
                samples,
                k=sample_k,
                seed=args.seed,
                max_iterations=args.max_iterations,
                output_dir=args.output_dir,
            )

        paths = save_experiment_run(
            metrics=result["metrics"],
            rows=result["rows"],
            output_dir=args.output_dir,
            condition=result["metrics"].get("condition"),
        )
        print(result["metrics"])
        print(f"json: {paths['json']}")
        print(f"csv: {paths['csv']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
