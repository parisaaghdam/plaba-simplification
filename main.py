from __future__ import annotations

import argparse
import sys

# Ensure non-ASCII output (e.g. the "<=" symbol in metric notes) prints on
# Windows consoles that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.evaluate import evaluate_on_samples, save_experiment_run
from app.graph import simplify_with_refinement
from app.plaba_data import load_sentence_level_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Single medical text to simplify")
    parser.add_argument("--eval", action="store_true", help="Run a batch evaluation")
    parser.add_argument("--data", type=str, default="data/plaba/val.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--k", type=int, default=20, help="Number of samples to evaluate (--eval)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (--eval)")
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

    if args.eval:
        samples = load_sentence_level_samples(args.data)
        result = evaluate_on_samples(samples, k=args.k, seed=args.seed)
        paths = save_experiment_run(
            metrics=result["metrics"],
            rows=result["rows"],
            output_dir=args.output_dir,
        )
        print(result["metrics"])
        print(f"json: {paths['json']}")
        print(f"csv: {paths['csv']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
