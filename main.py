from __future__ import annotations

import argparse

from app.evaluate import evaluate_on_samples, save_experiment_run
from app.graph import simplify_with_refinement
from app.plaba_data import load_sentence_level_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Single medical text to simplify")
    parser.add_argument("--eval", action="store_true", help="Run random 20-sample evaluation")
    parser.add_argument("--data", type=str, default="data/plaba/val.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
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
        result = evaluate_on_samples(samples, k=20, seed=42)
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
