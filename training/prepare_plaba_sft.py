"""Prepare PLABA data for supervised fine-tuning (SFT) of the simplifier.

Reads the PLABA CSV splits (columns: question, pmid, input_text, target_text,
Adaptation_Version, Question_Type) and writes instruction-tuning datasets:

- ``<split>.sft.jsonl``    -> chat format: {"messages": [...]} (used by TRL SFTTrainer)
- ``<split>.alpaca.jsonl`` -> {"instruction", "input", "output"} (human-readable)

Each (input_text, target_text) pair becomes one training example. A source with
multiple human references therefore yields multiple examples, which acts as light
data augmentation and exposes the model to varied valid simplifications.

This script is CPU-only and safe to run on any machine (no GPU needed).

Usage:
    python training/prepare_plaba_sft.py \
        --train data/plaba/train.csv \
        --val data/plaba/val.csv \
        --out-dir data/plaba/sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Kept in sync with the system prompt in app/agents.py:run_simplifier so that
# the model is fine-tuned on the same instruction it will see at inference time.
SIMPLIFIER_INSTRUCTION = (
    "You are a medical text simplifier writing for a general (lay) audience. "
    "Rewrite the source text into plain language while preserving all medical "
    "meaning. Use common words and short sentences, prefer active voice, define "
    "acronyms on first use, do not omit important facts, and do not add "
    "unsupported facts."
)


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip()


def build_examples(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    required = {"input_text", "target_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    examples: list[dict] = []
    for _, row in df.iterrows():
        source = _clean(row["input_text"])
        target = _clean(row["target_text"])
        if not source or not target:
            continue
        examples.append({"source": source, "target": target})
    return examples


def write_chat_jsonl(examples: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "messages": [
                    {"role": "system", "content": SIMPLIFIER_INSTRUCTION},
                    {"role": "user", "content": f"Source text:\n{ex['source']}"},
                    {"role": "assistant", "content": ex["target"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_alpaca_jsonl(examples: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "instruction": SIMPLIFIER_INSTRUCTION,
                "input": ex["source"],
                "output": ex["target"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_split(name: str, csv_path: Path, out_dir: Path) -> int:
    examples = build_examples(csv_path)
    write_chat_jsonl(examples, out_dir / f"{name}.sft.jsonl")
    write_alpaca_jsonl(examples, out_dir / f"{name}.alpaca.jsonl")
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PLABA SFT datasets.")
    parser.add_argument("--train", default="data/plaba/train.csv")
    parser.add_argument("--val", default="data/plaba/val.csv")
    parser.add_argument("--test", default="data/plaba/test.csv")
    parser.add_argument("--out-dir", default="data/plaba/sft")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        csv_path = Path(path)
        if not csv_path.exists():
            print(f"skip {name}: {csv_path} not found")
            continue
        count = process_split(name, csv_path, out_dir)
        print(f"{name}: {count} examples -> {out_dir / (name + '.sft.jsonl')}")

    print(f"\nDone. SFT datasets written to: {out_dir}")


if __name__ == "__main__":
    main()
