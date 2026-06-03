"""Merge a trained LoRA adapter into the base model and prepare for Ollama.

After training (train_qlora.py), this:
1. Loads the base model in full precision and applies the LoRA adapter.
2. Merges the adapter weights into the base ("merge_and_unload").
3. Saves a standalone merged model directory.
4. Writes an Ollama Modelfile pointing at the merged weights.

To actually run the merged model in Ollama you must convert the merged
Hugging Face model to GGUF (via llama.cpp's convert_hf_to_gguf.py), then:
    ollama create plaba-simplifier -f outputs/plaba-merged/Modelfile

Usage:
    python training/merge_and_export.py \
        --base-model mistralai/Mistral-7B-Instruct-v0.3 \
        --adapter-dir outputs/plaba-mistral-qlora \
        --merged-dir outputs/plaba-merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a medical text simplifier writing for a general (lay) audience. "
    "Rewrite the source text into plain language while preserving all medical "
    "meaning. Use common words and short sentences, prefer active voice, define "
    "acronyms on first use, do not omit important facts, and do not add "
    "unsupported facts."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter and export for Ollama.")
    p.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter-dir", default="outputs/plaba-mistral-qlora")
    p.add_argument("--merged-dir", default="outputs/plaba-merged")
    p.add_argument("--gguf-name", default="plaba-simplifier.gguf")
    return p.parse_args()


def write_modelfile(merged_dir: Path, gguf_name: str) -> None:
    modelfile = merged_dir / "Modelfile"
    content = (
        f"FROM ./{gguf_name}\n\n"
        f'SYSTEM """{SYSTEM_PROMPT}"""\n\n'
        "PARAMETER temperature 0.3\n"
        "PARAMETER top_p 0.9\n"
    )
    modelfile.write_text(content, encoding="utf-8")
    print(f"Wrote Ollama Modelfile -> {modelfile}")


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = Path(args.merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model (full precision for clean merge)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
    )
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to: {merged_dir}")

    write_modelfile(merged_dir, args.gguf_name)
    print(
        "\nNext steps for Ollama:\n"
        "  1) Convert to GGUF with llama.cpp:\n"
        f"     python convert_hf_to_gguf.py {merged_dir} "
        f"--outfile {merged_dir / args.gguf_name} --outtype q8_0\n"
        f"  2) ollama create plaba-simplifier -f {merged_dir / 'Modelfile'}\n"
        "  3) Run the graph with: set USE_OLLAMA_SIMPLIFIER=1\n"
    )


if __name__ == "__main__":
    main()
