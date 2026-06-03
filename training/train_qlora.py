"""QLoRA fine-tuning of a small open model (default Mistral-7B-Instruct) on PLABA.

This trains a LoRA adapter on top of a 4-bit quantized base model, so it fits on
a single ~16 GB GPU (e.g. a free Google Colab T4). It is NOT meant to run on CPU.

Prerequisites (install on the GPU machine / Colab):
    pip install -r training/requirements-train.txt

Prepare data first (CPU is fine):
    python training/prepare_plaba_sft.py

Train:
    python training/train_qlora.py \
        --train data/plaba/sft/train.sft.jsonl \
        --val data/plaba/sft/val.sft.jsonl \
        --base-model mistralai/Mistral-7B-Instruct-v0.3 \
        --output-dir outputs/plaba-mistral-qlora

The trained adapter is saved to <output-dir>. Use training/merge_and_export.py to
merge it into the base weights and prepare it for Ollama.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune the PLABA simplifier.")
    p.add_argument("--train", default="data/plaba/sft/train.sft.jsonl")
    p.add_argument("--val", default="data/plaba/sft/val.sft.jsonl")
    p.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--output-dir", default="outputs/plaba-mistral-qlora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Imported here so the file can be read/inspected without the heavy GPU stack.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. QLoRA fine-tuning of a 7B model needs a GPU "
            "(~16 GB VRAM). Run this on a CUDA machine or a free Google Colab T4."
        )

    bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    # 1) Load datasets (chat format produced by prepare_plaba_sft.py).
    data_files = {"train": args.train, "validation": args.val}
    dataset = load_dataset("json", data_files=data_files)

    # 2) Tokenizer (Mistral Instruct ships a chat template SFTTrainer will apply).
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Load base model in 4-bit (QLoRA).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype,
    )
    model.config.use_cache = False

    # 4) LoRA adapter config (attention + MLP projection layers of Mistral/Llama).
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # 5) Training configuration. Higher LR is standard for QLoRA.
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16_supported,
        fp16=not bf16_supported,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # 6) Save the LoRA adapter + tokenizer.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to: {args.output_dir}")
    print("Next: merge + export for Ollama with training/merge_and_export.py")


if __name__ == "__main__":
    main()
