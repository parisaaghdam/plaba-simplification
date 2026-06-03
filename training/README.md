# Fine-tuning the Simplifier on PLABA (beginner guide)

This folder fine-tunes a small open model (default **Mistral-7B-Instruct**) to
simplify medical text, using the PLABA dataset. The trained model then plugs into
the LangGraph app as the `simplifier` agent.

## Why these steps?

Fine-tuning a 7B model needs a **GPU** (~16 GB VRAM). Most laptops don't have one,
so the plan is:

1. **Prepare data** — runs anywhere (CPU is fine).
2. **Train** — on a free GPU (Google Colab T4) or any CUDA machine.
3. **Run locally with Ollama** — no GPU needed for inference of a small model.

We use **QLoRA**: the big model is loaded in 4-bit (small memory), and we only
train a tiny add-on ("LoRA adapter"). This is fast, cheap, and fits a free GPU.

---

## Step 1 — Prepare the data (local, CPU)

```bash
python training/prepare_plaba_sft.py
```

This reads `data/plaba/{train,val,test}.csv` and writes to `data/plaba/sft/`:

- `*.sft.jsonl` — chat format `{"messages": [...]}` used for training
- `*.alpaca.jsonl` — human-readable `{instruction, input, output}`

Each `(input_text, target_text)` pair becomes one training example
(≈635 train / 138 val / 148 test).

## Step 2 — Train (GPU / Google Colab)

On Colab: upload `data/plaba/sft/` and the `training/` folder, pick a **T4 GPU**
runtime, then:

```bash
pip install -r training/requirements-train.txt

python training/train_qlora.py \
  --train data/plaba/sft/train.sft.jsonl \
  --val   data/plaba/sft/val.sft.jsonl \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --output-dir outputs/plaba-mistral-qlora
```

Notes:
- The script **refuses to run without a GPU** (it would take weeks on CPU).
- Mistral is gated on Hugging Face — accept the license and run
  `huggingface-cli login` first. Free alternatives that need no gating:
  `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct` (also gated).
- Output = a small **LoRA adapter** in `--output-dir`.

Tunable flags: `--epochs`, `--batch-size`, `--grad-accum`, `--learning-rate`,
`--max-length`, `--lora-r`, `--lora-alpha`.

## Step 3 — Merge + export for Ollama

```bash
python training/merge_and_export.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter-dir outputs/plaba-mistral-qlora \
  --merged-dir  outputs/plaba-merged
```

This merges the adapter into the base model and writes an Ollama `Modelfile`.
Then convert to GGUF (with llama.cpp) and register it with Ollama:

```bash
# from a llama.cpp checkout
python convert_hf_to_gguf.py outputs/plaba-merged \
  --outfile outputs/plaba-merged/plaba-simplifier.gguf --outtype q8_0

ollama create plaba-simplifier -f outputs/plaba-merged/Modelfile
```

## Step 4 — Use the fine-tuned model in the app

Tell the graph to route the **simplifier** agent to your Ollama model
(the analyzer and quality gate still use the API model):

```powershell
# Windows PowerShell
$env:USE_OLLAMA_SIMPLIFIER = "1"
$env:OLLAMA_SIMPLIFIER_MODEL = "plaba-simplifier"   # optional, this is the default
python main.py --text "Hyperkalemia is a frequent clinical abnormality..."
```

```bash
# macOS / Linux
export USE_OLLAMA_SIMPLIFIER=1
python main.py --text "Hyperkalemia is a frequent clinical abnormality..."
```

Requires `pip install langchain-ollama` and the Ollama app running.

---

## How it connects to the code

- `app/llm_factory.py` — `get_simplifier_model()` returns a `ChatOllama` model
  when `USE_OLLAMA_SIMPLIFIER=1`, otherwise the default OpenAI model.
- `app/graph.py` — the `simplifier` node uses that model; when Ollama is active it
  reads the model's **plain-text** output instead of structured JSON.
- The training instruction in `prepare_plaba_sft.py` mirrors the simplifier's
  system prompt in `app/agents.py`, so training and inference match.

## Evaluating the fine-tuned model

After Step 4, run the usual evaluation to compare against the API baseline:

```bash
python main.py --eval --data data/plaba/val.csv
```

Compare `avg_sari` and `avg_fk_grade` against the runs in `outputs/experiments/`.
