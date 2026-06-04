#!/bin/bash
# SLURM job: merge the trained LoRA adapter into the base model and convert it
# to a single GGUF file that Ollama can run. Submit AFTER training has finished.
#
# Submit with:   sbatch training/hpc_merge.sbatch
# Result:        outputs/plaba-merged/plaba-simplifier.gguf  (download this to your PC)

#SBATCH --job-name=plaba-merge
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs outputs

BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR="outputs/plaba-simplifier-qlora"
MERGED_DIR="outputs/plaba-merged"
GGUF_NAME="plaba-simplifier.gguf"

# --- environment (reuse the venv built during training) ----------------------
VENV="$REPO_DIR/.venv-train"
source "$VENV/bin/activate"
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"

# --- 1) merge the adapter into the base model --------------------------------
python training/merge_and_export.py \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$ADAPTER_DIR" \
    --merged-dir "$MERGED_DIR" \
    --gguf-name "$GGUF_NAME"

# --- 2) convert the merged HF model to GGUF (q8_0 needs no C++ build) ---------
if [ ! -d llama.cpp ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

python llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" \
    --outfile "$MERGED_DIR/$GGUF_NAME" \
    --outtype q8_0

echo "Done. GGUF written to $MERGED_DIR/$GGUF_NAME"
ls -lh "$MERGED_DIR/$GGUF_NAME"
