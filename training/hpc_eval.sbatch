#!/bin/bash
# SLURM batch script: run batch evaluation on TU Dresden HPC (Alpha GPU).
#
# Uses your fine-tuned merged model on GPU (no Ollama needed).
# Analyzer + quality gate still use OpenAI API (needs network + API key).
#
# BEFORE submitting, set your OpenAI key in the shell:
#   export OPENAI_API_KEY="sk-..."
#
# Submit from the repo root on login1.alpha:
#   sbatch training/hpc_eval.sbatch
#
# Watch progress:
#   squeue --me
#   tail -f logs/plaba-eval-<jobid>.out
#   tail -f logs/plaba-eval-<jobid>.err
#
# Results:
#   outputs/experiments/eval-<timestamp>.json
#   outputs/experiments/eval-<timestamp>.csv
#   outputs/experiments/eval-inprogress.csv  (updated after each sample)

#SBATCH --job-name=plaba-eval
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --account=p_your_project

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs outputs/experiments

# --- OpenAI key (required for analyzer + quality gate) -----------------------
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: Set OPENAI_API_KEY before sbatch:"
    echo "  export OPENAI_API_KEY=\"sk-...\""
    echo "  sbatch training/hpc_eval.sbatch"
    exit 1
fi

# --- Python environment (reuse training venv + install app deps) -------------
module purge
module load release/24.04 || true
module load Python/3.11.5 || module load Python || true

VENV="$REPO_DIR/.venv-train"
if [ ! -d "$VENV" ]; then
    echo "ERROR: .venv-train not found. Run training first (hpc_train.sbatch)."
    exit 1
fi
source "$VENV/bin/activate"

# App dependencies (langgraph, langchain-openai, etc.) — idempotent install.
pip install -q -r requirements.txt

export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"
mkdir -p "$HF_HOME"

# --- Use fine-tuned model on GPU (merged weights from hpc_merge.sbatch) -------
export USE_HF_SIMPLIFIER=1
export HF_SIMPLIFIER_PATH="${HF_SIMPLIFIER_PATH:-$REPO_DIR/outputs/plaba-merged}"
export EVAL_VERBOSE=1
export PYTHONUNBUFFERED=1

if [ ! -f "$HF_SIMPLIFIER_PATH/config.json" ]; then
    echo "ERROR: Merged model not found at $HF_SIMPLIFIER_PATH"
    echo "Run: sbatch training/hpc_merge.sbatch"
    exit 1
fi

# --- Run evaluation ----------------------------------------------------------
nvidia-smi || true
echo "HF simplifier path: $HF_SIMPLIFIER_PATH"
echo "Starting eval: k=20, seed=42, max_iterations=2"

python main.py \
    --eval \
    --data data/plaba/val.csv \
    --k 20 \
    --seed 42 \
    --max-iterations 2 \
    --output-dir outputs/experiments

echo "Done. Check outputs/experiments/ for eval-*.json and eval-*.csv"
