#!/bin/bash
# SLURM: run the full experiment suite (5 baselines + ablation) on test.csv.
#
# BEFORE submitting:
#   export OPENAI_API_KEY="sk-..."
#
# Submit:
#   sbatch training/hpc_suite.sbatch
#
# Runs all conditions on the full test set (~148 examples each).
# Quality gate uses GPT-4o; simplifier uses fine-tuned HF model on GPU.

#SBATCH --job-name=plaba-suite
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs outputs/experiments

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: export OPENAI_API_KEY before sbatch"
    exit 1
fi

module purge
module load release/24.04 || true
module load Python/3.11.5 || module load Python || true

VENV="$REPO_DIR/.venv-train"
source "$VENV/bin/activate"
pip install -q -r requirements.txt

export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"
export USE_HF_SIMPLIFIER=1
export HF_SIMPLIFIER_PATH="${HF_SIMPLIFIER_PATH:-$REPO_DIR/outputs/plaba-merged}"
export QUALITY_GATE_MODEL="${QUALITY_GATE_MODEL:-gpt-4o}"
export EVAL_VERBOSE=1
export PYTHONUNBUFFERED=1

nvidia-smi || true
python main.py \
    --suite \
    --all \
    --data data/plaba/test.csv \
    --max-iterations 2 \
    --output-dir outputs/experiments

echo "Done. See outputs/experiments/suite-summary-*.json"
