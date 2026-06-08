#!/bin/bash
# SLURM: run the full experiment suite (5 baselines + ablation) on test.csv.
#
# BEFORE submitting:
#   export OPENAI_API_KEY="sk-..."
#
# Submit (all 5 conditions):
#   sbatch training/hpc_suite.sbatch
#
# Resume remaining conditions (skip ones already done):
#   export SUITE_CONDITIONS="single-finetuned,pipeline-mini,pipeline-finetuned,ablation-no-analyzer"
#   sbatch training/hpc_suite.sbatch
#
# Runs all conditions on the full test set (~110 sentence groups).
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

module --force purge
module load release/24.04 GCCcore/13.2.0 Python/3.11.5

VENV="$REPO_DIR/.venv-train"
source "$VENV/bin/activate"
echo "Venv Python: $(which python) ($(python --version))"
pip install -q -r requirements.txt
# SARI metric (evaluate-metric/sari) needs these; fail fast if still missing.
python -c "import sacremoses, sacrebleu"

# Fine-tuned inference needs CUDA-enabled PyTorch. Python 3.13 venvs keep +cpu wheels.
ensure_cuda_torch() {
    if python -c "import torch; assert torch.cuda.is_available()"; then
        python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available: True')"
        return 0
    fi
    echo "Replacing CPU PyTorch with CUDA build..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    for CU in cu124 cu121; do
        echo "Trying PyTorch wheel index: $CU"
        pip install -q torch --index-url "https://download.pytorch.org/whl/$CU" || continue
        if python -c "import torch; assert torch.cuda.is_available()"; then
            python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available: True')"
            return 0
        fi
        pip uninstall -y torch 2>/dev/null || true
    done
    return 1
}

if ! ensure_cuda_torch; then
    PY_MM="$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
    echo "ERROR: CUDA not visible to PyTorch on this GPU node."
    echo "Venv Python: $PY_MM (need 3.11 + CUDA torch). Run once on login node:"
    echo "  bash training/hpc_fix_venv.sh"
    nvidia-smi || true
    exit 1
fi

export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"
export USE_HF_SIMPLIFIER=1
export HF_SIMPLIFIER_PATH="${HF_SIMPLIFIER_PATH:-$REPO_DIR/outputs/plaba-merged}"
export QUALITY_GATE_MODEL="${QUALITY_GATE_MODEL:-gpt-4o}"
export EVAL_VERBOSE=1
export PYTHONUNBUFFERED=1

nvidia-smi || true
echo "Warming up SARI + BERTScore metric backends (one-time load)..."
python -c "from app.metrics import warmup_metrics; warmup_metrics()"
CONDITIONS_ARGS=()
if [ -n "${SUITE_CONDITIONS:-}" ]; then
    CONDITIONS_ARGS=(--conditions "$SUITE_CONDITIONS")
    echo "Resuming subset: $SUITE_CONDITIONS"
fi

python main.py \
    --suite \
    --all \
    --data data/plaba/test.csv \
    --max-iterations 2 \
    --output-dir outputs/experiments \
    "${CONDITIONS_ARGS[@]}"

echo "Done. See outputs/experiments/suite-summary-*.json"
