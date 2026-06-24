#!/bin/bash
# SLURM: run the full experiment suite (5 conditions) on test.csv.
#
# Fully open-source — no OpenAI API key required for 4 of 5 conditions.
# NOTE: the "single-mini" baseline condition uses GPT-4o-mini as the simplifier
#       (it is a commercial-model baseline for comparison). Set OPENAI_API_KEY
#       if you want to include it; omit to skip it.
#
# All other conditions use open-source HF models:
#   Analyzer    : Qwen/Qwen2.5-7B-Instruct  (GPU, cached)
#   Quality gate: Qwen/Qwen2.5-7B-Instruct  (GPU, shared with analyzer)
#   Simplifier  : fine-tuned 7B model        (GPU, outputs/plaba-merged)
#
# GPU memory (A100 40 GB, fp16):
#   Qwen2.5-7B  (analyzer + gate, cached): ~14 GB
#   Fine-tuned simplifier:                 ~14 GB
#   BERTScore + overhead:                  ~ 4 GB
#   Total:                                 ~32 GB  ✓
#
# PRE-REQUISITE: run hpc_download.sbatch once to cache Qwen2.5-7B-Instruct.
#
# Submit (all 5 conditions):
#   sbatch training/hpc_suite.sbatch
#
# Submit (skip single-mini — fully open-source, no API key needed):
#   export SUITE_CONDITIONS="single-finetuned,pipeline-mini,pipeline-finetuned,ablation-no-analyzer"
#   sbatch training/hpc_suite.sbatch
#
# Resume remaining conditions after a failed run:
#   export SUITE_CONDITIONS="pipeline-finetuned,ablation-no-analyzer"
#   sbatch training/hpc_suite.sbatch

#SBATCH --job-name=plaba-suite
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --account=p_your_project

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs outputs/experiments

# --- Python environment ------------------------------------------------------
module --force purge
module load release/24.04 GCCcore/13.2.0 Python/3.11.5

VENV="$REPO_DIR/.venv-train"
source "$VENV/bin/activate"
echo "Venv Python: $(which python) ($(python --version))"
pip install -q -r requirements.txt
python -c "import sacremoses, sacrebleu"

# --- CUDA check --------------------------------------------------------------
ensure_cuda_torch() {
    if python -c "import torch; assert torch.cuda.is_available()"; then
        python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.get_device_name(0))"
        return 0
    fi
    echo "Replacing CPU PyTorch with CUDA build..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    for CU in cu124 cu121; do
        pip install -q torch --index-url "https://download.pytorch.org/whl/$CU" || continue
        if python -c "import torch; assert torch.cuda.is_available()"; then
            python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.get_device_name(0))"
            return 0
        fi
        pip uninstall -y torch 2>/dev/null || true
    done
    return 1
}

if ! ensure_cuda_torch; then
    echo "ERROR: CUDA not visible. Run: bash training/hpc_fix_venv.sh"
    nvidia-smi || true
    exit 1
fi

# --- HF cache on scratch -----------------------------------------------------
export HF_HOME="${HF_HOME:-/scratch/ws/$(whoami)-hf-cache}"
mkdir -p "$HF_HOME"

# --- Model paths -------------------------------------------------------------
MERGED_MODEL="${HF_SIMPLIFIER_PATH:-$REPO_DIR/outputs/plaba-merged}"
OPEN_MODEL="${HF_OPEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

if [ ! -f "$MERGED_MODEL/config.json" ]; then
    echo "ERROR: Fine-tuned model not found at $MERGED_MODEL"
    echo "Run: sbatch training/hpc_merge.sbatch"
    exit 1
fi

# --- Fully open-source routing -----------------------------------------------
# Analyzer and quality gate use Qwen2.5-7B-Instruct (shared GPU weights).
# Simplifier uses the fine-tuned merged model.
# The experiment_context in experiments.py saves/restores these per condition.
export USE_HF_ANALYZER=1
export HF_ANALYZER_PATH="$OPEN_MODEL"
export USE_HF_SIMPLIFIER=1
export HF_SIMPLIFIER_PATH="$MERGED_MODEL"
export USE_HF_QUALITY_GATE=1
export HF_QUALITY_GATE_PATH="$OPEN_MODEL"
export EVAL_VERBOSE=1
export PYTHONUNBUFFERED=1

# Warn if single-mini condition is included but no API key is set.
CONDS_TO_RUN="${SUITE_CONDITIONS:-all}"
if [[ "$CONDS_TO_RUN" == "all" || "$CONDS_TO_RUN" == *"single-mini"* ]]; then
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "WARNING: OPENAI_API_KEY not set."
        echo "  The 'single-mini' condition (GPT-4o-mini baseline) will fail."
        echo "  To skip it, set: export SUITE_CONDITIONS=\"single-finetuned,pipeline-mini,pipeline-finetuned,ablation-no-analyzer\""
        echo "  Continuing in 5 seconds..."
        sleep 5
    fi
fi

# --- Warmup metric backends --------------------------------------------------
nvidia-smi || true
echo "Analyzer + quality gate : $OPEN_MODEL"
echo "Simplifier              : $MERGED_MODEL"
echo "Warming up SARI + BERTScore metric backends..."
python -c "from app.metrics import warmup_metrics; warmup_metrics()"

# --- Run suite ---------------------------------------------------------------
CONDITIONS_ARGS=()
if [ -n "${SUITE_CONDITIONS:-}" ]; then
    CONDITIONS_ARGS=(--conditions "$SUITE_CONDITIONS")
    echo "Running subset: $SUITE_CONDITIONS"
else
    echo "Running all 5 conditions on full test set"
fi

python main.py \
    --suite \
    --all \
    --data data/plaba/test.csv \
    --max-iterations 2 \
    --output-dir outputs/experiments \
    "${CONDITIONS_ARGS[@]}"

echo "Done. See outputs/experiments/suite-summary-*.json"
