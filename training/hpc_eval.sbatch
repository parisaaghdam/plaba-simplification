#!/bin/bash
# SLURM batch script: batch evaluation on TU Dresden HPC (Alpha GPU).
#
# Fully open-source — no OpenAI API key required.
#   Analyzer    : Qwen/Qwen2.5-7B-Instruct  (HF, GPU)
#   Simplifier  : fine-tuned 7B model        (HF, GPU, outputs/plaba-merged)
#   Quality gate: Qwen/Qwen2.5-7B-Instruct  (HF, GPU — shared weights with analyzer)
#
# GPU memory budget on A100 40 GB (fp16):
#   Qwen2.5-7B (analyzer + quality gate, cached): ~14 GB
#   Fine-tuned simplifier:                        ~14 GB
#   BERTScore + overhead:                         ~ 4 GB
#   Total:                                        ~32 GB  ✓
#
# PRE-REQUISITE: download Qwen2.5-7B-Instruct to HF cache before submitting:
#   sbatch training/hpc_download.sbatch
#
# Submit from repo root on login1.alpha:
#   sbatch training/hpc_eval.sbatch
#
# Watch progress:
#   squeue --me
#   tail -f logs/plaba-eval-<jobid>.out
#
# Results:
#   outputs/experiments/eval-<timestamp>.json
#   outputs/experiments/eval-<timestamp>.csv

#SBATCH --job-name=plaba-eval
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=04:00:00
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
if [ ! -d "$VENV" ]; then
    echo "ERROR: .venv-train not found. Run: bash training/hpc_fix_venv.sh"
    exit 1
fi
source "$VENV/bin/activate"

pip install -q -r requirements.txt

# --- HF cache on scratch (avoid filling $HOME quota) -------------------------
export HF_HOME="${HF_HOME:-/scratch/ws/$(whoami)-hf-cache}"
mkdir -p "$HF_HOME"

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

# --- Model paths -------------------------------------------------------------
MERGED_MODEL="${HF_SIMPLIFIER_PATH:-$REPO_DIR/outputs/plaba-merged}"
OPEN_MODEL="${HF_OPEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

if [ ! -f "$MERGED_MODEL/config.json" ]; then
    echo "ERROR: Fine-tuned model not found at $MERGED_MODEL"
    echo "Run: sbatch training/hpc_merge.sbatch"
    exit 1
fi

# --- Fully open-source routing (no OpenAI) -----------------------------------
export USE_HF_ANALYZER=1
export HF_ANALYZER_PATH="$OPEN_MODEL"
export USE_HF_SIMPLIFIER=1
export HF_SIMPLIFIER_PATH="$MERGED_MODEL"
export USE_HF_QUALITY_GATE=1
export HF_QUALITY_GATE_PATH="$OPEN_MODEL"
export EVAL_VERBOSE=1
export PYTHONUNBUFFERED=1

# --- Run evaluation ----------------------------------------------------------
nvidia-smi || true
echo "Analyzer + quality gate : $OPEN_MODEL"
echo "Simplifier              : $MERGED_MODEL"
echo "Starting eval: k=20, seed=42, max_iterations=2"

python main.py \
    --eval \
    --data data/plaba/val.csv \
    --k 20 \
    --seed 42 \
    --max-iterations 2 \
    --output-dir outputs/experiments

echo "Done. Check outputs/experiments/ for eval-*.json and eval-*.csv"
