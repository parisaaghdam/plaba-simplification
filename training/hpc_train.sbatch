#!/bin/bash
# SLURM batch script for QLoRA fine-tuning on the TU Dresden HPC (ZIH).
#
# Submit with:   sbatch training/hpc_train.sbatch
# Watch it with: squeue --me        (and: tail -f logs/plaba-qlora-<jobid>.out)
#
# Adjust --partition / --account to match what you have access to. Check with:
#   sinfo -s                 # available partitions
#   ws_list                  # your workspaces
# GPU partitions on ZIH are typically:  alpha (A100 40GB)  or  capella (H100).

#SBATCH --job-name=plaba-qlora
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --account=p_your_project   # uncomment + set if your project requires it
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@tu-dresden.de

set -euo pipefail

# --- 0) Where the code/data live (run sbatch from the repo root) -------------
REPO_DIR="$(pwd)"
mkdir -p logs outputs

# --- 1) Python environment ---------------------------------------------------
# Load a Python module (check available ones with: module spider Python).
module purge
module load release/24.04 || true
module load Python/3.11.5 || module load Python || true

# Create the virtual environment once; reuse it on later runs.
VENV="$REPO_DIR/.venv-train"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    # Torch with CUDA wheels (cu121 works on ZIH A100/H100 nodes).
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -r training/requirements-train.txt
else
    source "$VENV/bin/activate"
fi

# --- 2) Caches on scratch/workspace, not your small $HOME --------------------
# Point HF caches at a workspace so large model downloads don't fill $HOME.
# Replace this with a real workspace path, e.g. created via:
#   ws_allocate -F horse plaba 30
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# Only needed for GATED models (Mistral/Llama). Qwen is open and needs no token.
# export HF_TOKEN="hf_xxx"

# --- 3) Train ----------------------------------------------------------------
nvidia-smi || true
python training/train_qlora.py \
    --train data/plaba/sft/train.sft.jsonl \
    --val data/plaba/sft/val.sft.jsonl \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --output-dir outputs/plaba-simplifier-qlora \
    --epochs 3

echo "Done. Adapter saved to outputs/plaba-simplifier-qlora"
