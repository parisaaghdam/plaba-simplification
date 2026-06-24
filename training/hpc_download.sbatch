#!/bin/bash
# SLURM: pre-download Qwen2.5-7B-Instruct to the HF cache on scratch.
#
# Run this ONCE before hpc_eval.sbatch or hpc_suite.sbatch.
# Downloads ~14 GB of model weights to $HF_HOME so compute jobs
# do not download during the GPU job (login nodes have internet; compute nodes may not).
#
# Submit from repo root:
#   sbatch training/hpc_download.sbatch

#SBATCH --job-name=plaba-download
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module --force purge
module load release/24.04 GCCcore/13.2.0 Python/3.11.5

VENV="$(pwd)/.venv-train"
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-/scratch/ws/$(whoami)-hf-cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"

MODEL="${HF_OPEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
echo "Downloading $MODEL ..."

python - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "$MODEL"
print(f"Downloading tokenizer for {model_id}...")
AutoTokenizer.from_pretrained(model_id)
print("Downloading model weights (fp16, ~14 GB)...")
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
print("Download complete.")
EOF

echo "Model cached at $HF_HOME"
echo "You can now submit: sbatch training/hpc_eval.sbatch"
echo "             or   : sbatch training/hpc_suite.sbatch"
