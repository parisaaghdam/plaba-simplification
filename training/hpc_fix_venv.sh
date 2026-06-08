#!/bin/bash
# One-time fix: recreate .venv-train with Python 3.11 + CUDA PyTorch on Alpha.
# Run on login1.alpha from the repo root:
#   bash training/hpc_fix_venv.sh
#
# Why: a Python 3.13 venv cannot install CUDA PyTorch wheels (stays 2.x+cpu).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

module --force purge
module load release/24.04 GCCcore/13.2.0 Python/3.11.5

echo "Module Python: $(which python) ($(python --version))"

if [ -d ".venv-train" ]; then
    BACKUP=".venv-train.bak.$(date +%Y%m%d%H%M%S)"
    echo "Backing up existing venv to $BACKUP"
    mv .venv-train "$BACKUP"
fi

python -m venv .venv-train
source .venv-train/bin/activate

pip install --upgrade pip
echo "Installing CUDA PyTorch (cu121)..."
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r training/requirements-train.txt
pip install -r requirements.txt

python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
python -c "import sacremoses, sacrebleu; print('SARI deps OK')"

echo ""
echo "Done. CUDA may show False on the login node — that is normal."
echo "Submit the suite job from a shell where OPENAI_API_KEY is set:"
echo "  export OPENAI_API_KEY=\"sk-...\""
echo "  export SUITE_CONDITIONS=\"single-finetuned,pipeline-mini,pipeline-finetuned,ablation-no-analyzer\""
echo "  sbatch training/hpc_suite.sbatch"
