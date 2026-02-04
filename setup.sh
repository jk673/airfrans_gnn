#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
#  AirfRANS GNN — one-command environment setup
#
#  Usage:
#    bash setup.sh           # core dependencies only
#    bash setup.sh --all     # core + optuna + test
#    bash setup.sh --optuna  # core + optuna
#    bash setup.sh --test    # core + test
# ─────────────────────────────────────────────────────
set -euo pipefail

EXTRAS="optuna"
for arg in "$@"; do
    case "$arg" in
        --all)    EXTRAS="all" ;;
        --optuna) EXTRAS="optuna" ;;
        --test)   EXTRAS="test" ;;
        --core)   EXTRAS="" ;;
        --help|-h)
            echo "Usage: bash setup.sh [--all|--optuna|--test|--core]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash setup.sh [--all|--optuna|--test|--core]"
            exit 1
            ;;
    esac
done

# ── 1. Check for uv ───────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo ">> uv not found. Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ">> uv $(uv --version)"

# ── 2. Install dependencies ──────────────────────────
if [ -n "$EXTRAS" ]; then
    echo ">> Installing with extras: [$EXTRAS] (default: optuna)"
    uv sync --extra "$EXTRAS"
else
    echo ">> Installing core dependencies only (selected via --core)"
    uv sync
fi

# ── 3. Verify key imports ────────────────────────────
echo "" 
echo ">> Verifying installation ..."
uv run python -c "
import torch
print(f'  torch          {torch.__version__}  (CUDA {torch.version.cuda})')
import torch_geometric
print(f'  torch-geometric {torch_geometric.__version__}')
import torch_scatter
print(f'  torch-scatter   OK')
import torch_sparse
print(f'  torch-sparse    OK')
import torch_cluster
print(f'  torch-cluster   OK')
import numpy; print(f'  numpy          {numpy.__version__}')
import scipy; print(f'  scipy          {scipy.__version__}')
import matplotlib; print(f'  matplotlib     {matplotlib.__version__}')
import wandb; print(f'  wandb          {wandb.__version__}')
print()
print('  All core packages imported successfully.')
"

echo ""
echo ">> Setup complete. Run notebooks with:  uv run jupyter notebook"

# ── 4. Register Jupyter kernel ───────────────────────
echo ""
echo ">> Installing Jupyter kernel 'AirfRANS GNN' ..."
uv run python -m ipykernel install --user --name airfrans-gnn --display-name "AirfRANS GNN"
