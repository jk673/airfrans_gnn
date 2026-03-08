#!/usr/bin/env bash
# Usage:
#   source bin/setup_env_rtx6000pro.sh          # setup + activate venv
#   source bin/setup_env_rtx6000pro.sh --recreate  # rebuild venv from scratch
#   bash   bin/setup_env_rtx6000pro.sh           # setup only (prints activation hint)
set -euo pipefail

# Detect whether the script is being sourced or executed
_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _SOURCED=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_VERSION="3.11"
RECREATE_VENV=0

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
# Use 'return' when sourced so we don't kill the user's shell
fail()  { echo -e "${RED}[FAIL]${NC} $*"; if [[ $_SOURCED -eq 1 ]]; then return 1 2>/dev/null; else exit 1; fi; }

usage() {
    cat <<EOF
Usage: $0 [--recreate]

Options:
  --recreate   Rebuild $VENV_DIR before syncing dependencies.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recreate)
            RECREATE_VENV=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            fail "Unknown argument: $1"
            ;;
    esac
done

[[ -f "$PROJECT_DIR/pyproject.toml" ]] || fail "Missing $PROJECT_DIR/pyproject.toml"
[[ -f "$PROJECT_DIR/uv.lock" ]] || fail "Missing $PROJECT_DIR/uv.lock"

echo "=================================================================="
echo " AirfRANS GNN - RTX PRO 6000 Remote Environment Setup"
echo "=================================================================="
echo ""

info "Repository root: $REPO_ROOT"
info "Dependency project: $PROJECT_DIR"
info "Target virtualenv: $VENV_DIR"

info "Checking for uv..."
if command -v uv >/dev/null 2>&1; then
    ok "uv already installed: $(uv --version)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || fail "uv installation failed"
    ok "uv installed: $(uv --version)"
fi

info "Checking for Python $PYTHON_VERSION..."
if uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then
    ok "Python $PYTHON_VERSION found: $(uv python find "$PYTHON_VERSION")"
else
    info "Installing Python $PYTHON_VERSION via uv..."
    uv python install "$PYTHON_VERSION"
    ok "Python $PYTHON_VERSION installed: $(uv python find "$PYTHON_VERSION")"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
    DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
    GPU_MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)"

    ok "GPU detected: $GPU_NAME"
    ok "Driver version: $DRIVER_VERSION"
    ok "Driver CUDA API level: ${DRIVER_CUDA:-unknown}"
    ok "GPU memory: $GPU_MEMORY"

    case "$GPU_NAME" in
        *"Blackwell"*|*"RTX PRO 6000"*)
            ok "Blackwell-class GPU detected; CUDA 12.8 PyTorch wheels are the right baseline here."
            ;;
        *)
            warn "This script is tuned for RTX PRO 6000 / Blackwell hosts, but it can still work on other NVIDIA GPUs."
            ;;
    esac
else
    warn "nvidia-smi not found. The script will continue, but only CPU validation will be possible."
fi

if [[ $RECREATE_VENV -eq 1 ]]; then
    info "Recreating virtual environment at $VENV_DIR"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --clear
else
    info "Ensuring virtual environment exists at $VENV_DIR"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --allow-existing
fi

info "Syncing dependencies from $PROJECT_DIR with the locked CUDA 12.8 stack..."
(
    export VIRTUAL_ENV="$VENV_DIR"
    export PATH="$VENV_DIR/bin:$PATH"
    cd "$REPO_ROOT"
    uv sync --project "$PROJECT_DIR" --active --frozen --all-extras
)
ok "Dependencies installed into $VENV_DIR"

info "Verifying Python, PyTorch, PyG, and CUDA..."
"$VENV_DIR/bin/python" - <<'PY'
import importlib
import sys

required = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "numpy",
    "scipy",
    "matplotlib",
    "tqdm",
    "wandb",
    "dotenv",
    "flask",
    "pytest",
    "optuna",
]

GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
NC     = "\033[0m"

missing = []
for name in required:
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"  {RED}✗{NC} {name}")
        missing.append(name)
    else:
        # Show version if available
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            label = f"{name} {ver}" if ver else name
        except Exception:
            label = name
        print(f"  {GREEN}✓{NC} {label}")

if missing:
    print(f"\nMissing packages: {missing}")
    sys.exit(1)
print()

import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.version.cuda != "12.8":
    print(f"Unexpected torch CUDA runtime: {torch.version.cuda}")
    sys.exit(1)

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    x = torch.randn((2048, 2048), device="cuda")
    y = torch.randn((2048, 2048), device="cuda")
    z = (x @ y).mean().item()
    torch.cuda.synchronize()

    print(f"GPU: {name}")
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    print(f"Visible GPU memory (GiB): {total_gb:.1f}")
    print(f"CUDA matmul smoke test: {z:.6f}")
else:
    print("CUDA is not available to PyTorch.")
    sys.exit(1)
PY
ok "Python and CUDA validation passed"

if [[ -d "$REPO_ROOT/Dataset" ]]; then
    ok "Dataset directory found at $REPO_ROOT/Dataset"
elif [[ -d "$REPO_ROOT/prebuilt_edges_v2" ]]; then
    ok "Preprocessed data directory found at $REPO_ROOT/prebuilt_edges_v2"
else
    warn "No Dataset/ or prebuilt_edges_v2/ directory found under $REPO_ROOT"
fi

if [[ ! -f "$REPO_ROOT/.env" ]]; then
    info "Creating template .env in $REPO_ROOT"
    cat > "$REPO_ROOT/.env" <<'EOF'
# AirfRANS GNN environment variables
# Uncomment and set as needed:

# WANDB_API_KEY=your_key_here
# WANDB_MODE=disabled
EOF
    ok "Template .env created"
else
    ok ".env already exists"
fi

echo ""
echo "=================================================================="
echo -e " ${GREEN}Remote setup complete${NC}"
echo "=================================================================="
echo ""

# Activate venv automatically when sourced; otherwise print hint
if [[ $_SOURCED -eq 1 ]]; then
    info "Activating virtualenv..."
    source "$VENV_DIR/bin/activate"
    ok "Virtualenv activated (python → $(which python))"
    echo ""
    echo "Ready to go:"
    echo "  python scripts/train.py"
    echo "  python dashboard/app.py"
else
    echo "Activate:"
    echo "  source $VENV_DIR/bin/activate"
    echo ""
    echo "Or re-run this script with source to auto-activate:"
    echo "  source $0"
fi
echo ""
echo "If the existing environment is broken, rebuild it with:"
echo "  source $0 --recreate"
