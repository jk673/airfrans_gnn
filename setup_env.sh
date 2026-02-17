#!/usr/bin/env bash
# ===========================================================================
# setup_env.sh — AirfRANS GNN 전체 환경 설치 스크립트
#
# 수행 내용:
#   1. uv 설치 (없으면)
#   2. Python 3.11 확보 (uv python install)
#   3. .venv 생성 + 의존성 설치
#   4. AirfRANS 데이터셋 존재 여부 확인
#   5. 설치 검증
#
# 사용법:
#   chmod +x setup_env.sh && ./setup_env.sh
# ===========================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo "=================================================================="
echo " AirfRANS GNN — Environment Setup"
echo "=================================================================="
echo ""

# ------------------------------------------------------------------
# 1. uv 설치
# ------------------------------------------------------------------
info "Checking for uv..."
if command -v uv &>/dev/null; then
    ok "uv already installed: $(uv --version)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed: $(uv --version)"
    else
        fail "uv installation failed. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
    fi
fi

# ------------------------------------------------------------------
# 2. Python 3.11 확보
# ------------------------------------------------------------------
info "Checking for Python 3.11..."
if uv python find 3.11 &>/dev/null; then
    ok "Python 3.11 found: $(uv python find 3.11)"
else
    info "Installing Python 3.11 via uv..."
    uv python install 3.11
    ok "Python 3.11 installed: $(uv python find 3.11)"
fi

# ------------------------------------------------------------------
# 3. pyproject.toml에 python-dotenv 추가 (누락된 의존성)
# ------------------------------------------------------------------
if ! grep -q 'python-dotenv' "$PROJECT_DIR/pyproject.toml"; then
    info "Adding missing dependency: python-dotenv to pyproject.toml"
    sed -i 's/"psutil",/"psutil",\n    "python-dotenv",/' "$PROJECT_DIR/pyproject.toml"
    ok "python-dotenv added to pyproject.toml"
else
    ok "python-dotenv already in pyproject.toml"
fi

# ------------------------------------------------------------------
# 4. .venv 생성 + 의존성 설치
# ------------------------------------------------------------------
info "Creating virtual environment and installing dependencies..."
info "(PyTorch CUDA 12.4 wheels + PyG extensions — this may take a few minutes)"
echo ""

uv sync --python 3.11

echo ""
ok "All dependencies installed"

# ------------------------------------------------------------------
# 5. CUDA 확인
# ------------------------------------------------------------------
info "Checking CUDA availability..."
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    ok "NVIDIA driver: $DRIVER_VER"

    CUDA_OK=$("$PROJECT_DIR/.venv/bin/python" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [ "$CUDA_OK" = "True" ]; then
        GPU_NAME=$("$PROJECT_DIR/.venv/bin/python" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        ok "CUDA available — GPU: $GPU_NAME"
    else
        warn "torch.cuda.is_available() = False — training will use CPU"
    fi
else
    warn "nvidia-smi not found — training will use CPU"
fi

# ------------------------------------------------------------------
# 6. 설치 검증: 핵심 패키지 import 테스트
# ------------------------------------------------------------------
info "Verifying critical imports..."

"$PROJECT_DIR/.venv/bin/python" -c "
import sys
errors = []

packages = {
    'torch':              'import torch',
    'torch_geometric':    'import torch_geometric',
    'torch_scatter':      'import torch_scatter',
    'torch_sparse':       'import torch_sparse',
    'torch_cluster':      'import torch_cluster',
    'numpy':              'import numpy',
    'scipy':              'import scipy',
    'matplotlib':         'import matplotlib',
    'tqdm':               'import tqdm',
    'wandb':              'import wandb',
    'dotenv':             'import dotenv',
}

for name, stmt in packages.items():
    try:
        exec(stmt)
        print(f'  [OK] {name}')
    except ImportError as e:
        errors.append(name)
        print(f'  [FAIL] {name}: {e}')

if errors:
    print(f'\nFailed imports: {errors}')
    sys.exit(1)
else:
    print('\nAll imports OK')
"
ok "All critical packages verified"

# ------------------------------------------------------------------
# 7. 데이터셋 확인
# ------------------------------------------------------------------
echo ""
if [ -d "$PROJECT_DIR/Dataset" ]; then
    ok "Dataset directory found"
elif [ -d "$PROJECT_DIR/prebuilt_edges_v2" ]; then
    ok "prebuilt_edges_v2 directory found (preprocessed data)"
else
    warn "No Dataset/ or prebuilt_edges_v2/ directory found."
    warn "Training requires the AirfRANS dataset. To download:"
    echo "    The dataset will be auto-downloaded on first run, or you can"
    echo "    manually place it in $PROJECT_DIR/Dataset/"
fi

# ------------------------------------------------------------------
# 8. .env 파일 확인
# ------------------------------------------------------------------
if [ ! -f "$PROJECT_DIR/.env" ]; then
    info "Creating template .env file..."
    cat > "$PROJECT_DIR/.env" <<'ENVEOF'
# AirfRANS GNN environment variables
# Uncomment and set as needed:

# WANDB_API_KEY=your_key_here
# WANDB_MODE=disabled
ENVEOF
    ok "Template .env created (edit as needed)"
else
    ok ".env file exists"
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=================================================================="
echo -e " ${GREEN}Setup complete!${NC}"
echo "=================================================================="
echo ""
echo " To activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo " To start training:"
echo "   python scripts/train.py"
echo ""
echo " To start training with custom options:"
echo "   python scripts/train.py --epochs 50 --lr 3e-4 --wandb-mode disabled"
echo ""
echo " For all options:"
echo "   python scripts/train.py --help"
echo "=================================================================="
