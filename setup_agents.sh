#!/bin/bash
set -e

echo "============================================"
echo "  AI Dev Tools Installer"
echo "  Node.js + Codex CLI + Claude Code"
echo "============================================"
echo ""

# ── 색상 정의 ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_step() { echo -e "\n${GREEN}[✓]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
print_err()  { echo -e "${RED}[✗]${NC} $1"; }

# ── 1. Node.js 설치 (nvm 경유) ──
print_step "Node.js 설치 중 (nvm 사용)..."

export NVM_DIR="$HOME/.nvm"

if [ ! -d "$NVM_DIR" ]; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
else
    print_warn "nvm이 이미 설치되어 있습니다. 업데이트 스킵."
fi

# nvm 로드
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 최신 LTS 설치 및 기본값 설정
nvm install --lts
nvm use --lts
nvm alias default lts/*

NODE_VER=$(node -v)
NPM_VER=$(npm -v)
print_step "Node.js ${NODE_VER} / npm ${NPM_VER} 설치 완료"

# ── 2. OpenAI Codex CLI 설치 ──
print_step "OpenAI Codex CLI 설치 중..."
npm install -g @openai/codex@latest
CODEX_VER=$(codex --version 2>/dev/null || echo "버전 확인 불가")
print_step "Codex CLI 설치 완료 (${CODEX_VER})"

# ── 3. Claude Code 설치 ──
print_step "Anthropic Claude Code 설치 중..."
npm install -g @anthropic-ai/claude-code@latest
CLAUDE_VER=$(claude --version 2>/dev/null || echo "버전 확인 불가")
print_step "Claude Code 설치 완료 (${CLAUDE_VER})"

# ── 4. 셸 설정 파일에 nvm 경로 추가 확인 ──
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q 'NVM_DIR' "$SHELL_RC" 2>/dev/null; then
        print_warn "nvm 경로를 ${SHELL_RC}에 추가합니다..."
        cat >> "$SHELL_RC" << 'EOF'

# nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
EOF
    fi
fi

# ── 완료 요약 ──
echo ""
echo "============================================"
echo "  설치 완료 요약"
echo "============================================"
echo "  Node.js  : $(node -v)"
echo "  npm      : $(npm -v)"
echo "  Codex CLI: $(codex --version 2>/dev/null || echo 'N/A')"
echo "  Claude   : $(claude --version 2>/dev/null || echo 'N/A')"
echo "============================================"
echo ""
print_warn "새 터미널을 열거나 'source ${SHELL_RC:-~/.bashrc}'를 실행하세요."
echo ""