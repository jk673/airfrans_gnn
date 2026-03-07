#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

DATASET_ROOT="Dataset"
TASK="full"
DOWNSAMPLED_DIR="downsampled_graphs_v2"
EDGES_DIR="prebuilt_edges_v2"
PYTHON_BIN="python"

usage() {
  cat <<'EOF'
Usage:
  setup_proc_data.sh [options]

Options:
  --dataset-root <path>   Dataset root path (default: Dataset)
  --task <scarce|full|all>  AirfRANS task split (default: scarce)
  --downsampled-dir <path>  Output dir for downsampled graphs (default: downsampled_graphs_v2)
  --edges-dir <path>      Output dir for edge-built graphs (default: prebuilt_edges_v2)
  --python <bin>          Python executable (default: python)
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --downsampled-dir)
      DOWNSAMPLED_DIR="$2"
      shift 2
      ;;
    --edges-dir)
      EDGES_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

TASKS=()
case "${TASK}" in
  scarce|full)
    TASKS=("${TASK}")
    ;;
  all)
    TASKS=("scarce" "full")
    ;;
  *)
    echo "Invalid --task value: ${TASK}. Use 'scarce', 'full', or 'all'." >&2
    exit 1
    ;;
esac

RAW_DIR="${DATASET_ROOT}/raw"
ZIP_PATH="${RAW_DIR}/AirfRANS.zip"
PT_PATH="${RAW_DIR}/AirfRANS.pt"
MANIFEST_PATH="${RAW_DIR}/manifest.json"

cd "${REPO_ROOT}"

if [[ -f "${PT_PATH}" && -f "${MANIFEST_PATH}" ]]; then
  echo "[1/3] Found extracted dataset files. Skipping unzip."
elif [[ -f "${ZIP_PATH}" ]]; then
  echo "[1/3] Unzipping ${ZIP_PATH} -> ${RAW_DIR}"
  mkdir -p "${RAW_DIR}"
  unzip -o "${ZIP_PATH}" -d "${RAW_DIR}"
else
  echo "Missing dataset archive: ${ZIP_PATH}" >&2
  exit 1
fi

for CURRENT_TASK in "${TASKS[@]}"; do
  echo "[2/3] Running downsampling for task=${CURRENT_TASK}"
  "${PYTHON_BIN}" -m preprocessing.downsample_airfrans_v2 \
    --root "${DATASET_ROOT}" \
    --task "${CURRENT_TASK}" \
    --out-dir "${DOWNSAMPLED_DIR}"

  echo "[3/3] Building edges for task=${CURRENT_TASK}"
  "${PYTHON_BIN}" -m preprocessing.edges_from_downsampled_v2 \
    --in-dir "${DOWNSAMPLED_DIR}" \
    --out-dir "${EDGES_DIR}" \
    --task "${CURRENT_TASK}"
done

echo "Done. Outputs:"
for CURRENT_TASK in "${TASKS[@]}"; do
  echo "  - Downsampled: ${DOWNSAMPLED_DIR}/${CURRENT_TASK}"
  echo "  - Edges:       ${EDGES_DIR}/${CURRENT_TASK}"
done
