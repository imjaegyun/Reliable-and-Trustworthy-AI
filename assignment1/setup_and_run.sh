#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="assignment1"
PYTHON_VERSION="3.10"
CUDA_TAG=""
SETUP_ONLY=0
REINSTALL_TORCH=0
TEST_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash setup_and_run.sh [script-options] [test.py options]

Script options:
  --env NAME         Conda environment name (default: assignment1)
  --python VERSION   Python version for the conda env (default: 3.10)
  --cuda TAG         Install CUDA PyTorch wheels first (example: cu124)
  --setup-only       Create/update the environment and install packages only
  --reinstall-torch  Reinstall torch/torchvision before installing requirements
  -h, --help         Show this help message

Examples:
  bash setup_and_run.sh --fast
  bash setup_and_run.sh --env viscot-a1 --cuda cu124 --fast
  bash setup_and_run.sh --setup-only

Any unrecognized arguments are forwarded to test.py.
EOF
}

while (($#)); do
    case "$1" in
        --env)
            ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cuda)
            CUDA_TAG="$2"
            shift 2
            ;;
        --setup-only)
            SETUP_ONLY=1
            shift
            ;;
        --reinstall-torch)
            REINSTALL_TORCH=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            TEST_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command was not found. Please install conda first."
    exit 1
fi

env_exists=0
if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
    env_exists=1
fi

if [[ "$env_exists" -eq 0 ]]; then
    echo "[INFO] Creating conda environment: $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
    echo "[INFO] Reusing existing conda environment: $ENV_NAME"
fi

echo "[INFO] Upgrading pip in $ENV_NAME"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

if [[ "$REINSTALL_TORCH" -eq 1 ]]; then
    echo "[INFO] Reinstalling torch/torchvision"
    conda run -n "$ENV_NAME" python -m pip uninstall -y torch torchvision || true
fi

if [[ -n "$CUDA_TAG" ]]; then
    echo "[INFO] Installing CUDA-enabled torch/torchvision from PyTorch index: $CUDA_TAG"
    conda run -n "$ENV_NAME" python -m pip install --upgrade \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
        torch torchvision
fi

echo "[INFO] Installing project requirements"
conda run -n "$ENV_NAME" python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "$SETUP_ONLY" -eq 1 ]]; then
    echo "[INFO] Environment setup complete."
    echo "[INFO] Run the project with:"
    echo "       conda run -n $ENV_NAME python $ROOT_DIR/test.py"
    exit 0
fi

echo "[INFO] Running assignment test.py"
echo "[INFO] Command: conda run -n $ENV_NAME python $ROOT_DIR/test.py ${TEST_ARGS[*]:-}"
conda run -n "$ENV_NAME" python "$ROOT_DIR/test.py" "${TEST_ARGS[@]}"
