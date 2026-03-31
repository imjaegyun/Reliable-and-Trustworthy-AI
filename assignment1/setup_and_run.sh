#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="assignment1"
PYTHON_VERSION="3.10"
CUDA_TAG=""
SETUP_ONLY=0
TEST_ARGS=()
PYTORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"

usage() {
    cat <<'EOF'
Usage:
  ./setup_and_run.sh [script-options] [test.py options]

Script options:
  --env NAME         Conda environment name (default: assignment1)
  --python VERSION   Python version for the conda env (default: 3.10)
  --cuda TAG         CUDA runtime for PyTorch (examples: 12.4, 12.1, cu124, cpu)
  --setup-only       Create/update the environment and install packages only
  -h, --help         Show this help message

Examples:
  ./setup_and_run.sh
  ./setup_and_run.sh --fast
  ./setup_and_run.sh --cuda 12.1 --fast
  ./setup_and_run.sh --setup-only

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

normalize_cuda_tag() {
    local raw="$1"
    case "$raw" in
        "" )
            echo ""
            ;;
        cpu|CPU )
            echo "cpu"
            ;;
        12.4|cu124|CU124 )
            echo "12.4"
            ;;
        12.1|cu121|CU121 )
            echo "12.1"
            ;;
        11.8|cu118|CU118 )
            echo "11.8"
            ;;
        * )
            echo "$raw"
            ;;
    esac
}

detect_default_cuda_tag() {
    local driver_version
    local driver_major

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "cpu"
        return
    fi

    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
    if [[ -z "$driver_version" ]]; then
        echo "12.1"
        return
    fi

    driver_major="${driver_version%%.*}"

    if [[ "$driver_major" =~ ^[0-9]+$ ]]; then
        if (( driver_major >= 550 )); then
            echo "12.4"
        elif (( driver_major >= 530 )); then
            echo "12.1"
        else
            echo "11.8"
        fi
    else
        echo "12.1"
    fi
}

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

CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")

if [[ -z "$CUDA_TAG" ]]; then
    CUDA_TAG="$(detect_default_cuda_tag)"
fi

CUDA_TAG="$(normalize_cuda_tag "$CUDA_TAG")"

echo "[INFO] Verifying conda environment before package install"
"${CONDA_RUN[@]}" python -c "
import os
import sys

print(
    '[INFO] Env ready | '
    f'env=${ENV_NAME} | '
    f'python={sys.executable} | '
    f'python_version={sys.version.split()[0]} | '
    f'conda_prefix={os.environ.get(\"CONDA_PREFIX\", \"unknown\")} | '
    f'requested_cuda=${CUDA_TAG}'
)
"

echo "[INFO] Upgrading pip in $ENV_NAME"
"${CONDA_RUN[@]}" python -m pip install --upgrade pip

echo "[INFO] Removing conflicting torch packages if present"
"${CONDA_RUN[@]}" python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
conda remove -y -n "$ENV_NAME" pytorch torchvision torchaudio pytorch-cuda cpuonly >/dev/null 2>&1 || true

if [[ "$CUDA_TAG" == "cpu" ]]; then
    echo "[INFO] Installing CPU PyTorch in $ENV_NAME"
    conda install -y -n "$ENV_NAME" -c pytorch \
        "pytorch=${PYTORCH_VERSION}" "torchvision=${TORCHVISION_VERSION}" cpuonly
else
    echo "[INFO] Installing PyTorch with pytorch-cuda=$CUDA_TAG in $ENV_NAME"
    conda install -y -n "$ENV_NAME" -c pytorch -c nvidia \
        "pytorch=${PYTORCH_VERSION}" "torchvision=${TORCHVISION_VERSION}" "pytorch-cuda=${CUDA_TAG}"
fi

echo "[INFO] Installing project requirements"
"${CONDA_RUN[@]}" python -m pip install -r "$ROOT_DIR/requirements.txt"

echo "[INFO] Verifying PyTorch runtime inside conda environment"
"${CONDA_RUN[@]}" python -c '
import sys
import torch

cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0
current_device = torch.cuda.current_device() if cuda_available else "cpu"

print(
    "[INFO] Env check | "
    f"python={sys.executable} | "
    f"torch={torch.__version__} | "
    f"torch_cuda={torch.version.cuda} | "
    f"cuda_available={cuda_available} | "
    f"device_count={device_count} | "
    f"current_device={current_device}"
)
'

if [[ "$CUDA_TAG" != "cpu" ]]; then
    CUDA_OK="$("${CONDA_RUN[@]}" python -c 'import torch; print("1" if torch.cuda.is_available() else "0")')"
    if [[ "$CUDA_OK" != "1" ]]; then
        echo "[WARN] CUDA runtime was requested, but torch.cuda.is_available() is False in $ENV_NAME"
        echo "[WARN] This usually means the selected runtime is newer than the installed NVIDIA driver."
        echo "[WARN] Try rerunning with a lower runtime, for example: ./setup_and_run.sh --cuda 12.1"
    fi
fi

if [[ "$SETUP_ONLY" -eq 1 ]]; then
    echo "[INFO] Environment setup complete."
    echo "[INFO] Run the project with:"
    echo "       conda run --no-capture-output -n $ENV_NAME python $ROOT_DIR/test.py"
    exit 0
fi

echo "[INFO] Running assignment test.py"
echo "[INFO] Command: conda run --no-capture-output -n $ENV_NAME python $ROOT_DIR/test.py ${TEST_ARGS[*]:-}"
"${CONDA_RUN[@]}" python "$ROOT_DIR/test.py" "${TEST_ARGS[@]}"
