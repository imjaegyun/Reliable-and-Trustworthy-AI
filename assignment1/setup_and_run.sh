#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="assignment1"
PYTHON_VERSION="3.10"
CUDA_TAG=""
CUDA_EXPLICIT=0
SETUP_ONLY=0
TEST_ARGS=()
PYTORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
MAX_RETRIES=3

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
  ./setup_and_run.sh --cuda 12.4 --fast
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
            CUDA_EXPLICIT=1
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
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "cpu"
        return
    fi

    echo "12.4"
}

remove_conflicting_torch() {
    echo "[INFO] Removing conflicting torch packages if present"
    "${CONDA_RUN[@]}" python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
    conda remove -y -n "$ENV_NAME" pytorch torchvision torchaudio pytorch-cuda cpuonly >/dev/null 2>&1 || true
}

runtime_to_wheel_tag() {
    local tag="$1"
    case "$tag" in
        cpu)
            echo "cpu"
            ;;
        12.4)
            echo "cu124"
            ;;
        12.1)
            echo "cu121"
            ;;
        11.8)
            echo "cu118"
            ;;
        *)
            echo ""
            ;;
    esac
}

run_with_retries() {
    local description="$1"
    shift

    local attempt=1
    while true; do
        if "$@"; then
            return 0
        fi

        if (( attempt >= MAX_RETRIES )); then
            echo "[ERROR] ${description} failed after ${MAX_RETRIES} attempts"
            return 1
        fi

        echo "[WARN] ${description} failed on attempt ${attempt}/${MAX_RETRIES}. Retrying in 5 seconds..."
        sleep 5
        attempt=$((attempt + 1))
    done
}

install_pytorch_runtime() {
    local tag="$1"
    local wheel_tag

    remove_conflicting_torch
    wheel_tag="$(runtime_to_wheel_tag "$tag")"

    if [[ -z "$wheel_tag" ]]; then
        echo "[ERROR] Unsupported CUDA runtime tag: $tag"
        return 1
    fi

    echo "[INFO] Installing PyTorch wheels (${wheel_tag}) in $ENV_NAME"
    run_with_retries "PyTorch ${wheel_tag} installation" \
        "${CONDA_RUN[@]}" python -m pip install --upgrade \
            --index-url "https://download.pytorch.org/whl/${wheel_tag}" \
            "torch==${PYTORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
}

print_torch_env_check() {
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
}

cuda_runtime_ok() {
    "${CONDA_RUN[@]}" python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'
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
run_with_retries "pip upgrade" "${CONDA_RUN[@]}" python -m pip install --upgrade pip

if [[ "$CUDA_TAG" == "cpu" ]]; then
    install_pytorch_runtime "cpu"
    print_torch_env_check
else
    ATTEMPT_TAGS=("$CUDA_TAG")
    if [[ "$CUDA_EXPLICIT" -eq 0 ]]; then
        case "$CUDA_TAG" in
            12.4)
                ATTEMPT_TAGS+=(12.1 11.8)
                ;;
            12.1)
                ATTEMPT_TAGS+=(11.8)
                ;;
        esac
    fi

    CUDA_READY=0
    for TRY_TAG in "${ATTEMPT_TAGS[@]}"; do
        install_pytorch_runtime "$TRY_TAG"
        print_torch_env_check
        if cuda_runtime_ok; then
            CUDA_TAG="$TRY_TAG"
            CUDA_READY=1
            break
        fi

        echo "[WARN] CUDA runtime $TRY_TAG was installed but torch.cuda.is_available() is False"
    done

    if [[ "$CUDA_READY" -ne 1 ]]; then
        echo "[ERROR] Failed to enable CUDA in $ENV_NAME."
        echo "[ERROR] Tried runtimes: ${ATTEMPT_TAGS[*]}"
        echo "[ERROR] Upgrade the NVIDIA driver or rerun with ./setup_and_run.sh --cuda cpu"
        exit 1
    fi
fi

echo "[INFO] Installing project requirements"
run_with_retries "project requirements installation" \
    "${CONDA_RUN[@]}" python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "$SETUP_ONLY" -eq 1 ]]; then
    echo "[INFO] Environment setup complete."
    echo "[INFO] Run the project with:"
    echo "       conda run --no-capture-output -n $ENV_NAME python $ROOT_DIR/test.py"
    exit 0
fi

echo "[INFO] Running assignment test.py"
echo "[INFO] Command: conda run --no-capture-output -n $ENV_NAME python $ROOT_DIR/test.py ${TEST_ARGS[*]:-}"
"${CONDA_RUN[@]}" python "$ROOT_DIR/test.py" "${TEST_ARGS[@]}"
