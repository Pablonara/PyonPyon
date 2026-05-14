#!/bin/bash
# Setup script for qwen3_rl GRPO training on RTX PRO 6000 Blackwell (SM120).
#
# Tested on Ubuntu 22.04 with NVIDIA driver 580.x, CUDA 12.8 system toolkit.
# Creates a Python 3.12 venv with all dependencies and workarounds for:
#   - FlashInfer JIT compilation on SM120 (needs system CUDA 13.0 headers)
#   - FLA TileLang bug on Blackwell (FLA_TILELANG=0)
#   - vLLM _decompose_size_nodes bug (patched in vllm_backend.py)
#   - tilelang libcudart_stub.so conflict (patched in vllm_backend.py)
#
# Usage:
#   scp setup_server.sh root@<host>:~/
#   ssh root@<host> 'bash setup_server.sh'

set -euo pipefail

VENV="/root/rl_venv312"
CUDA_TOOLKIT_VERSION="13-0"  # system CUDA for FlashInfer JIT headers

echo "=== Step 1: System packages ==="
# Python 3.12 from deadsnakes PPA
if ! command -v python3.12 &>/dev/null; then
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y python3.12 python3.12-venv python3.12-dev
fi
python3.12 --version

# CUDA 13.0 toolkit (provides cooperative_groups.h + curand.h for FlashInfer JIT)
if [ ! -d "/usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.}" ]; then
    echo "Installing CUDA ${CUDA_TOOLKIT_VERSION//-/.} toolkit..."
    apt-get install -y "cuda-toolkit-${CUDA_TOOLKIT_VERSION}"
fi
echo "CUDA toolkit: /usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.}"

# ninja for FlashInfer JIT
if ! command -v ninja &>/dev/null; then
    apt-get install -y ninja-build
fi

echo ""
echo "=== Step 2: Python venv ==="
if [ -d "$VENV" ]; then
    echo "Venv already exists at $VENV, skipping creation"
else
    python3.12 -m venv "$VENV"
    echo "Created venv at $VENV"
fi

# Install uv for fast package management
"$VENV/bin/pip" install -q uv
UV="$VENV/bin/uv"
export VIRTUAL_ENV="$VENV"

echo ""
echo "=== Step 3: PyTorch + Triton (cu130) ==="
$UV pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
echo "torch: $($VENV/bin/python -c 'import torch; print(torch.__version__)')"
echo "triton: $($VENV/bin/python -c 'import triton; print(triton.__version__)')"

echo ""
echo "=== Step 4: ML stack ==="
$UV pip install transformers peft accelerate datasets trl sentencepiece xformers bitsandbytes

echo ""
echo "=== Step 5: vLLM ==="
$UV pip install vllm==0.21.0

echo ""
echo "=== Step 6: Unsloth (from GitHub for latest) ==="
$UV pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" \
                "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"

echo ""
echo "=== Step 7: FLA (flash-linear-attention) ==="
$UV pip install flash-linear-attention==0.5.0

echo ""
echo "=== Step 8: causal-conv1d (CUDA kernel, build from source) ==="
CUDA_HOME="/usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.}" \
    "$VENV/bin/pip" install causal-conv1d --no-build-isolation

echo ""
echo "=== Step 9: FA3 (prebuilt wheel for cu130 + torch 2.11) ==="
$UV pip install flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2110

echo ""
echo "=== Step 10: FlashInfer (for vLLM sampling kernels) ==="
$UV pip install flashinfer-python==0.6.11.post1 flashinfer-cubin==0.6.11.post1

echo ""
echo "=== Step 11: nvidia-cuda-nvcc for pip (FlashInfer JIT fallback) ==="
# FlashInfer JIT needs nvcc 13+ for sm_120f. System CUDA 13.0 provides headers,
# but pip nvcc 13.2 provides the compiler binary with SM120 support.
$UV pip install "nvidia-cuda-nvcc>=13" "nvidia-cuda-crt>=13"

echo ""
echo "=== Step 12: Symlinks for FlashInfer JIT ==="
CU13="$VENV/lib/python3.12/site-packages/nvidia/cu13"
if [ -d "$CU13" ]; then
    # lib64 symlink (FlashInfer linker expects lib64/)
    test -e "$CU13/lib64" || ln -s "$CU13/lib" "$CU13/lib64"
    # libcudart.so soname link
    test -e "$CU13/lib/libcudart.so" || ln -sf libcudart.so.13 "$CU13/lib/libcudart.so" 2>/dev/null || true
    # libcuda.so stub for linking
    mkdir -p "$CU13/lib/stubs"
    test -e "$CU13/lib/stubs/libcuda.so" || ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so "$CU13/lib/stubs/libcuda.so"
    echo "FlashInfer symlinks created"
else
    echo "WARNING: nvidia/cu13 not found at $CU13"
fi

echo ""
echo "=== Step 13: Verify installation ==="
"$VENV/bin/python" -c "
import sys; print(f'Python: {sys.version}')
import torch; print(f'torch: {torch.__version__}, CUDA: {torch.version.cuda}')
import triton; print(f'triton: {triton.__version__}')
import vllm; print(f'vllm: {vllm.__version__}')
import fla; print(f'fla: {fla.__version__}')
try:
    import causal_conv1d; print(f'causal_conv1d: {causal_conv1d.__version__}')
except: print('causal_conv1d: NOT INSTALLED')
try:
    import flash_attn_3; print('flash_attn_3: available')
except: print('flash_attn_3: NOT INSTALLED')
try:
    import flashinfer; print('flashinfer: available')
except: print('flashinfer: NOT INSTALLED')
import unsloth; print(f'unsloth: {unsloth.__version__}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Required env vars for training:"
echo "  UNSLOTH_CE_LOSS_TARGET_GB=4"
echo "  FLA_TILELANG=0                      # TileLang bugs on Blackwell"
echo "  VLLM_ENABLE_V1_MULTIPROCESSING=0    # no EngineCore subprocess"
echo "  UNSLOTH_VLLM_STANDBY=1              # unsloth CuMemAllocator patch"
echo "  FLASHINFER_WORKSPACE_BASE=/root      # prevent unsloth cache redirect"
echo "  CUDA_HOME=/usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.}"
echo "  LD_LIBRARY_PATH=$CU13/lib:/usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.}/lib64"
echo ""
echo "Example training command:"
echo "  cd /root/qwen3_rl_code && PYTHONPATH=/root/qwen3_rl_code \\"
echo "    UNSLOTH_CE_LOSS_TARGET_GB=4 FLA_TILELANG=0 \\"
echo "    VLLM_ENABLE_V1_MULTIPROCESSING=0 UNSLOTH_VLLM_STANDBY=1 \\"
echo "    FLASHINFER_WORKSPACE_BASE=/root \\"
echo "    CUDA_HOME=/usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.} \\"
echo "    LD_LIBRARY_PATH=$CU13/lib:/usr/local/cuda-${CUDA_TOOLKIT_VERSION//-/.}/lib64 \\"
echo "    $VENV/bin/python -u -m qwen3_rl.scripts.rl_train \\"
echo "    --model Qwen/Qwen3.5-0.8B --backend vllm --env reverse_string \\"
echo "    --group-size 8 --prompts-per-iter 4 --num-iters 100"
