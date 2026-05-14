#!/bin/bash
# Deploy code to remote server and launch GRPO training in screen.
#
# Usage:
#   ./deploy_and_train.sh
#   ENV=reverse_string BACKEND=vllm ./deploy_and_train.sh
#   ENV=string_match LR=1e-4 NUM_ITERS=200 ./deploy_and_train.sh
set -e

SERVER="${SERVER:?Set SERVER (e.g. SERVER=root@yourhost)}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/root/qwen3_rl_code}"
VENV="${VENV:-/root/rl_venv312/bin/python}"

# Configurable via env vars (defaults shown)
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
ENV="${ENV:-string_match}"
BACKEND="${BACKEND:-hf}"
MAX_TURNS="${MAX_TURNS:-1}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_ITERS="${NUM_ITERS:-100}"
LR="${LR:-1e-4}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_SUBSET="${EVAL_SUBSET:-32}"
PROMPTS_PER_ITER="${PROMPTS_PER_ITER:-1}"
SEED="${SEED:-42}"
LOG_NAME="${LOG_NAME:-rl_train}"
SKIP_TESTS="${SKIP_TESTS:-}"

echo "=== Uploading code ==="
ssh -o StrictHostKeyChecking=no $SERVER -p $PORT "rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR/{qwen3_rl/{template,rollout,env,loss,scripts},tests}"

for dir in "" template rollout env loss scripts; do
    if [ -z "$dir" ]; then
        scp -P $PORT qwen3_rl/*.py $SERVER:$REMOTE_DIR/qwen3_rl/
    else
        scp -P $PORT qwen3_rl/$dir/*.py $SERVER:$REMOTE_DIR/qwen3_rl/$dir/
    fi
done

scp -P $PORT tests/*.py $SERVER:$REMOTE_DIR/tests/
scp -P $PORT qwen35_chat_template.jinja $SERVER:$REMOTE_DIR/

# Ensure nvcc 13 lib layout is linker-friendly (FlashInfer JIT expects lib64/ and libcudart.so)
VENV_ROOT="${VENV%/bin/*}"
CU13="$VENV_ROOT/lib/python3.12/site-packages/nvidia/cu13"
ssh $SERVER -p $PORT "test -d $CU13 && { \
  test -e $CU13/lib64 || ln -s $CU13/lib $CU13/lib64; \
  test -e $CU13/lib/libcudart.so || ln -sf libcudart.so.13 $CU13/lib/libcudart.so; \
  mkdir -p $CU13/lib/stubs; \
  test -e $CU13/lib/stubs/libcuda.so || ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so $CU13/lib/stubs/libcuda.so; \
} || echo 'nvidia-cuda-nvcc not installed, skipping lib64 setup'"

echo "=== Running import smoke test ==="
ssh $SERVER -p $PORT "cd $REMOTE_DIR && $VENV -m qwen3_rl.scripts.rollout_debug"

if [ -z "$SKIP_TESTS" ]; then
    echo "=== Running CPU tests ==="
    ssh $SERVER -p $PORT "cd $REMOTE_DIR && $VENV -m pytest tests/test_trajectory.py tests/test_template.py tests/test_grpo_loss.py tests/test_multi_turn_rollout.py -v"
fi

echo "=== Launching training in screen '$LOG_NAME' ==="
# SM120 (Blackwell) needs nvcc 13+ for FlashInfer JIT — pip-installed at nvidia/cu13
CUDA_ENV="UNSLOTH_CE_LOSS_TARGET_GB=4"
CUDA_ENV="$CUDA_ENV FLA_TILELANG=0"
CUDA_ENV="$CUDA_ENV VLLM_ENABLE_V1_MULTIPROCESSING=0"
CUDA_ENV="$CUDA_ENV UNSLOTH_VLLM_STANDBY=1"
CUDA_ENV="$CUDA_ENV FLASHINFER_WORKSPACE_BASE=/root"
CUDA_ENV="$CUDA_ENV CUDA_HOME=/usr/local/cuda-13.0"
CUDA_ENV="$CUDA_ENV LD_LIBRARY_PATH=/root/rl_venv312/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-13.0/lib64:\$LD_LIBRARY_PATH"
CMD="cd $REMOTE_DIR && $CUDA_ENV $VENV -u -m qwen3_rl.scripts.rl_train"
CMD="$CMD --model $MODEL --lora-r 16 --group-size $GROUP_SIZE --num-iters $NUM_ITERS"
CMD="$CMD --max-turns $MAX_TURNS --max-tokens-per-turn $MAX_TOKENS --lr $LR"
CMD="$CMD --max-seq-len $MAX_SEQ_LEN --env $ENV --backend $BACKEND"
CMD="$CMD --eval-every $EVAL_EVERY --eval-subset $EVAL_SUBSET"
CMD="$CMD --prompts-per-iter $PROMPTS_PER_ITER --seed $SEED"

ssh $SERVER -p $PORT "screen -dmS $LOG_NAME bash -c '$CMD 2>&1 | tee /root/${LOG_NAME}.log'"

echo ""
echo "=== Training launched ==="
echo "  env=$ENV  backend=$BACKEND  max_turns=$MAX_TURNS  G=$GROUP_SIZE  lr=$LR"
echo "  Monitor: ssh $SERVER -p $PORT 'tail -f /root/${LOG_NAME}.log'"
