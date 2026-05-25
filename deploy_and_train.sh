#!/usr/bin/env bash
# Deploy local qwen3_rl code and launch a remote GRPO run.
set -euo pipefail

SERVER="${SERVER:-ubuntu@154.54.100.242}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/qwen3_rl_code}"
VENV_DIR="${VENV_DIR:-/home/ubuntu/rl_venv312}"
PYTHON="$VENV_DIR/bin/python"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
ENV="${ENV:-reverse_string}"
BACKEND="${BACKEND:-vllm}"
MAX_TURNS="${MAX_TURNS:-1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-$GROUP_SIZE}"
NUM_ITERS="${NUM_ITERS:-100}"
LR="${LR:-5e-6}"
LOG_NAME="${LOG_NAME:-qwen_vllm_4096}"
OUTPUT_DIR="${OUTPUT_DIR:-$REMOTE_DIR/rl_output/$LOG_NAME}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_SUBSET="${EVAL_SUBSET:-64}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_TOP_K="${EVAL_TOP_K:-20}"
PROMPTS_PER_ITER="${PROMPTS_PER_ITER:-4}"
SEED="${SEED:-42}"
SKIP_TESTS="${SKIP_TESTS:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ASYNC_REFILL="${ASYNC_REFILL:-0}"
ASYNC_CROSS_STEP="${ASYNC_CROSS_STEP:-1}"
ASYNC_MAX_INFLIGHT="${ASYNC_MAX_INFLIGHT:-64}"
ASYNC_POOL_TARGET_GROUPS="${ASYNC_POOL_TARGET_GROUPS:-16}"
ASYNC_MAX_OFF_POLICY_STEPS="${ASYNC_MAX_OFF_POLICY_STEPS:-16}"
ASYNC_DROP_STALE="${ASYNC_DROP_STALE:-1}"
TRACE_DIR="${TRACE_DIR:-$REMOTE_DIR/rl_output/rollout_traces}"
TRACE_MAX_PER_ITER="${TRACE_MAX_PER_ITER:-8}"
REVERSE_MIN_LEN="${REVERSE_MIN_LEN:-4}"
REVERSE_MAX_LEN="${REVERSE_MAX_LEN:-8}"
REVERSE_HARD_MIN_LEN="${REVERSE_HARD_MIN_LEN:-}"
REVERSE_HARD_MAX_LEN="${REVERSE_HARD_MAX_LEN:-}"
REVERSE_HARD_PROB="${REVERSE_HARD_PROB:-0}"
REVERSE_CORRECT_THINK_BONUS="${REVERSE_CORRECT_THINK_BONUS:-0}"
REVERSE_THINK_BONUS_CAP_TOKENS="${REVERSE_THINK_BONUS_CAP_TOKENS:-1024}"
REVERSE_WRONG_LENGTH_PENALTY="${REVERSE_WRONG_LENGTH_PENALTY:-0}"
REVERSE_WRONG_LENGTH_CAP_TOKENS="${REVERSE_WRONG_LENGTH_CAP_TOKENS:-4096}"

ssh_remote() {
    ssh -o StrictHostKeyChecking=no -p "$PORT" "$SERVER" "$@"
}

scp_remote() {
    scp -P "$PORT" "$@"
}

remote_cu13() {
    ssh_remote "$PYTHON - <<'PY'
import os, site
for root in site.getsitepackages() + [site.getusersitepackages()]:
    path = os.path.join(root, 'nvidia', 'cu13')
    if os.path.exists(os.path.join(path, 'bin', 'nvcc')):
        print(path)
        break
PY"
}

echo "=== Uploading code to $SERVER:$REMOTE_DIR ==="
ssh_remote "mkdir -p $REMOTE_DIR/{qwen3_rl/{template,rollout,env,loss,scripts,sft},tests,logs,rl_output/rollout_traces}"
for dir in "" template rollout env loss scripts sft; do
    if [ -z "$dir" ]; then
        scp_remote qwen3_rl/*.py "$SERVER:$REMOTE_DIR/qwen3_rl/"
    else
        scp_remote qwen3_rl/$dir/*.py "$SERVER:$REMOTE_DIR/qwen3_rl/$dir/"
    fi
done
scp_remote tests/*.py "$SERVER:$REMOTE_DIR/tests/"
scp_remote qwen35_chat_template.jinja "$SERVER:$REMOTE_DIR/"
scp_remote sitecustomize.py "$SERVER:$REMOTE_DIR/"

CU13="$(remote_cu13)"
if [ -z "$CU13" ]; then
    echo "ERROR: pip CUDA 13 toolkit not found in $VENV_DIR. Run setup_server.sh first." >&2
    exit 1
fi
ssh_remote "test -e $CU13/lib64 || ln -s $CU13/lib $CU13/lib64; test -e $CU13/lib/libcudart.so || ln -sf libcudart.so.13 $CU13/lib/libcudart.so; mkdir -p $CU13/lib/stubs; test -e $CU13/lib/stubs/libcuda.so || ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so $CU13/lib/stubs/libcuda.so"

RUNTIME_ENV="PATH=$CU13/bin:$VENV_DIR/bin:\$PATH PYTHONPATH=$REMOTE_DIR CUDA_HOME=$CU13 LD_LIBRARY_PATH=$CU13/lib:\${LD_LIBRARY_PATH:-} UNSLOTH_CE_LOSS_TARGET_GB=4 FLA_TILELANG=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NO_USAGE_STATS=1 UNSLOTH_VLLM_STANDBY=1"

echo "=== Running import smoke test ==="
ssh_remote "cd $REMOTE_DIR && $RUNTIME_ENV $PYTHON -m qwen3_rl.scripts.rollout_debug"

if [ -z "$SKIP_TESTS" ]; then
    echo "=== Running CPU tests ==="
    ssh_remote "cd $REMOTE_DIR && $RUNTIME_ENV $PYTHON -m pytest tests/test_trajectory.py tests/test_template.py tests/test_grpo_loss.py tests/test_mock_repl.py tests/test_multi_turn_rollout.py tests/test_trainer_eval.py tests/test_trace.py tests/test_rollout_viewer.py tests/test_vllm_backend_env.py tests/test_reverse_string.py tests/test_reward_text.py tests/test_sft_jsonl.py -v"
fi

echo "=== Launching tmux session '$LOG_NAME' ==="
CMD="cd $REMOTE_DIR && $RUNTIME_ENV $PYTHON -u -m qwen3_rl.scripts.rl_train"
CMD="$CMD --model $MODEL --lora-r 16 --group-size $GROUP_SIZE --num-generations $NUM_GENERATIONS --num-iters $NUM_ITERS"
CMD="$CMD --max-turns $MAX_TURNS --max-tokens-per-turn $MAX_TOKENS --lr $LR"
CMD="$CMD --temperature $TEMPERATURE --top-p $TOP_P"
if [ -n "$TOP_K" ]; then
    CMD="$CMD --top-k $TOP_K"
fi
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --max-seq-len $MAX_SEQ_LEN --env $ENV --backend $BACKEND"
CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-lora-rank $MAX_LORA_RANK"
CMD="$CMD --eval-every $EVAL_EVERY --eval-subset $EVAL_SUBSET"
CMD="$CMD --eval-temperature $EVAL_TEMPERATURE --eval-top-p $EVAL_TOP_P --eval-top-k $EVAL_TOP_K"
CMD="$CMD --prompts-per-iter $PROMPTS_PER_ITER --seed $SEED"
CMD="$CMD --rollout-trace-dir $TRACE_DIR --rollout-trace-max-per-iter $TRACE_MAX_PER_ITER"
if [ "$ENV" = "reverse_string" ]; then
    CMD="$CMD --reverse-min-len $REVERSE_MIN_LEN --reverse-max-len $REVERSE_MAX_LEN"
    if [ -n "$REVERSE_HARD_MIN_LEN" ] || [ -n "$REVERSE_HARD_MAX_LEN" ]; then
        CMD="$CMD --reverse-hard-min-len $REVERSE_HARD_MIN_LEN --reverse-hard-max-len $REVERSE_HARD_MAX_LEN"
    fi
    CMD="$CMD --reverse-hard-prob $REVERSE_HARD_PROB"
    CMD="$CMD --reverse-correct-think-bonus $REVERSE_CORRECT_THINK_BONUS"
    CMD="$CMD --reverse-think-bonus-cap-tokens $REVERSE_THINK_BONUS_CAP_TOKENS"
    CMD="$CMD --reverse-wrong-length-penalty $REVERSE_WRONG_LENGTH_PENALTY"
    CMD="$CMD --reverse-wrong-length-cap-tokens $REVERSE_WRONG_LENGTH_CAP_TOKENS"
fi
if [ "$ENFORCE_EAGER" = "1" ]; then
    CMD="$CMD --enforce-eager"
fi
if [ "$ASYNC_REFILL" = "1" ]; then
    CMD="$CMD --async-refill"
fi
if [ "$ASYNC_CROSS_STEP" = "1" ]; then
    CMD="$CMD --async-cross-step"
fi
if [ -n "$ASYNC_MAX_INFLIGHT" ]; then
    CMD="$CMD --async-max-inflight $ASYNC_MAX_INFLIGHT"
fi
if [ -n "$ASYNC_POOL_TARGET_GROUPS" ]; then
    CMD="$CMD --async-pool-target-groups $ASYNC_POOL_TARGET_GROUPS"
fi
CMD="$CMD --async-max-off-policy-steps $ASYNC_MAX_OFF_POLICY_STEPS"
if [ "$ASYNC_DROP_STALE" = "0" ]; then
    CMD="$CMD --no-async-drop-stale"
fi

ssh_remote "tmux kill-session -t $LOG_NAME 2>/dev/null || true; tmux new-session -d -s $LOG_NAME \"bash -lc '$CMD 2>&1 | tee $REMOTE_DIR/logs/${LOG_NAME}.log'\""

cat <<MSG
=== Training launched ===
server=$SERVER  session=$LOG_NAME
model=$MODEL backend=$BACKEND env=$ENV M=$PROMPTS_PER_ITER num_generations=$NUM_GENERATIONS max_tokens=$MAX_TOKENS max_seq_len=$MAX_SEQ_LEN train_sampling=temp:$TEMPERATURE/top_p:$TOP_P/top_k:${TOP_K:-disabled} eval_sampling=temp:$EVAL_TEMPERATURE/top_p:$EVAL_TOP_P/top_k:$EVAL_TOP_K async_refill=$ASYNC_REFILL async_cross_step=$ASYNC_CROSS_STEP async_max_inflight=${ASYNC_MAX_INFLIGHT:-default}
Monitor: ssh -p $PORT $SERVER 'tail -f $REMOTE_DIR/logs/${LOG_NAME}.log'
Viewer:  http://${SERVER#*@}:8765/
MSG
