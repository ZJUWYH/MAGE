#!/bin/bash

# Serve Qwen3-235B-A22B-Instruct-2507-FP8 as the AGENT model via vLLM.
# Consumed by `python -m agentdojo.scripts.benchmark --model LOCAL
# --model-id Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` — which reads
# `LOCAL_LLM_PORT` (default 8000) to reach this server.
#
# Hardware: 4x H100 80GB (320GB total) — minimum viable config.
# Note: 8 GPUs recommended for full context length support.
#
# Example (run detached):
#   screen -dmS vllm-agent bash -c "bash scripts/vllm_qwen.sh"

HOST="localhost"
PORT="${PORT:-8000}"

# --- Hardware Configuration ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"    # Max out GPU memory for this large model
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"                    # Reduced from 262K to fit in 4 GPUs
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"                         # Limit concurrent sequences to save memory

# --- Performance Tuning ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

start_vllm_server() {
    local model_path=$1
    local served_name_arg=""
    if [ -n "$2" ]; then
        served_name_arg="--served-model-name $2"
    fi

    echo "Starting VLLM OpenAI API server for model: $model_path on port $PORT"
    echo "Config: TP=$TENSOR_PARALLEL_SIZE, MaxLen=$MAX_MODEL_LEN, MaxSeqs=$MAX_NUM_SEQS"

    python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      $served_name_arg \
      --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
      --port $PORT \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --max-model-len $MAX_MODEL_LEN \
      --max-num-seqs $MAX_NUM_SEQS \
      --disable-log-requests \
      --disable-log-stats \
      --dtype auto \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser hermes
}

mkdir -p log
start_vllm_server "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" >> "log/vllm_qwen3_${PORT}_$(date +%Y%m%d_%H%M).log" 2>&1
