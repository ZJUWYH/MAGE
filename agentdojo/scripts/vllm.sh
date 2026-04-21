#!/bin/bash

# Serve Llama-3.3-70B-Instruct as the AGENT model via vLLM.
# Consumed by `python -m agentdojo.scripts.benchmark --model LOCAL` —
# which reads `LOCAL_LLM_PORT` (default 8000) to reach this server.
#
# Example (run detached):
#   screen -dmS vllm bash -c "bash scripts/vllm.sh"

HOST="localhost"
PORT="${PORT:-8000}"

# --- Hardware Configuration ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3,4,5}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-bfloat16}"

start_vllm_server() {
    local model_path=$1
    local served_name_arg=""
    if [ -n "$2" ]; then
        served_name_arg="--served-model-name $2"
    fi

    echo "Starting VLLM OpenAI API server for model: $model_path on port $PORT"
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      $served_name_arg \
      --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
      --port $PORT \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --disable-log-requests \
      --disable-log-stats \
      --dtype $DTYPE \
      --enable-auto-tool-choice \
      --tool-call-parser pythonic
}

mkdir -p log
start_vllm_server "meta-llama/Llama-3.3-70B-Instruct" >> "log/vllm_${PORT}_$(date +%Y%m%d_%H%M).log" 2>&1
