#!/bin/bash


# --- Qwen3-235B-A22B-Instruct-2507-FP8 Configuration ---
# cd ../rllm
# source .venv/bin/activate
# cd ../memagent
# Model: MoE with 235B total params, 22B active params per token
# Hardware: 4x H100 80GB (320GB total) - minimum viable config
# Note: 8 GPUs recommended for full context length support

# --- Server Parameters ---
HOST="localhost"
PORT="8002"

# --- Hardware Configuration ---
export CUDA_VISIBLE_DEVICES="2"
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95    # Max out GPU memory for this large model
MAX_MODEL_LEN=32768            # Reduced from 262K to fit in 4 GPUs (adjust as needed)
MAX_NUM_SEQS=16                # Limit concurrent sequences to save memory

# --- Performance Tuning ---
export OMP_NUM_THREADS=1       # Reduce CPU contention for better throughput
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Use FlashAttention for H100

#==============================================================================
#                             HELPER FUNCTIONS
#==============================================================================

# --- Function to start the VLLM server ---
# Takes the model path and an optional served model name as arguments.
start_vllm_server() {
    local model_path=$1
    local served_name_arg=""
    if [ -n "$2" ]; then
        served_name_arg="--served-model-name $2"
    fi

    echo "Starting VLLM OpenAI API server for model: $model_path"
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
      --dtype bfloat16 \
      --trust-remote-code
}


# start_vllm_server "ckpt/judge_only" "Qwen-4B-judge_only" >> log/vllm_qwen3_$(date +%Y%m%d_%H%M).log 2>&1

# start_vllm_server "ckpt/Qwen3-4B-Rewardv4-agentdojov2" "Qwen3-4B-Rewardv4-agentdojov2" >> log/vllm_qwen_$(date +%Y%m%d_%H%M).log 2>&1
    #   --trust-remote-code \
    #   --enable-auto-tool-choice \
    #   --tool-call-parser hermes \
    #   --reasoning-parser deepseek_r1 

start_vllm_server "Qwen/Qwen3-4B-Instruct-2507" >> log/vllm_qwen3_$(date +%Y%m%d_%H%M).log 2>&1


# start_vllm_server "ckpt/Qwen3-4B-stacv2v3/step_300_hf" "Qwen3-4B-stacv2v3" >> log/vllm_qwen3_$(date +%Y%m%d_%H%M).log 2>&1