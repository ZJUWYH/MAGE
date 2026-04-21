#!/usr/bin/env bash
# Evaluation entry point for memagent on the AgentDojo benchmark.
#
# Three sections: (1) remote API models, (2) local vLLM-served models,
# (3) naive judge (no memory). Comment in/out the blocks you need.
#
# Example:
#     ts=$(date +%Y%m%d_%H%M%S)
#     model_short_name=Qwen3-4B
#     bash scripts/test.sh > log/test_${model_short_name}_${ts}.log 2>&1


# ============================================================
# Section 1: Remote API models (OpenAI-compatible)
# ============================================================

python -m core.test \
    --model gpt-5-mini \
    --n_parallel 1 \
    --indexes 0,1,2,3,4 \
    --split test \
    --memory_dir mem_res/agentdojov2_test \
    --targeted_env agentdojo \
    --dataset_name agentdojov2

python -m core.test \
    --model gpt-5.1 \
    --n_parallel 1 \
    --indexes 309 \
    --split test \
    --memory_dir mem_res/agentdojov2_test \
    --targeted_env agentdojo \
    --dataset_name agentdojov2


# ============================================================
# Section 2: Local vLLM-served models (start vllm_qwen.sh first)
# ============================================================

python -m core.test \
    --model Qwen3-4B-agentdojov2-final \
    --n_parallel 1 \
    --split test \
    --indexes 0,1,2,3,4 \
    --memory_dir mem_res/agentdojov2_test \
    --targeted_env agentdojo \
    --dataset_name agentdojov2 \
    --is_local_model --port 8002

python -m core.test \
    --model Qwen3-4B-agentdojov2-final \
    --n_parallel 1 \
    --split train \
    --indexes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --memory_dir mem_res/agentdojov2_train \
    --targeted_env agentdojo \
    --dataset_name agentdojov2 \
    --is_local_model --port 8002


# ============================================================
# Section 3: Naive judge (no memory agent)
# ============================================================

python -m core.test_naive \
    --model gpt-5-mini \
    --n_parallel 1 \
    --split test \
    --indexes 0,1,2,3,4 \
    --targeted_env agentdojo \
    --dataset_name agentdojov2

python -m core.test_naive \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --n_parallel 1 \
    --split test \
    --indexes 0,1,2,3,4 \
    --targeted_env agentdojo \
    --dataset_name agentdojov2 \
    --is_local_model --port 8002
