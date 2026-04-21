# ts=$(date +%Y%m%d_%H%M%S)
# model_short_name=Qwen3-4B
# bash scripts/train_agentdojo.sh > log/train_${model_short_name}_${ts}.log 2>&1
# memory v3 is decay reward + ablation reward + format penalty
# memory v4, fix the step record error, remove the tool call redundancy in context and tool call
# memory v5, try to make the memory more concise and less redundant
# memory v6, add the ablation reward back
# memory v7, gamma = 0.5, lambda = 0, tau = 1
# memory v8, gamma = 0.5, lambda = 0, tau = 0.5
# memory v9, gamma = 0.5, lambda = 0, tau = 0.5
# agentdojo v3, filter the harmless samples, and add penalty for false positive
# agentdojo v4, increase the max response length to 4096, less benign samples, reward even. Revise the prompt
# agentdojo v5, use agentdojov3,, leess dataset for sweep
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=0,1,2,3

lambda=0.0
normalize_reward=True
GAMMAS=(0.0)
TAUS=(1.0)

for gamma in "${GAMMAS[@]}"; do
    for tau in "${TAUS[@]}"; do
        CKPT_DIR="ckpt/Qwen3-4B-agentdojov2-final-gamma${gamma}-lambda${lambda}-tau${tau}-norm${normalize_reward}"
        MEMORY_DIR="mem_res/agentdojov2-final-train-gamma${gamma}-lambda${lambda}-tau${tau}-norm${normalize_reward}"
        EXP_NAME="Qwen3-4B-agentdojov2-final-gamma${gamma}-lambda${lambda}-tau${tau}-norm${normalize_reward}"

        echo "Starting run with gamma=${gamma}, lambda=${lambda}, tau=${tau}, normalize_reward=${normalize_reward}"

        # First, clean the memory dir
        mkdir -p "$MEMORY_DIR"
        rm -rf "$MEMORY_DIR"/*

        # Then, train the model
        python3 -m core.train \
            data.train_batch_size=32 \
            data.max_prompt_length=4096 \
            data.max_response_length=4096 \
            actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
            actor_rollout_ref.actor.ppo_mini_batch_size=32 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0.0 \
            actor_rollout_ref.actor.clip_ratio_low=0.2 \
            actor_rollout_ref.actor.clip_ratio_high=0.28 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.mode="async" \
            actor_rollout_ref.rollout.enforce_eager=False \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.rollout.top_p=1.0 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
            actor_rollout_ref.rollout.n=8 \
            actor_rollout_ref.rollout.val_kwargs.n=1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.adv_estimator=grpo \
            rllm.compact_filtering.enable=False \
            rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
            rllm.compact_filtering.mask_max_response_length_exceeded=True \
            rllm.compact_filtering.mask_max_turns_exceeded=False \
            rllm.compact_filtering.mask_timeout=True \
            rllm.rejection_sample.enable=False \
            rllm.rejection_sample.multiplier=1.0 \
            rllm.stepwise_advantage.enable=True \
            rllm.stepwise_advantage.mode=per_step \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name='memagent' \
            trainer.experiment_name=$EXP_NAME \
            trainer.val_before_train=True \
            trainer.default_local_dir=$CKPT_DIR \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.save_freq=150 \
            trainer.test_freq=10 \
            trainer.default_hdfs_dir=null \
            trainer.total_epochs=2 \
            rllm.workflow.use_workflow=True \
            +workflow.use_memory_reward=True \
            +workflow.gamma_mem=${gamma} \
            +workflow.lambda_ablation=${lambda} \
            +workflow.tau_length=${tau} \
            +workflow.normalize_reward=${normalize_reward} \
            +workflow.memory_dir=$MEMORY_DIR \
            +workflow.targeted_env=agentdojo
        pkill -9 -f 'ray::WorkerDict'

        ITER_FILE="$CKPT_DIR/latest_checkpointed_iteration.txt"
        if [ -f "$ITER_FILE" ]; then
            GLOBAL_STEP=$(cat "$ITER_FILE")
            LOCAL_DIR="$CKPT_DIR/global_step_${GLOBAL_STEP}/actor"
            echo "merge ckpt at global_step_${GLOBAL_STEP}"
            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$LOCAL_DIR" \
                --target_dir "$CKPT_DIR"
            echo "ALL Done."
        else
            echo "No checkpoint iteration file found at $ITER_FILE, skipping merge."
        fi
    done
done
