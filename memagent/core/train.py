"""
Training script for memagent security analysis workflow.

Uses AgentTrainer with Hydra configuration to train the joint MemAgent + JudgeAgent
workflow via RL (GRPO/PPO). Reward comes from the environment step-wise:
  - JudgeAgent steps: +1 for correct judgment, -1 for incorrect
  - MemAgent steps: 0 (no gradient; future-ready for memory rewards)

Usage:
    python -m core.train [hydra overrides...]

    Or via the shell script:
    bash scripts/train_agentdojo.sh
"""

import os

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

from core.workflow import JudgeAgentNaiveWorkflow, MemTarAgentWorkflow


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    workflow_type = config.workflow.get("workflow_type", "memagent")
    targeted_env = config.workflow.get("targeted_env", "agentdojo")
    if targeted_env == "agentdojo":
        train_dataset = DatasetRegistry.load_dataset("agentdojov2", "train")
        test_dataset = DatasetRegistry.load_dataset("agentdojov2", "test")
    elif targeted_env == "stac":
        train_dataset = DatasetRegistry.load_dataset("stac_v2v3", "train")
        test_dataset = DatasetRegistry.load_dataset("stac_v2v3", "test")
    else:
        raise ValueError(f"Invalid targeted environment: {targeted_env}")

    if train_dataset is None:
        raise RuntimeError(
            f"Train dataset not found for '{targeted_env}'. Run the dataset preprocessing "
            f"and 'python -m dataset.json_to_rllm_format' first."
        )
    if test_dataset is None:
        raise RuntimeError(
            f"Test dataset not found for '{targeted_env}'. Run the dataset preprocessing "
            f"and 'python -m dataset.json_to_rllm_format' first."
        )

    print(f"[Train] Dataset '{targeted_env}': train={len(train_dataset)}, test={len(test_dataset)}")

    if workflow_type == "naive":
        workflow_class = JudgeAgentNaiveWorkflow
        workflow_args = {
            "targeted_env": targeted_env,
        }
    else:
        workflow_class = MemTarAgentWorkflow
        workflow_args = {
            "use_memory_reward": config.workflow.use_memory_reward,
            "gamma_mem": config.workflow.get("gamma_mem", 0.0),
            "lambda_ablation": config.workflow.get("lambda_ablation", 0.0),
            "tau_length": config.workflow.get("tau_length", 0.0),
            "model_name": os.path.basename(config.trainer.default_local_dir) if getattr(config.trainer, "default_local_dir", None) else None,
            "memory_dir": config.workflow.memory_dir,
            "targeted_env": targeted_env,
        }

    trainer = AgentTrainer(
        workflow_class=workflow_class,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
