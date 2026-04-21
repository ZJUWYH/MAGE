# Memagent

Memagent is a joint-training framework for prompt-injection defense on
tool-using LLM agents. Two collaborating agents are trained together with
reinforcement learning:

- **MemAgent** — maintains a rolling, natural-language *security memory*
  summarizing suspicious context observed so far.
- **JudgeAgent** — uses that memory (plus the current proposed tool call)
  to emit an `APPROVE` / `DENY` decision before the target agent actually
  invokes the tool.

Both agents share a single base model and are optimized jointly via
GRPO/PPO on top of the `rllm` + `verl` stack. Evaluation uses attack
benchmarks from AgentDojo and STAC.

A separate naive baseline (`JudgeAgentNaive`) is also provided: a single
judge with no memory, evaluated end-to-end for comparison.

## Repository layout

- `core/` — agents (`agent.py`), environments (`env.py`, `live_env.py`,
  `live_naive_env.py`), workflows (`workflow.py`, `live_workflow.py`,
  `live_naive_workflow.py`), prompt templates (`prompt.py`), persistent
  memory store (`memory.py`), and the three entry points:
  - `train.py` — RL training (Hydra + `AgentTrainer`).
  - `test.py` — offline evaluation of the memory + judge workflow.
  - `test_naive.py` — offline evaluation of the naive judge baseline.
- `dataset/` — preprocessing scripts that register datasets via
  `DatasetRegistry` (`agentdojov2`, `agentdojov3`, `stac_v2v3`). Raw
  datasets are not shipped with this repo (see `.gitattributes`); only
  `dataset/json_to_rllm_format.py` is included as the shared conversion
  helper.
- `scripts/` — shell wrappers that call the entry points with the
  hyperparameters used in the paper:
  - `train_agentdojo.sh` — train on AgentDojo.
  - `train_stac.sh` — train on STAC.
  - `test.sh` — consolidated evaluation (three sections: remote API,
    local vLLM, naive judge).
  - `vllm_qwen.sh` — start an OpenAI-compatible vLLM server on
    `localhost:8002` for local-model evaluation.

## Setup

1. Python ≥ 3.10. CUDA GPUs are required for training (the default
   `scripts/train_agentdojo.sh` uses 4× GPUs; adjust
   `CUDA_VISIBLE_DEVICES`, `trainer.n_gpus_per_node`, and batch sizes to
   your hardware).
2. Install [`rllm`](https://github.com/) (the agent-RL training runtime)
   and its `verl` dependency. The shell scripts assume the `rllm` repo
   lives as a sibling of this one (`../rllm`) and that its virtualenv is
   at `../rllm/.venv`.
3. (Optional) Install vLLM for local-model serving and evaluation.

## Preparing datasets

Raw datasets live under `dataset/<name>/` and are excluded from the
release tarball. To reproduce:

1. Place the raw trajectory/attack data into `dataset/agentdojov2/` or
   `dataset/stac/`.
2. Run the builder for that dataset (e.g. `python -m dataset.agentdojov2`
   or `python -m dataset.stac`) to produce the `all_samples.json` /
   `test.json` artifacts.
3. Register the train/test splits for `rllm` consumption:

   ```bash
   python -m dataset.json_to_rllm_format
   ```

## Training

From the repository root:

```bash
bash scripts/train_agentdojo.sh   # target: AgentDojo (agentdojov2)
bash scripts/train_stac.sh        # target: STAC (stac_v2v3)
```

The reward-shaping hyperparameters live at the top of each script
(`GAMMAS`, `TAUS`, `lambda`, `normalize_reward`). Each run writes
checkpoints into `ckpt/<exp_name>/` and per-step memory traces into
`mem_res/<exp_name>/`. After training, the script merges the final FSDP
checkpoint into a HuggingFace-loadable directory.

## Evaluation

```bash
bash scripts/test.sh
```

`scripts/test.sh` is organized into three sections. Edit the file and
comment in/out the blocks you need:

1. **Remote API models** — calls OpenAI-compatible endpoints (e.g.
   `gpt-5-mini`, `gpt-5.1`).
2. **Local vLLM-served models** — requires `scripts/vllm_qwen.sh`
   already running. Add `--is_local_model --port 8002`.
3. **Naive judge (no memory)** — uses `core.test_naive` for the
   single-agent baseline.

Typical offline evaluation on a single trajectory index:

```bash
python -m core.test \
    --model gpt-5-mini \
    --n_parallel 1 \
    --indexes 0,1,2,3,4 \
    --split test \
    --memory_dir mem_res/agentdojov2_test \
    --targeted_env agentdojo \
    --dataset_name agentdojov2
```

## Serving local models

To run a trained or base Qwen checkpoint behind an OpenAI-compatible
endpoint at `http://localhost:8002/v1`:

```bash
bash scripts/vllm_qwen.sh
```

Edit the bottom of the script to point at the checkpoint path you want
served (defaults to `Qwen/Qwen3-4B-Instruct-2507`).

## Integration with AgentDojo

The companion AgentDojo fork imports this package to run the judge at
attack-evaluation time:

```python
from memagent.core.live_workflow import LiveShadowMemoryWorkflow      # memory + judge
from memagent.core.live_naive_workflow import LiveNaiveJudgeWorkflow  # naive baseline
from memagent.core.memory import MemoryManager
```

These module names are stable — renaming them requires updating the
corresponding `agentdojo/src/agentdojo/agent_pipeline/shadowmemory.py`
and `naivejudge.py`.
