# ShadowMem: Safeguarding LLM Agents against Long-Horizon Threats via Shadow Memory

This repository contains the official code release for the paper *Safeguarding LLM Agents against Long-Horizon Threats via Shadow Memory*. **ShadowMem** is a defensive framework that maintains a dedicated, safety-focused agentic memory — inspired by the *shadow stack* abstraction in systems security — to distill and retain safety-critical context across an agent's full execution trajectory, and uses this shadow memory to proactively assess pending actions before they are executed.

![ShadowMem overview](assets/mage.png)

## Repository layout

| Folder        | Role |
| ------------- | ---- |
| `rllm/`       | RL post-training backbone. Provides the training loop (verl / GRPO / PPO backends) used to train the shadow-memory judge. |
| `memagent/`   | Shadow-memory judge implementation — core logic, prompts, dataset adapters, and training/evaluation scripts. Imported as a Python module by the pipelines. |
| `agentdojo/`  | Benchmark harness for evaluating prompt-injection attacks and defenses on LLM agents. Extended in this paper to cover long-horizon threats. |

## Installation

The three components share a single Python environment. Install `rllm` first (it pulls in the RL/serving stack), then `agentdojo` in the same environment. `memagent` is a pure Python module and needs no install step — it is picked up via `PYTHONPATH` or by running scripts from the `memagent/` directory.

Requirements: Python ≥ 3.10, CUDA GPUs for training, `uv` or `pip`.

```bash
# 1. Create and activate a fresh environment (Python 3.10+)
uv venv --python 3.10
source .venv/bin/activate

# 2. Install rllm in editable mode with the verl training backend.
#    See rllm/README.md for backend/GPU notes and optional tinker backend.
cd rllm
uv pip install -e ".[verl]"
cd ..

# 3. Install agentdojo in editable mode.
#    See agentdojo/README.md for optional extras (transformers detector, docs, etc.).
cd agentdojo
uv pip install -e .
cd ..

# 4. memagent needs no install — import paths resolve from the repo root.
```

For detailed per-project setup (Docker, specific CUDA/torch versions, optional extras), consult the docs inside each folder (`rllm/README.md`, `memagent/README.md`, `agentdojo/README.md`).

## Usage

- `memagent/` — training and testing of the shadow-memory judge. See `memagent/README.md`.
- `agentdojo/` — further live-agent evaluation with the trained judge in the loop. See `agentdojo/ATTACK_DEFENSE.md` for the attack/defense commands used in the paper.
- `rllm/` — minor extensions to the trainer for judge metrics and ASR/BU reporting. See `rllm/diff.md` for a summary of changes against the upstream `rllm` codebase.

## Models & datasets

Released on Hugging Face:

- PI judge model: https://huggingface.co/HuntingQuasar/pi-model
- STAC judge model: https://huggingface.co/HuntingQuasar/stac-model
- PI dataset: https://huggingface.co/datasets/HuntingQuasar/pi-dataset
- STAC dataset: https://huggingface.co/datasets/HuntingQuasar/stac-dataset


## Citation

If you use ShadowMem in your research, please cite:

```bibtex
@inproceedings{wang2026shadowmem,
  author    = {Yuhui Wang and Tanqiu Jiang and Jiacheng Liang and Charles Fleming and Ting Wang},
  title     = {Safeguarding {LLM} Agents against Long-Horizon Threats via Shadow Memory},
  booktitle = {Proceedings of the 2026 ACM SIGSAC Conference on Computer and Communications Security},
  series    = {CCS '26},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3830454.3846601},
  url       = {https://doi.org/10.1145/3830454.3846601}
}
```
