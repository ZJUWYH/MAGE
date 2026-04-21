# Attack & Defense Reference

This document explains how to run prompt-injection attacks and defenses against
the agentdojo benchmark for the **banking** and **slack** suites. All examples
use the banking suite — substitute `-s slack` for the slack equivalent.

Every command assumes your conda environment is already active and the package
is importable (e.g. `pip install -e .` from the repo root).

---

## 1. Running baseline attacks

Four baseline attacks are registered under `agentdojo.attacks`:

| Attack name              | Module                                  |
| ------------------------ | --------------------------------------- |
| `important_instructions` | `important_instructions_attacks.py`     |
| `tool_knowledge`         | `important_instructions_attacks.py`     |
| `ignore_previous`        | `baseline_attacks.py`                   |
| `injecagent`             | `baseline_attacks.py`                   |

Run a benchmark with one of them:

```bash
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model GPT_4O_MINI_2024_07_18 \
  --attack important_instructions \
  -f
```

Results land under `runs/<model_name>/banking/<user_task>/<attack>/<injection_task>.json`.

---

## 2. Running the long-horizon attack pipeline

The `search_attack_pipeline` module generates rationalization snippets with an
attack model and benchmarks them against a target agent model in a single run.
It supports round-based rewrites (`--max-rewrites`) to iteratively retry failed
attacks.

```bash
python -m agentdojo.attacks.search_attack_pipeline \
  --suite banking \
  --user-task-id user_task_0 user_task_1 \
  --injection-task-id injection_task_0 injection_task_1 \
  --max-workers 2 \
  --max-rewrites 1 \
  --agent-model-name gpt-4o-mini-2024-07-18 \
  --attack-model-name gpt-5-mini
```

Generated snippets are cached under
`res/long_horizon/<agent_model>/banking/v1.2.1/<user_task>_<injection_task>/`
and reused by the `long_horizon` attack in step 1.

---

## 3. Running defenses via the CLI benchmark

Six defenses are wired into `AgentPipeline.from_config`:

| Defense                        | Notes                                               |
| ------------------------------ | --------------------------------------------------- |
| `repeat_user_prompt`           | Prompt-only                                         |
| `transformers_pi_detector`     | Requires the `transformers` extra                   |
| `spotlighting_with_delimiting` | Prompt-only                                         |
| `melon`                        | Double-LLM check                                    |
| `shadowmemory`                 | MemAgent + JudgeAgent; needs a served judge model   |
| `naivejudge`                   | JudgeAgent only; needs a served judge model         |

Launch a benchmark with any defense:

```bash
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model GPT_4O_MINI_2024_07_18 \
  --attack long_horizon \
  --defense shadowmemory \
  -f
```

### Port map

There are up to three OpenAI-compatible endpoints in play, each bound to a
different port so the agent model and the defense judges can coexist:

| Port | Role                        | Served by                     | Env var override        |
| ---- | --------------------------- | ----------------------------- | ----------------------- |
| 8000 | **Agent model**             | `scripts/vllm.sh` *or* `scripts/vllm_qwen.sh` | `LOCAL_LLM_PORT` |
| 8001 | **`naivejudge` judge**      | *user-supplied* endpoint      | `NAIVEJUDGE_BASE_URL`   |
| 8002 | **`shadowmemory` judge**    | *user-supplied* endpoint      | `SHADOWMEMORY_BASE_URL` |

Only the **agent** endpoint is provided by this repo:

```bash
# Llama-3.3-70B on port 8000 (default)
screen -dmS vllm-agent bash -c "bash scripts/vllm.sh"

# OR Qwen3-235B-A22B-FP8 on port 8000 (default)
screen -dmS vllm-agent bash -c "bash scripts/vllm_qwen.sh"
```

Use one or the other — both bind to `LOCAL_LLM_PORT` (8000) and are consumed
by `--model LOCAL`. Both scripts also accept a `PORT` env var if you need a
non-default port.

The **judge** endpoints for `shadowmemory` and `naivejudge` are *not* served
by `scripts/vllm*.sh`. Bring up your own judge server (e.g. from the sibling
`memagent` repo) on port 8001 / 8002 and, if needed, point the defense at a
different URL or model:

```bash
export SHADOWMEMORY_BASE_URL=http://localhost:8002/v1
export SHADOWMEMORY_MODEL=Qwen/Qwen3-4B-Instruct-2507
# Optional: where per-task memories are persisted
export SHADOWMEMORY_MEMORY_DIR=/path/to/memagent/mem_res/agentdojo_live

# export NAIVEJUDGE_BASE_URL=http://localhost:8001/v1
```

`shadowmemory` also requires the sibling packages **memagent** and **rllm** to
be importable. `shadowmemory.py` adds `../memagent` and `../rllm` to
`sys.path` at import time, so clone them next to the `agentdojo/` directory.

---

## 4. File layout

| Path              | Purpose                                                   |
| ----------------- | --------------------------------------------------------- |
| `runs/`           | Benchmark outputs per model / suite / task                |
| `res/`            | Generated attack artifacts (snippets, example policies)   |
| `log/`            | vLLM server logs                                          |
| `scripts/run.sh`  | Reference commands — uncomment the section you need       |
