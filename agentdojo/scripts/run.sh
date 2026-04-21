# scripts/run.sh — reference commands for attack + defense on banking.
# Activate your conda env first. Swap `-s banking` for `-s slack` to target
# the slack suite.
#
# Port convention (see ATTACK_DEFENSE.md for the full table):
#   8000  AGENT model           (LOCAL_LLM_PORT, used by --model LOCAL)
#         — served by scripts/vllm.sh or scripts/vllm_qwen.sh
#   8001  naivejudge defense    (NAIVEJUDGE_BASE_URL) — user-supplied endpoint
#   8002  shadowmemory defense  (SHADOWMEMORY_BASE_URL) — user-supplied endpoint

# -------------------------------------------------------------------------
# 1. Baseline benchmark (no attack / no defense)
# -------------------------------------------------------------------------
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model GPT_4O_MINI_2024_07_18 \
  -f

# -------------------------------------------------------------------------
# 2. Baseline attacks
#    Available attacks: important_instructions, tool_knowledge,
#                       ignore_previous, injecagent
# -------------------------------------------------------------------------
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model GPT_4O_MINI_2024_07_18 \
  --attack important_instructions \
  -f

# -------------------------------------------------------------------------
# 3. long_horizon (search-driven rationalization) attack pipeline.
#    Generates snippets under res/long_horizon/<agent_model>/banking/v1.2.1/
#    and immediately benchmarks them. The outer `for` loop retries the whole
#    pipeline N times — later tries reuse snippets from earlier tries and
#    focus compute on pairs that still haven't broken through.
# -------------------------------------------------------------------------
for i in {1..3}; do
  python -m agentdojo.attacks.search_attack_pipeline \
    --suite banking \
    --user-task-id user_task_0 user_task_1 \
    --injection-task-id injection_task_0 injection_task_1 \
    --max-workers 2 \
    --max-rewrites 1 \
    --agent-model-name gpt-4o-mini-2024-07-18 \
    --attack-model-name gpt-5-mini
done

# Same loop but running the attack AGAINST a defense. Available `--defense`
# values: repeat_user_prompt, transformers_pi_detector,
# spotlighting_with_delimiting, melon, shadowmemory. (shadowmemory needs a
# judge endpoint — see block 5.)
for i in {1..3}; do
  python -m agentdojo.attacks.search_attack_pipeline \
    --suite banking \
    --user-task-id user_task_0 user_task_1 \
    --injection-task-id injection_task_0 injection_task_1 \
    --max-workers 2 \
    --max-rewrites 1 \
    --agent-model-name gpt-4o-mini-2024-07-18 \
    --attack-model-name gpt-5-mini \
    --defense shadowmemory
done

# -------------------------------------------------------------------------
# 4. Run the agent model locally via vLLM (port 8000).
#    Pick ONE of these scripts — both serve an agent model on LOCAL_LLM_PORT.
# -------------------------------------------------------------------------
screen -dmS vllm-agent bash -c "bash scripts/vllm.sh"         # Llama-3.3-70B
# screen -dmS vllm-agent bash -c "bash scripts/vllm_qwen.sh"  # Qwen3-235B-FP8

# -------------------------------------------------------------------------
# 5. Defenses.
#    Available defenses: repeat_user_prompt, transformers_pi_detector,
#                        spotlighting_with_delimiting, melon,
#                        shadowmemory, naivejudge
#
#    `shadowmemory` / `naivejudge` need an OpenAI-compatible JUDGE endpoint
#    on ports 8002 / 8001 respectively. That endpoint is NOT served by
#    scripts/vllm*.sh — bring up your own judge server (e.g. from the
#    sibling memagent repo) and export SHADOWMEMORY_BASE_URL /
#    NAIVEJUDGE_BASE_URL if you need to override the defaults.
# -------------------------------------------------------------------------
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model GPT_4O_MINI_2024_07_18 \
  --attack long_horizon \
  --defense shadowmemory \
  -f
