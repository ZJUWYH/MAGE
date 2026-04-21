"""
NaiveJudge – agentdojo BasePipelineElement that gates tool execution
using the rllm JudgeAgentNaive workflow (no memory, full trajectory).

This is the ablation baseline for ShadowMemory: it uses the full
conversation context instead of accumulated memory for judging.

Architecture
------------
  ToolsExecutionLoop:
    [NaiveJudge]    <-- sits BEFORE ToolsExecutor
    [ToolsExecutor] <-- only runs when NaiveJudge APPROVEs
    [LLM]

For every assistant message that contains tool calls, NaiveJudge:
  1. Converts the full agentdojo message list to plain dicts and passes that
     entire list as judge ``context`` (pending tool calls from the last
     assistant message are still passed separately as ``pending_tool_calls``).
  2. Runs the rllm LiveNaiveJudgeWorkflow (JudgeAgentNaive only)
     synchronously.
  3. On APPROVE  → returns messages unchanged; ToolsExecutor executes.
  4. On DENY     → depends on deny_mode:
                   "stop"    – raises AbortAgentError
                   "replace" – appends fake tool results with rationale
"""

import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import requests

# ---------------------------------------------------------------------------
# Ensure the memagent / rllm packages are importable from the agentdojo venv.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]   # …/9/
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "rllm"), str(_REPO_ROOT / "memagent")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rllm.engine.rollout.openai_engine import OpenAIEngine

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.pi_detector import normalize_content_to_text
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.token_usage import add_token_usage
from agentdojo.types import ChatAssistantMessage, ChatMessage, ChatToolResultMessage, text_content_block_from_string

from memagent.core.live_naive_workflow import LiveNaiveJudgeWorkflow

BLOCKED_TEMPLATE = (
    "[BLOCKED] Tool execution denied by naivejudge security monitor.\n Continue with the original user intention."
    "Rationale: {rationale}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_from_server(base_url: str) -> str | None:
    """Query an OpenAI-compatible /models endpoint and return the first model id."""
    try:
        url = base_url.rstrip("/") + "/models"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        model_name = response.json()["data"][0]["id"]
        logging.info(f"[NaiveJudge] Auto-detected model from server: {model_name}")
        return model_name
    except Exception as exc:
        logging.warning(f"[NaiveJudge] Could not fetch model from {base_url}: {exc}")
        return None


def _function_call_to_dict(tc: FunctionCall) -> dict[str, Any]:
    """Convert agentdojo FunctionCall to the plain dict expected by memagent."""
    return {"function": tc.function, "args": dict(tc.args)}


def _agentdojo_messages_to_plain(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert agentdojo ChatMessage sequence to plain dicts for judge context."""
    plain: list[dict[str, Any]] = []
    for msg in messages:
        m: dict[str, Any] = {"role": msg["role"]}

        # Flatten content blocks → plain string
        raw_content = msg.get("content")
        text = normalize_content_to_text(raw_content)
        if text:
            m["content"] = text

        # Tool calls on assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            m["tool_calls"] = [_function_call_to_dict(tc) for tc in tool_calls]

        # Single tool_call on tool-result messages
        tool_call = msg.get("tool_call")
        if tool_call:
            m["tool_call"] = _function_call_to_dict(tool_call)

        plain.append(m)
    return plain


def _run_async(coro):
    """
    Run an async coroutine from synchronous code.

    Handles the case where an event loop is already running (e.g., inside
    Jupyter or certain test runners) by spawning a fresh thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NaiveJudge(BasePipelineElement):
    """
    Agentdojo pipeline element that gates tool execution using the rllm
    JudgeAgentNaive workflow (LiveNaiveJudgeWorkflow) — no memory.

    This is the ablation baseline for ShadowMemory.

    Parameters
    ----------
    model : str | None
        Model name served by the rllm / vLLM backend.
        Resolved in order: argument → NAIVEJUDGE_MODEL env var →
        queried from the vLLM server at base_url → "Qwen/Qwen3-4B-Instruct-2507".
    base_url : str
        Base URL of the OpenAI-compatible inference endpoint.
        Defaults to NAIVEJUDGE_BASE_URL env var, or
        "http://localhost:8000/v1".
    api_key : str
        API key for the inference endpoint (often "EMPTY" for local).
    max_workers : int
        Thread pool size for the async executor used by rllm.
    deny_mode : "stop" | "replace"
        Controls what happens when the workflow returns DENY.
        ``"stop"``    – raise :class:`AbortAgentError` to immediately halt
                        the entire pipeline.
                        **This is the default.**
        ``"replace"`` – replace every blocked tool call's result with a
                        fixed ``BLOCKED_TEMPLATE`` message and let the LLM
                        continue.
    """

    name = "naivejudge"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_workers: int = 2,
        deny_mode: Literal["stop", "replace"] = "stop",
    ) -> None:
        _base_url = (
            base_url
            or os.getenv("NAIVEJUDGE_BASE_URL")
            or "http://localhost:8000/v1"
        )
        _api_key = api_key or os.getenv("NAIVEJUDGE_API_KEY", "EMPTY")

        _model = (
            model
            or os.getenv("NAIVEJUDGE_MODEL")
            or _get_model_from_server(_base_url)
            or "Qwen/Qwen3-4B-Instruct-2507"
        )

        self._engine = OpenAIEngine(
            model=_model,
            base_url=_base_url,
            api_key=_api_key,
            sampling_params={"max_tokens": 4096},
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._workflow = LiveNaiveJudgeWorkflow(
            rollout_engine=self._engine,
            executor=self._executor,
        )

        self._deny_mode: Literal["stop", "replace"] = deny_mode

    # ------------------------------------------------------------------
    # BasePipelineElement.query
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:

        # Only intercept when the last message is an assistant with tool_calls
        if not messages:
            return query, runtime, env, messages, extra_args
        last = messages[-1]
        if last["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        tool_calls: list[FunctionCall] | None = last.get("tool_calls")
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        # ------------------------------------------------------------------
        # Full conversation as judge context; pending calls from last message
        # ------------------------------------------------------------------
        plain_msgs = _agentdojo_messages_to_plain(messages)
        task_dict = {
            "context": plain_msgs,
            "pending_tool_calls": [_function_call_to_dict(tc) for tc in tool_calls],
        }

        # ------------------------------------------------------------------
        # Run the rllm workflow (synchronously from the agentdojo pipeline)
        # ------------------------------------------------------------------
        uid = str(uuid4())
        try:
            episode = _run_async(
                self._workflow.run_with_termination_handling(task_dict, uid=uid)
            )
            # LiveNaiveJudgeWorkflow produces exactly 1 trajectory (judge only)
            assert episode is not None and len(episode.trajectories) == 1, (
                f"[NaiveJudge] expected 1 trajectory (judge), "
                f"got {len(episode.trajectories) if episode else 'None'}"
            )
            for traj in episode.trajectories:
                assert len(traj.steps) == 1, (
                    f"[NaiveJudge] expected 1 step per trajectory '{traj.name}', "
                    f"got {len(traj.steps)}"
                )
            # Accumulate defense token usage from JudgeAgentNaive LLM call.
            _defense_input = 0
            _defense_output = 0
            for traj in episode.trajectories:
                mo = traj.steps[0].model_output
                if mo is not None:
                    _defense_input += getattr(mo, "prompt_length", 0) or 0
                    _defense_output += getattr(mo, "completion_length", 0) or 0
            if _defense_input or _defense_output:
                add_token_usage(
                    extra_args,
                    input_tokens=_defense_input,
                    output_tokens=_defense_output,
                    scope="defense",
                )
        except Exception as exc:
            # On any backend failure, fail safe: do NOT block the tool.
            print(f"[NaiveJudge] backend error, defaulting to APPROVE: {exc}")
            return query, runtime, env, messages, extra_args

        # ------------------------------------------------------------------
        # Read back results
        # ------------------------------------------------------------------
        decision = (self._workflow.env.decision or "APPROVE").upper()
        rationale = self._workflow.env.rationale or ""

        # ------------------------------------------------------------------
        # Act on the decision
        # ------------------------------------------------------------------
        extra_args.setdefault("naivejudge_decisions", []).append(
            {
                "decision": decision,
                "rationale": rationale,
                "tool_calls": [_function_call_to_dict(tc) for tc in tool_calls],
            }
        )

        if decision == "DENY":
            if self._deny_mode == "stop":
                abort_msg = (
                    f"[BLOCKED] Prompt injection detected by naivejudge security monitor. "
                    f"Rationale: {rationale}"
                )
                raise AbortAgentError(abort_msg, list(messages), env, extra_args)

            # deny_mode == "replace"
            blocked_content = BLOCKED_TEMPLATE.format(rationale=rationale)
            fake_results: list[ChatToolResultMessage] = [
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(blocked_content)],
                    tool_call_id=tc.id,
                    tool_call=tc,
                    error=f"Tool execution blocked by naivejudge security monitor. Rationale: {rationale}",)
                for tc in tool_calls
            ]
            stop_message = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(
                    "[BLOCKED] Prompt injection detected by naivejudge security monitor. "
                    f"Rationale: {rationale}"
                )],
                tool_calls=None,
            )
            return query, runtime, env, [*messages, *fake_results, stop_message], extra_args

        # APPROVE – let ToolsExecutor run as normal
        return query, runtime, env, messages, extra_args
