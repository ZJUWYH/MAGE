"""
ShadowMemory – agentdojo BasePipelineElement that gates tool execution
using the rllm MemAgent + JudgeAgent workflow.

Architecture
------------
  ToolsExecutionLoop:
    [ShadowMemory]  <-- sits BEFORE ToolsExecutor
    [ToolsExecutor] <-- only runs when ShadowMemory APPROVEs
    [LLM]

For every assistant message that contains tool calls, ShadowMemory:
  1. Converts the agentdojo messages to the memagent segment format
     (reusing segment_single_trajectory / clean_message from
      memagent/dataset/agentdojo.py).
  2. Injects the current per-task memory and runs the rllm
     LiveShadowMemoryWorkflow (MemAgent → JudgeAgent) synchronously.
  3. On APPROVE  → returns messages unchanged; ToolsExecutor executes.
  4. On DENY     → appends a fake ChatToolResultMessage for every
                   blocked tool call that contains the rationale.
                   ToolsExecutor then sees role="tool" as the last
                   message and skips execution.  The target LLM is
                   called next and can adapt.
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
# Adjust paths if the repo layout differs.
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

from memagent.core.live_workflow import LiveShadowMemoryWorkflow
from memagent.core.memory import MemoryManager
from memagent.dataset.agentdojo import segment_single_trajectory

BLOCKED_TEMPLATE = (
    "[BLOCKED] Tool execution denied by shadowmemory security monitor.\n Continue with the original user intention."
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
        logging.info(f"[ShadowMemory] Auto-detected model from server: {model_name}")
        return model_name
    except Exception as exc:
        logging.warning(f"[ShadowMemory] Could not fetch model from {base_url}: {exc}")
        return None


def _function_call_to_dict(tc: FunctionCall) -> dict[str, Any]:
    """Convert agentdojo FunctionCall to the plain dict expected by memagent."""
    return {"function": tc.function, "args": dict(tc.args)}


def _agentdojo_messages_to_plain(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """
    Convert agentdojo ChatMessage sequence to plain dicts that
    segment_single_trajectory() can process.

    Agentdojo messages use typed FunctionCall objects and structured
    content blocks; memagent preprocessing expects plain strings/dicts.
    """
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

    # A loop is already running – execute in a dedicated thread that has
    # its own event loop to avoid "This event loop is already running".
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShadowMemory(BasePipelineElement):
    """
    Agentdojo pipeline element that gates tool execution using the rllm
    MemAgent + JudgeAgent workflow (LiveShadowMemoryWorkflow).

    Parameters
    ----------
    model : str | None
        Model name served by the rllm / vLLM backend.
        Resolved in order: argument → SHADOWMEMORY_MODEL env var →
        queried from the vLLM server at base_url → "Qwen/Qwen3-4B-Instruct-2507".
    base_url : str
        Base URL of the OpenAI-compatible inference endpoint.
        Defaults to SHADOWMEMORY_BASE_URL env var, or
        "http://localhost:8000/v1".
    api_key : str
        API key for the inference endpoint (often "EMPTY" for local).
    max_workers : int
        Thread pool size for the async executor used by rllm.
    memory_dir : str | None
        If set, per-task memory is persisted to this directory.
    deny_mode : "stop" | "replace"
        Controls what happens when the workflow returns DENY.
        ``"stop"``    – raise :class:`AbortAgentError` to immediately halt
                        the entire pipeline (like MELON's raise_on_injection).
                        **This is the default.**
        ``"replace"`` – replace every blocked tool call's result with a
                        fixed ``BLOCKED_TEMPLATE`` message and let the LLM
                        continue (the original behaviour).
    """

    name = "shadowmemory"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_workers: int = 2,
        memory_dir: str | None = None,
        deny_mode: Literal["stop", "replace"] = "stop",
    ) -> None:
        _base_url = (
            base_url
            or os.getenv("SHADOWMEMORY_BASE_URL")
            or "http://localhost:8000/v1"
        )
        _api_key = api_key or os.getenv("SHADOWMEMORY_API_KEY", "EMPTY")

        _model = (
            model
            or os.getenv("SHADOWMEMORY_MODEL")
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
        self._workflow = LiveShadowMemoryWorkflow(
            rollout_engine=self._engine,
            executor=self._executor,
        )

        self._memory_dir = Path(memory_dir) if memory_dir else None
        self._deny_mode: Literal["stop", "replace"] = deny_mode
        # Maps task_key → MemoryManager.  task_key derived from the initial
        # user query so it is stable across ToolsExecutionLoop iterations.
        self._memories: dict[str, MemoryManager] = {}

    # ------------------------------------------------------------------
    # Per-task memory management
    # ------------------------------------------------------------------

    def _task_key(self, messages: Sequence[ChatMessage]) -> str:
        """Derive a stable key from benchmark metadata via the active TraceLogger."""
        logger = Logger.get()
        ctx = getattr(logger, "context", None)
        if ctx:
            pipeline_name = (ctx.get("pipeline_name") or "").replace("/", "_")
            suite_name = ctx.get("suite_name") or ""
            user_task_id = ctx.get("user_task_id") or ""
            attack_type = ctx.get("attack_type") or ""
            injection_task_id = ctx.get("injection_task_id") or ""

            parts = [pipeline_name, suite_name, user_task_id, attack_type, injection_task_id]
            key = "_".join(p for p in parts if p)
            if key:
                return key

        # Fallback: derive from first user message (for non-benchmark usage)
        for msg in messages:
            if msg["role"] == "user":
                return str(normalize_content_to_text(msg.get("content")))[:256]
        return "__default__"

    def _get_memory(self, task_key: str) -> MemoryManager:
        if task_key not in self._memories:
            path = None
            if self._memory_dir:
                safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in task_key)
                path = self._memory_dir / safe[:128]
            self._memories[task_key] = MemoryManager(memory_path=path)
        return self._memories[task_key]

    def _is_first_tool_call(self, messages: Sequence[ChatMessage]) -> bool:
        """True when no tool results exist yet → start of a new task."""
        return not any(m.get("role") == "tool" for m in messages)

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
        # Resolve the per-task memory manager
        # ------------------------------------------------------------------
        task_key = self._task_key(messages)
        memory_mgr = self._get_memory(task_key)
        if self._is_first_tool_call(messages):
            memory_mgr.reset()

        # ------------------------------------------------------------------
        # Convert messages → segment using memagent preprocessing
        # ------------------------------------------------------------------
        plain_msgs = _agentdojo_messages_to_plain(messages)
        segments = segment_single_trajectory(plain_msgs)
        if not segments:
            # Unexpected – no assistant-with-tool-calls found; pass through
            return query, runtime, env, messages, extra_args

        # Take the last segment (the current pending tool calls)
        segment = segments[-1]
        task_dict = {
            "context": segment.get("context", []),
            "pending_tool_calls": segment.get("tool_calls", []),
        }

        # ------------------------------------------------------------------
        # Inject the per-task MemoryManager into the env before running.
        # The env reads/writes through it directly (no string-level sync).
        # This mirrors how MemTarAgentWorkflow sets env.memory_manager.memory_path
        # before reset().
        # ------------------------------------------------------------------
        self._workflow.env.memory_manager = memory_mgr

        # ------------------------------------------------------------------
        # Run the rllm workflow (synchronously from the agentdojo pipeline)
        # ------------------------------------------------------------------
        uid = str(uuid4())
        try:
            episode = _run_async(
                self._workflow.run_with_termination_handling(task_dict, uid=uid)
            )
            # LiveShadowMemoryWorkflow always produces exactly 2 trajectories
            # (one for MemAgent, one for JudgeAgent), each with exactly 1 step.
            # Assert this invariant to prevent accidental token double-counting
            # if the workflow structure ever changes unexpectedly.
            assert episode is not None and len(episode.trajectories) == 2, (
                f"[ShadowMemory] expected 2 trajectories (mem + judge), "
                f"got {len(episode.trajectories) if episode else 'None'}"
            )
            for traj in episode.trajectories:
                assert len(traj.steps) == 1, (
                    f"[ShadowMemory] expected 1 step per trajectory '{traj.name}', "
                    f"got {len(traj.steps)}"
                )
            # Accumulate defense token usage from MemAgent + JudgeAgent LLM calls.
            # ModelOutput.prompt_length / completion_length come directly from
            # response.usage.prompt_tokens / completion_tokens in OpenAIEngine.
            _defense_input = 0
            _defense_output = 0
            for traj in episode.trajectories:
                mo = traj.steps[0].model_output
                if mo is not None:
                    _defense_input += getattr(mo, "prompt_length", 0) or 0
                    _defense_output += getattr(mo, "completion_length", 0) or 0
                    # print(f"[DEBUG] traj={traj!r} mo.prompt_length={getattr(mo, 'prompt_length', None)!r} mo.completion_length={getattr(mo, 'completion_length', None)!r}")
            if _defense_input or _defense_output:
                add_token_usage(
                    extra_args,
                    input_tokens=_defense_input,
                    output_tokens=_defense_output,
                    scope="defense",
                )
        except Exception as exc:
            # On any backend failure, fail safe: do NOT block the tool.
            print(f"[ShadowMemory] backend error, defaulting to APPROVE: {exc}")
            return query, runtime, env, messages, extra_args

        # ------------------------------------------------------------------
        # Read back results – memory_mgr is already up to date because the
        # env wrote through it directly during the memory_overwrite step.
        # ------------------------------------------------------------------
        decision = (self._workflow.env.decision or "APPROVE").upper()
        rationale = self._workflow.env.rationale or ""

        # ------------------------------------------------------------------
        # Act on the decision
        # ------------------------------------------------------------------
        extra_args.setdefault("shadowmemory_decisions", []).append(
            {
                "decision": decision,
                "rationale": rationale,
                "tool_calls": [_function_call_to_dict(tc) for tc in tool_calls],
            }
        )

        if decision == "DENY":
            if self._deny_mode == "stop":
                abort_msg = (
                    f"[BLOCKED] Prompt injection detected by shadowmemory security monitor. "
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
                    error=f"Tool execution blocked by shadowmemory security monitor. Rationale: {rationale}",)
                for tc in tool_calls
            ]
            stop_message = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(
                    "[BLOCKED] Prompt injection detected by shadowmemory security monitor. "
                    f"Rationale: {rationale}"
                )],
                tool_calls=None,
            )
            # Append fake tool results + assistant stop message.
            # The stop message (tool_calls=None) causes ToolsExecutor and
            # the LLM to pass through, and ToolsExecutionLoop breaks on
            # the next iteration.  Tokens from any subsequent LLM call
            # remain under the "original" (task) scope.
            return query, runtime, env, [*messages, *fake_results, stop_message], extra_args

        # APPROVE – let ToolsExecutor run as normal
        return query, runtime, env, messages, extra_args
