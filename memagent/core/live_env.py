"""
LiveShadowMemoryEnv: single-segment, inference-only env for the
live agentdojo integration.

Unlike MemTarAgentEnv (which iterates over a full pre-recorded trajectory),
this env handles exactly *one* segment per reset() call.

Memory ownership follows the same pattern as MemTarAgentEnv: the env owns a
MemoryManager instance.  The caller (ShadowMemory pipeline element) swaps in
the right per-task MemoryManager *before* calling reset(), so that memory
persists correctly across ToolsExecutionLoop iterations without any manual
string-level sync.

Expected task dict keys
-----------------------
context           : list[dict]  – cleaned conversation context
pending_tool_calls: list[dict]  – tool calls proposed by the target agent

(No "memory" key – the env reads/writes through self.memory_manager directly.)
"""

from pathlib import Path
from typing import Any, Optional

from rllm.agents.agent import Action
from rllm.environments.base.base_env import BaseEnv

from core.memory import MemoryManager


class LiveShadowMemoryEnv(BaseEnv):
    """
    Single-segment env for live shadowmemory inference.

    Step actions
    ------------
    memory_overwrite  : updates memory_manager, keeps same segment (done=False)
    judge             : records APPROVE/DENY decision, marks done=True
    """

    def __init__(self, memory_path: Optional[Path] = None, task: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        # Owns memory exactly like MemTarAgentEnv.
        # Caller may replace self.memory_manager before reset() to inject a
        # different per-task manager without constructing a new env instance.
        self.memory_manager = MemoryManager(memory_path=memory_path)

        self.context: list[dict] = []
        self.pending_tool_calls: list[dict] = []
        self.decision: Optional[str] = None
        self.rationale: Optional[str] = None

        if task is not None:
            self._load_task(task)

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self, task: Optional[dict] = None) -> tuple[dict, dict]:
        """Reset segment state.  Does NOT reset the memory_manager so that
        memory accumulated across prior segments (within the same agentdojo
        task) is preserved.  The caller resets memory_manager when a new
        agentdojo task begins."""
        if task is not None:
            self._load_task(task)

        self.decision = None
        self.rationale = None

        observation = self._build_observation()
        info = {"total_segments": 1, "segment_idx": 0}
        return observation, info

    def step(self, action: Action) -> tuple[dict, float, bool, dict]:
        tool_call = action.action if isinstance(action, Action) else action
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        info: dict[str, Any] = {}
        reward = 0.0

        if tool_name == "memory_overwrite":
            self.memory_manager.overwrite(str(args.get("memory", "")))
            done = False
            info["action_type"] = "memory_overwrite"
            info["memory_content"] = self.memory_manager.get()

        elif tool_name == "judge":
            self.decision = str(args.get("decision", "DENY")).upper()
            self.rationale = str(args.get("rationale", ""))
            done = True
            info["action_type"] = "judge"
            info["decision"] = self.decision
            info["rationale"] = self.rationale

        else:
            done = False
            info["action_type"] = "unknown"
            info["unknown_tool_name"] = tool_name

        observation = self._build_observation()
        return observation, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_task(self, task: dict) -> None:
        self.context = task.get("context", [])
        self.pending_tool_calls = task.get("pending_tool_calls", [])

    def _build_observation(self) -> dict:
        return {
            "context": self.context,
            "memory": self.memory_manager.get(),
            "pending_tool_calls": self.pending_tool_calls,
        }

    @staticmethod
    def from_dict(info: dict) -> "LiveShadowMemoryEnv":
        return LiveShadowMemoryEnv(task=info)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
