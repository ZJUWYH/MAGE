"""
LiveNaiveJudgeEnv: single-segment, inference-only env for the
live agentdojo integration of the naive judge (no memory).

Unlike LiveShadowMemoryEnv, this env has no memory management at all.
It handles exactly *one* segment per reset() call.

Expected task dict keys
-----------------------
context           : list[dict]  – cleaned conversation context
pending_tool_calls: list[dict]  – tool calls proposed by the target agent
"""

from typing import Any, Optional

from rllm.agents.agent import Action
from rllm.environments.base.base_env import BaseEnv


class LiveNaiveJudgeEnv(BaseEnv):
    """
    Single-segment env for live naive judge inference (no memory).

    Step actions
    ------------
    judge : records APPROVE/DENY decision, marks done=True
    """

    def __init__(self, task: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)

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
        """Reset segment state for a new segment."""
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

        if tool_name == "judge":
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
            "pending_tool_calls": self.pending_tool_calls,
        }

    @staticmethod
    def from_dict(info: dict) -> "LiveNaiveJudgeEnv":
        return LiveNaiveJudgeEnv(task=info)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
