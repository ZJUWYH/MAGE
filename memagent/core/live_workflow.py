"""
LiveShadowMemoryWorkflow: a single-cycle (one segment) rllm workflow
for live agentdojo inference.

Runs exactly one MemAgent step followed by one JudgeAgent step, then
terminates with ENV_DONE.  Memory is not owned here – it is passed in
via the task dict and read back from env.memory after the workflow
completes.

Usage (from ShadowMemory pipeline element)
------------------------------------------
    engine   = OpenAIEngine(model=..., base_url=..., api_key=...)
    workflow = LiveShadowMemoryWorkflow(rollout_engine=engine, executor=executor)
    task     = {"context": [...], "pending_tool_calls": [...], "memory": "..."}
    episode  = asyncio.run(workflow.run_with_termination_handling(task, uid="..."))
    decision = workflow.env.decision   # "APPROVE" or "DENY"
    rationale= workflow.env.rationale
    memory   = workflow.env.memory     # updated memory after MemAgent ran
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rllm.agents.agent import Episode, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.workflows.timing_mixin import TimingTrackingMixin
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

from core.agent import JudgeAgent, MemAgent
from core.live_env import LiveShadowMemoryEnv


class LiveShadowMemoryWorkflow(TimingTrackingMixin, Workflow):
    """
    Single-segment shadowmemory workflow.

    Runs one MemAgent step (memory_overwrite) then one JudgeAgent step
    (judge) on a single provided segment and terminates.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine = None,
        executor: ThreadPoolExecutor = None,
        agent_args: dict = None,
        judge_agent_args: dict = None,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.agent = MemAgent(**(agent_args or {}))
        self.judge_agent = JudgeAgent(**(judge_agent_args or {}))
        self.env = LiveShadowMemoryEnv()

    # ------------------------------------------------------------------
    # Workflow interface
    # ------------------------------------------------------------------

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Run one mem-judge cycle on the provided single segment."""
        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0, False, info)

        # ----------------------------------------------------------
        # Step 1: MemAgent analyses context and updates memory
        # ----------------------------------------------------------
        mem_output: ModelOutput | None = None
        for _attempt in range(3):
            mem_output = await self.timed_llm_call(
                self.agent.chat_completions,
                application_id=uid,
                **kwargs,
            )
            if self.agent.can_parse(mem_output.text):
                break

        mem_action = self.agent.update_from_model(
            mem_output.text, model_output=mem_output
        )
        observation, _, _, info = await self.timed_env_call(
            self.env.step, mem_action
        )
        self.agent.update_from_env(
            observation, 0, False, info, phase="update step reward"
        )

        # ----------------------------------------------------------
        # Step 2: JudgeAgent decides APPROVE / DENY
        # ----------------------------------------------------------
        self.judge_agent.update_from_env(
            observation, 0, False, info, phase="update messages"
        )

        judge_output: ModelOutput | None = None
        for _attempt in range(3):
            judge_output = await self.timed_llm_call(
                self.judge_agent.chat_completions,
                application_id=uid,
                **kwargs,
            )
            if self.judge_agent.can_parse(judge_output.text):
                break

        judge_action = self.judge_agent.update_from_model(
            judge_output.text, model_output=judge_output
        )
        observation, _, done, info = await self.timed_env_call(
            self.env.step, judge_action
        )
        self.judge_agent.update_from_env(
            observation, 0, done, info, phase="update step reward"
        )

        # Always terminates after a single judge step
        raise TerminationEvent(TerminationReason.ENV_DONE)

    def reset(
        self, task: dict | None = None, uid: str | None = None
    ) -> tuple[Any, dict]:
        self.start_timing()

        self.uid = uid
        self.task = task
        self._completed_trajectories = []

        self.agent.reset()
        self.agent.trajectory.task = task
        self.agent.trajectory.name = "mem"

        self.judge_agent.reset()
        self.judge_agent.trajectory.task = task
        self.judge_agent.trajectory.name = "judge"

        return self.env.reset(task)

    def compute_trajectory_reward(self, trajectory: Trajectory) -> None:
        trajectory.reward = sum(step.reward for step in trajectory.steps)
