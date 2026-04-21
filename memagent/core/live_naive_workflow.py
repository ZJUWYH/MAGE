"""
LiveNaiveJudgeWorkflow: a single-cycle (one segment) rllm workflow
for live agentdojo inference of the naive judge (no memory).

Runs exactly one JudgeAgentNaive step, then terminates with ENV_DONE.
No memory agent is involved — this is the ablation baseline.

Usage (from NaiveJudge pipeline element)
----------------------------------------
    engine   = OpenAIEngine(model=..., base_url=..., api_key=...)
    workflow  = LiveNaiveJudgeWorkflow(rollout_engine=engine, executor=executor)
    task      = {"context": [...], "pending_tool_calls": [...]}
    episode   = asyncio.run(workflow.run_with_termination_handling(task, uid="..."))
    decision  = workflow.env.decision   # "APPROVE" or "DENY"
    rationale = workflow.env.rationale
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rllm.agents.agent import Episode, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.workflows.timing_mixin import TimingTrackingMixin
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

from core.agent import JudgeAgentNaive
from core.live_naive_env import LiveNaiveJudgeEnv


class LiveNaiveJudgeWorkflow(TimingTrackingMixin, Workflow):
    """
    Single-segment naive judge workflow (no memory).

    Runs one JudgeAgentNaive step (judge decision) on a single provided
    segment and terminates.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine = None,
        executor: ThreadPoolExecutor = None,
        judge_agent_args: dict = None,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.judge_agent = JudgeAgentNaive(**(judge_agent_args or {}))
        self.env = LiveNaiveJudgeEnv()

    # ------------------------------------------------------------------
    # Workflow interface
    # ------------------------------------------------------------------

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Run one judge cycle on the provided single segment."""
        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)
        self.judge_agent.update_from_env(observation, 0, False, info, phase="update messages")

        # ----------------------------------------------------------
        # Single step: JudgeAgentNaive decides APPROVE / DENY
        # ----------------------------------------------------------
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

        self.judge_agent.reset()
        self.judge_agent.trajectory.task = task
        self.judge_agent.trajectory.name = "judge"

        return self.env.reset(task)

    def compute_trajectory_reward(self, trajectory: Trajectory) -> None:
        trajectory.reward = sum(step.reward for step in trajectory.steps)
