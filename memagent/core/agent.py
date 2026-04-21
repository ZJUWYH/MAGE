import copy
import json
from typing import Any, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.parser import QwenToolParser, get_tool_parser
from core.prompt import MEMORY_SYSTEM_PROMPT_TOOLS, JUDGE_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT_NAIVE, format_context, format_memory, format_action


# ============================================
# Tool Definitions (MCP-style)
# ============================================

MEMORY_OVERWRITE_TOOL = {
    "type": "function",
    "name": "memory_overwrite",
    "description": "Overwrite the current memory with new security analysis findings.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory": {
                "type": "string",
                "description": "Complete memory content to store"
            }
        },
        "required": ["memory"]
    }
}

# Tool schema as JSON string for parser
MEMORY_OVERWRITE_TOOL_SCHEMA = json.dumps([MEMORY_OVERWRITE_TOOL], indent=2)

JUDGE_TOOL = {
    "type": "function",
    "name": "judge",
    "description": "Make a security decision on the proposed action.",
    "parameters": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["APPROVE", "DENY"],
                "description": "Security decision: APPROVE if action is safe, DENY if suspicious"
            },
            "rationale": {
                "type": "string",
                "description": "Brief explanation for the decision"
            }
        },
        "required": ["decision", "rationale"]
    }
}

# Tool schema as JSON string for parser
JUDGE_TOOL_SCHEMA = json.dumps([JUDGE_TOOL], indent=2)


class MemAgent(BaseAgent):
    """
    An agent that only has access to the memory_overwrite tool.
    Adapted from rllm's BaseAgent pattern.
    
    This agent maintains minimal state: system prompt + last observation only
    (no history accumulation, similar to FrozenLakeAgent with accumulate_history=False).
    """
    
    def __init__(
        self, 
        system_prompt: str = MEMORY_SYSTEM_PROMPT_TOOLS,
        parser_name: str = "qwen"
    ):
        """Initialize the MemAgent.
        
        Args:
            system_prompt: The system prompt for the agent.
            parser_name: Name of the tool parser to use ('qwen' or 'r1').
        """
        self.base_system_prompt = system_prompt
        
        # Initialize tool parser
        parser_class = get_tool_parser(parser_name)
        self.tool_parser = parser_class()
        
        # Build system prompt with tool definition
        tool_prompt = self.tool_parser.get_tool_prompt(MEMORY_OVERWRITE_TOOL_SCHEMA)
        self.system_prompt = system_prompt + "\n" + tool_prompt
        
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation: Any = None
        self.step_count: int = 0
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, phase: str = "all", **kwargs):
        """Update agent state after environment step.
        
        Args:
            observation: Dict with "context", "memory", "pending_tool_calls".
            reward: Reward from the environment step.
            done: Whether the episode is done.
            info: Info dict from the environment step.
            phase: Controls what gets updated:
                - "update step reward": only update the last step's reward/done/info.
                - "update messages": only append a new user message from observation.
                - "all": do both (default, preserves backwards compatibility).
        """
        if phase in ("update step reward", "all"):
            if self._trajectory.steps:
                cur_step = self._trajectory.steps[-1]
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info
        
        if phase in ("update messages", "all"):
            context = observation.get("context", [])
            memory = observation.get("memory", "")
            pending_tool_calls = observation.get("pending_tool_calls", [])
            
            user_content = f"""## Current Memory
{format_memory(memory)}

## Conversation Context
{format_context(context)}

## Pending Tool Calls (to analyze)
{format_action(pending_tool_calls) if pending_tool_calls else "No pending tool calls."}

Analyze the context and pending tool calls. Record your security analysis findings using the memory_overwrite tool."""

            self.messages.append({"role": "user", "content": user_content})
            self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Update agent state after model response.
        
        Parses the memory_overwrite tool call from the response and creates a Step.
        
        Args:
            response: The model's response text
            **kwargs: Additional arguments, including model_output for RL training
            
        Returns:
            Action containing the parsed tool call
        """
        model_output: Optional[ModelOutput] = kwargs.get("model_output")
        
        # Parse tool call from response
        tool_call = self._parse_memory_tool(response)
        
        # Add assistant message to history BEFORE creating Step,
        # so chat_completions snapshot includes it (matches FrozenLakeAgent pattern).
        self.messages.append({"role": "assistant", "content": response})

        # Create Step WITH model_output for RL training
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_call,
            model_response=response,
            observation=self.current_observation,
            model_output=model_output,  # Critical for RL!
        )
        self._trajectory.steps.append(new_step)
        self.step_count += 1

        return Action(action=tool_call)

    def _parse_memory_tool(self, response: str) -> dict:
        """Parse memory_overwrite tool call from model response using rllm's ToolParser.
        
        Uses QwenToolParser or R1ToolParser to extract tool calls in the standard format:
        <tool_call>{"name": "memory_overwrite", "arguments": {"memory": "..."}}</tool_call>
        
        Args:
            response: The model's response text
            
        Returns:
            Dict with "name" and "args" keys
        """
        try:
            # Use rllm's tool parser
            tool_calls = self.tool_parser.parse(response)
            
            if tool_calls:
                # Get the first tool call (should be memory_overwrite)
                tool_call = tool_calls[0]
                if tool_call.name.lower() == "memory_overwrite":
                    return {
                        "name": tool_call.name.lower(),
                        "args": tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                    }
        except Exception as e:
            # Log parsing error but don't fail
            print(f"Warning: Failed to parse tool call: {e}")
        
        # Fallback: use entire response as memory content
        response = response.replace("<tool_call>", "").replace("</tool_call>", "")
        return {"name": "memory_overwrite", "args": {"memory": response.strip()}, "_parse_failed": True}

    def can_parse(self, response: str) -> bool:
        """Check if the response contains a parseable tool call."""
        try:
            tool_calls = self.tool_parser.parse(response)
            return bool(tool_calls)
        except Exception:
            return False

    def reset(self):
        """Reset the agent's internal state."""
        self._trajectory = Trajectory()
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.step_count = 0
        self.current_observation = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return messages for model completion.

        During inference (last msg is user): returns [system, user].
        After update_from_model (last msg is assistant): returns [system, user, assistant]
        so the Step snapshot has the correct input/output split for RL training.
        """
        if len(self.messages) <= 1:
            return self.messages
        if self.messages[-1]["role"] == "assistant" and len(self.messages) >= 3:
            return [self.messages[0], self.messages[-2], self.messages[-1]]
        else:
            return [self.messages[0], self.messages[-1]]

    @property
    def trajectory(self) -> Trajectory:
        """Return the agent's trajectory."""
        return self._trajectory


class JudgeAgent(BaseAgent):
    """
    A judge agent that decides whether an action is safe based on memory and proposed action.
    
    The judge agent has access to the judge tool: judge(decision="APPROVE"/"DENY", rationale="...")
    It works together with the MemAgent in the workflow.
    
    Key difference from MemAgent: JudgeAgent only uses memory + pending_tool_calls from the
    observation, ignoring the context. This ensures the judge makes decisions based on the
    security analysis (memory) rather than raw conversation history.
    """
    
    def __init__(
        self, 
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
        parser_name: str = "qwen"
    ):
        """Initialize the JudgeAgent.
        
        Args:
            system_prompt: The system prompt for the agent.
            parser_name: Name of the tool parser to use ('qwen' or 'r1').
        """
        self.base_system_prompt = system_prompt
        
        # Initialize tool parser
        parser_class = get_tool_parser(parser_name)
        self.tool_parser = parser_class()
        
        # Build system prompt with tool definition
        tool_prompt = self.tool_parser.get_tool_prompt(JUDGE_TOOL_SCHEMA)
        self.system_prompt = system_prompt + "\n" + tool_prompt
        
        self._trajectory = Trajectory(name="judge")
        self.messages: list[dict[str, Any]] = []
        self.current_observation: Any = None
        self.step_count: int = 0
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, phase: str, **kwargs):
        """Update agent state after environment step.
        
        The observation is a dictionary with keys:
        - "context": conversation context (IGNORED by JudgeAgent)
        - "memory": the latest memory content from MemAgent
        - "pending_tool_calls": the proposed action to judge
        
        JudgeAgent only uses memory + pending_tool_calls, not context.
        """
        if phase == "update messages":
            # Extract only what JudgeAgent needs (ignore context)
            memory = observation.get("memory", "")
            pending_tool_calls = observation.get("pending_tool_calls", [])
            
            user_content = f"""## Security Analysis (Memory)
    {format_memory(memory)}

    ## Proposed Action
    {format_action(pending_tool_calls) if pending_tool_calls else "No pending action."}

    Based on the security analysis above, make your APPROVE or DENY decision using the judge tool."""

            self.messages.append({"role": "user", "content": user_content})
            self.current_observation = observation
                # Update reward/done for the last step if exists
        elif phase == "update step reward":
            if self._trajectory.steps:
                cur_step = self._trajectory.steps[-1]
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Update agent state after model response.
        
        Parses the judge tool call from the response and creates a Step.
        
        Args:
            response: The model's response text
            **kwargs: Additional arguments, including model_output for RL training
            
        Returns:
            Action containing the parsed tool call
        """
        model_output: Optional[ModelOutput] = kwargs.get("model_output")
        
        # Parse tool call from response
        tool_call = self._parse_judge_tool(response)
        
        # Add assistant message to history BEFORE creating Step,
        # so chat_completions snapshot includes it (matches FrozenLakeAgent pattern).
        self.messages.append({"role": "assistant", "content": response})

        # Create Step WITH model_output for RL training
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_call,
            model_response=response,
            observation=self.current_observation,
            model_output=model_output,  # Critical for RL!
        )
        self._trajectory.steps.append(new_step)
        self.step_count += 1

        return Action(action=tool_call)

    def _parse_judge_tool(self, response: str) -> dict:
        """Parse judge tool call from model response using rllm's ToolParser.
        
        Uses QwenToolParser or R1ToolParser to extract tool calls in the standard format:
        <tool_call>{"name": "judge", "arguments": {"decision": "...", "rationale": "..."}}</tool_call>
        
        Args:
            response: The model's response text
            
        Returns:
            Dict with "name" and "args" keys
        """
        try:
            # Use rllm's tool parser
            tool_calls = self.tool_parser.parse(response)
            
            if tool_calls:
                # Get the first tool call (should be judge)
                tool_call = tool_calls[0]
                if tool_call.name.lower() == "judge":
                    return {
                        "name": tool_call.name.lower(),
                        "args": tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                    }
        except Exception as e:
            # Log parsing error but don't fail
            print(f"Warning: Failed to parse judge tool call: {e}")
        
        # Fallback: try to extract decision from response text
        decision = "APPROVE"  # Default to APPROVE
        rationale = response.strip()
        
        # Simple pattern matching for APPROVE/DENY
        response_upper = response.upper()
        if "APPROVE" in response_upper:
            decision = "APPROVE"
        elif "DENY" in response_upper:
            decision = "DENY"
        
        return {"name": "judge", "args": {"decision": decision, "rationale": rationale}, "_parse_failed": True}

    def can_parse(self, response: str) -> bool:
        """Check if the response contains a parseable tool call."""
        try:
            tool_calls = self.tool_parser.parse(response)
            return bool(tool_calls)
        except Exception:
            return False

    def reset(self):
        """Reset the agent's internal state."""
        self._trajectory = Trajectory(name="judge")
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.step_count = 0
        self.current_observation = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return messages for model completion.

        During inference (last msg is user): returns [system, user].
        After update_from_model (last msg is assistant): returns [system, user, assistant]
        so the Step snapshot has the correct input/output split for RL training.
        """
        if len(self.messages) <= 1:
            return self.messages
        if self.messages[-1]["role"] == "assistant" and len(self.messages) >= 3:
            return [self.messages[0], self.messages[-2], self.messages[-1]]
        else:
            return [self.messages[0], self.messages[-1]]

    @property
    def trajectory(self) -> Trajectory:
        """Return the agent's trajectory."""
        return self._trajectory




class JudgeAgentNaive(BaseAgent):
    """
    A naive judge agent that decides whether an action is safe based only on
    the conversation context and proposed action (no memory).
    """

    def __init__(
        self, 
        system_prompt: str = JUDGE_SYSTEM_PROMPT_NAIVE,
        parser_name: str = "qwen"
    ):
        """Initialize the JudgeAgentNaive."""
        self.base_system_prompt = system_prompt
        
        # Initialize tool parser
        parser_class = get_tool_parser(parser_name)
        self.tool_parser = parser_class()
        
        # Build system prompt with tool definition
        tool_prompt = self.tool_parser.get_tool_prompt(JUDGE_TOOL_SCHEMA)
        self.system_prompt = system_prompt + "\n" + tool_prompt
        
        self._trajectory = Trajectory(name="judge")
        self.messages: list[dict[str, Any]] = []
        self.current_observation: Any = None
        self.step_count: int = 0
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, phase: str = "all", **kwargs):
        """Update agent state. Uses context + pending_tool_calls only (no memory)."""
        if phase in ("update step reward", "all"):
            if self._trajectory.steps:
                cur_step = self._trajectory.steps[-1]
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info

        if phase in ("update messages", "all"):
            context = observation.get("context", [])
            pending_tool_calls = observation.get("pending_tool_calls", [])

            user_content = f"""## Conversation Context
{format_context(context)}

## Proposed Action
{format_action(pending_tool_calls) if pending_tool_calls else "No pending action."}

Based on the context and proposed action above, make your APPROVE or DENY decision using the judge tool."""

            self.messages.append({"role": "user", "content": user_content})
            self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Update agent state after model response."""
        model_output: Optional[ModelOutput] = kwargs.get("model_output")
        
        tool_call = self._parse_judge_tool(response)
        
        # Add assistant message to history
        self.messages.append({"role": "assistant", "content": response})
        
        # Create Step WITH model_output for RL training
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_call,
            model_response=response,
            observation=self.current_observation,
            model_output=model_output,  # Critical for RL!
        )
        self._trajectory.steps.append(new_step)
        self.step_count += 1
        
        return Action(action=tool_call)

    def _parse_judge_tool(self, response: str) -> dict:
        """Parse judge tool call from model response using rllm's ToolParser.
        
        Uses QwenToolParser or R1ToolParser to extract tool calls in the standard format:
        <tool_call>{"name": "judge", "arguments": {"decision": "...", "rationale": "..."}}</tool_call>
        
        Args:
            response: The model's response text
            
        Returns:
            Dict with "name" and "args" keys
        """
        try:
            # Use rllm's tool parser
            tool_calls = self.tool_parser.parse(response)
            
            if tool_calls:
                # Get the first tool call (should be judge)
                tool_call = tool_calls[0]
                if tool_call.name.lower() == "judge":
                    return {
                        "name": tool_call.name.lower(),
                        "args": tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                    }
        except Exception as e:
            # Log parsing error but don't fail
            print(f"Warning: Failed to parse judge tool call: {e}")
        
        # Fallback: try to extract decision from response text
        decision = "DENY"  # Default to DENY for safety
        rationale = response.strip()
        
        # Simple pattern matching for APPROVE/DENY
        response_upper = response.upper()
        if "APPROVE" in response_upper:
            decision = "APPROVE"
        elif "DENY" in response_upper:
            decision = "DENY"
        
        return {"name": "judge", "args": {"decision": decision, "rationale": rationale}, "_parse_failed": True}

    def can_parse(self, response: str) -> bool:
        """Check if the response contains a parseable tool call."""
        try:
            tool_calls = self.tool_parser.parse(response)
            return bool(tool_calls)
        except Exception:
            return False

    def reset(self):
        """Reset the agent's internal state."""
        self._trajectory = Trajectory(name="judge")
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.step_count = 0
        self.current_observation = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return all messages for inference — naive agent sees full history."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Return the agent's trajectory."""
        return self._trajectory