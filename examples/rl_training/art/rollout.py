"""ART rollout function with Aviato sandbox execution.

This module implements the core rollout function that:
1. Uses OpenAI-compatible API with tool calling
2. Executes tools (execute_code, submit_solution) in Aviato sandboxes
3. Builds an ART Trajectory with the conversation history
4. Computes binary reward (1.0 if tests pass, 0.0 otherwise)

The rollout loop allows max 3 attempts per problem, with errors
returned as tool responses so the agent can observe and retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

import art

from .tools import (
    EXECUTE_CODE_NAME,
    ROLLOUT_TOOLS,
    SUBMIT_SOLUTION_NAME,
    ExecuteCodeArgs,
    SubmitSolutionArgs,
)

if TYPE_CHECKING:
    from aviato import Sandbox

MAX_TOOL_CALLS = 3
EXECUTION_TIMEOUT_SECONDS = 30.0


@dataclass
class Problem:
    """A coding problem for the agent to solve.

    Attributes:
        task_id: Unique identifier for the problem
        prompt: Problem description/instructions
        test_code: Code that tests the solution (runs after solution code)
        test_imports: Optional imports needed for test code
    """

    task_id: str
    prompt: str
    test_code: str
    test_imports: str = ""


@dataclass
class RolloutConfig:
    """Configuration for rollout execution.

    Attributes:
        model: Model identifier (e.g., "gpt-4o", "gpt-4o-mini")
        base_url: OpenAI-compatible API base URL
        api_key: API key for authentication
        max_attempts: Maximum tool calls before giving up
        execution_timeout: Timeout for code execution in seconds
    """

    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    max_attempts: int = MAX_TOOL_CALLS
    execution_timeout: float = EXECUTION_TIMEOUT_SECONDS


def build_system_message(problem: Problem) -> str:
    """Build the system message with problem context."""
    return f"""You are a Python programming assistant. Solve the given coding problem.

You have access to two tools:
1. execute_code: Test your code in an isolated sandbox. Use this to debug.
2. submit_solution: Submit your final answer. This runs all test cases.

Guidelines:
- Read the problem carefully before writing code
- Test your solution with execute_code before submitting
- Handle edge cases (empty inputs, etc.)
- You can make up to {MAX_TOOL_CALLS} tool calls total
- Only submit when confident your solution is correct

Problem:
{problem.prompt}
"""


async def execute_code_in_sandbox(
    sandbox: Sandbox,
    code: str,
    timeout: float = EXECUTION_TIMEOUT_SECONDS,
) -> str:
    """Execute code in sandbox and return output or error message.

    Args:
        sandbox: Aviato sandbox instance
        code: Python code to execute
        timeout: Execution timeout in seconds

    Returns:
        Output string: stdout on success, error message on failure
    """
    try:
        process = sandbox.exec(
            ["python", "-c", code],
            timeout_seconds=timeout,
        )
        result = process.result()

        if result.returncode == 0:
            output = result.stdout.strip() if result.stdout else "(no output)"
            return output
        else:
            error = result.stderr.strip() if result.stderr else "(unknown error)"
            return f"Error (exit code {result.returncode}):\n{error}"

    except TimeoutError:
        return f"Error: Execution timed out after {timeout} seconds"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


async def run_tests_in_sandbox(
    sandbox: Sandbox,
    solution_code: str,
    test_code: str,
    test_imports: str = "",
    timeout: float = EXECUTION_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Run solution against test cases in sandbox.

    Args:
        sandbox: Aviato sandbox instance
        solution_code: The solution code to test
        test_code: Test code that validates the solution
        test_imports: Optional imports for test code
        timeout: Execution timeout in seconds

    Returns:
        Tuple of (passed: bool, output: str)
    """
    full_code = f"{test_imports}\n\n{solution_code}\n\n{test_code}"

    try:
        process = sandbox.exec(
            ["python", "-c", full_code],
            timeout_seconds=timeout,
        )
        result = process.result()

        if result.returncode == 0:
            output = result.stdout.strip() if result.stdout else "All tests passed"
            return True, output
        else:
            error = result.stderr.strip() if result.stderr else "(unknown error)"
            return False, f"Tests failed (exit code {result.returncode}):\n{error}"

    except TimeoutError:
        return False, f"Error: Tests timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"


async def handle_tool_call(
    tool_call: ChatCompletionMessageToolCall,
    sandbox: Sandbox,
    problem: Problem,
    config: RolloutConfig,
) -> tuple[str, bool, bool]:
    """Handle a single tool call.

    Args:
        tool_call: The tool call to handle
        sandbox: Aviato sandbox instance
        problem: The coding problem
        config: Rollout configuration

    Returns:
        Tuple of (result_content, is_submission, passed)
    """
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON arguments: {e}", False, False

    if name == EXECUTE_CODE_NAME:
        typed_args: ExecuteCodeArgs = args
        result = await execute_code_in_sandbox(
            sandbox, typed_args["code"], config.execution_timeout
        )
        return result, False, False

    elif name == SUBMIT_SOLUTION_NAME:
        typed_args_submit: SubmitSolutionArgs = args
        passed, result = await run_tests_in_sandbox(
            sandbox,
            typed_args_submit["code"],
            problem.test_code,
            problem.test_imports,
            config.execution_timeout,
        )
        return result, True, passed

    else:
        return f"Error: Unknown tool '{name}'", False, False


async def rollout(
    problem: Problem,
    sandbox: Sandbox,
    config: RolloutConfig | None = None,
) -> art.Trajectory:
    """Execute a single rollout for a coding problem.

    This function runs an agent loop where the model can use tools
    to explore and test its solution before submitting.

    Args:
        problem: The coding problem to solve
        sandbox: Aviato sandbox instance (must be running)
        config: Optional rollout configuration

    Returns:
        ART Trajectory with conversation history and reward
    """
    if config is None:
        config = RolloutConfig()

    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": build_system_message(problem)},
        {"role": "user", "content": "Please solve this problem."},
    ]

    messages_and_choices: list[ChatCompletionMessageParam | Choice] = list(messages)
    tool_call_count = 0
    reward = 0.0
    submitted = False

    while tool_call_count < config.max_attempts and not submitted:
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            tools=ROLLOUT_TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        messages_and_choices.append(choice)

        assistant_message = choice.message
        if assistant_message.tool_calls:
            assistant_msg: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [tc.model_dump() for tc in assistant_message.tool_calls],
            }
            messages.append(assistant_msg)

            for tool_call in assistant_message.tool_calls:
                tool_call_count += 1
                result_content, is_submission, passed = await handle_tool_call(
                    tool_call, sandbox, problem, config
                )

                tool_response: ChatCompletionMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content,
                }
                messages.append(tool_response)
                messages_and_choices.append(tool_response)

                if is_submission:
                    submitted = True
                    reward = 1.0 if passed else 0.0
                    break

                if tool_call_count >= config.max_attempts:
                    break
        else:
            if assistant_message.content:
                messages.append({"role": "assistant", "content": assistant_message.content})
            break

    trajectory = art.Trajectory(
        messages_and_choices=messages_and_choices,
        tools=ROLLOUT_TOOLS,
        reward=reward,
        metadata={
            "task_id": problem.task_id,
            "tool_calls": tool_call_count,
            "submitted": submitted,
        },
    )

    return trajectory.finish()
