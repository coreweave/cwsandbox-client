"""ART rollout function with Aviato sandbox execution.

This module implements the core rollout function that:
1. Uses OpenAI Responses API with tool calling
2. Executes tools (execute_code, submit_solution) in Aviato sandboxes
3. Builds an ART Trajectory with the conversation history
4. Computes binary reward (1.0 if tests pass, 0.0 otherwise)

The rollout loop allows max 20 tool calls per problem, with errors
returned as tool responses so the agent can observe and retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import Function
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.responses import Response, ResponseFunctionToolCall

import art

from .tools import (
    EXECUTE_CODE_NAME,
    ROLLOUT_TOOLS,
    SUBMIT_SOLUTION_NAME,
    ExecuteCodeArgs,
    SubmitSolutionArgs,
    to_responses_format,
)

if TYPE_CHECKING:
    from aviato import Sandbox

MAX_TOOL_CALLS = 20
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

CRITICAL RULES:
- You MUST use tools to write and test code. NEVER output code as plain text.
- You MUST call submit_solution before finishing. Every problem requires a submission.
- Do NOT ask clarifying questions. Make reasonable assumptions and solve the problem.
- If a term is unfamiliar, implement your best interpretation.

Workflow:
1. Use execute_code to develop and test your solution
2. Once tests pass, call submit_solution with your final code
3. You have up to {MAX_TOOL_CALLS} tool calls total

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


def _normalize_output_for_input(response: Response) -> list[dict[str, Any]]:
    """Convert response output items to valid input items for re-feeding.

    The Responses API output items need minor transformation to be valid input:
    - ResponseOutputMessage: extract content and convert to message format
    - ResponseFunctionToolCall: pass through (already valid as input)
    - ResponseReasoningItem: include for reasoning models (required before function_call)
    - Other items (refusal, error): convert to message format

    Args:
        response: The API response object

    Returns:
        List of input items ready to append to conversation
    """
    input_items: list[dict[str, Any]] = []

    for item in response.output:
        if item.type == "message":
            content_parts = []
            for part in item.content:
                if part.type == "output_text":
                    content_parts.append({"type": "text", "text": part.text})
                elif part.type == "refusal":
                    content_parts.append({"type": "text", "text": f"[Refusal: {part.refusal}]"})
            if content_parts:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": content_parts,
                })
        elif item.type == "reasoning":
            input_items.append({
                "type": "reasoning",
                "id": item.id,
                "summary": getattr(item, "summary", None),
            })
        elif item.type == "function_call":
            input_items.append({
                "type": "function_call",
                "id": item.id,
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            })

    return input_items


def _response_output_to_choice(response: Response) -> Choice:
    """Convert Responses API output to a Chat Completions Choice object for ART.

    ART expects Choice objects with ChatCompletionMessage for trajectory building.
    This function constructs the equivalent Choice from Responses API output.

    Args:
        response: The API response object

    Returns:
        Choice object compatible with ART Trajectory
    """
    text_content = response.output_text or ""

    tool_calls: list[ChatCompletionMessageToolCall] = []
    for item in response.output:
        if item.type == "function_call":
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=item.call_id,
                    type="function",
                    function=Function(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                )
            )

    message = ChatCompletionMessage(
        role="assistant",
        content=text_content if text_content else None,
        tool_calls=cast(Any, tool_calls) if tool_calls else None,
    )

    finish_reason: Literal["stop", "tool_calls"] = "tool_calls" if tool_calls else "stop"

    return Choice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )


async def handle_tool_call(
    tool_call: ResponseFunctionToolCall,
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
    name = tool_call.name
    arguments = tool_call.arguments
    if not isinstance(arguments, str):
        return "Error: Tool arguments must be a string", False, False
    try:
        args = json.loads(arguments)
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

    input_items: list[dict[str, Any]] = [
        {"type": "message", "role": "system", "content": build_system_message(problem)},
        {"type": "message", "role": "user", "content": "Please solve this problem."},
    ]

    messages_and_choices: list[dict[str, Any] | Choice] = [
        {"role": "system", "content": build_system_message(problem)},
        {"role": "user", "content": "Please solve this problem."},
    ]
    tool_call_count = 0
    reward = 0.0
    submitted = False

    while tool_call_count < config.max_attempts and not submitted:
        response = await client.responses.create(
            model=config.model,
            input=cast(Any, input_items),
            tools=to_responses_format(ROLLOUT_TOOLS),
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        if response.status != "completed":
            error_msg = response.error.message if response.error else "Unknown error"
            raise RuntimeError(f"Response not completed: {response.status} - {error_msg}")

        choice = _response_output_to_choice(response)
        messages_and_choices.append(choice)

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if function_calls:
            normalized_items = _normalize_output_for_input(response)
            input_items.extend(normalized_items)

            for tool_call in function_calls:
                tool_call_count += 1
                result_content, is_submission, passed = await handle_tool_call(
                    tool_call, sandbox, problem, config
                )

                tool_output_item: dict[str, Any] = {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result_content,
                }
                input_items.append(tool_output_item)

                tool_response_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tool_call.call_id,
                    "content": result_content,
                }
                messages_and_choices.append(tool_response_msg)

                if is_submission:
                    submitted = True
                    reward = 1.0 if passed else 0.0
                    break

                if tool_call_count >= config.max_attempts:
                    break
        else:
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
