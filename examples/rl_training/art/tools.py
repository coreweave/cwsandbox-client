"""Tool definitions for ART rollout with Aviato sandboxes.

This module defines the tool schemas for the multi-step rollout:
- execute_code: Run Python code in sandbox, return stdout/stderr
- submit_solution: Final submission, runs all test cases
"""

from __future__ import annotations

from typing import Any, TypedDict

from openai.types.chat import ChatCompletionToolParam
from openai.types.responses import FunctionToolParam

EXECUTE_CODE_NAME = "execute_code"
SUBMIT_SOLUTION_NAME = "submit_solution"


class ExecuteCodeArgs(TypedDict):
    """Arguments for execute_code tool."""

    code: str


class SubmitSolutionArgs(TypedDict):
    """Arguments for submit_solution tool."""

    code: str


EXECUTE_CODE_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": EXECUTE_CODE_NAME,
        "description": (
            "Execute Python code in an isolated sandbox and return the output. "
            "Use this to test your solution before submitting. "
            "Returns stdout if successful, or the error message if execution fails."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
}

SUBMIT_SOLUTION_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": SUBMIT_SOLUTION_NAME,
        "description": (
            "Submit your final solution. This runs the code against all test cases. "
            "Only submit when you are confident your solution is correct. "
            "You can only submit once per problem."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Your final Python solution code",
                }
            },
            "required": ["code"],
        },
    },
}

ROLLOUT_TOOLS: list[ChatCompletionToolParam] = [EXECUTE_CODE_TOOL, SUBMIT_SOLUTION_TOOL]


def to_responses_format(
    tools: list[ChatCompletionToolParam],
) -> list[FunctionToolParam]:
    """Convert ChatCompletionToolParam to OpenAI Responses API format.

    The Responses API uses a flattened structure with strict mode enabled:
    - type, name, description, parameters at top level (not nested under "function")
    - strict: True for structured outputs
    - additionalProperties: False in parameters schema
    """
    result: list[FunctionToolParam] = []
    for tool in tools:
        func = tool["function"]
        params: dict[str, Any] = {**func["parameters"], "additionalProperties": False}
        result.append(
            FunctionToolParam(
                type="function",
                name=func["name"],
                description=func.get("description"),
                parameters=params,
                strict=True,
            )
        )
    return result


ROLLOUT_TOOLS_RESPONSES = to_responses_format(ROLLOUT_TOOLS)
