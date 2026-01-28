"""Unit tests for ART tool definitions and format conversion."""

from __future__ import annotations

from examples.rl_training.art.tools import (
    EXECUTE_CODE_NAME,
    EXECUTE_CODE_TOOL,
    ROLLOUT_TOOLS,
    ROLLOUT_TOOLS_RESPONSES,
    SUBMIT_SOLUTION_NAME,
    SUBMIT_SOLUTION_TOOL,
    to_responses_format,
)


class TestRolloutTools:
    """Tests for original ROLLOUT_TOOLS in ChatCompletionToolParam format."""

    def test_rollout_tools_count(self) -> None:
        assert len(ROLLOUT_TOOLS) == 2

    def test_execute_code_tool_has_nested_function(self) -> None:
        assert EXECUTE_CODE_TOOL["type"] == "function"
        assert "function" in EXECUTE_CODE_TOOL
        assert EXECUTE_CODE_TOOL["function"]["name"] == EXECUTE_CODE_NAME

    def test_submit_solution_tool_has_nested_function(self) -> None:
        assert SUBMIT_SOLUTION_TOOL["type"] == "function"
        assert "function" in SUBMIT_SOLUTION_TOOL
        assert SUBMIT_SOLUTION_TOOL["function"]["name"] == SUBMIT_SOLUTION_NAME


class TestToResponsesFormat:
    """Tests for to_responses_format() conversion function."""

    def test_converts_all_tools(self) -> None:
        converted = to_responses_format(ROLLOUT_TOOLS)
        assert len(converted) == len(ROLLOUT_TOOLS)

    def test_flattens_structure(self) -> None:
        converted = to_responses_format(ROLLOUT_TOOLS)
        for tool in converted:
            assert "function" not in tool
            assert "name" in tool
            assert "parameters" in tool

    def test_has_required_fields(self) -> None:
        converted = to_responses_format(ROLLOUT_TOOLS)
        required_fields = {"type", "name", "description", "parameters", "strict"}
        for tool in converted:
            assert set(tool.keys()) == required_fields

    def test_strict_is_true(self) -> None:
        converted = to_responses_format(ROLLOUT_TOOLS)
        for tool in converted:
            assert tool["strict"] is True

    def test_additional_properties_false(self) -> None:
        converted = to_responses_format(ROLLOUT_TOOLS)
        for tool in converted:
            assert tool["parameters"]["additionalProperties"] is False

    def test_preserves_tool_names(self) -> None:
        converted = to_responses_format(ROLLOUT_TOOLS)
        names = {tool["name"] for tool in converted}
        assert names == {EXECUTE_CODE_NAME, SUBMIT_SOLUTION_NAME}

    def test_preserves_parameters_schema(self) -> None:
        converted = to_responses_format(ROLLOUT_TOOLS)
        execute_tool = next(t for t in converted if t["name"] == EXECUTE_CODE_NAME)
        assert execute_tool["parameters"]["type"] == "object"
        assert "code" in execute_tool["parameters"]["properties"]
        assert "code" in execute_tool["parameters"]["required"]

    def test_preserves_descriptions(self) -> None:
        converted = to_responses_format(ROLLOUT_TOOLS)
        execute_tool = next(t for t in converted if t["name"] == EXECUTE_CODE_NAME)
        assert "Execute Python code" in execute_tool["description"]


class TestRolloutToolsResponses:
    """Tests for pre-converted ROLLOUT_TOOLS_RESPONSES constant."""

    def test_is_precomputed(self) -> None:
        assert ROLLOUT_TOOLS_RESPONSES is not None
        assert len(ROLLOUT_TOOLS_RESPONSES) == 2

    def test_matches_conversion(self) -> None:
        fresh_conversion = to_responses_format(ROLLOUT_TOOLS)
        assert ROLLOUT_TOOLS_RESPONSES == fresh_conversion

    def test_execute_code_schema_shape(self) -> None:
        execute_tool = next(
            t for t in ROLLOUT_TOOLS_RESPONSES if t["name"] == EXECUTE_CODE_NAME
        )
        assert execute_tool == {
            "type": "function",
            "name": "execute_code",
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
                "additionalProperties": False,
            },
            "strict": True,
        }

    def test_submit_solution_schema_shape(self) -> None:
        submit_tool = next(
            t for t in ROLLOUT_TOOLS_RESPONSES if t["name"] == SUBMIT_SOLUTION_NAME
        )
        assert submit_tool == {
            "type": "function",
            "name": "submit_solution",
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
                "additionalProperties": False,
            },
            "strict": True,
        }


class TestBackwardCompatibility:
    """Tests ensuring original tools remain unchanged for ART compatibility."""

    def test_original_execute_code_unchanged(self) -> None:
        assert EXECUTE_CODE_TOOL["type"] == "function"
        assert EXECUTE_CODE_TOOL["function"]["name"] == EXECUTE_CODE_NAME
        assert "additionalProperties" not in EXECUTE_CODE_TOOL["function"]["parameters"]
        assert "strict" not in EXECUTE_CODE_TOOL

    def test_original_submit_solution_unchanged(self) -> None:
        assert SUBMIT_SOLUTION_TOOL["type"] == "function"
        assert SUBMIT_SOLUTION_TOOL["function"]["name"] == SUBMIT_SOLUTION_NAME
        assert "additionalProperties" not in SUBMIT_SOLUTION_TOOL["function"]["parameters"]
        assert "strict" not in SUBMIT_SOLUTION_TOOL
