"""Unit tests for ART rollout with mocked LLM and sandbox."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from examples.rl_training.art.rollout import (
    Problem,
    RolloutConfig,
    _normalize_output_for_input,
    _response_output_to_choice,
    build_system_message,
    execute_code_in_sandbox,
    handle_tool_call,
    rollout,
    run_tests_in_sandbox,
)
from examples.rl_training.art.tools import (
    EXECUTE_CODE_NAME,
    ROLLOUT_TOOLS,
    SUBMIT_SOLUTION_NAME,
)


@pytest.fixture
def sample_problem() -> Problem:
    """Create a sample problem for testing."""
    return Problem(
        task_id="test_001",
        prompt="Write a function that adds two numbers.",
        test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
        test_imports="",
    )


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock sandbox that returns successful execution."""
    sandbox = MagicMock()

    mock_process = MagicMock()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "test output"
    mock_result.stderr = ""
    mock_process.result.return_value = mock_result

    sandbox.exec.return_value = mock_process
    return sandbox


@pytest.fixture
def failing_sandbox() -> MagicMock:
    """Create a mock sandbox that returns failed execution."""
    sandbox = MagicMock()

    mock_process = MagicMock()
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "NameError: name 'add' is not defined"
    mock_process.result.return_value = mock_result

    sandbox.exec.return_value = mock_process
    return sandbox


class TestProblem:
    """Tests for Problem dataclass."""

    def test_problem_creation(self) -> None:
        problem = Problem(
            task_id="test_001",
            prompt="Write a function",
            test_code="assert func() == 42",
        )
        assert problem.task_id == "test_001"
        assert problem.prompt == "Write a function"
        assert problem.test_code == "assert func() == 42"
        assert problem.test_imports == ""

    def test_problem_with_imports(self) -> None:
        problem = Problem(
            task_id="test_002",
            prompt="Use math",
            test_code="assert compute() == 3.14",
            test_imports="import math",
        )
        assert problem.test_imports == "import math"


class TestRolloutConfig:
    """Tests for RolloutConfig dataclass."""

    def test_default_config(self) -> None:
        config = RolloutConfig()
        assert config.model == "gpt-4o-mini"
        assert config.base_url is None
        assert config.api_key is None
        assert config.max_attempts == 3
        assert config.execution_timeout == 30.0

    def test_custom_config(self) -> None:
        config = RolloutConfig(
            model="gpt-4o",
            base_url="http://localhost:8000",
            api_key="test-key",
            max_attempts=5,
            execution_timeout=60.0,
        )
        assert config.model == "gpt-4o"
        assert config.base_url == "http://localhost:8000"
        assert config.api_key == "test-key"
        assert config.max_attempts == 5
        assert config.execution_timeout == 60.0


class TestBuildSystemMessage:
    """Tests for system message construction."""

    def test_build_system_message_includes_problem(self, sample_problem: Problem) -> None:
        message = build_system_message(sample_problem)
        assert sample_problem.prompt in message
        assert "execute_code" in message
        assert "submit_solution" in message


class TestExecuteCodeInSandbox:
    """Tests for execute_code_in_sandbox function."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, mock_sandbox: MagicMock) -> None:
        result = await execute_code_in_sandbox(mock_sandbox, "print('hello')")
        assert result == "test output"
        mock_sandbox.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_execution(self, failing_sandbox: MagicMock) -> None:
        result = await execute_code_in_sandbox(failing_sandbox, "bad_code")
        assert "Error (exit code 1)" in result
        assert "NameError" in result

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        sandbox = MagicMock()
        mock_process = MagicMock()
        mock_process.result.side_effect = TimeoutError("timed out")
        sandbox.exec.return_value = mock_process

        result = await execute_code_in_sandbox(sandbox, "while True: pass", timeout=1.0)
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_empty_output(self) -> None:
        sandbox = MagicMock()
        mock_process = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_process.result.return_value = mock_result
        sandbox.exec.return_value = mock_process

        result = await execute_code_in_sandbox(sandbox, "x = 1")
        assert result == "(no output)"


class TestRunTestsInSandbox:
    """Tests for run_tests_in_sandbox function."""

    @pytest.mark.asyncio
    async def test_tests_pass(self, mock_sandbox: MagicMock) -> None:
        passed, output = await run_tests_in_sandbox(
            mock_sandbox,
            "def add(a, b): return a + b",
            "assert add(1, 2) == 3",
        )
        assert passed is True
        assert output == "test output"

    @pytest.mark.asyncio
    async def test_tests_fail(self, failing_sandbox: MagicMock) -> None:
        passed, output = await run_tests_in_sandbox(
            failing_sandbox,
            "def add(a, b): return a - b",
            "assert add(1, 2) == 3",
        )
        assert passed is False
        assert "Tests failed" in output

    @pytest.mark.asyncio
    async def test_includes_imports(self, mock_sandbox: MagicMock) -> None:
        await run_tests_in_sandbox(
            mock_sandbox,
            "def compute(): return math.pi",
            "assert compute() == math.pi",
            test_imports="import math",
        )
        call_args = mock_sandbox.exec.call_args
        code = call_args[0][0][2]
        assert "import math" in code


class TestHandleToolCall:
    """Tests for handle_tool_call function.

    Note: handle_tool_call now expects ResponseFunctionToolCall objects from the
    Responses API, which have .name and .arguments directly (not nested under .function).
    """

    @pytest.mark.asyncio
    async def test_execute_code_tool(
        self, mock_sandbox: MagicMock, sample_problem: Problem
    ) -> None:
        tool_call = MagicMock()
        tool_call.name = EXECUTE_CODE_NAME
        tool_call.arguments = json.dumps({"code": "print('hello')"})
        tool_call.call_id = "call_123"
        tool_call.type = "function_call"

        config = RolloutConfig()
        result, is_submission, passed = await handle_tool_call(
            tool_call, mock_sandbox, sample_problem, config
        )

        assert result == "test output"
        assert is_submission is False
        assert passed is False

    @pytest.mark.asyncio
    async def test_submit_solution_tool_pass(
        self, mock_sandbox: MagicMock, sample_problem: Problem
    ) -> None:
        tool_call = MagicMock()
        tool_call.name = SUBMIT_SOLUTION_NAME
        tool_call.arguments = json.dumps({"code": "def add(a, b): return a + b"})
        tool_call.call_id = "call_456"
        tool_call.type = "function_call"

        config = RolloutConfig()
        result, is_submission, passed = await handle_tool_call(
            tool_call, mock_sandbox, sample_problem, config
        )

        assert is_submission is True
        assert passed is True

    @pytest.mark.asyncio
    async def test_submit_solution_tool_fail(
        self, failing_sandbox: MagicMock, sample_problem: Problem
    ) -> None:
        tool_call = MagicMock()
        tool_call.name = SUBMIT_SOLUTION_NAME
        tool_call.arguments = json.dumps({"code": "def add(a, b): return a - b"})
        tool_call.call_id = "call_789"
        tool_call.type = "function_call"

        config = RolloutConfig()
        result, is_submission, passed = await handle_tool_call(
            tool_call, failing_sandbox, sample_problem, config
        )

        assert is_submission is True
        assert passed is False
        assert "Tests failed" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self, mock_sandbox: MagicMock, sample_problem: Problem) -> None:
        tool_call = MagicMock()
        tool_call.name = "unknown_tool"
        tool_call.arguments = "{}"
        tool_call.call_id = "call_unknown"
        tool_call.type = "function_call"

        config = RolloutConfig()
        result, is_submission, passed = await handle_tool_call(
            tool_call, mock_sandbox, sample_problem, config
        )

        assert "Unknown tool" in result
        assert is_submission is False
        assert passed is False

    @pytest.mark.asyncio
    async def test_invalid_json_arguments(
        self, mock_sandbox: MagicMock, sample_problem: Problem
    ) -> None:
        tool_call = MagicMock()
        tool_call.name = EXECUTE_CODE_NAME
        tool_call.arguments = "not valid json"
        tool_call.call_id = "call_invalid"
        tool_call.type = "function_call"

        config = RolloutConfig()
        result, is_submission, passed = await handle_tool_call(
            tool_call, mock_sandbox, sample_problem, config
        )

        assert "Invalid JSON" in result
        assert is_submission is False
        assert passed is False


class TestNormalizeOutputForInput:
    """Tests for _normalize_output_for_input() helper.

    This function converts Responses API output items to valid input items
    for re-feeding to the API in multi-turn conversations.
    """

    def test_text_message_only(self) -> None:
        """Test normalizing a message with only text content."""
        mock_text = MagicMock()
        mock_text.type = "output_text"
        mock_text.text = "Hello, I will help you."

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [mock_text]

        mock_response = MagicMock()
        mock_response.output = [mock_message]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hello, I will help you."}]

    def test_function_call_only(self) -> None:
        """Test normalizing a function call without accompanying message."""
        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.id = "item_call_123"
        mock_function_call.call_id = "call_123"
        mock_function_call.name = EXECUTE_CODE_NAME
        mock_function_call.arguments = '{"code": "print(1)"}'

        mock_response = MagicMock()
        mock_response.output = [mock_function_call]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 1
        assert result[0]["type"] == "function_call"
        assert result[0]["id"] == "item_call_123"
        assert result[0]["call_id"] == "call_123"
        assert result[0]["name"] == EXECUTE_CODE_NAME
        assert result[0]["arguments"] == '{"code": "print(1)"}'

    def test_mixed_message_and_function_call(self) -> None:
        """Test normalizing output with both text message and function call."""
        mock_text = MagicMock()
        mock_text.type = "output_text"
        mock_text.text = "Let me test this code."

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [mock_text]

        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.id = "item_call_456"
        mock_function_call.call_id = "call_456"
        mock_function_call.name = EXECUTE_CODE_NAME
        mock_function_call.arguments = '{"code": "x = 1"}'

        mock_response = MagicMock()
        mock_response.output = [mock_message, mock_function_call]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 2
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "assistant"
        assert result[1]["type"] == "function_call"
        assert result[1]["call_id"] == "call_456"

    def test_refusal_content(self) -> None:
        """Test normalizing a message with refusal content."""
        mock_refusal = MagicMock()
        mock_refusal.type = "refusal"
        mock_refusal.refusal = "I cannot execute harmful code."

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [mock_refusal]

        mock_response = MagicMock()
        mock_response.output = [mock_message]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [
            {"type": "text", "text": "[Refusal: I cannot execute harmful code.]"}
        ]

    def test_multiple_text_parts(self) -> None:
        """Test normalizing a message with multiple text parts."""
        mock_text1 = MagicMock()
        mock_text1.type = "output_text"
        mock_text1.text = "First part."

        mock_text2 = MagicMock()
        mock_text2.type = "output_text"
        mock_text2.text = "Second part."

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [mock_text1, mock_text2]

        mock_response = MagicMock()
        mock_response.output = [mock_message]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 1
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["text"] == "First part."
        assert result[0]["content"][1]["text"] == "Second part."

    def test_empty_message_content_skipped(self) -> None:
        """Test that messages with no recognized content are not added."""
        mock_unknown = MagicMock()
        mock_unknown.type = "unknown_type"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [mock_unknown]

        mock_response = MagicMock()
        mock_response.output = [mock_message]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 0

    def test_multiple_function_calls(self) -> None:
        """Test normalizing output with multiple function calls."""
        mock_call1 = MagicMock()
        mock_call1.type = "function_call"
        mock_call1.id = "item_call_1"
        mock_call1.call_id = "call_1"
        mock_call1.name = EXECUTE_CODE_NAME
        mock_call1.arguments = '{"code": "x = 1"}'

        mock_call2 = MagicMock()
        mock_call2.type = "function_call"
        mock_call2.id = "item_call_2"
        mock_call2.call_id = "call_2"
        mock_call2.name = SUBMIT_SOLUTION_NAME
        mock_call2.arguments = '{"code": "def add(a, b): return a + b"}'

        mock_response = MagicMock()
        mock_response.output = [mock_call1, mock_call2]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 2
        assert result[0]["name"] == EXECUTE_CODE_NAME
        assert result[1]["name"] == SUBMIT_SOLUTION_NAME

    def test_reasoning_item_preserved(self) -> None:
        """Test that reasoning items are preserved for Codex models.

        When using reasoning models (gpt-5.x-codex), function_call items must
        be accompanied by their corresponding reasoning item in multi-turn
        conversations.
        """
        mock_reasoning = MagicMock()
        mock_reasoning.type = "reasoning"
        mock_reasoning.id = "rs_abc123"
        mock_reasoning.summary = None

        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.id = "fc_def456"
        mock_function_call.call_id = "call_789"
        mock_function_call.name = EXECUTE_CODE_NAME
        mock_function_call.arguments = '{"code": "print(1)"}'

        mock_response = MagicMock()
        mock_response.output = [mock_reasoning, mock_function_call]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 2
        assert result[0]["type"] == "reasoning"
        assert result[0]["id"] == "rs_abc123"
        assert result[0]["summary"] is None
        assert result[1]["type"] == "function_call"
        assert result[1]["call_id"] == "call_789"

    def test_reasoning_with_summary(self) -> None:
        """Test reasoning items with summary text are preserved."""
        mock_reasoning = MagicMock()
        mock_reasoning.type = "reasoning"
        mock_reasoning.id = "rs_xyz789"
        mock_reasoning.summary = [{"type": "summary_text", "text": "Thinking about the problem..."}]

        mock_response = MagicMock()
        mock_response.output = [mock_reasoning]

        result = _normalize_output_for_input(mock_response)

        assert len(result) == 1
        assert result[0]["type"] == "reasoning"
        assert result[0]["id"] == "rs_xyz789"
        assert result[0]["summary"] == [{"type": "summary_text", "text": "Thinking about the problem..."}]


class TestResponseOutputToChoice:
    """Tests for _response_output_to_choice() helper.

    This function converts Responses API output to ART-compatible Choice objects.
    """

    def test_text_only_response(self) -> None:
        """Test converting response with only text to Choice."""
        mock_response = MagicMock()
        mock_response.output_text = "The answer is 42."
        mock_response.output = []

        choice = _response_output_to_choice(mock_response)

        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content == "The answer is 42."
        assert choice.message.tool_calls is None
        assert choice.finish_reason == "stop"

    def test_function_call_response(self) -> None:
        """Test converting response with function call to Choice."""
        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.call_id = "call_123"
        mock_function_call.name = EXECUTE_CODE_NAME
        mock_function_call.arguments = '{"code": "print(1)"}'

        mock_response = MagicMock()
        mock_response.output_text = ""
        mock_response.output = [mock_function_call]

        choice = _response_output_to_choice(mock_response)

        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content is None
        assert choice.finish_reason == "tool_calls"
        assert len(choice.message.tool_calls) == 1
        assert choice.message.tool_calls[0].id == "call_123"
        assert choice.message.tool_calls[0].type == "function"
        assert choice.message.tool_calls[0].function.name == EXECUTE_CODE_NAME
        assert choice.message.tool_calls[0].function.arguments == '{"code": "print(1)"}'

    def test_mixed_text_and_function_call(self) -> None:
        """Test converting response with both text and function call."""
        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.call_id = "call_456"
        mock_function_call.name = SUBMIT_SOLUTION_NAME
        mock_function_call.arguments = '{"code": "def solve(): pass"}'

        mock_response = MagicMock()
        mock_response.output_text = "Here is my solution."
        mock_response.output = [mock_function_call]

        choice = _response_output_to_choice(mock_response)

        assert choice.message.content == "Here is my solution."
        assert choice.finish_reason == "tool_calls"
        assert len(choice.message.tool_calls) == 1

    def test_multiple_function_calls(self) -> None:
        """Test converting response with multiple function calls."""
        mock_call1 = MagicMock()
        mock_call1.type = "function_call"
        mock_call1.call_id = "call_1"
        mock_call1.name = EXECUTE_CODE_NAME
        mock_call1.arguments = '{"code": "x = 1"}'

        mock_call2 = MagicMock()
        mock_call2.type = "function_call"
        mock_call2.call_id = "call_2"
        mock_call2.name = EXECUTE_CODE_NAME
        mock_call2.arguments = '{"code": "y = 2"}'

        mock_response = MagicMock()
        mock_response.output_text = ""
        mock_response.output = [mock_call1, mock_call2]

        choice = _response_output_to_choice(mock_response)

        assert choice.finish_reason == "tool_calls"
        assert len(choice.message.tool_calls) == 2
        assert choice.message.tool_calls[0].id == "call_1"
        assert choice.message.tool_calls[1].id == "call_2"

    def test_empty_output_text_treated_as_none(self) -> None:
        """Test that empty output_text becomes None in message content."""
        mock_response = MagicMock()
        mock_response.output_text = ""
        mock_response.output = []

        choice = _response_output_to_choice(mock_response)

        assert choice.message.content is None
        assert choice.finish_reason == "stop"

    def test_non_function_call_items_ignored(self) -> None:
        """Test that non-function_call items are not included in tool_calls."""
        mock_message = MagicMock()
        mock_message.type = "message"

        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.call_id = "call_789"
        mock_function_call.name = EXECUTE_CODE_NAME
        mock_function_call.arguments = '{"code": "z = 3"}'

        mock_response = MagicMock()
        mock_response.output_text = "Some text"
        mock_response.output = [mock_message, mock_function_call]

        choice = _response_output_to_choice(mock_response)

        assert len(choice.message.tool_calls) == 1
        assert choice.message.tool_calls[0].id == "call_789"


def _create_mock_response_with_function_call(
    name: str, arguments: str, call_id: str, text_content: str = ""
) -> MagicMock:
    """Create a mock Responses API response with a function call."""
    mock_function_call = MagicMock()
    mock_function_call.type = "function_call"
    mock_function_call.name = name
    mock_function_call.arguments = arguments
    mock_function_call.call_id = call_id
    mock_function_call.id = f"item_{call_id}"

    mock_message = MagicMock()
    mock_message.type = "message"
    mock_message.content = []
    if text_content:
        mock_text = MagicMock()
        mock_text.type = "output_text"
        mock_text.text = text_content
        mock_message.content = [mock_text]

    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.error = None
    mock_response.output = [mock_message, mock_function_call]
    mock_response.output_text = text_content

    return mock_response


def _create_mock_response_text_only(text_content: str) -> MagicMock:
    """Create a mock Responses API response with only text (no function calls)."""
    mock_text = MagicMock()
    mock_text.type = "output_text"
    mock_text.text = text_content

    mock_message = MagicMock()
    mock_message.type = "message"
    mock_message.content = [mock_text]

    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.error = None
    mock_response.output = [mock_message]
    mock_response.output_text = text_content

    return mock_response


class TestRollout:
    """Tests for the main rollout function.

    Note: The rollout function now uses the Responses API (client.responses.create)
    instead of the Chat Completions API. The response format is different:
    - response.output contains a list of items (messages, function_calls)
    - function_call items have .name, .arguments, .call_id directly
    - response.status indicates completion status
    """

    @pytest.mark.asyncio
    async def test_successful_rollout_single_submission(
        self, sample_problem: Problem, mock_sandbox: MagicMock
    ) -> None:
        """Test rollout where agent submits correct solution on first try."""
        mock_response = _create_mock_response_with_function_call(
            name=SUBMIT_SOLUTION_NAME,
            arguments=json.dumps({"code": "def add(a, b): return a + b"}),
            call_id="call_123",
            text_content="I'll submit a solution.",
        )

        with patch("examples.rl_training.art.rollout.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            config = RolloutConfig(api_key="test-key")
            trajectory = await rollout(sample_problem, mock_sandbox, config)

        assert trajectory.reward == 1.0
        assert trajectory.metadata["task_id"] == "test_001"
        assert trajectory.metadata["submitted"] is True
        assert trajectory.tools == ROLLOUT_TOOLS

    @pytest.mark.asyncio
    async def test_failed_submission_reward_zero(
        self, sample_problem: Problem, failing_sandbox: MagicMock
    ) -> None:
        """Test rollout where agent submits but tests fail."""
        mock_response = _create_mock_response_with_function_call(
            name=SUBMIT_SOLUTION_NAME,
            arguments=json.dumps({"code": "def add(a, b): return a - b"}),
            call_id="call_456",
            text_content="Here's my solution.",
        )

        with patch("examples.rl_training.art.rollout.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            config = RolloutConfig(api_key="test-key")
            trajectory = await rollout(sample_problem, failing_sandbox, config)

        assert trajectory.reward == 0.0
        assert trajectory.metadata["submitted"] is True

    @pytest.mark.asyncio
    async def test_max_attempts_without_submission(
        self, sample_problem: Problem, mock_sandbox: MagicMock
    ) -> None:
        """Test rollout stops after max attempts without submission."""
        mock_response = _create_mock_response_with_function_call(
            name=EXECUTE_CODE_NAME,
            arguments=json.dumps({"code": "print('testing')"}),
            call_id="call_789",
            text_content="Let me test this.",
        )

        with patch("examples.rl_training.art.rollout.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            config = RolloutConfig(api_key="test-key", max_attempts=2)
            trajectory = await rollout(sample_problem, mock_sandbox, config)

        assert trajectory.reward == 0.0
        assert trajectory.metadata["submitted"] is False
        assert trajectory.metadata["tool_calls"] == 2

    @pytest.mark.asyncio
    async def test_no_tool_calls(self, sample_problem: Problem, mock_sandbox: MagicMock) -> None:
        """Test rollout where agent responds without using tools."""
        mock_response = _create_mock_response_text_only(
            "The solution is def add(a, b): return a + b"
        )

        with patch("examples.rl_training.art.rollout.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            config = RolloutConfig(api_key="test-key")
            trajectory = await rollout(sample_problem, mock_sandbox, config)

        assert trajectory.reward == 0.0
        assert trajectory.metadata["submitted"] is False
        assert trajectory.metadata["tool_calls"] == 0


class TestToolSchemas:
    """Tests for tool schema definitions."""

    def test_rollout_tools_count(self) -> None:
        assert len(ROLLOUT_TOOLS) == 2

    def test_execute_code_tool_schema(self) -> None:
        execute_tool = next(t for t in ROLLOUT_TOOLS if t["function"]["name"] == EXECUTE_CODE_NAME)
        assert execute_tool["type"] == "function"
        assert "code" in execute_tool["function"]["parameters"]["properties"]
        assert "code" in execute_tool["function"]["parameters"]["required"]

    def test_submit_solution_tool_schema(self) -> None:
        submit_tool = next(
            t for t in ROLLOUT_TOOLS if t["function"]["name"] == SUBMIT_SOLUTION_NAME
        )
        assert submit_tool["type"] == "function"
        assert "code" in submit_tool["function"]["parameters"]["properties"]
        assert "code" in submit_tool["function"]["parameters"]["required"]
