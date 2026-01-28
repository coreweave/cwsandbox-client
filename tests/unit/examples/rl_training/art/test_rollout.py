"""Unit tests for ART rollout with mocked LLM and sandbox."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from examples.rl_training.art.rollout import (
    Problem,
    RolloutConfig,
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
    """Tests for handle_tool_call function."""

    @pytest.mark.asyncio
    async def test_execute_code_tool(
        self, mock_sandbox: MagicMock, sample_problem: Problem
    ) -> None:
        tool_call = MagicMock()
        tool_call.function.name = EXECUTE_CODE_NAME
        tool_call.function.arguments = json.dumps({"code": "print('hello')"})

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
        tool_call.function.name = SUBMIT_SOLUTION_NAME
        tool_call.function.arguments = json.dumps({"code": "def add(a, b): return a + b"})

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
        tool_call.function.name = SUBMIT_SOLUTION_NAME
        tool_call.function.arguments = json.dumps({"code": "def add(a, b): return a - b"})

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
        tool_call.function.name = "unknown_tool"
        tool_call.function.arguments = "{}"

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
        tool_call.function.name = EXECUTE_CODE_NAME
        tool_call.function.arguments = "not valid json"

        config = RolloutConfig()
        result, is_submission, passed = await handle_tool_call(
            tool_call, mock_sandbox, sample_problem, config
        )

        assert "Invalid JSON" in result
        assert is_submission is False
        assert passed is False


class TestRollout:
    """Tests for the main rollout function."""

    @pytest.mark.asyncio
    async def test_successful_rollout_single_submission(
        self, sample_problem: Problem, mock_sandbox: MagicMock
    ) -> None:
        """Test rollout where agent submits correct solution on first try."""
        mock_choice = MagicMock()
        mock_choice.message.content = "I'll submit a solution."
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = SUBMIT_SOLUTION_NAME
        mock_tool_call.function.arguments = json.dumps({"code": "def add(a, b): return a + b"})
        mock_tool_call.model_dump.return_value = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": SUBMIT_SOLUTION_NAME,
                "arguments": '{"code": "def add(a, b): return a + b"}',
            },
        }
        mock_choice.message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("examples.rl_training.art.rollout.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
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
        mock_choice = MagicMock()
        mock_choice.message.content = "Here's my solution."
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = SUBMIT_SOLUTION_NAME
        mock_tool_call.function.arguments = json.dumps({"code": "def add(a, b): return a - b"})
        mock_tool_call.model_dump.return_value = {
            "id": "call_456",
            "type": "function",
            "function": {
                "name": SUBMIT_SOLUTION_NAME,
                "arguments": '{"code": "def add(a, b): return a - b"}',
            },
        }
        mock_choice.message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("examples.rl_training.art.rollout.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
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
        mock_choice = MagicMock()
        mock_choice.message.content = "Let me test this."
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_789"
        mock_tool_call.function.name = EXECUTE_CODE_NAME
        mock_tool_call.function.arguments = json.dumps({"code": "print('testing')"})
        mock_tool_call.model_dump.return_value = {
            "id": "call_789",
            "type": "function",
            "function": {"name": EXECUTE_CODE_NAME, "arguments": '{"code": "print(\'testing\')"}'},
        }
        mock_choice.message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("examples.rl_training.art.rollout.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            config = RolloutConfig(api_key="test-key", max_attempts=2)
            trajectory = await rollout(sample_problem, mock_sandbox, config)

        assert trajectory.reward == 0.0
        assert trajectory.metadata["submitted"] is False
        assert trajectory.metadata["tool_calls"] == 2

    @pytest.mark.asyncio
    async def test_no_tool_calls(self, sample_problem: Problem, mock_sandbox: MagicMock) -> None:
        """Test rollout where agent responds without using tools."""
        mock_choice = MagicMock()
        mock_choice.message.content = "The solution is def add(a, b): return a + b"
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("examples.rl_training.art.rollout.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
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
