"""ART core types for trajectory-based RL training.

This module provides the Trajectory and TrainableModel abstractions for
multi-step RL training with code execution rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@dataclass
class Trajectory:
    """A trajectory of messages and model choices for RL training.

    Trajectories capture the full interaction history for a rollout,
    including model completions and feedback messages. This enables
    credit assignment across multi-step problem solving.

    Attributes:
        messages_and_choices: List of messages and model responses.
            Messages are dicts with "role" and "content".
            Choices are ChatCompletion Choice objects.
        reward: Final reward for the trajectory (0.0 to 1.0).
        finished: Whether finish() has been called.
    """

    messages_and_choices: list[Any] = field(default_factory=list)
    reward: float = 0.0
    finished: bool = False

    def finish(self) -> Trajectory:
        """Mark trajectory as complete and return self.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If trajectory was already finished.
        """
        if self.finished:
            raise RuntimeError("Trajectory already finished")
        self.finished = True
        return self


@runtime_checkable
class TrainableModel(Protocol):
    """Protocol for models that can be used in ART rollouts.

    TrainableModel provides a unified interface for different model backends
    (OpenAI, vLLM, local transformers) to be used with ART's trajectory system.
    """

    def openai_client(self) -> AsyncOpenAI:
        """Get an OpenAI-compatible async client for chat completions.

        Returns:
            AsyncOpenAI client (or compatible) for generating completions.
        """
        ...
