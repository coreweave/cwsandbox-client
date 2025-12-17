"""Unit tests for aviato._defaults module."""

import dataclasses

import pytest

from aviato import SandboxDefaults


class TestSandboxDefaults:
    """Tests for SandboxDefaults dataclass."""

    def test_defaults_are_reasonable(self) -> None:
        """Test SandboxDefaults has reasonable default values."""
        defaults = SandboxDefaults()

        # Container image should be non-empty and look like a docker image
        assert defaults.container_image
        assert ":" in defaults.container_image or "/" in defaults.container_image

        # Command and args should be non-empty (sandbox needs something to run)
        assert defaults.command
        assert isinstance(defaults.args, tuple)

        # Base URL should be a valid HTTP(S) URL
        assert defaults.base_url.startswith(("http://", "https://"))

        # Timeout should be positive
        assert 0 < defaults.request_timeout_seconds

        # max_lifetime_seconds can be None (backend-controlled) or positive
        assert defaults.max_lifetime_seconds is None or defaults.max_lifetime_seconds > 0

        # Tags should default to empty tuple (immutable)
        assert defaults.tags == ()

        # Runway/tower IDs should default to None (no filtering)
        assert defaults.runway_ids is None
        assert defaults.tower_ids is None

    def test_defaults_are_immutable(self) -> None:
        """Test SandboxDefaults is frozen/immutable."""
        defaults = SandboxDefaults(container_image="python:3.11")

        assert dataclasses.is_dataclass(defaults)

        with pytest.raises(dataclasses.FrozenInstanceError):
            defaults.container_image = "python:3.12"  # type: ignore[misc]

    def test_merge_tags_empty_base(self) -> None:
        """Test merge_tags with no default tags."""
        defaults = SandboxDefaults()

        result = defaults.merge_tags(["new-tag"])

        assert result == ["new-tag"]

    def test_merge_tags_empty_additional(self) -> None:
        """Test merge_tags with no additional tags."""
        defaults = SandboxDefaults(tags=("default-tag",))

        result = defaults.merge_tags(None)

        assert result == ["default-tag"]

    def test_merge_tags_both(self) -> None:
        """Test merge_tags combines both sources."""
        defaults = SandboxDefaults(tags=("default-1", "default-2"))

        result = defaults.merge_tags(["additional-1", "additional-2"])

        assert result == ["default-1", "default-2", "additional-1", "additional-2"]

    def test_with_overrides_creates_new(self) -> None:
        """Test with_overrides creates a new instance."""
        defaults = SandboxDefaults(container_image="python:3.11")

        new_defaults = defaults.with_overrides(container_image="python:3.12")

        assert defaults.container_image == "python:3.11"  # original unchanged
        assert new_defaults.container_image == "python:3.12"

    def test_with_overrides_partial(self) -> None:
        """Test with_overrides preserves other fields."""
        defaults = SandboxDefaults(
            container_image="python:3.11",
            base_url="http://example.com",
            request_timeout_seconds=60.0,
        )

        new_defaults = defaults.with_overrides(request_timeout_seconds=120.0)

        assert new_defaults.container_image == "python:3.11"
        assert new_defaults.base_url == "http://example.com"
        assert new_defaults.request_timeout_seconds == 120.0

    def test_merge_env_vars_empty_base(self) -> None:
        """Test merge_env_vars with no default env vars."""
        defaults = SandboxDefaults()

        result = defaults.merge_env_vars({"LOG_LEVEL": "info"})

        assert result == {"LOG_LEVEL": "info"}

    def test_merge_env_vars_with_additional(self) -> None:
        """Test merge_env_vars with additional env vars."""
        defaults = SandboxDefaults(
            env_vars={
                "LOG_LEVEL": "info",
                "REGION": "us-west",
            },
        )

        result = defaults.merge_env_vars({
            "LOG_LEVEL": "debug",
            "MODEL": "gpt2",
        })

        assert result == {
            "LOG_LEVEL": "debug",  # Overridden
            "REGION": "us-west",  # Preserved
            "MODEL": "gpt2",  # Added
        }
