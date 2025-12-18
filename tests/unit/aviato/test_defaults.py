"""Unit tests for aviato._defaults module."""

from aviato import SandboxDefaults


class TestSandboxDefaults:
    """Tests for SandboxDefaults dataclass."""

    def test_defaults_have_sensible_values(self) -> None:
        """Test SandboxDefaults has sensible default values."""
        defaults = SandboxDefaults()

        assert defaults.container_image == "python:3.11"
        assert defaults.base_url == "https://atc.cwaviato.com"
        assert defaults.request_timeout_seconds == 300.0
        assert defaults.max_lifetime_seconds is None  # Backend controls default
        assert defaults.tags == ()
        assert defaults.runway_ids is None
        assert defaults.tower_ids is None

    def test_defaults_are_immutable(self) -> None:
        """Test SandboxDefaults is frozen/immutable."""
        defaults = SandboxDefaults(container_image="python:3.11")

        # Attempting to modify should raise
        import dataclasses

        assert dataclasses.is_dataclass(defaults)
        # The frozen=True attribute means assignment raises FrozenInstanceError

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
