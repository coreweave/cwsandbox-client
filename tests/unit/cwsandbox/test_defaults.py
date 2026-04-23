# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._defaults module."""

import dataclasses
import math

import pytest

from cwsandbox import NetworkOptions, ResourceOptions, SandboxDefaults, Secret


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

        # Profile/runner IDs should default to None (no filtering)
        assert defaults.profile_ids is None
        assert defaults.profile_names is None
        assert defaults.runner_ids is None

        # Network should default to None (backend defaults)
        assert defaults.network is None

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

    def test_merge_environment_variables_empty_base(self) -> None:
        """Test merge_environment_variables with no default environment variables."""
        defaults = SandboxDefaults()

        result = defaults.merge_environment_variables({"LOG_LEVEL": "info"})

        assert result == {"LOG_LEVEL": "info"}

    def test_merge_environment_variables_with_additional(self) -> None:
        """Test merge_environment_variables with additional environment variables."""
        defaults = SandboxDefaults(
            environment_variables={
                "LOG_LEVEL": "info",
                "REGION": "us-west",
            },
        )

        result = defaults.merge_environment_variables(
            {
                "LOG_LEVEL": "debug",
                "MODEL": "gpt2",
            }
        )

        assert result == {
            "LOG_LEVEL": "debug",  # Overridden
            "REGION": "us-west",  # Preserved
            "MODEL": "gpt2",  # Added
        }

    def test_annotations_default_empty(self) -> None:
        """Test annotations defaults to empty dict."""
        defaults = SandboxDefaults()
        assert defaults.annotations == {}

    def test_annotations_can_be_set(self) -> None:
        """Test annotations can be set on construction."""
        defaults = SandboxDefaults(annotations={"team": "platform", "env": "staging"})
        assert defaults.annotations == {"team": "platform", "env": "staging"}

    def test_annotations_immutable(self) -> None:
        """Test annotations field cannot be reassigned."""
        defaults = SandboxDefaults(annotations={"team": "platform"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            defaults.annotations = {"team": "other"}  # type: ignore[misc]

    def test_merge_annotations_delegates_to_merge_dicts(self) -> None:
        """Test merge_annotations uses same logic as merge_environment_variables."""
        defaults = SandboxDefaults(
            annotations={"team": "platform", "env": "staging"},
        )
        result = defaults.merge_annotations({"env": "production", "version": "v2"})
        assert result == {
            "team": "platform",
            "env": "production",
            "version": "v2",
        }

    def test_with_overrides_annotations(self) -> None:
        """Test with_overrides can change annotations."""
        defaults = SandboxDefaults(annotations={"team": "platform"})
        new_defaults = defaults.with_overrides(annotations={"team": "infra"})
        assert defaults.annotations == {"team": "platform"}
        assert new_defaults.annotations == {"team": "infra"}

    def test_network_can_be_set(self) -> None:
        """Test network can be set in SandboxDefaults."""
        network = NetworkOptions(ingress_mode="public", exposed_ports=(8080,))
        defaults = SandboxDefaults(network=network)

        assert defaults.network is network
        assert defaults.network.ingress_mode == "public"
        assert defaults.network.exposed_ports == (8080,)

    def test_with_overrides_network(self) -> None:
        """Test with_overrides can change network."""
        network1 = NetworkOptions(egress_mode="internet")
        network2 = NetworkOptions(egress_mode="isolated")
        defaults = SandboxDefaults(network=network1)

        new_defaults = defaults.with_overrides(network=network2)

        assert defaults.network is network1  # original unchanged
        assert new_defaults.network is network2

    def test_resources_accepts_resource_options(self) -> None:
        """Test resources field accepts ResourceOptions."""
        opts = ResourceOptions(
            requests={"cpu": "1", "memory": "256Mi"},
            limits={"cpu": "8", "memory": "2Gi"},
        )
        defaults = SandboxDefaults(resources=opts)
        assert isinstance(defaults.resources, ResourceOptions)
        assert defaults.resources.requests == {"cpu": "1", "memory": "256Mi"}
        assert defaults.resources.limits == {"cpu": "8", "memory": "2Gi"}

    def test_resources_accepts_flat_dict(self) -> None:
        """Test resources field still accepts flat dicts for backward compat."""
        defaults = SandboxDefaults(resources={"cpu": "2", "memory": "4Gi"})
        assert defaults.resources == {"cpu": "2", "memory": "4Gi"}

    def test_with_overrides_resources_to_resource_options(self) -> None:
        """Test with_overrides replaces flat dict resources with ResourceOptions."""
        defaults = SandboxDefaults(resources={"cpu": "2", "memory": "4Gi"})
        opts = ResourceOptions(
            requests={"cpu": "1", "memory": "256Mi"},
            limits={"cpu": "8", "memory": "2Gi"},
        )

        new_defaults = defaults.with_overrides(resources=opts)

        assert defaults.resources == {"cpu": "2", "memory": "4Gi"}
        assert isinstance(new_defaults.resources, ResourceOptions)
        assert new_defaults.resources.requests == {"cpu": "1", "memory": "256Mi"}
        assert new_defaults.resources.limits == {"cpu": "8", "memory": "2Gi"}

    def test_secrets_can_be_set(self) -> None:
        """Test secrets can be set in SandboxDefaults."""
        secret = Secret(store="wandb", name="HF_TOKEN")
        defaults = SandboxDefaults(secrets=(secret,))

        assert defaults.secrets is not None
        assert len(defaults.secrets) == 1
        assert defaults.secrets[0].store == "wandb"
        assert defaults.secrets[0].name == "HF_TOKEN"

    def test_poll_retry_defaults(self) -> None:
        """Poll retry config has sensible wall-clock and per-call defaults."""
        defaults = SandboxDefaults()
        assert defaults.poll_retry_budget_seconds == 30.0
        assert defaults.poll_rpc_timeout_seconds == 15.0

    def test_poll_retry_can_be_overridden(self) -> None:
        """Poll retry fields accept explicit values on construction."""
        defaults = SandboxDefaults(
            poll_retry_budget_seconds=60.0,
            poll_rpc_timeout_seconds=5.0,
        )
        assert defaults.poll_retry_budget_seconds == 60.0
        assert defaults.poll_rpc_timeout_seconds == 5.0

    def test_poll_retry_budget_zero_disables_retry(self) -> None:
        """Budget of 0 is accepted - callers use it to opt out of retry."""
        defaults = SandboxDefaults(poll_retry_budget_seconds=0.0)
        assert defaults.poll_retry_budget_seconds == 0.0

    def test_with_overrides_preserves_poll_fields(self) -> None:
        """with_overrides preserves unrelated poll fields."""
        defaults = SandboxDefaults(
            poll_retry_budget_seconds=60.0,
            poll_rpc_timeout_seconds=5.0,
        )
        new_defaults = defaults.with_overrides(container_image="python:3.12")
        assert new_defaults.poll_retry_budget_seconds == 60.0
        assert new_defaults.poll_rpc_timeout_seconds == 5.0

    def test_with_overrides_can_change_poll_fields(self) -> None:
        """with_overrides can change poll fields independently."""
        defaults = SandboxDefaults()
        new_defaults = defaults.with_overrides(
            poll_retry_budget_seconds=45.0,
            poll_rpc_timeout_seconds=10.0,
        )
        assert defaults.poll_retry_budget_seconds == 30.0  # original unchanged
        assert new_defaults.poll_retry_budget_seconds == 45.0
        assert new_defaults.poll_rpc_timeout_seconds == 10.0

    def test_poll_retry_budget_seconds_rejects_negative(self) -> None:
        """Negative poll_retry_budget_seconds raises ValueError."""
        with pytest.raises(ValueError, match="poll_retry_budget_seconds"):
            SandboxDefaults(poll_retry_budget_seconds=-1.0)

    def test_poll_retry_budget_seconds_rejects_nan(self) -> None:
        """NaN poll_retry_budget_seconds raises ValueError."""
        with pytest.raises(ValueError, match="poll_retry_budget_seconds"):
            SandboxDefaults(poll_retry_budget_seconds=math.nan)

    def test_poll_retry_budget_seconds_rejects_inf(self) -> None:
        """Infinite poll_retry_budget_seconds raises ValueError."""
        with pytest.raises(ValueError, match="poll_retry_budget_seconds"):
            SandboxDefaults(poll_retry_budget_seconds=math.inf)

    def test_poll_retry_budget_seconds_zero_allowed(self) -> None:
        """Zero poll_retry_budget_seconds is accepted (disables retry)."""
        defaults = SandboxDefaults(poll_retry_budget_seconds=0.0)
        assert defaults.poll_retry_budget_seconds == 0.0

    def test_poll_rpc_timeout_seconds_rejects_zero(self) -> None:
        """Zero poll_rpc_timeout_seconds raises ValueError - must be strictly positive."""
        with pytest.raises(ValueError, match="poll_rpc_timeout_seconds"):
            SandboxDefaults(poll_rpc_timeout_seconds=0.0)

    def test_poll_rpc_timeout_seconds_rejects_negative(self) -> None:
        """Negative poll_rpc_timeout_seconds raises ValueError."""
        with pytest.raises(ValueError, match="poll_rpc_timeout_seconds"):
            SandboxDefaults(poll_rpc_timeout_seconds=-5.0)

    def test_poll_rpc_timeout_seconds_rejects_nan(self) -> None:
        """NaN poll_rpc_timeout_seconds raises ValueError."""
        with pytest.raises(ValueError, match="poll_rpc_timeout_seconds"):
            SandboxDefaults(poll_rpc_timeout_seconds=math.nan)


class TestSandboxDefaultsFromDict:
    """Tests for SandboxDefaults.from_dict()."""

    def test_from_dict_none_returns_default(self) -> None:
        """from_dict(None) returns default SandboxDefaults."""
        defaults = SandboxDefaults.from_dict(None)
        assert defaults.container_image == "python:3.11"

    def test_from_dict_basic_fields(self) -> None:
        """from_dict passes through simple fields."""
        defaults = SandboxDefaults.from_dict(
            {
                "container_image": "ubuntu:22.04",
                "max_lifetime_seconds": 3600,
            }
        )
        assert defaults.container_image == "ubuntu:22.04"
        assert defaults.max_lifetime_seconds == 3600

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """from_dict silently ignores keys not in SandboxDefaults."""
        defaults = SandboxDefaults.from_dict(
            {
                "container_image": "ubuntu:22.04",
                "not_a_real_field": "ignored",
            }
        )
        assert defaults.container_image == "ubuntu:22.04"

    def test_from_dict_coerces_lists_to_tuples(self) -> None:
        """from_dict converts lists to tuples for tuple fields."""
        defaults = SandboxDefaults.from_dict(
            {
                "tags": ["tag1", "tag2"],
                "args": ["-f", "/dev/null"],
                "profile_ids": ["r1"],
                "profile_names": ["n1", "n2"],
                "runner_ids": ["t1", "t2"],
            }
        )
        assert defaults.tags == ("tag1", "tag2")
        assert defaults.args == ("-f", "/dev/null")
        assert defaults.profile_ids == ("r1",)
        assert defaults.profile_names == ("n1", "n2")
        assert defaults.runner_ids == ("t1", "t2")

    def test_from_dict_rejects_bare_string_profile_names(self) -> None:
        """from_dict raises on bare string for profile_names (not a sequence)."""
        with pytest.raises(TypeError, match="profile_names must be a sequence of strings"):
            SandboxDefaults.from_dict({"profile_names": "not-a-list"})

    def test_from_dict_coerces_network_dict(self) -> None:
        """from_dict converts network dict to NetworkOptions."""
        defaults = SandboxDefaults.from_dict(
            {
                "network": {"ingress_mode": "public", "exposed_ports": [8080]},
            }
        )
        assert defaults.network is not None
        assert isinstance(defaults.network, NetworkOptions)
        assert defaults.network.ingress_mode == "public"

    def test_from_dict_coerces_secrets_dicts(self) -> None:
        """from_dict converts secret dicts to Secret objects."""
        defaults = SandboxDefaults.from_dict(
            {
                "secrets": [
                    {"store": "wandb", "name": "HF_TOKEN"},
                    {
                        "store": "wandb",
                        "name": "db-creds",
                        "field": "password",
                        "env_var": "DB_PASS",
                    },
                ],
            }
        )
        assert defaults.secrets is not None
        assert len(defaults.secrets) == 2
        assert isinstance(defaults.secrets[0], Secret)
        assert defaults.secrets[0].env_var == "HF_TOKEN"
        assert defaults.secrets[1].field == "password"

    def test_from_dict_preserves_existing_dataclasses(self) -> None:
        """from_dict passes through Secret and NetworkOptions instances."""
        net = NetworkOptions(egress_mode="internet")
        secret = Secret(store="wandb", name="HF_TOKEN")
        defaults = SandboxDefaults.from_dict(
            {
                "network": net,
                "secrets": [secret],
            }
        )
        assert defaults.network is net
        assert defaults.secrets is not None
        assert defaults.secrets[0] is secret

    def test_from_dict_network_none_preserved(self) -> None:
        """from_dict preserves network=None without crashing."""
        defaults = SandboxDefaults.from_dict({"network": None})
        assert defaults.network is None

    def test_from_dict_secrets_none_preserved(self) -> None:
        """from_dict preserves secrets=None without crashing."""
        defaults = SandboxDefaults.from_dict({"secrets": None})
        assert defaults.secrets is None

    def test_from_dict_drops_none_for_non_optional_fields(self) -> None:
        """from_dict drops None for args/tags/environment_variables, using defaults."""
        defaults = SandboxDefaults.from_dict(
            {
                "tags": None,
                "args": None,
                "environment_variables": None,
            }
        )
        assert defaults.tags == ()
        assert defaults.args == ("-c", 'trap "exit 0" TERM INT; sleep infinity & wait')
        assert defaults.environment_variables == {}

    def test_from_dict_drops_none_for_scalar_fields(self) -> None:
        """from_dict drops None for scalar non-optional fields."""
        defaults = SandboxDefaults.from_dict(
            {
                "container_image": None,
                "command": None,
                "base_url": None,
                "temp_dir": None,
                "request_timeout_seconds": None,
            }
        )
        assert defaults.container_image == "python:3.11"
        assert defaults.command == "/bin/sh"
        assert defaults.base_url == "https://api.cwsandbox.com"
        assert defaults.temp_dir == "/tmp"
        assert defaults.request_timeout_seconds == 300.0

    def test_from_dict_flat_resources_dict(self) -> None:
        """from_dict passes through a flat resources dict."""
        defaults = SandboxDefaults.from_dict({"resources": {"cpu": "2", "memory": "4Gi"}})
        assert defaults.resources == {"cpu": "2", "memory": "4Gi"}

    def test_from_dict_nested_resources_dict(self) -> None:
        """from_dict passes through a nested resources dict with requests/limits."""
        defaults = SandboxDefaults.from_dict(
            {
                "resources": {
                    "requests": {"cpu": "1", "memory": "256Mi"},
                    "limits": {"cpu": "8", "memory": "2Gi"},
                },
            }
        )
        assert isinstance(defaults.resources, dict)
        assert "requests" in defaults.resources
        assert "limits" in defaults.resources

    def test_from_dict_preserves_resource_options(self) -> None:
        """from_dict preserves ResourceOptions instance."""
        from cwsandbox._types import ResourceOptions

        opts = ResourceOptions(requests={"cpu": "1"}, limits={"cpu": "4"})
        defaults = SandboxDefaults.from_dict({"resources": opts})
        assert defaults.resources is opts

    def test_from_dict_rejects_bare_string_for_tuple_fields(self) -> None:
        """from_dict raises TypeError when a bare string is passed for a tuple field."""
        with pytest.raises(TypeError, match="must be a sequence of strings"):
            SandboxDefaults.from_dict({"tags": "prod"})

    def test_from_dict_poll_retry_fields(self) -> None:
        """from_dict round-trips poll retry fields."""
        defaults = SandboxDefaults.from_dict(
            {
                "poll_retry_budget_seconds": 60.0,
                "poll_rpc_timeout_seconds": 5.0,
            }
        )
        assert defaults.poll_retry_budget_seconds == 60.0
        assert defaults.poll_rpc_timeout_seconds == 5.0

    def test_from_dict_drops_none_for_poll_fields(self) -> None:
        """from_dict drops None for poll fields so dataclass defaults apply."""
        defaults = SandboxDefaults.from_dict(
            {
                "poll_retry_budget_seconds": None,
                "poll_rpc_timeout_seconds": None,
            }
        )
        assert defaults.poll_retry_budget_seconds == 30.0
        assert defaults.poll_rpc_timeout_seconds == 15.0


class TestResolveSelector:
    """Tests for the _resolve_selector helper."""

    def test_override_wins_over_default(self) -> None:
        """Explicit override (non-empty) takes precedence over default."""
        from cwsandbox._defaults import _resolve_selector

        assert _resolve_selector(["a", "b"], None) == ["a", "b"]

    def test_empty_override_preserves_clear_intent(self) -> None:
        """Explicit empty list clears the default; does NOT fall through."""
        from cwsandbox._defaults import _resolve_selector

        assert _resolve_selector([], ("x",)) == []

    def test_none_override_falls_back_to_nonempty_default(self) -> None:
        """Unset override resolves to the default when the default is non-empty."""
        from cwsandbox._defaults import _resolve_selector

        assert _resolve_selector(None, ("x",)) == ["x"]

    def test_empty_default_collapses_to_none(self) -> None:
        """When override is None and default is empty, result is None."""
        from cwsandbox._defaults import _resolve_selector

        assert _resolve_selector(None, ()) is None

    def test_both_none_returns_none(self) -> None:
        """When override and default are both None, result is None."""
        from cwsandbox._defaults import _resolve_selector

        assert _resolve_selector(None, None) is None

    def test_bare_string_override_raises(self) -> None:
        """A bare string override raises TypeError (would otherwise split into characters)."""
        from cwsandbox._defaults import _resolve_selector

        with pytest.raises(TypeError, match="override must be a sequence of strings"):
            _resolve_selector("prod", None)

    def test_bare_string_default_raises(self) -> None:
        """A bare string default raises TypeError (would otherwise split into characters)."""
        from cwsandbox._defaults import _resolve_selector

        with pytest.raises(TypeError, match="default must be a sequence of strings"):
            _resolve_selector(None, "prod")


class TestValidatePollConfig:
    """Tests for the _validate_poll_config helper.

    Ensures invalid NaN/inf/negative/zero values cannot reach the retry
    loop (where NaN would silently defeat the retry_deadline check).
    """

    @pytest.mark.parametrize(
        "bad_budget",
        [math.nan, math.inf, -math.inf, -1.0, -0.0001],
    )
    def test_rejects_invalid_budget(self, bad_budget: float) -> None:
        """Invalid poll_retry_budget_seconds values raise ValueError."""
        from cwsandbox._defaults import _validate_poll_config

        with pytest.raises(ValueError, match="poll_retry_budget_seconds"):
            _validate_poll_config(bad_budget, 15.0)

    def test_accepts_zero_budget(self) -> None:
        """Budget of 0 is accepted (callers opt out of retry)."""
        from cwsandbox._defaults import _validate_poll_config

        _validate_poll_config(0.0, 15.0)  # no raise

    def test_accepts_positive_budget(self) -> None:
        """Positive budget values pass."""
        from cwsandbox._defaults import _validate_poll_config

        _validate_poll_config(30.0, 15.0)  # no raise

    @pytest.mark.parametrize(
        "bad_rpc_timeout",
        [math.nan, math.inf, -math.inf, -1.0, -0.5, 0.0],
    )
    def test_rejects_invalid_rpc_timeout(self, bad_rpc_timeout: float) -> None:
        """Invalid poll_rpc_timeout_seconds values raise ValueError.

        Unlike budget, rpc_timeout must be strictly positive: a zero-timeout
        RPC would never complete.
        """
        from cwsandbox._defaults import _validate_poll_config

        with pytest.raises(ValueError, match="poll_rpc_timeout_seconds"):
            _validate_poll_config(30.0, bad_rpc_timeout)

    def test_accepts_positive_rpc_timeout(self) -> None:
        """Positive rpc_timeout values pass."""
        from cwsandbox._defaults import _validate_poll_config

        _validate_poll_config(30.0, 0.001)  # no raise, any positive finite value
