# CHANGELOG


## v0.18.0 (2026-04-20)

### Features

- **tests**: Pin e2e sandboxes to specific runner(s)
  ([`a8b7e41`](https://github.com/coreweave/cwsandbox-client/commit/a8b7e4149ae563d9b6d8282e1ce8b2066f128e00))

Engineers rolling out runner-side changes need to verify behavior against only the affected
  runner(s). Before this, `sandbox_defaults` left `runner_ids=None` and every e2e sandbox was
  auto-scheduled, forcing ad-hoc conftest edits to target a runner.

Add a pytest CLI flag `--cwsandbox-runner-ids=<csv>` and env var `CWSANDBOX_TEST_RUNNER_IDS` that
  feed into `sandbox_defaults.runner_ids`. CLI wins over env; passing the flag empty clears the env
  for a single invocation. A session-scoped autouse fixture calls `list_runners()` once to fail fast
  on unknown IDs, naming the source (CLI vs env) in the error. When no targeting is configured the
  fixture returns immediately - no discovery call, no added latency, byte-identical to the prior
  default.

Scope is intentionally narrow: only `runner_ids` is touched. `container_image`, `resources`, `tags`,
  and `max_lifetime_seconds` remain as they were.


## v0.17.0 (2026-04-17)

### Bug Fixes

- Update DEFAULT_BASE_URL from cw-sandbox.com to api.cwsandbox.com
  ([`f52364f`](https://github.com/coreweave/cwsandbox-client/commit/f52364ff2ca334ba6bc72bd93620f1509d712e94))

DNS migrated from gateway.cw-sandbox.com to api.cwsandbox.com as part of the cw-sandbox.com →
  cwsandbox.com domain consolidation.

### Documentation

- Remove local MkDocs and GitHub Pages infrastructure
  ([`c1c1058`](https://github.com/coreweave/cwsandbox-client/commit/c1c10582e5b4175455938f2c6b184bcc9d3182f5))

Remove the local MkDocs/GitHub Pages docs setup from this repo and point users to the canonical docs
  site instead.

This change: - removes MkDocs configuration and GitHub Pages publishing infrastructure - removes
  local tutorial/guide/API docs content that was only used by the MkDocs site - updates repository
  docs to point to https://docs.coreweave.com/products/coreweave-sandbox/client - keeps
  docs/README.md as a minimal pointer to the external docs site instead of a full landing page -
  removes docs-tooling lockfile entries tied to the old local docs build

API reference is no longer built in this repo with mkdocstrings. The replacement lives in
  coreweave/docs: the generate-cwsandbox-api-ref.yaml workflow generates Mintlify MDX API reference
  pages from cwsandbox source using Griffe for a given release tag and opens a PR in the docs repo.

### Features

- Add TERMINATING status and _Stopping lifecycle state
  ([`3f96b9a`](https://github.com/coreweave/cwsandbox-client/commit/3f96b9ac9ca1dc7c33c774954783947dd5045e7f))

Handle SANDBOX_STATUS_TERMINATING=9 from the backend proto. Add _Stopping lifecycle state, shared
  _stop_task pattern, and _stop_owned provenance for raise_on_termination. Bump vendored proto stubs
  from BSR.

- Rename aviato → sandbox (align with server rebrand)
  ([`b3027eb`](https://github.com/coreweave/cwsandbox-client/commit/b3027eb21fc900a85b5c42501fb00e62bab4f26b))

Update vendored proto stubs and SDK to match the server-side rename from aviato/v1beta1 to
  sandbox/v1beta2:

- Proto: atc_pb2 → gateway_pb2, ATCService → GatewayService, ATCStreamingService →
  GatewayStreamingService - Public API: tower_ids → runner_ids, runway_ids → profile_ids,
  tower_group_id → runner_group_id - CLI flags: --tower-id → --runner-id, --runway-id → --profile-id
  - All gRPC service paths updated to coreweave.sandbox.v1beta2.*

Generated stubs from aviato repo's `make buf-gen-python` output. All 658 unit tests pass.

- Rename discovery API (Tower/Runway -> Runner/Profile)
  ([`5a25996`](https://github.com/coreweave/cwsandbox-client/commit/5a259960c8dfd56b05f7d9dbf508753b1ff783f0))

Align public types and functions with v1beta2 proto rename.


## v0.16.0 (2026-04-07)

### Chores

- Add .worktreeinclude for Claude Code worktrees
  ([`a9c6a2d`](https://github.com/coreweave/cwsandbox-client/commit/a9c6a2d0b6336e2bd23ab0494e86d9e49906250b))

Ensures .env is automatically copied into new worktrees created by Claude Code, so auth credentials
  are available without manual setup.

### Continuous Integration

- Add commitlint workflow to validate PR commits
  ([`a2fdcfa`](https://github.com/coreweave/cwsandbox-client/commit/a2fdcfa12af4e1fa74d6904c99164a43561e482d))

The pre-commit hook enforces conventional commit messages locally, but CI only checked the PR title
  (via action-semantic-pull-request in lint-pr.yaml). This left a gap where non-conforming commit
  messages could land in main through squash-less merges.

Add a commitlint workflow that validates every commit in a PR against the same conventional commit
  types configured in .pre-commit-config.yaml. The workflow and config follow the established
  pattern used across other organization repositories.

### Features

- Add ResourceOptions type and K8s quantity parser
  ([`a635a04`](https://github.com/coreweave/cwsandbox-client/commit/a635a04cb633f8d6742481ff54cdc7c0f67b2e5e))

Support separate resource requests and limits for Burstable QoS pods, which is needed for the
  overcommit feature. GPU remains a top-level field since GPU overcommit is not supported by the
  backend.

New modules: - _quantity.py: Kubernetes quantity parser using stdlib Decimal, handles SI and binary
  suffixes with validation - _resources.py: Centralized normalization handling ResourceOptions,
  nested dicts, and flat dict backward compatibility

ResourceOptions is a frozen dataclass with requests, limits, and gpu fields, following the same
  immutable pattern as NetworkOptions and Secret.

Includes unit tests covering quantity parsing, validation, and resource normalization logic.

- Wire ResourceOptions through SDK entry points and gRPC
  ([`4f15e10`](https://github.com/coreweave/cwsandbox-client/commit/4f15e105dd21481b3fd742135c19e8673ae09cce))

Accept ResourceOptions alongside plain dicts for the resources parameter across all public entry
  points (Sandbox.__init__, Sandbox.run, Session.sandbox, Session.function, RemoteFunction,
  SandboxDefaults).

Flat resource dicts are normalized to Guaranteed QoS ResourceOptions via normalize_resources() in
  Sandbox.__init__. ResourceOptions instances map to the new resource_limits/resource_requests proto
  fields (31/32) while legacy dicts pass through to the old resources field for backward compat.

GPU configuration is mapped identically into both proto fields. Response properties
  (resource_limits, resource_requests) are extracted from StartSandboxResponse when the backend
  echoes them back.

SandboxDefaults.from_dict() preserves ResourceOptions instances rather than flattening them to plain
  dicts.


## v0.15.0 (2026-04-06)

### Chores

- **proto**: Add discovery stubs and bump BSR pins to 20260402
  ([`13cebcb`](https://github.com/coreweave/cwsandbox-client/commit/13cebcbed75a156f76ef33056340a47946db5159))

Add discovery_pb2.py, discovery_pb2.pyi, and discovery_pb2_grpc.py to the vendored proto directory
  for the new DiscoveryService (ListAvailableTowers, GetAvailableTower, ListRunways, GetRunway).

Bump BSR version pins from 20260306 (b06c7779a62f) to 20260402 (69e6fe070d98), which also updates
  the existing ATC stubs with upstream changes. Generalize validate_protobuf_version to check all
  *_pb2.py files instead of only atc_pb2.py.

### Features

- **discovery**: Add Discovery API for tower and runway introspection
  ([`45067d9`](https://github.com/coreweave/cwsandbox-client/commit/45067d9016bf4fa09df5ddb96798c40662e9383b))

Add module-level sync functions (list_towers, get_tower, list_runways, get_runway) that query the
  backend DiscoveryService via gRPC. Returns frozen dataclass types (Tower, TowerResources, Runway,
  IngressMode, EgressMode) with auto-pagination, error translation, and input validation.

Client-side filtering for ingress/egress modes and tower capacity is included as a workaround until
  backend support lands (coreweave/aviato#571, coreweave/aviato#572).

Includes TowerNotFoundError and RunwayNotFoundError exceptions, format_bytes/format_cpu utilities,
  77 unit tests, 23 integration tests, example script, and AGENTS.md documentation.

Closes #47


## v0.14.0 (2026-04-03)

### Documentation

- Update AGENTS.md for accuracy and best practices
  ([`4b692a1`](https://github.com/coreweave/cwsandbox-client/commit/4b692a132b57380a54f4bed8321dd9ae8d79aaa9))

Remove redundant "sleep infinity" from code examples - the SDK default command already keeps
  sandboxes alive. Fix stale documentation: update auth section to match pluggable auth mode system,
  correct cwsandbox.result() to cwsandbox.results(), add TerminalSession to Waitable type alias, and
  remove hardcoded test counts that go stale.

### Features

- Remove python<3.14 constraint ([#99](https://github.com/coreweave/cwsandbox-client/pull/99),
  [`a4775d8`](https://github.com/coreweave/cwsandbox-client/commit/a4775d8d77981a609878a4528189080a3517cdf5))

* feat: remove python<3.14 constraint

* chore: Add <4 to python version


## v0.13.0 (2026-04-02)

### Chores

- **ci**: Remove Cloudsmith publish step from release workflow
  ([`1efc31f`](https://github.com/coreweave/cwsandbox-client/commit/1efc31f95a21f930c978062eff7a88c27c7ad701))

The coreweave/actions OIDC actions can no longer be resolved, causing publish-cloudsmith to fail on
  every release since 2026-03-24. Now that packages are distributed via PyPI (#79), the Cloudsmith
  step is redundant.

### Features

- **auth**: Allow register auth mode
  ([`f381366`](https://github.com/coreweave/cwsandbox-client/commit/f381366c05892a5edf1ca104de0825f8524f97b4))

- **auth**: Register one active auth mode, no chain
  ([`efa9c23`](https://github.com/coreweave/cwsandbox-client/commit/efa9c23b368b37d08cfece1be01d2accdc91394c))

No more chain of auth modes. Use the registered auth mode directly.

- **auth**: Remove builtin wandb auth and update doc
  ([`fe6c474`](https://github.com/coreweave/cwsandbox-client/commit/fe6c47490050ed5444152c8e281dc99df31859c0))


## v0.12.2 (2026-03-31)

### Bug Fixes

- Correct team slug in CODEOWNERS
  ([`165a3f1`](https://github.com/coreweave/cwsandbox-client/commit/165a3f11537cfa9035d18442d9e9cee30bf144cc))

The CODEOWNERS file referenced @coreweave/appliedtraining which does not match the actual GitHub
  team slug (applied-training). This prevented automatic review assignment on PRs.

### Chores

- Add 7-day package cooldown via uv exclude-newer
  ([`ce9896a`](https://github.com/coreweave/cwsandbox-client/commit/ce9896ad8280b2e8158680fc63d3438145f2f85c))

Supply chain security measure that prevents uv from resolving packages published within the last 7
  days, giving the community time to detect and remove malicious versions before they reach our
  environment.


## v0.12.1 (2026-03-26)

### Bug Fixes

- Widen protobuf dependency to >=5,<7 ([#92](https://github.com/coreweave/cwsandbox-client/pull/92),
  [`b8e5e5a`](https://github.com/coreweave/cwsandbox-client/commit/b8e5e5aabe3fae56009caf33bc0511130e4ec7cb))

A customer reported that their training codebase uses protobuf 5 and hit a version conflict with
  cwsandbox requiring >=6.33.5. An audit of ~25 ML training frameworks shows protobuf v5 is the
  common ground for PyTorch-based stacks: most don't constrain protobuf at all, and the few that do
  (Composer <5.30, vLLM >=5.29.6) target v5.

Frameworks audited include DeepSpeed, Megatron-LM, TRL, Accelerate, Transformers, torchtitan, verl,
  NeMo, Composer, LLM Foundry, Axolotl, LLaMA-Factory, Unsloth, vLLM, SGLang, Ray, PyTorch
  Lightning, PEFT, OpenRLHF, W&B, TensorBoard, and others. Only TensorFlow HEAD requires protobuf
  v6+, but TF environments are separate from PyTorch training.

Use buf.build plugin version 26.1.0 (protobuf 5.26.1) to generate proto stubs that predate the
  ValidateProtobufRuntimeVersion check introduced in 5.27.0. This removes the runtime version
  pinning from generated stubs, allowing protobuf >=5,<7 as the dependency range. Both protobuf 5.x
  and 6.x pass the full unit test suite. A CI matrix job verifies compatibility with both major
  versions.


## v0.12.0 (2026-03-25)

### Bug Fixes

- **sandbox**: Always enqueue output sentinel on cancellation
  ([`e680f4c`](https://github.com/coreweave/cwsandbox-client/commit/e680f4c3751eb2793f06d8ad8ec1fec57818369f))

CancelledError is a BaseException, bypassing the except Exception handler. The `if not cancelled`
  guard in the finally block skipped the output_queue sentinel, leaving StreamReader consumers
  hanging forever. Remove the guard so the sentinel is always enqueued.

Fixes both _exec_streaming_async and _exec_streaming_tty_async.

### Documentation

- Add interactive shells guide
  ([`1ecd912`](https://github.com/coreweave/cwsandbox-client/commit/1ecd91289362164060488f425b358bafcab6ef2d))

Cover SDK shell() usage, terminal resize, stdin/stdout streaming, and a comparison table for
  choosing between exec and shell.

### Features

- **cli**: Add shell command
  ([`0838e2a`](https://github.com/coreweave/cwsandbox-client/commit/0838e2a15ef2c4232db9a63bffc8c923b69d2107))

Remote debugging and interactive workflows need direct terminal access to sandboxes without writing
  Python.

Add `cwsandbox sh` with raw mode, SIGWINCH resize, and terminal state restore on exit.

- **sandbox**: Add TTY exec and shell method
  ([`8ff5de4`](https://github.com/coreweave/cwsandbox-client/commit/8ff5de4106c47895f70297aa3ac14c2c2f5d8483))

Add _exec_streaming_tty_async for raw-byte TTY streaming with resize support, and the public shell()
  method that returns a TerminalSession. Widen _on_exec_complete to accept TerminalResult.

- **types**: Add terminal types
  ([`94e55a5`](https://github.com/coreweave/cwsandbox-client/commit/94e55a53e858b66074ae6a8a8bdf778bf00cb612))

Make StreamReader a Generic[_S] parameterized over str (text streams) and bytes (raw TTY output).
  Add TerminalResult and TerminalSession types for interactive TTY sessions. Update Waitable type
  alias and __all__ exports.

### Testing

- **sandbox**: Add TTY streaming tests
  ([`2602959`](https://github.com/coreweave/cwsandbox-client/commit/2602959d7ce458cb4dfafd9656a52d98447b0af5))

Cover _exec_streaming_tty_async paths: happy path, server errors, gRPC transport errors, stdin
  forwarding, resize, and early failure propagation. Fold TestShellCancellation into the new class
  with shared helpers to reduce setup duplication.

- **sandbox**: Add xfail test for cancelled TTY session output hang
  ([`8d665d8`](https://github.com/coreweave/cwsandbox-client/commit/8d665d87ebdc19fe3026adab742dd394ec2e12ff))

Demonstrates a bug where cancelling a TerminalSession future leaves the output StreamReader hanging
  forever. The CancelledError path in _exec_streaming_tty_async skips the output_queue sentinel (`if
  not cancelled`), so consumers block indefinitely.

The fix is to remove the `if not cancelled` guard so the sentinel is always enqueued. Cross-review
  confirmed no double-sentinel risk.

Marked xfail(strict=True) — will flip to XPASS once the guard is removed, signaling the marker
  should be dropped.


## v0.11.2 (2026-03-24)

### Bug Fixes

- Enforce keyword args on NetworkOptions and Secret
  ([#90](https://github.com/coreweave/cwsandbox-client/pull/90),
  [`60d8fd7`](https://github.com/coreweave/cwsandbox-client/commit/60d8fd72d1a9e14087e5ffcaaa247d1130b329e4))


## v0.11.1 (2026-03-24)

### Bug Fixes

- Enforce click version to be >=8.2 ([#89](https://github.com/coreweave/cwsandbox-client/pull/89),
  [`1139ba4`](https://github.com/coreweave/cwsandbox-client/commit/1139ba4b8ef0a7654e21bce843c27cfe9eca78ef))


## v0.11.0 (2026-03-24)

### Features

- Add secrets support on Session and Sandbox
  ([#78](https://github.com/coreweave/cwsandbox-client/pull/78),
  [`061b32d`](https://github.com/coreweave/cwsandbox-client/commit/061b32d465a6a16ff40e5a34982dbf1d428ddedd))

---------

Co-authored-by: NavarrePratt <npratt@coreweave.com>


## v0.10.0 (2026-03-24)

### Documentation

- Broken relative links in PyPI readme
  ([#80](https://github.com/coreweave/cwsandbox-client/pull/80),
  [`cc9a9d5`](https://github.com/coreweave/cwsandbox-client/commit/cc9a9d5c6ba3f34886741c71901a8209c9abcc8d))

### Features

- Add annotations support to Sandbox, Session, and Function
  ([`0dcac5d`](https://github.com/coreweave/cwsandbox-client/commit/0dcac5de7891b0abbcf3a0dfec20480e059c1008))

Wire Kubernetes pod annotations through the full SDK surface: SandboxDefaults, Sandbox.run(),
  Session.sandbox(), and @session.function(). Annotations merge with defaults where explicit keys
  win, matching the environment_variables pattern.

Sent as pod_annotations in the gRPC StartSandboxRequest. Backend does not yet echo annotations in
  responses (tracked separately), so e2e round-trip verification is deferred.


## v0.9.0 (2026-03-12)

### Features

- **ci**: Publish to pypi ([#79](https://github.com/coreweave/cwsandbox-client/pull/79),
  [`db69a2e`](https://github.com/coreweave/cwsandbox-client/commit/db69a2ef0c38ba12b4472835e5569eda5f741b14))

* feat(ci): publish to pypi

* chore(ci): remove uv install step for pypi publish job

* chore(ci): update job name


## v0.8.2 (2026-03-11)

### Bug Fixes

- **auth**: Allow wandb entity and project to be optional
  ([#76](https://github.com/coreweave/cwsandbox-client/pull/76),
  [`321a00b`](https://github.com/coreweave/cwsandbox-client/commit/321a00b515bdb055cff4f27c80525a9cab931cc7))

* Optionally send the x-entity-id and x-project-name headers when entity and project are specified
  in the environment

* Change the environment variable lookup from WANDB_ENTITY_NAME to WANDB_ENTITY

* Change the environment variable lookup from WANDB_PROJECT_NAME to WANDB_PROJECT


## v0.8.1 (2026-03-11)

### Bug Fixes

- Bump vendored protos to b06c7779a62f
  ([#77](https://github.com/coreweave/cwsandbox-client/pull/77),
  [`69aed76`](https://github.com/coreweave/cwsandbox-client/commit/69aed7617350c001c0baa6258d62793bf7fdd1bc))


## v0.8.0 (2026-03-10)

### Bug Fixes

- **cli**: Close stderr stream on exec failure
  ([`f68a6a8`](https://github.com/coreweave/cwsandbox-client/commit/f68a6a8f5e87d6481534f25ec5b31699397a9f85))

Use StreamReader.close() to unblock the stderr drain thread when stdout iteration raises, preventing
  the thread from hanging.

### Documentation

- Add logging guide and update CLI quickstart
  ([`22ee9b7`](https://github.com/coreweave/cwsandbox-client/commit/22ee9b7600b78f9417c896754f0b51882b37de5d))

Add docs/guides/logging.md covering stream_logs() usage patterns. Update CLI quickstart with logs
  command examples and cross-references.

- Standardize docstrings and export RemoteFunction
  ([`5188b37`](https://github.com/coreweave/cwsandbox-client/commit/5188b37cdabcf6411d75a62b6b62a1074093bae9))

Prepare public API docstrings for auto-generation in CW public docs. Add Attributes/Examples
  sections for structured parsing. Export RemoteFunction for type annotation support.

### Features

- **cli**: Add logs command
  ([`a9175ca`](https://github.com/coreweave/cwsandbox-client/commit/a9175ca2166c5289a0385c3b8fb4d9424db80c97))

Add `cwsandbox logs <sandbox-id>` for streaming container logs. Supports --follow, --tail, --since,
  and --timestamps options.

- **sandbox**: Add stream_logs method
  ([`613605c`](https://github.com/coreweave/cwsandbox-client/commit/613605c9da18a1f73295dc2b2d627feea7baa5da))

Add stream_logs() for streaming container logs via gRPC bidirectional streaming. Supports follow
  mode, tail lines, since-time filtering, and server-side timestamps. Uses bounded queues for
  backpressure in long-lived follow streams.

- **types**: Add StreamReader.close()
  ([`354b24d`](https://github.com/coreweave/cwsandbox-client/commit/354b24d78e2263752ed7345d23afaf6956ecf6ff))

Add close() method to StreamReader for tearing down background producers. Accepts an optional cancel
  callback invoked on first close; subsequent calls are idempotent.


## v0.7.0 (2026-03-09)

### Documentation

- Add CLI quickstart guide
  ([`e48aa6c`](https://github.com/coreweave/cwsandbox-client/commit/e48aa6cd3da3bada997154d0cab7179db2620b0d))

Cover installation, ls/exec usage, and JSON output for scripting.

### Features

- **cli**: Add exec command
  ([`d3b7b3d`](https://github.com/coreweave/cwsandbox-client/commit/d3b7b3db8c5c35ddd0514928e7a25bc9e169d850))

Add `cwsandbox exec <sandbox-id> <command>` for running commands in sandboxes with real-time
  stdout/stderr streaming. Supports --cwd and --timeout options.

Update conftest make_process helper to use real event loops for thread-safe stream iteration in
  tests.

- **cli**: Add list command
  ([`22935db`](https://github.com/coreweave/cwsandbox-client/commit/22935dbc75ea463e2927bd339392c93f0d85fbb6))

Add `cwsandbox ls` with table and JSON output formats. Supports filtering by --status, --tag,
  --runway-id, and --tower-id.

- **cli**: Scaffold CLI with Click entry point
  ([`51480d8`](https://github.com/coreweave/cwsandbox-client/commit/51480d894e7d619588993162d8d10cf9d6d77df2))

Managing sandboxes currently requires writing Python scripts for every operation. A CLI enables
  quick terminal workflows — listing, executing, and inspecting sandboxes — without boilerplate.

Add the cwsandbox CLI framework as an optional dependency with a Click group, __main__ entry point,
  and ImportError fallback when Click is not installed.


## v0.6.3 (2026-03-03)

### Bug Fixes

- Honor base_url when using `session.list()`
  ([`7600dd4`](https://github.com/coreweave/cwsandbox-client/commit/7600dd414521b26c3b8d976d726ee23c6f8a6635))

`Session` currently strongly asserts `base_url`, which gets passed to `Sandbox` as an explicit
  parameter.

In this pr, we set `Session` `base_url` to `None` when using the default, so `Sandbox` uses its
  evaluation logic to determine which value to use


## v0.6.2 (2026-02-27)

### Bug Fixes

- Update default base URL to atc.cw-sandbox.com
  ([`5ccee3e`](https://github.com/coreweave/cwsandbox-client/commit/5ccee3e97b107cc457c40195502a84bffaba3026))

The cwaviato.com domain is being retired in favor of cw-sandbox.com.


## v0.6.1 (2026-02-27)

### Bug Fixes

- Sync uv.lock during semantic-release version bump
  ([`fdac3c6`](https://github.com/coreweave/cwsandbox-client/commit/fdac3c68239df82a9b48405e8c1f2b482ec286b8))

The release workflow bumps pyproject.toml but never regenerates the lock file, leaving uv.lock stale
  after every release. This also caused uv to resolve the now-yanked grpcio 1.78.1.

Update build_command to run `uv lock && git add uv.lock` before `uv build` so the lock file is
  included in the release commit. Follows the pattern established in coreweave/bonectl.

Also refreshes uv.lock: resolves grpcio to 1.78.0 (1.78.1 was yanked for breaking gcloud serverless
  environments) and syncs the package version from 0.5.0 to 0.6.0.


## v0.6.0 (2026-02-27)

### Chores

- Add GitHub issue templates for bug reports and feature requests
  ([`10ddcc1`](https://github.com/coreweave/cwsandbox-client/commit/10ddcc1e331bc92a47c5c151199a52749ddd4df0))

Structured YAML-based issue forms to help external contributors file well-structured reports once
  the repo goes public.

### Refactoring

- Vendor proto stubs, remove buf.build registry dependency
  ([`1fd6bb5`](https://github.com/coreweave/cwsandbox-client/commit/1fd6bb5133bae0dd51b2a6b9b67dc117082ff29c))

Vendor the generated protobuf/gRPC stubs directly into src/cwsandbox/_proto/ so the published wheel
  is self-contained. Users install from a single package index with no buf.build configuration
  required.

- Vendor atc and streaming proto stubs into src/cwsandbox/_proto/ - Rewrite all imports from
  coreweave.aviato.v1beta1 to cwsandbox._proto - Replace coreweave-aviato-grpc-python (buf.build)
  with googleapis-common-protos (PyPI) for google.api imports - Bump protobuf pin to >=6.33.5 to
  match generated code - Add scripts/update-protos.sh for refreshing vendored stubs - Update
  mise.toml dev tasks and DEVELOPMENT.md

BREAKING CHANGE: The coreweave.aviato.v1beta1 import path is no longer available. Use
  cwsandbox._proto for direct proto access (internal API).

### Breaking Changes

- The coreweave.aviato.v1beta1 import path is no longer available. Use cwsandbox._proto for direct
  proto access (internal API).


## v0.5.0 (2026-02-27)

### Refactoring

- Rename package from aviato to cwsandbox
  ([`59bc899`](https://github.com/coreweave/cwsandbox-client/commit/59bc899e3aa575f2c76629fb79d4b0257dc73d20))

Rename the Python SDK from "Aviato" to "CoreWeave Sandbox" (cwsandbox) to align with CW product
  naming for public release.

- Move src/aviato/ to src/cwsandbox/ and all test directories - Rename public identifiers: imports,
  env vars (CWSANDBOX_API_KEY, CWSANDBOX_BASE_URL), exceptions (CWSandboxError hierarchy), W&B
  metric namespace (cwsandbox/*) - Update 67 SPDX headers from aviato-client to cwsandbox-client -
  Update REUSE.toml, CI workflows, mkdocs config, mise task runner - Update all documentation and
  examples

Backend references unchanged: atc.cwaviato.com endpoint, coreweave-aviato-grpc-python proto package,
  coreweave.aviato.v1beta1 proto imports.

BREAKING CHANGE: all imports, env vars, and exception names changed. No backward compatibility
  layer.

### Breaking Changes

- All imports, env vars, and exception names changed. No backward compatibility layer.


## v0.4.0 (2026-02-26)

### Bug Fixes

- **sandbox**: Add lock for exec statistics
  ([`d281eb4`](https://github.com/coreweave/cwsandbox-client/commit/d281eb486cb06b14ba07c429b463482fe580b803))

Concurrent callbacks from the background event loop can corrupt _exec_count and related counters
  when multiple operations complete simultaneously.

Protect all exec statistics with a threading.Lock.

- **sandbox**: Cache shutdown task in stdin loop
  ([`5b9fa78`](https://github.com/coreweave/cwsandbox-client/commit/5b9fa7816f6f950a06f2032de61c44f9b064376d))

The shutdown_task was recreated every iteration of the stdin forwarding loop, wasting resources.
  Worse, cancelling and recreating it could miss the event being set between iterations.

Cache the task and skip cancelling it in the pending-task cleanup.

### Chores

- Update ruff pre-commit hook to v0.14.10 and fix formatting
  ([`430311b`](https://github.com/coreweave/cwsandbox-client/commit/430311ba49d147e55bdf03efe53b9115483acb0b))

Align pre-commit ruff version with project dependency (v0.8.3 → v0.14.10) to resolve formatting
  disagreements between pre-commit and CI checks.

### Features

- **types**: Propagate exceptions in StreamReader
  ([`3b94574`](https://github.com/coreweave/cwsandbox-client/commit/3b94574ef3e2bad0f7a70d0a2eef2d31614b4f41))

StreamReader can only yield values or signal completion. Background producers have no way to
  communicate errors — they must silently drop them.

Widen the queue type from `str | None` to `str | Exception | None`. When an Exception is dequeued,
  it is re-raised to the consumer and the reader is marked exhausted.

### Refactoring

- **sandbox**: Widen exec queue types
  ([`ab79d7a`](https://github.com/coreweave/cwsandbox-client/commit/ab79d7af52225c9fbe9164e47ff4348e072e0fb5))

StreamReader now propagates exceptions from the queue (6e64f5e). The exec queue type annotations
  must match to allow exceptions to flow through to the consumer.


## v0.3.0 (2026-02-25)

### Documentation

- Update get_status() for terminal sandbox caching
  ([`00f588a`](https://github.com/coreweave/cwsandbox-client/commit/00f588afc426827f1b47474ea60261572524eded))

Reflect that get_status() now returns cached status for terminal sandboxes (COMPLETED, FAILED,
  TERMINATED) without an RPC, since terminal states are immutable.

### Features

- **sandbox**: Add _LifecycleState sealed type and transition helpers
  ([`27c3c23`](https://github.com/coreweave/cwsandbox-client/commit/27c3c2370f55497954ec7e8430ab1733db911a04))

Introduce frozen dataclass variants for each sandbox lifecycle phase: _NotStarted, _Starting,
  _Running, and _Terminal. These will replace the four independent instance variables (_sandbox_id,
  _status, _stopped, _returncode) with a single _state field that makes invalid combinations
  unrepresentable.

Add _lifecycle_state_from_info() builder, _apply_sandbox_info() for protobuf response parsing with
  monotonicity guard and source-aware returncode handling, _is_done property, and
  _raise_or_return_for_terminal helper.

Includes comprehensive unit tests (test_lifecycle.py) covering all state variants, transitions,
  guards, and edge cases.

### Refactoring

- **sandbox**: Replace lifecycle variables with _LifecycleState
  ([`cc327e5`](https://github.com/coreweave/cwsandbox-client/commit/cc327e5c567b044167201bae0dbc0344d3227ca5))

Replace four independent instance variables (_sandbox_id, _status, _stopped, _returncode) with a
  single self._state: _LifecycleState field.

Key changes: - Rewrite _from_sandbox_info to route through _apply_sandbox_info - Consolidate five
  status-setting locations into _apply_sandbox_info - Migrate all 17 _stopped guard sites to
  _is_done property checks - Migrate all 3 _stopped write sites to state transitions - Update
  property accessors to extract from _state with fallbacks - Unlock get_status() for terminal
  sandboxes (returns cached, no RPC) - Add fast-path terminal checks in wait methods - Delete
  _stopped workaround and its TODO comment - Remove google.protobuf.timestamp_pb2 import (now
  handled internally)

The _stopped boolean was a client-side workaround because the backend previously discarded sandbox
  records after stop(). Now that the backend persists terminal states, the client queries
  authoritative status directly. Invalid state combinations (e.g. status=RUNNING + stopped=True) are
  structurally impossible with the sealed type.

No public API changes. All 527 unit tests pass.


## v0.2.0 (2026-02-24)

### Bug Fixes

- **e2e**: Fix session terminal status filter test
  ([`a67c923`](https://github.com/coreweave/cwsandbox-client/commit/a67c9234c9259f1294cae5352a6a3bfa176dfa83))

Create sandbox outside session context manager so session.close() doesn't call stop() on it, which
  would change the status from COMPLETED to TERMINATED — causing the status="completed" filter to
  miss it.

- **e2e**: Relax session status assertion and add terminal filter test
  ([`6e50509`](https://github.com/coreweave/cwsandbox-client/commit/6e50509c0133714ed913b1685538bdd2f41d0e53))

- Relax include_stopped status assertion in session test to match sandbox test — DB records may have
  stale status - Add test_session_list_terminal_status_filter for parity with
  test_sandbox_list_terminal_status_filter

- **e2e**: Relax status assertions for DB-backed sandbox records
  ([`528fb5c`](https://github.com/coreweave/cwsandbox-client/commit/528fb5cb4fedec4c45cf488ad3447d581ca1f96d))

The backend DB fallback may return stale or unspecified statuses: - List with include_stopped
  returns the DB record's status at write time (e.g. "creating"), not the final terminal status -
  from_id DB fallback may return UNSPECIFIED for the status field

Relax assertions to match actual backend behavior.

- **list**: Address PR review feedback for include_stopped
  ([`db4e72a`](https://github.com/coreweave/cwsandbox-client/commit/db4e72a66b14590a24cf3bd0ee7d54fb39c670a9))

- Bump coreweave-aviato-grpc-python to 1.78.1 which includes the include_stopped proto field (field
  6) on ListSandboxesRequest - Remove try/except fallback and set include_stopped directly via
  request_kwargs instead of post-construction assignment - Fix docstring: replace "or a terminal
  status filter is used" with accurate description of backend behavior - Make unit tests
  unconditionally assert on include_stopped field instead of conditionally checking if the proto
  field exists - Add list_stopped_sandboxes.py example script demonstrating the include_stopped
  parameter with both Sandbox and Session APIs

- **list**: Address second round of PR review feedback
  ([`6348615`](https://github.com/coreweave/cwsandbox-client/commit/6348615aa48d5e7b4d92fc7d5887a266482333a8))

- Use definitive docstring wording for terminal status auto-widening in both Sandbox.list() and
  Session.list() — the backend always widens the DB query for terminal status filters - Add --create
  step to list_stopped_sandboxes.py example so users can create sandboxes first, then observe them
  with --include-stopped

- **sandbox**: Mark terminal sandboxes as stopped in _from_sandbox_info
  ([`69e21db`](https://github.com/coreweave/cwsandbox-client/commit/69e21dbf68b59eab6cfb9c025063ffa2bb9bf173))

Sandboxes reconstructed via _from_sandbox_info (from list/get responses) now have _stopped=True when
  their status is terminal (completed, failed, terminated). This prevents unnecessary stop RPCs when
  session.close() cleans up adopted terminal sandboxes.

### Code Style

- Apply ruff formatting to example and integration tests
  ([`bae3e31`](https://github.com/coreweave/cwsandbox-client/commit/bae3e3111e1627d0da1c9a6869ec7a9f355bd653))

- Apply ruff formatting to list_stopped_sandboxes example
  ([`96a2514`](https://github.com/coreweave/cwsandbox-client/commit/96a251462607221c967aff5c85cbffb2ebb8133a))

### Documentation

- Remove architecture-revealing language from integration tests
  ([`b6742a5`](https://github.com/coreweave/cwsandbox-client/commit/b6742a541aaee1caa2beca1268e0f633d6a5c285))

Remove references to DB, cache, and other internal architecture details from test comments and
  AGENTS.md to prepare for public repository.

- Remove architecture-revealing language from public-facing code
  ([`a62df68`](https://github.com/coreweave/cwsandbox-client/commit/a62df688c26f89cdbe1c1164051793b62700ae77))

Remove references to "persistent storage" from docstrings and examples to avoid leaking internal
  architecture details in the public repository.

### Features

- **list**: Add include_stopped parameter to Sandbox.list() and Session.list()
  ([`d61c6f1`](https://github.com/coreweave/cwsandbox-client/commit/d61c6f1a195856bc57917bdaee16840286e4ef91))

The backend now excludes terminal sandboxes (completed, failed, terminated) from default List
  results. Add include_stopped parameter (default False) that tells the server to include terminal
  sandboxes from persistent DB storage.

Handles forward-compatibility with the proto package: the include_stopped field (proto field 6) is
  set via try/except since the published proto package may not yet include it.

### Testing

- **e2e**: Add integration tests for DB-backed List/Get behavior
  ([`c0180db`](https://github.com/coreweave/cwsandbox-client/commit/c0180db79ad9ae4261d47c56bbb5b401ca8c689c))

Add e2e tests validating backend changes from aviato PRs #242-248: - Default list excludes terminal
  sandboxes (completed/failed/terminated) - list(include_stopped=True) returns terminal sandboxes
  from persistent DB - Terminal status filter (e.g., status="completed") returns stopped sandboxes -
  from_id() returns stopped sandboxes via DB fallback (was NOT_FOUND before) -
  Session.list(include_stopped=True) passes parameter through correctly

Also update AGENTS.md to document the new include_stopped parameter and the updated from_id()
  behavior for stopped sandboxes.


## v0.1.1 (2026-02-18)

### Bug Fixes

- **examples**: Read sandbox_id after auto-start in swebench runner
  ([`1bbee6c`](https://github.com/coreweave/cwsandbox-client/commit/1bbee6cfe80c446a7e43f16932a55d846bb60c02))

session.sandbox() returns an unstarted sandbox with no ID yet. Move the sandbox_id read to after
  write_file, which triggers auto-start and populates the ID from the backend.


## v0.1.0 (2026-02-18)

### Bug Fixes

- Address PR 36 review feedback
  ([`f39a95a`](https://github.com/coreweave/cwsandbox-client/commit/f39a95af7c2c67f24ef81a1a622df2a7d1839af0))

- Remove unused TYPE_CHECKING import and empty conditional block - Replace fixed time.sleep(3) with
  polling loop for W&B API sync (10 attempts at 1s intervals, exits early when data available) - Add
  netrc support to has_wandb_credentials() by reusing _read_api_key_from_netrc() from _auth.py

- Update exec() to handle raw bytes stdout/stderr from protobuf change
  ([#9](https://github.com/coreweave/cwsandbox-client/pull/9),
  [`1786292`](https://github.com/coreweave/cwsandbox-client/commit/1786292cbba960d15eddbf44a399f84ca9de6069))

- **AGENTS**: Reference AGENTS.md and not CLAUDE.md
  ([`10ec479`](https://github.com/coreweave/cwsandbox-client/commit/10ec479a9ea530198289d81b473baaefee100ba8))

- **auth**: Pass gRPC auth metadata directly, remove interceptors
  ([`095def9`](https://github.com/coreweave/cwsandbox-client/commit/095def90e6e5c05b29b48fe16405b551eec4176c))

gRPC auth interceptors silently fail when nest_asyncio patches the event loop, causing 403 errors in
  ART training. The streaming exec code already worked around this by passing metadata directly.

Replace interceptor-based auth with direct metadata passing on all gRPC calls. Add
  resolve_auth_metadata() helper that returns metadata as lowercase key-value tuples ready for gRPC.
  Remove AuthInterceptor, create_auth_interceptors(), and the interceptors parameter from
  create_channel().

### Build System

- **deps**: Replace ConnectRPC with gRPC transport
  ([`d8c285a`](https://github.com/coreweave/cwsandbox-client/commit/d8c285a78ce6292d6538ba47a933a54710b14d61))

Switch from coreweave-aviato-connectrpc-python + httpx + h2 to coreweave-aviato-grpc-python +
  grpcio. The gRPC SDK provides native bidirectional streaming needed for stdin support, eliminating
  the HTTP/2 workaround that ConnectRPC required.

Add grpc-stubs (type stubs) and httpx (integration test networking) to dev dependencies.

- **deps**: Update gRPC proto package to 1.78.0.1
  ([`343ac1b`](https://github.com/coreweave/cwsandbox-client/commit/343ac1b48919d0cc297cb45f739ebdc12498c9fe))

Update coreweave-aviato-grpc-python from 1.76.0.2 to 1.78.0.1 which includes the merged stdin
  streaming protos (ExecStreamRequest stdin/ close/resize oneofs, ExecStreamResponse ready signal).

Validated against staging with tower v0.14.1: - 434 unit tests pass - 72 integration tests pass
  (including all 19 stdin streaming tests) - All 14 example scripts pass

- **deps**: Upgrade connectrpc to 0.8.1, add httpx and protobuf
  ([`f70efbf`](https://github.com/coreweave/cwsandbox-client/commit/f70efbf103bef30b352ae9bcd4c2f504d391e718))

Upgrade coreweave-aviato-connectrpc-python from 0.6.0 to 0.8.1 which includes the new
  interceptor-based auth pattern and pyqwest HTTP client. Add httpx and protobuf as explicit
  dependencies.

### Chores

- Add applied training as codeowners ([#25](https://github.com/coreweave/cwsandbox-client/pull/25),
  [`ee76ffb`](https://github.com/coreweave/cwsandbox-client/commit/ee76ffb267b9b94068064dde43e3e1357c9a3aff))

- Add wandb optional dependency and tooling config
  ([`d1b901c`](https://github.com/coreweave/cwsandbox-client/commit/d1b901cafe08437d492b871f4cdf2742670949ae))

- Add wandb as optional dependency in [project.optional-dependencies] - Update uv.lock with wandb
  and transitive dependencies - Add wandb/ directory to gitignore (local run artifacts)

- Apply ruff formatting
  ([`89614ec`](https://github.com/coreweave/cwsandbox-client/commit/89614ec4608400a204de61bf3a8c54a82f2e9239))

Run ruff format to ensure consistent code style across the codebase.

- Update project configuration and documentation
  ([#22](https://github.com/coreweave/cwsandbox-client/pull/22),
  [`c36e8b8`](https://github.com/coreweave/cwsandbox-client/commit/c36e8b843e3031ea20598c52a21c2160fb06ff0c))

* chore: update project configuration and documentation

Add mise.toml for task running: - mise run check: format, lint, typecheck, test - mise run test:e2e:
  integration tests - mise run test:e2e:parallel: parallel integration tests

Update READMEs and CLAUDE.md files for AI-assisted development. Update project metadata and
  .gitignore patterns.

* refactor(api): update all .get() usages to .result()

Complete the OperationRef.get() to .result() migration across the entire codebase:

- Update src/aviato/_session.py and _function.py source code - Update all docstrings to reference
  .result() instead of .get() - Update integration tests to use .result() - Update unit tests
  including mock assertions - Update all example scripts - Update all README files and CLAUDE.md
  documentation - Update docs/guides/ documentation files

This is the final branch in the refactoring stack, completing the migration to a consistent
  .result() API for blocking on OperationRef values. Process now inherits from OperationRef, so both
  use .result().

* docs: update CLAUDE.md files for aviato.result() rename

Update documentation to reflect the rename of aviato.get() to aviato.result() for API consistency.

* docs: remove references to deleted example files

Remove stale references to list_sandboxes.py and kwargs_validation_demo.py from the examples
  documentation, as these files no longer exist.

* docs: rename CLAUDE.md to AGENTS.md with symlinks

Migrate to tool-agnostic AGENTS.md naming convention while maintaining backwards compatibility via
  symlinks. This addresses PR feedback about not over-indexing on Claude-specific tooling.

Changes: - Rename all 5 CLAUDE.md files to AGENTS.md - Create CLAUDE.md -> AGENTS.md symlinks for
  backwards compatibility - Update header and intro to be tool-agnostic - Update internal
  cross-references to use AGENTS.md

- Update protos for 20251229 update ([#14](https://github.com/coreweave/cwsandbox-client/pull/14),
  [`0a22945`](https://github.com/coreweave/cwsandbox-client/commit/0a22945cef4aed8c18f22f18fe22a416175db4f5))

- **dev**: Add local proto development and lint tasks
  ([`29eb7ec`](https://github.com/coreweave/cwsandbox-client/commit/29eb7ec70036db27f3314a84af513974f8e5cbdd))

Add mise tasks for working with locally-generated protos from the backend repo (dev:local-protos,
  dev:revert-protos) and no-sync variants of lint/typecheck/test that preserve local proto installs.

Also add top-level lint, typecheck, and check tasks as convenience aliases.

- **docs**: Add SWEBench guide to docs site
  ([#33](https://github.com/coreweave/cwsandbox-client/pull/33),
  [`dc94c55`](https://github.com/coreweave/cwsandbox-client/commit/dc94c55933b34d445510476c3970302e43ad9893))

* docs: add SWE-bench guide to mkdocs navigation

The SWE-bench evaluation guide was added in commit 29df2b8 but was not included in the mkdocs.yml
  nav, so it did not appear in published docs.

* docs: add mkdocs.yml update guidance to AGENTS.md

Remind contributors to update mkdocs.yml nav when adding new documentation files to prevent
  omissions in site navigation.

- Add note in AGENTS.md Design Documentation section - Add step 7 to docs/guides/AGENTS.md Writing
  New Guides

- **mise**: Simplify task configuration
  ([`add0e96`](https://github.com/coreweave/cwsandbox-client/commit/add0e96be01a83c552da48d777199dff618bde91))

The check task ran sequentially via inline commands, and install used the older uv pip workflow.

Use depends for parallel task execution in check. Update install to use uv sync which is the
  idiomatic uv approach. Remove redundant ci task alias.

### Code Style

- Fix unsorted import block in wandb example
  ([`7caba13`](https://github.com/coreweave/cwsandbox-client/commit/7caba13357d14fc1699ab52b27d496d41ccfcb0f))

### Continuous Integration

- Add release pipeline, CI checks, and commit linting
  ([`fc5ee6c`](https://github.com/coreweave/cwsandbox-client/commit/fc5ee6cbfc14913ba297765d550a33083cda7b6d))

Release pipeline (.github/workflows/release.yaml): - Two-job workflow triggered on push to main -
  Job 1 (release): app token, version bump, build, GitHub Release - Job 2 (publish-cloudsmith): OIDC
  token, publish to Cloudsmith - Concurrency group prevents race conditions from rapid pushes

CI workflow (.github/workflows/ci.yaml): - Single-job PR quality gate: format, lint, typecheck, unit
  tests - Uses .python-version file for consistent Python 3.11

Commit message enforcement: - conventional-pre-commit hook validates local commit messages -
  lint-pr.yaml validates PR titles via action-semantic-pull-request - Allowed types match
  semantic-release config

Foundation: - .python-version: pin Python 3.11 - pyproject.toml: requires-python bumped to
  >=3.11,<3.14 - src/aviato/__init__.py: add __version__ for runtime access - pyproject.toml:
  python-semantic-release config with build_command

### Documentation

- Add code fences to docstring examples
  ([`ed445e4`](https://github.com/coreweave/cwsandbox-client/commit/ed445e47fe8ac270d63f53bf4b070ebc6ce32e67))

Docstring examples rendered as plain text in mkdocstrings because they lacked explicit code block
  markers. This made the API reference harder to read and prevented syntax highlighting.

Wrap all docstring examples in ```python fences so mkdocstrings renders them with proper formatting
  and syntax highlighting.

- Add contributing guidelines and commit standards
  ([`7499eba`](https://github.com/coreweave/cwsandbox-client/commit/7499eba2d02c6d247239d80ea445a1d7c568e0bf))

Establish formal contribution guidelines to set clear expectations for code quality, testing,
  documentation, and commit practices. These documents provide a foundation for maintaining
  consistency as the project grows and more contributors join.

The commit guidelines enforce atomic commits and conventional commit format, which will improve code
  review efficiency and make the project history more navigable.

- Add DEVELOPMENT.md and consolidate dev documentation
  ([`c9f8d17`](https://github.com/coreweave/cwsandbox-client/commit/c9f8d1703df188eb8b8818ec1f810227f857da92))

Development instructions were scattered across AGENTS.md, README.md, and required manual discovery.
  New contributors had to piece together setup steps from multiple files.

Create a dedicated DEVELOPMENT.md with comprehensive setup, workflow, and testing documentation.
  Update other files to reference it, reducing duplication and providing a single source of truth
  for development onboarding.

- Add MkDocs documentation site
  ([`4980f43`](https://github.com/coreweave/cwsandbox-client/commit/4980f43a04fefcda84f00a1f39282aa389ed0586))

The project lacked hosted documentation, making it difficult for users to discover the API and
  understand usage patterns without reading source code directly.

Set up MkDocs with Material theme and mkdocstrings to auto-generate API documentation from
  docstrings. A GitHub Actions workflow deploys to GitHub Pages on pushes to main.

- Add RL training guide and examples
  ([`c164565`](https://github.com/coreweave/cwsandbox-client/commit/c16456531a014806589be1ccf60268439b465ca1))

Add comprehensive RL training guide covering code execution rewards, TRL GRPO integration, and
  multi-step agent rollouts with ART (Agent Rollout Trainer). Includes runnable examples:
  reward_function.py for basic sandbox-based reward computation, trl_grpo_integration.py for TRL
  trainer integration, and the full ART framework with rollout engine, tool execution, and tests.

- Add sandbox lifecycle guide
  ([`2560d57`](https://github.com/coreweave/cwsandbox-client/commit/2560d5781c405ee89fc76d7c55bb2ed3b0331bbd))

Covers sandbox lifecycle states with mermaid diagram, creation patterns (Sandbox.run vs
  session.sandbox), auto-start behavior, waiting strategies, stop vs delete semantics, reconnection
  caveats, and under-the-hood architecture.

Positioned first in the Guides nav as the hub guide that links to sync-vs-async, sessions,
  cleanup-patterns, and troubleshooting.

- Add SPDX license header policy to CLAUDE.md
  ([`726cb59`](https://github.com/coreweave/cwsandbox-client/commit/726cb59ca0b8a7b7fe9777b7692966dad9f9edf6))

Agents creating new files were skipping license headers because CLAUDE.md had no instructions about
  them. Add a "License Headers" section documenting which license applies where, header formats for
  Python and Markdown files, and the reuse lint validation command. References CONTRIBUTING.md for
  the full policy.

- Add tutorial section for onboarding
  ([`cc9d7e4`](https://github.com/coreweave/cwsandbox-client/commit/cc9d7e425194e66cf6a56f2f72c2546437898fb0))

Add 8-part tutorial covering sandbox basics through cleanup, and update mkdocs.yml navigation.

- Add wandb integration example
  ([`3df65b7`](https://github.com/coreweave/cwsandbox-client/commit/3df65b76ce6a8026b1577e4caf8de7918fc5038a))

Add wandb_integration.py demonstrating: - Session with report_to="wandb" for automatic metrics -
  Exec statistics tracking - Custom metric logging via session.log_metrics()

Update examples/AGENTS.md with example documentation.

- Rename sandbox-config.md to sandbox-configuration.md
  ([`a88c8df`](https://github.com/coreweave/cwsandbox-client/commit/a88c8df396cf50539e794ed9a343817c78405f9f))

- Rewrite sync-vs-async guide with tabbed pairs
  ([`1a2cd24`](https://github.com/coreweave/cwsandbox-client/commit/1a2cd240f5cbda4fa8df753f9adf7697f74e35fd))

Restructure the guide to show consistent sync/async pairs for every SDK operation using
  pymdownx.tabbed content tabs. Each operation now has a side-by-side Sync/Async tab showing
  .result() vs await patterns.

Key changes: - Add pymdownx.tabbed extension to mkdocs.yml - Add OperationRef core concept section
  with event loop warning - Add auto-start behavior documentation - Exhaustive tabbed pairs for all
  operations: sandbox lifecycle, exec, file I/O, discovery (list/from_id/delete), session methods,
  streaming, stdin, and remote functions - Clarify Sandbox.run() blocks (calls start().result()), so
  async code should use Sandbox() constructor + async with instead - Document sync-only APIs:
  wait(), get_status(), aviato.results(), aviato.wait() - Update parallel execution section with
  session.sandbox() and collected start() ref patterns - Add error handling section linking to
  troubleshooting guide

- **exec**: Add stdin streaming documentation and examples
  ([`5e1f3e2`](https://github.com/coreweave/cwsandbox-client/commit/5e1f3e28d5edb70f256ef105cfcee4a7b3895b5c))

Add stdin streaming section to the execution guide covering StreamWriter methods, multiple writes,
  interactive Python, combined stdin/stdout streaming, EOF-dependent commands, and async usage.

Add examples/stdin_streaming.py with six sync demos and one async demo. Update AGENTS.md for gRPC
  backend, StreamWriter type, and stdin parameter on exec().

- **guides**: Fix code examples to use correct API patterns
  ([`82c99fa`](https://github.com/coreweave/cwsandbox-client/commit/82c99fab0140548175bd07f020b0d3849fc1f8bc))

Update guide examples to properly import and use SandboxDefaults

### Features

- Add authentication mechanism for W&B SaaS
  ([`74e1d3b`](https://github.com/coreweave/cwsandbox-client/commit/74e1d3be5ebcfe2693b573e947b813e544ca7b42))

- Add environment variable support
  ([`44ccc7e`](https://github.com/coreweave/cwsandbox-client/commit/44ccc7e14be84d106a6787e2e0b3cf579263e868))

- Add environment variable support ([#6](https://github.com/coreweave/cwsandbox-client/pull/6),
  [`580862d`](https://github.com/coreweave/cwsandbox-client/commit/580862d60c033afe7d38a28c3c1c435af5417f2e))

- Initial implementation of the aviato Python SDK
  ([#1](https://github.com/coreweave/cwsandbox-client/pull/1),
  [`e860000`](https://github.com/coreweave/cwsandbox-client/commit/e86000055d036242fc6ea6b516a1046149c505d9))

* feat: Initial implementation of the aviato Python SDK

Python client library for creating and managing Aviato sandboxes with: - Sandbox lifecycle
  management (create, exec, file I/O, stop) - Session-based sandbox pooling - @session.function()
  decorator for remote Python execution - Async-first API with context manager support

Includes unit/integration tests, examples, and documentation.

* fix(docs): Update to use new results formatting

* fix(sandbox): Handle container-image defaults through create()

- SandboxDefaults container image was being overridden by the default env var if you didn't pass
  container_image directly

* fix(sandbox): Use 'stopped' flag in del warning

* fix(examples): Remove nonexistent 'write_file' return value

* fix(docs): Use ExecResult's stderr for text format

* fix(function): Match only on '@<obj>.function()' decorators

* fix(session): Use exception group when closing sandboxes

* feat: Add FunctionSerializationError with verbose message

* chore: Relax python env requirements

* fix: Add buffer to client-side timeout on exec calls

User supplied timeout is given to the backend, we want that to trigger only when the command takes
  not to long. The user shouldn't need to account for round trip time

* fix(session): 'zip' with 'strict=True' when closing

- **auth**: Add AuthHeaderInterceptor for connectrpc auth
  ([`c2cb5f1`](https://github.com/coreweave/cwsandbox-client/commit/c2cb5f113000f9537090aa6af45ee08fd07844ae))

Add _AuthHeaderInterceptor class and create_auth_interceptors() function to support the new
  connectrpc interceptor-based authentication pattern. This replaces the previous httpx
  session-based header injection.

- **examples**: Add SWE-bench evaluation example
  ([#30](https://github.com/coreweave/cwsandbox-client/pull/30),
  [`29df2b8`](https://github.com/coreweave/cwsandbox-client/commit/29df2b8e23f85bc49ae56f0c6cab4005621411d4))

Add a complete example demonstrating how to run SWE-bench evaluations using Aviato's parallel
  sandbox execution. The script uses Session and ThreadPoolExecutor to run multiple evaluation
  instances concurrently.

Key features: - Pulls pre-built images from Epoch AI's GHCR registry - Supports gold patches for
  validation or custom predictions - Configurable parallelism, timeout, and dataset selection -
  Produces SWE-bench compatible report files - Per-instance sandbox cleanup with fire-and-forget
  stop pattern

- **exec**: Add stdin streaming support to exec
  ([`d5a751f`](https://github.com/coreweave/cwsandbox-client/commit/d5a751f6b3bf21b90de3f68d08f2f338d6d8d25c))

Enable sending input to running commands via exec(stdin=True). When enabled, Process.stdin provides
  a StreamWriter with write(), writeline(), and close() methods backed by a bounded asyncio.Queue
  for backpressure.

The request_generator coordinates with the server through a ready/shutdown event protocol: it waits
  for a StreamingExecReady response before sending stdin data, and terminates when the process exits
  or the shutdown event fires. Large writes are chunked into 64KB pieces. The generator's natural
  completion signals gRPC to half-close the send direction.

- **exec**: Migrate Sandbox RPCs from ConnectRPC to gRPC
  ([`e47adfa`](https://github.com/coreweave/cwsandbox-client/commit/e47adfab501145e623ebfb2473e2c3238443936a))

Replace ConnectRPC client with gRPC channel/stub pattern across all Sandbox operations. Add
  _translate_rpc_error() to map gRPC status codes to the existing Aviato exception hierarchy.

Key changes: - Unary RPCs (Start, Get, Stop, List, Delete, RetrieveFile, AddFile) now use
  ATCServiceStub with per-call timeout - Streaming exec uses ATCStreamingServiceStub with a
  dedicated channel and request iterator pattern for proper half-close semantics - Auth metadata
  passed directly to streaming calls (interceptors do not work with request iterators in gRPC) -
  Streaming channel created lazily with double-checked locking

- **function**: Report sandbox creation to session reporter
  ([`1496526`](https://github.com/coreweave/cwsandbox-client/commit/1496526643d03fc1a4eb733f65f5a9c0d0c83ea6))

Wire RemoteFunction to report sandbox creation events through the session reporter when executing
  remote functions.

- **network**: Add gRPC channel management and rewrite auth interceptors
  ([`c9d1339`](https://github.com/coreweave/cwsandbox-client/commit/c9d1339b6aaf5127b4a317017749ad892577d0b8))

Add _network.py with parse_grpc_target() for URL-to-target conversion and create_channel() for
  secure/insecure async channel creation.

Rewrite _auth.py to replace ConnectRPC _AuthHeaderInterceptor with gRPC AuthInterceptor implementing
  both UnaryUnaryClientInterceptor and StreamStreamClientInterceptor. Metadata keys are normalized
  to lowercase per gRPC requirements.

- **sdk**: Add infrastructure filtering and public service properties
  ([#29](https://github.com/coreweave/cwsandbox-client/pull/29),
  [`6260bbf`](https://github.com/coreweave/cwsandbox-client/commit/6260bbfca30d49b1d39fe30706c7755df068d22f))

* feat(sandbox): add service_address and exposed_ports properties

Capture service_address and exposed_ports from StartSandboxResponse and expose them as read-only
  properties on the Sandbox class. These fields are needed for network-accessible sandboxes (SSH,
  web services).

- Initialize fields to None in __init__ and _from_sandbox_info - Capture from response in
  _start_async (empty string becomes None) - Add properties with docstrings explaining availability
  conditions - Add unit tests covering all scenarios

* feat(sdk): add runway_ids and tower_ids parameters

Add runway_ids and tower_ids parameters to Sandbox.run(), Session.sandbox(), Session.function(), and
  RemoteFunction for infrastructure filtering.

These parameters allow targeting specific runways or towers when creating sandboxes, useful for
  workloads requiring specific hardware or regions.

- **sdk**: Add NetworkOptions for typed network configuration
  ([`e0f0df6`](https://github.com/coreweave/cwsandbox-client/commit/e0f0df67527502c9181f8ce9ee9bd5207677eac8))

Add NetworkOptions dataclass for typed network configuration with ingress/egress modes and exposed
  ports. Accept both NetworkOptions and dict for flexibility.

Changes: - Add NetworkOptions frozen dataclass with ingress_mode, exposed_ports, egress_mode -
  Accept both NetworkOptions and dict for network parameter (dict auto-converts) - Add network field
  to SandboxDefaults for shared configuration - Add applied_ingress_mode and applied_egress_mode
  properties - Ensure client connections are properly closed in async methods - Add comprehensive
  unit and integration tests - Update documentation with usage examples

The network parameter now works in three places: - Sandbox.run(network=...) -
  Session.sandbox(network=...) - @session.function(network=...)

- **session**: Add report_to parameter and exec statistics
  ([`0d9d065`](https://github.com/coreweave/cwsandbox-client/commit/0d9d06589b9f0b7861b3b49bc3a644203450cac6))

Add metrics reporting integration to Session and Sandbox:

- Session accepts report_to parameter (Reporter protocol or "wandb") - Dynamic W&B auto-detection
  when report_to="wandb" is specified - Sandbox tracks startup time and exec statistics - exec()
  passes stats_callback to Process for automatic reporting - Session.log_metrics() for custom metric
  logging

Updates unit tests for new parameters and behavior.

- **session**: Add Session class and function decorator
  ([#21](https://github.com/coreweave/cwsandbox-client/pull/21),
  [`4dcf2b2`](https://github.com/coreweave/cwsandbox-client/commit/4dcf2b26cf78aad108c4a4d0aeb8dbd482a68617))

* feat(session): add Session class and function decorator

Session provides multi-sandbox management with shared defaults: - session.sandbox(): Create sandbox
  with session defaults - session.function(): Decorator for remote function execution -
  session.adopt(): Register existing sandbox for cleanup - session.list()/from_id(): Find and
  reconnect to sandboxes - Context manager for automatic cleanup of all sandboxes

RemoteFunction enables serverless-style execution: - func.remote(): Execute in sandbox, return
  OperationRef - func.map(): Parallel execution across inputs - func.local(): Local execution for
  testing - Serialization modes: JSON (default) or PICKLE

* refactor(session): update .get() to .result() in session docs/tests

Update docs/guides/sessions.md, docs/guides/remote-functions.md, and
  tests/unit/aviato/test_session.py to use the new .result() method instead of .get() for blocking
  on OperationRef values.

* fix(session): update .get() to .result() in session module

Complete the API rename from .get() to .result() in the session module and its tests, which was
  missed in the earlier refactor.

- Update Session.__exit__ to call .close().result() - Update all docstring examples to use .result()
  - Fix integration tests to use .result() pattern - Fix unit test mock assertions to check .result
  calls

* docs(guides): fix .map() examples to use tuple arguments

The .map() method expects an iterable of argument tuples, not bare values. Update examples to wrap
  single arguments in tuples.

* fix: update .get() to .result() in docstrings and examples

Address Copilot review comments. OperationRef uses .result() method, not .get().

* docs(remote-functions): update get() to results() function

Use aviato.results() for collecting multiple OperationRef results, consistent with the API rename in
  earlier commits.

* remove unused _cleanup_async method from Session

- **tests**: Add W&B integration tests
  ([`7d2c1f2`](https://github.com/coreweave/cwsandbox-client/commit/7d2c1f238cd45c6a6f546410c64b646467d169c3))

Add integration tests for W&B reporting functionality: - Session with report_to="wandb" creates and
  manages W&B runs - Exec statistics are reported automatically - Custom metrics via
  session.log_metrics() - Graceful handling when W&B is not configured

- **types**: Add stats_callback to Process for exec tracking
  ([`a51059c`](https://github.com/coreweave/cwsandbox-client/commit/a51059c1b1627e5cafc863a05ca56d02c82808e0))

Add optional stats_callback parameter to Process class for collecting execution statistics. Callback
  receives ExecStats dataclass containing timing, command, and result information. Thread-safe
  recording ensures callbacks complete before result() returns.

- **wandb**: Add per-sandbox metrics and lazy run detection
  ([`4c0a7f3`](https://github.com/coreweave/cwsandbox-client/commit/4c0a7f3c711df11e0652d3b2aa1e1d7ef7a99014))

Enhance WandbReporter for RL training workflows where sandbox metrics need to be tracked per-sandbox
  and logged at training steps.

- Per-sandbox exec count tracking via sandbox_id parameter - Startup time statistics (min/max/avg)
  across all sandboxes - Completion rate and failure rate derived metrics - Lazy wandb run
  detection: metrics collect even if wandb.init() is called after Session creation (common in
  training loops) - Session reporter is Optional (None when report_to=[]) - Cache
  is_wandb_available() at module level

Fix Process._record_stats() race condition where concurrent.futures.Future.set_result() notifies
  waiting threads before invoking done callbacks. The main thread's .result() could return before
  the stats callback executed, causing missing metrics. Move callback invocation inside _stats_lock
  to guarantee the callback has completed before _record_stats() returns.

- **wandb**: Add WandbReporter and detection utilities
  ([`2d0d047`](https://github.com/coreweave/cwsandbox-client/commit/2d0d047181e1bfc465fe3f072589ae11693e2467))

Add optional Weights & Biases integration for tracking sandbox metrics:

- WandbReporter: collects exec stats, sandbox lifecycle events, and custom metrics with automatic
  W&B run management - Detection utilities: is_wandb_available(), is_wandb_active() for runtime
  environment detection - Graceful degradation when wandb package not installed

Includes comprehensive unit tests for both modules.

### Refactoring

- **sandbox**: Decouple start lifecycle from construction
  ([`d314b53`](https://github.com/coreweave/cwsandbox-client/commit/d314b53b082c9cb5a17bdf9ca690e1d81e5843dc))

Decouple sandbox construction from starting to fix a cross-event-loop bug (RuntimeError: Task got
  Future attached to a different loop) that occurs when using `await sandbox` with nest_asyncio. The
  root cause: Sandbox.__await__() was the only async entrypoint that bypassed _LoopManager, calling
  gRPC directly on the caller's event loop using a channel created on the background loop.

API changes: - session.sandbox() no longer calls start() automatically - start() returns
  OperationRef[None] (was blocking, returned None) - Operations (exec, read_file, write_file, wait)
  auto-start if needed - Context managers (__enter__, __aenter__) auto-start if needed - `await
  sandbox` starts if needed, then waits for RUNNING - Sandbox.run() still starts automatically
  (unchanged)

Implementation: - start() uses _loop_manager.run_async() instead of run_sync() - Add
  _ensure_started_async() lazy-start guard to all operations - Fix __await__ routing through
  _loop_manager with asyncio.wrap_future() - Acquire _start_lock in _stop_async() to handle
  start/stop races - Move sandbox creation metric from _start_async to session.sandbox() - Add
  cross-event-loop regression tests verifying all async entrypoints route through _loop_manager

- **sandbox**: Deduplicate status polling across concurrent waiters
  ([`b2f4a13`](https://github.com/coreweave/cwsandbox-client/commit/b2f4a13a16d81438e253b4099dd732701fb5f7f8))

Multiple concurrent operations (exec, read_file, write_file) each called _wait_until_running_async
  independently, causing N*M redundant GetSandbox API calls when N operations were queued on a
  sandbox still starting up.

Introduce a shared asyncio.Task pattern for both _wait_until_running_async and
  _wait_until_complete_async. The first waiter creates a polling task; subsequent waiters join via
  asyncio.shield(), eliminating redundant polls.

Design: - asyncio.shield prevents one waiter's timeout/cancel from killing the shared poll -
  _do_poll_complete returns SandboxStatus instead of raising, so per-waiter raise_on_termination is
  preserved - Done callbacks clear task on failure to allow retry, and suppress "Task exception was
  never retrieved" warnings - _stop_async cancels polling tasks; waiters translate CancelledError to
  SandboxNotRunningError

Closes #23

- **sandbox**: Make wait_until_complete return OperationRef
  ([`8e100a5`](https://github.com/coreweave/cwsandbox-client/commit/8e100a5ba129e47ece5dc5ead18954a80f676313))

Change wait_until_complete() to return OperationRef[Sandbox] instead of Sandbox directly, enabling
  async usage via await alongside sync usage via .result(). Follows the same pattern as the start()
  method.

Update callers in tests to use .result() and update docs to reflect the new return type.

- **sdk**: Use interceptors, pyqwest HTTP/2, rename service to network
  ([`1d04186`](https://github.com/coreweave/cwsandbox-client/commit/1d0418617c784ff1c55cf77e477a2af2f17e6785))

Major refactoring of SDK internals: - Replace httpx.AsyncClient with connectrpc interceptors for
  auth - Use pyqwest HTTP/2 transport for streaming exec - Add timeout_ms to all ConnectRPC client
  instantiations - Add max_timeout_seconds to stop requests - Remove _close_client() method (no
  longer needed without httpx session) - Rename 'service' parameter to 'network' for proto
  compatibility - Add explicit type annotations for runway_ids and tower_ids - Update
  DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS comment

docs: update sandbox configuration for service to network rename

Update the sandbox configuration guide to use 'network' parameter instead of 'service', reflecting
  the proto compatibility change.

test(sdk): fix async mock patterns and add interceptor tests

Update tests for the new interceptor-based auth pattern: - Replace resolve_auth patches with
  create_auth_interceptors patches - Remove mock_auth.return_value.headers setup (no longer needed)
  - Update TestSandboxAuth to verify interceptor usage - Update service kwarg to network in kwargs
  validation test - Add tests for _AuthHeaderInterceptor and create_auth_interceptors()

- **types**: Introduce sync/async hybrid type system
  ([#18](https://github.com/coreweave/cwsandbox-client/pull/18),
  [`5eab345`](https://github.com/coreweave/cwsandbox-client/commit/5eab3451a74523ce13d9d4065d868bb2407253a7))

* refactor(types): introduce sync/async hybrid type system

Add core types for the new lazy-start model: - OperationRef[T]: Generic wrapper bridging Future to
  asyncio - Process: Handle for running commands with streaming support - ProcessResult: Completed
  process with stdout/stderr/returncode - StreamReader: Dual sync/async iterable for output
  streaming - Serialization: Enum for function serialization modes - ExecResult: Legacy type
  (removed in PR 3)

These types enable sync code to work with async operations via .get()/.result() or await, without
  requiring users to manage event loops.

* fix(types): address PR review feedback from Copilot

- Fix TimeoutError caching bug in Process._ensure_result: timeouts are no longer cached, allowing
  callers to retry with a longer timeout - Use explicit 'is not None' checks for exception type
  narrowing in wait() and result() methods - Add missing test coverage for Process.result() with
  timeout parameter - Simplify StreamReader.__next__ by removing unnecessary async wrapper - Clean
  up test imports to use single 'from concurrent.futures' import

* fix(docs): Clarify diff between OperationRef and Process

* refactor(types): rename OperationRef.get() to .result() and add Process inheritance

- Rename OperationRef.get() to .result() for consistency with concurrent.futures.Future API - Make
  Process inherit from OperationRef[ProcessResult] establishing proper type hierarchy - Process now
  inherits __await__ from parent instead of duplicating - Update all tests and documentation to use
  .result()

BREAKING CHANGE: OperationRef.get() has been removed. Use .result() instead.

* fix(docs): Correct comment in docstring
