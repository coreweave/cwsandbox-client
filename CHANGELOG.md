# CHANGELOG


## v0.11.0 (2026-03-24)

### Features

- Add secrets support on Session and Sandbox
  ([#78](https://github.com/coreweave/cwsandbox-client/pull/78),
  [`061b32d`](https://github.com/coreweave/cwsandbox-client/commit/061b32d465a6a16ff40e5a34982dbf1d428ddedd))


## v0.10.0 (2026-03-24)

### Documentation

- Broken relative links in PyPI readme
  ([#80](https://github.com/coreweave/cwsandbox-client/pull/80),
  [`cc9a9d5`](https://github.com/coreweave/cwsandbox-client/commit/cc9a9d5c6ba3f34886741c71901a8209c9abcc8d))

### Features

- Add annotations support to Sandbox, Session, and Function
  ([`0dcac5d`](https://github.com/coreweave/cwsandbox-client/commit/0dcac5de7891b0abbcf3a0dfec20480e059c1008))


## v0.9.0 (2026-03-12)

### Chores

- **ci**: Remove uv install step for pypi publish job
  ([#79](https://github.com/coreweave/cwsandbox-client/pull/79),
  [`db69a2e`](https://github.com/coreweave/cwsandbox-client/commit/db69a2ef0c38ba12b4472835e5569eda5f741b14))

- **ci**: Update job name ([#79](https://github.com/coreweave/cwsandbox-client/pull/79),
  [`db69a2e`](https://github.com/coreweave/cwsandbox-client/commit/db69a2ef0c38ba12b4472835e5569eda5f741b14))

### Features

- **ci**: Publish to pypi ([#79](https://github.com/coreweave/cwsandbox-client/pull/79),
  [`db69a2e`](https://github.com/coreweave/cwsandbox-client/commit/db69a2ef0c38ba12b4472835e5569eda5f741b14))


## v0.8.2 (2026-03-11)

### Bug Fixes

- **auth**: Allow wandb entity and project to be optional
  ([#76](https://github.com/coreweave/cwsandbox-client/pull/76),
  [`321a00b`](https://github.com/coreweave/cwsandbox-client/commit/321a00b515bdb055cff4f27c80525a9cab931cc7))


## v0.8.1 (2026-03-11)

### Bug Fixes

- Bump vendored protos to b06c7779a62f
  ([#77](https://github.com/coreweave/cwsandbox-client/pull/77),
  [`69aed76`](https://github.com/coreweave/cwsandbox-client/commit/69aed7617350c001c0baa6258d62793bf7fdd1bc))


## v0.8.0 (2026-03-10)

### Bug Fixes

- **cli**: Close stderr stream on exec failure
  ([`f68a6a8`](https://github.com/coreweave/cwsandbox-client/commit/f68a6a8f5e87d6481534f25ec5b31699397a9f85))

### Documentation

- Add logging guide and update CLI quickstart
  ([`22ee9b7`](https://github.com/coreweave/cwsandbox-client/commit/22ee9b7600b78f9417c896754f0b51882b37de5d))

- Standardize docstrings and export RemoteFunction
  ([`5188b37`](https://github.com/coreweave/cwsandbox-client/commit/5188b37cdabcf6411d75a62b6b62a1074093bae9))

### Features

- **cli**: Add logs command
  ([`a9175ca`](https://github.com/coreweave/cwsandbox-client/commit/a9175ca2166c5289a0385c3b8fb4d9424db80c97))

- **sandbox**: Add stream_logs method
  ([`613605c`](https://github.com/coreweave/cwsandbox-client/commit/613605c9da18a1f73295dc2b2d627feea7baa5da))

- **types**: Add StreamReader.close()
  ([`354b24d`](https://github.com/coreweave/cwsandbox-client/commit/354b24d78e2263752ed7345d23afaf6956ecf6ff))


## v0.7.0 (2026-03-09)

### Documentation

- Add CLI quickstart guide
  ([`e48aa6c`](https://github.com/coreweave/cwsandbox-client/commit/e48aa6cd3da3bada997154d0cab7179db2620b0d))

### Features

- **cli**: Add exec command
  ([`d3b7b3d`](https://github.com/coreweave/cwsandbox-client/commit/d3b7b3db8c5c35ddd0514928e7a25bc9e169d850))

- **cli**: Add list command
  ([`22935db`](https://github.com/coreweave/cwsandbox-client/commit/22935dbc75ea463e2927bd339392c93f0d85fbb6))

- **cli**: Scaffold CLI with Click entry point
  ([`51480d8`](https://github.com/coreweave/cwsandbox-client/commit/51480d894e7d619588993162d8d10cf9d6d77df2))


## v0.6.3 (2026-03-03)

### Bug Fixes

- Honor base_url when using `session.list()`
  ([`7600dd4`](https://github.com/coreweave/cwsandbox-client/commit/7600dd414521b26c3b8d976d726ee23c6f8a6635))


## v0.6.2 (2026-02-27)

### Bug Fixes

- Update default base URL to atc.cw-sandbox.com
  ([`5ccee3e`](https://github.com/coreweave/cwsandbox-client/commit/5ccee3e97b107cc457c40195502a84bffaba3026))


## v0.6.1 (2026-02-27)

### Bug Fixes

- Sync uv.lock during semantic-release version bump
  ([`fdac3c6`](https://github.com/coreweave/cwsandbox-client/commit/fdac3c68239df82a9b48405e8c1f2b482ec286b8))


## v0.6.0 (2026-02-27)

### Chores

- Add GitHub issue templates for bug reports and feature requests
  ([`10ddcc1`](https://github.com/coreweave/cwsandbox-client/commit/10ddcc1e331bc92a47c5c151199a52749ddd4df0))

### Refactoring

- Vendor proto stubs, remove buf.build registry dependency
  ([`1fd6bb5`](https://github.com/coreweave/cwsandbox-client/commit/1fd6bb5133bae0dd51b2a6b9b67dc117082ff29c))

### Breaking Changes

- The coreweave.aviato.v1beta1 import path is no longer available. Use cwsandbox._proto for direct
  proto access (internal API).


## v0.5.0 (2026-02-27)

### Refactoring

- Rename package from aviato to cwsandbox
  ([`59bc899`](https://github.com/coreweave/cwsandbox-client/commit/59bc899e3aa575f2c76629fb79d4b0257dc73d20))

### Breaking Changes

- All imports, env vars, and exception names changed. No backward compatibility layer.


## v0.4.0 (2026-02-26)

### Bug Fixes

- **sandbox**: Add lock for exec statistics
  ([`d281eb4`](https://github.com/coreweave/cwsandbox-client/commit/d281eb486cb06b14ba07c429b463482fe580b803))

- **sandbox**: Cache shutdown task in stdin loop
  ([`5b9fa78`](https://github.com/coreweave/cwsandbox-client/commit/5b9fa7816f6f950a06f2032de61c44f9b064376d))

### Chores

- Update ruff pre-commit hook to v0.14.10 and fix formatting
  ([`430311b`](https://github.com/coreweave/cwsandbox-client/commit/430311ba49d147e55bdf03efe53b9115483acb0b))

### Features

- **types**: Propagate exceptions in StreamReader
  ([`3b94574`](https://github.com/coreweave/cwsandbox-client/commit/3b94574ef3e2bad0f7a70d0a2eef2d31614b4f41))

### Refactoring

- **sandbox**: Widen exec queue types
  ([`ab79d7a`](https://github.com/coreweave/cwsandbox-client/commit/ab79d7af52225c9fbe9164e47ff4348e072e0fb5))


## v0.3.0 (2026-02-25)

### Documentation

- Update get_status() for terminal sandbox caching
  ([`00f588a`](https://github.com/coreweave/cwsandbox-client/commit/00f588afc426827f1b47474ea60261572524eded))

### Features

- **sandbox**: Add _LifecycleState sealed type and transition helpers
  ([`27c3c23`](https://github.com/coreweave/cwsandbox-client/commit/27c3c2370f55497954ec7e8430ab1733db911a04))

### Refactoring

- **sandbox**: Replace lifecycle variables with _LifecycleState
  ([`cc327e5`](https://github.com/coreweave/cwsandbox-client/commit/cc327e5c567b044167201bae0dbc0344d3227ca5))


## v0.2.0 (2026-02-24)

### Bug Fixes

- **e2e**: Fix session terminal status filter test
  ([`a67c923`](https://github.com/coreweave/cwsandbox-client/commit/a67c9234c9259f1294cae5352a6a3bfa176dfa83))

- **e2e**: Relax session status assertion and add terminal filter test
  ([`6e50509`](https://github.com/coreweave/cwsandbox-client/commit/6e50509c0133714ed913b1685538bdd2f41d0e53))

- **e2e**: Relax status assertions for DB-backed sandbox records
  ([`528fb5c`](https://github.com/coreweave/cwsandbox-client/commit/528fb5cb4fedec4c45cf488ad3447d581ca1f96d))

- **list**: Address PR review feedback for include_stopped
  ([`db4e72a`](https://github.com/coreweave/cwsandbox-client/commit/db4e72a66b14590a24cf3bd0ee7d54fb39c670a9))

- **list**: Address second round of PR review feedback
  ([`6348615`](https://github.com/coreweave/cwsandbox-client/commit/6348615aa48d5e7b4d92fc7d5887a266482333a8))

- **sandbox**: Mark terminal sandboxes as stopped in _from_sandbox_info
  ([`69e21db`](https://github.com/coreweave/cwsandbox-client/commit/69e21dbf68b59eab6cfb9c025063ffa2bb9bf173))

### Code Style

- Apply ruff formatting to example and integration tests
  ([`bae3e31`](https://github.com/coreweave/cwsandbox-client/commit/bae3e3111e1627d0da1c9a6869ec7a9f355bd653))

- Apply ruff formatting to list_stopped_sandboxes example
  ([`96a2514`](https://github.com/coreweave/cwsandbox-client/commit/96a251462607221c967aff5c85cbffb2ebb8133a))

### Documentation

- Remove architecture-revealing language from integration tests
  ([`b6742a5`](https://github.com/coreweave/cwsandbox-client/commit/b6742a541aaee1caa2beca1268e0f633d6a5c285))

- Remove architecture-revealing language from public-facing code
  ([`a62df68`](https://github.com/coreweave/cwsandbox-client/commit/a62df688c26f89cdbe1c1164051793b62700ae77))

### Features

- **list**: Add include_stopped parameter to Sandbox.list() and Session.list()
  ([`d61c6f1`](https://github.com/coreweave/cwsandbox-client/commit/d61c6f1a195856bc57917bdaee16840286e4ef91))

### Testing

- **e2e**: Add integration tests for DB-backed List/Get behavior
  ([`c0180db`](https://github.com/coreweave/cwsandbox-client/commit/c0180db79ad9ae4261d47c56bbb5b401ca8c689c))


## v0.1.1 (2026-02-18)

### Bug Fixes

- **examples**: Read sandbox_id after auto-start in swebench runner
  ([`1bbee6c`](https://github.com/coreweave/cwsandbox-client/commit/1bbee6cfe80c446a7e43f16932a55d846bb60c02))


## v0.1.0 (2026-02-18)

- Initial Release
