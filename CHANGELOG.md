# CHANGELOG


## v0.2.0 (2026-02-24)

### Bug Fixes

- **e2e**: Fix session terminal status filter test
  ([`a67c923`](https://github.com/coreweave/aviato-client/commit/a67c9234c9259f1294cae5352a6a3bfa176dfa83))

- **e2e**: Relax session status assertion and add terminal filter test
  ([`6e50509`](https://github.com/coreweave/aviato-client/commit/6e50509c0133714ed913b1685538bdd2f41d0e53))

- **e2e**: Relax status assertions for DB-backed sandbox records
  ([`528fb5c`](https://github.com/coreweave/aviato-client/commit/528fb5cb4fedec4c45cf488ad3447d581ca1f96d))

- **list**: Address PR review feedback for include_stopped
  ([`db4e72a`](https://github.com/coreweave/aviato-client/commit/db4e72a66b14590a24cf3bd0ee7d54fb39c670a9))

- **list**: Address second round of PR review feedback
  ([`6348615`](https://github.com/coreweave/aviato-client/commit/6348615aa48d5e7b4d92fc7d5887a266482333a8))

- **sandbox**: Mark terminal sandboxes as stopped in _from_sandbox_info
  ([`69e21db`](https://github.com/coreweave/aviato-client/commit/69e21dbf68b59eab6cfb9c025063ffa2bb9bf173))

### Code Style

- Apply ruff formatting to example and integration tests
  ([`bae3e31`](https://github.com/coreweave/aviato-client/commit/bae3e3111e1627d0da1c9a6869ec7a9f355bd653))

- Apply ruff formatting to list_stopped_sandboxes example
  ([`96a2514`](https://github.com/coreweave/aviato-client/commit/96a251462607221c967aff5c85cbffb2ebb8133a))

### Documentation

- Remove architecture-revealing language from integration tests
  ([`b6742a5`](https://github.com/coreweave/aviato-client/commit/b6742a541aaee1caa2beca1268e0f633d6a5c285))

- Remove architecture-revealing language from public-facing code
  ([`a62df68`](https://github.com/coreweave/aviato-client/commit/a62df688c26f89cdbe1c1164051793b62700ae77))

### Features

- **list**: Add include_stopped parameter to Sandbox.list() and Session.list()
  ([`d61c6f1`](https://github.com/coreweave/aviato-client/commit/d61c6f1a195856bc57917bdaee16840286e4ef91))

### Testing

- **e2e**: Add integration tests for DB-backed List/Get behavior
  ([`c0180db`](https://github.com/coreweave/aviato-client/commit/c0180db79ad9ae4261d47c56bbb5b401ca8c689c))


## v0.1.1 (2026-02-18)

### Bug Fixes

- **examples**: Read sandbox_id after auto-start in swebench runner
  ([`1bbee6c`](https://github.com/coreweave/aviato-client/commit/1bbee6cfe80c446a7e43f16932a55d846bb60c02))


## v0.1.0 (2026-02-18)

- Initial Release
