#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
#
# Refresh vendored protobuf/gRPC stubs for coreweave.sandbox.v1.
#
# Usage:
#   scripts/update-protos.sh --from-backend ../aviato
#       Generate with protobuf 26.1 / pyi / grpc plugins into a temp dir under
#       the backend checkout (or via BUF_PROTO_DIR), then vendor consumer stubs.
#   scripts/update-protos.sh --local ../aviato/gen/python
#       Copy already-generated stubs from a local gen/python tree (must be
#       generated with plugin <=26.1 to pass validate_protobuf_version).
#
# Only consumer-facing v1 modules are vendored (no runner_management, policy,
# placement_org_config, or network).

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$REPO_ROOT/src/cwsandbox/_proto"
SCRIPT_DIR="$REPO_ROOT/scripts"
BUF_GEN_TEMPLATE="$SCRIPT_DIR/buf.gen.python.yaml"
SPDX_HEADER='# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client'

# Consumer stubs only (flat layout under _proto/)
PROTO_FILES=(
    sandbox_pb2.py
    sandbox_pb2.pyi
    sandbox_pb2_grpc.py
    discovery_pb2.py
    discovery_pb2.pyi
    discovery_pb2_grpc.py
    settings_pb2.py
    settings_pb2.pyi
    settings_pb2_grpc.py
    sandbox_template_pb2.py
    sandbox_template_pb2.pyi
    sandbox_template_pb2_grpc.py
    volume_pb2.py
    volume_pb2.pyi
    volume_pb2_grpc.py
)

# Legacy beta stubs to delete on refresh
LEGACY_BETA_FILES=(
    gateway_pb2.py
    gateway_pb2.pyi
    gateway_pb2_grpc.py
    streaming_pb2.py
    streaming_pb2.pyi
    streaming_pb2_grpc.py
    secrets_pb2.py
    secrets_pb2.pyi
)

log() { printf '%s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

TMPDIR_CREATED=""
cleanup() {
    if [[ -n "$TMPDIR_CREATED" ]]; then
        rm -rf "$TMPDIR_CREATED"
    fi
}
trap cleanup EXIT

LOCAL_PATH=""
FROM_BACKEND=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            [[ -n "${2:-}" ]] || die "--local requires a path argument"
            LOCAL_PATH="$2"
            shift 2
            ;;
        --from-backend)
            [[ -n "${2:-}" ]] || die "--from-backend requires a path argument"
            FROM_BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            log "Usage: $0 --from-backend PATH | --local PATH"
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$LOCAL_PATH" || -n "$FROM_BACKEND" ]] || die "Specify --from-backend PATH or --local PATH"
[[ -z "$LOCAL_PATH" || -z "$FROM_BACKEND" ]] || die "Specify only one of --from-backend or --local"

copy_from_v1_tree() {
    local src="$1/coreweave/sandbox/v1"
    [[ -d "$src" ]] || die "Local proto source not found: $src"

    log "Copying from local: $src"
    for f in "${PROTO_FILES[@]}"; do
        if [[ -f "$src/$f" ]]; then
            cp "$src/$f" "$PROTO_DIR/$f"
        elif [[ "$f" == *.pyi ]]; then
            log "  WARN: skipping $f (not in local source) - type stubs may be stale"
        else
            die "Missing file: $src/$f"
        fi
    done
}

generate_from_backend() {
    local backend="$1"
    [[ -d "$backend" ]] || die "Backend path not found: $backend"
    [[ -f "$BUF_GEN_TEMPLATE" ]] || die "Missing template: $BUF_GEN_TEMPLATE"
    command -v buf >/dev/null || die "buf CLI is required to generate protos"

    TMPDIR_CREATED="$(mktemp -d)"
    local out="$TMPDIR_CREATED/out"
    mkdir -p "$out"

    # aviato buf module root is proto/ (see buf.yaml modules.path).
    local proto_mod="$backend/proto"
    [[ -d "$proto_mod/coreweave/sandbox/v1" ]] || die "Expected $proto_mod/coreweave/sandbox/v1"

    local tmp_template="$TMPDIR_CREATED/buf.gen.python.yaml"
    sed "s|out: gen/python|out: $out|" "$BUF_GEN_TEMPLATE" > "$tmp_template"

    log "Generating Python stubs (protobuf 26.1) from $proto_mod ..."
    (
        cd "$proto_mod"
        buf generate --template "$tmp_template" --path coreweave/sandbox/v1 --include-imports
    )

    copy_from_v1_tree "$out"
}

remove_legacy_beta() {
    for f in "${LEGACY_BETA_FILES[@]}"; do
        local filepath="$PROTO_DIR/$f"
        if [[ -f "$filepath" ]]; then
            log "Removing legacy beta stub: $f"
            rm -f "$filepath"
        fi
    done
}

inject_spdx_header() {
    for f in "${PROTO_FILES[@]}"; do
        local filepath="$PROTO_DIR/$f"
        [[ -f "$filepath" ]] || continue
        if head -1 "$filepath" | grep -q "SPDX-FileCopyrightText"; then
            continue
        fi
        local tmp
        tmp="$(mktemp)"
        printf '%s\n' "$SPDX_HEADER" | cat - "$filepath" > "$tmp"
        mv "$tmp" "$filepath"
    done
}

rewrite_imports() {
    # Rewrite Python import paths from the upstream package to our vendored location.
    # gRPC service paths (e.g. '/coreweave.sandbox.v1.SandboxService/CreateSandbox')
    # are protocol-level identifiers and must NOT be rewritten.
    for f in "${PROTO_FILES[@]}"; do
        local filepath="$PROTO_DIR/$f"
        [[ -f "$filepath" ]] || continue
        local tmp
        tmp="$(mktemp)"
        sed -E \
            -e 's/from coreweave\.sandbox\.v1 import/from cwsandbox._proto import/g' \
            -e 's/import coreweave\.sandbox\.v1\./import cwsandbox._proto./g' \
            "$filepath" > "$tmp"
        mv "$tmp" "$filepath"
    done
}

validate_imports() {
    local stale
    stale=$(grep -rn "from coreweave\.sandbox" "$PROTO_DIR/" 2>/dev/null || true)
    if [[ -n "$stale" ]]; then
        log "FAIL: stale Python imports found in vendored files:"
        log "$stale"
        exit 1
    fi
    log "OK: no stale Python imports"
}

validate_protobuf_version() {
    # Verify generated files use protobuf <=5.26.x, which predates the
    # ValidateProtobufRuntimeVersion check (introduced in 5.27.0).
    local bad_files
    bad_files=$(grep -rl "ValidateProtobufRuntimeVersion" "$PROTO_DIR"/*_pb2.py 2>/dev/null || true)
    if [[ -n "$bad_files" ]]; then
        log "FAIL: generated files contain ValidateProtobufRuntimeVersion:"
        log "$bad_files"
        log "  Use plugin version <=26.1.x (protobuf <=5.26.x) to avoid the runtime check"
        exit 1
    fi
    local version
    version=$(grep -h "Protobuf Python Version:" "$PROTO_DIR"/*_pb2.py 2>/dev/null \
        | head -1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' || echo "unknown")
    log "OK: protobuf version $version (no runtime version check)"
}

main() {
    [[ -d "$PROTO_DIR" ]] || die "Proto directory not found: $PROTO_DIR"

    if [[ -n "$FROM_BACKEND" ]]; then
        generate_from_backend "$FROM_BACKEND"
    else
        copy_from_v1_tree "$LOCAL_PATH"
    fi

    remove_legacy_beta
    rewrite_imports
    inject_spdx_header
    validate_imports
    validate_protobuf_version

    log ""
    log "Proto stubs updated in $PROTO_DIR (coreweave.sandbox.v1 consumer surface)"
}

main
