#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
#
# Refresh vendored protobuf/gRPC stubs from buf.build or a local sandbox checkout.
#
# Usage:
#   scripts/update-protos.sh                          # download from buf.build
#   scripts/update-protos.sh --local ../sandbox/gen/python  # copy from local path

set -euo pipefail

# ---------------------------------------------------------------------------
# Version pins - update these when bumping protos
# ---------------------------------------------------------------------------
# buf.build plugin versions (prefix differs per package, commit suffix is shared)
GRPC_VERSION="1.80.0.1.20260421205249+cbd4f819dd44"
PB_VERSION="26.1.0.2.20260421205249+cbd4f819dd44"
PYI_VERSION="26.1.0.2.20260421205249+cbd4f819dd44"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$REPO_ROOT/src/cwsandbox/_proto"
BUF_INDEX="https://buf.build/gen/python"
SPDX_HEADER='# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client'

# Files we vendor (everything else in the wheels is ignored)
PROTO_FILES=(
    gateway_pb2.py
    gateway_pb2.pyi
    gateway_pb2_grpc.py
    discovery_pb2.py
    discovery_pb2.pyi
    discovery_pb2_grpc.py
    secrets_pb2.py
    secrets_pb2.pyi
    streaming_pb2.py
    streaming_pb2.pyi
    streaming_pb2_grpc.py
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { printf '%s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

TMPDIR_CREATED=""
cleanup() {
    if [[ -n "$TMPDIR_CREATED" ]]; then
        rm -rf "$TMPDIR_CREATED"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
LOCAL_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            [[ -n "${2:-}" ]] || die "--local requires a path argument"
            LOCAL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            log "Usage: $0 [--local PATH]"
            log ""
            log "  --local PATH  Copy from local sandbox gen/python directory"
            log "                (default: download from buf.build)"
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Source: local path
# ---------------------------------------------------------------------------
copy_from_local() {
    local src="$1/coreweave/sandbox/v1beta2"
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

# ---------------------------------------------------------------------------
# Source: buf.build wheels
# ---------------------------------------------------------------------------
copy_from_buf() {
    TMPDIR_CREATED="$(mktemp -d)"
    local tmpdir="$TMPDIR_CREATED"

    log "Downloading wheels from buf.build..."
    pip3 download --no-deps --quiet --dest "$tmpdir" \
        --index-url "$BUF_INDEX" \
        "coreweave-sandbox-grpc-python==$GRPC_VERSION" \
        "coreweave-sandbox-protocolbuffers-python==$PB_VERSION" \
        "coreweave-sandbox-protocolbuffers-pyi==$PYI_VERSION"

    log "Wheel checksums (SHA256):"
    for whl in "$tmpdir"/*.whl; do
        log "  $(shasum -a 256 "$whl")"
    done

    log "Extracting proto files..."
    local extract_dir="$tmpdir/extracted"
    mkdir -p "$extract_dir"

    for whl in "$tmpdir"/*.whl; do
        unzip -q -o "$whl" -d "$extract_dir"
    done

    local src="$extract_dir/coreweave/sandbox/v1beta2"
    [[ -d "$src" ]] || die "Expected path not found in wheels: $src"

    for f in "${PROTO_FILES[@]}"; do
        [[ -f "$src/$f" ]] || die "Missing file in wheels: $src/$f"
        cp "$src/$f" "$PROTO_DIR/$f"
    done
}

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
inject_spdx_header() {
    for f in "${PROTO_FILES[@]}"; do
        local filepath="$PROTO_DIR/$f"
        # Skip if header already present
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
    # Only rewrites "from coreweave.sandbox.v1beta2 import" statements.
    # gRPC service paths (e.g. '/coreweave.sandbox.v1beta2.GatewayService/Start') are
    # protocol-level identifiers and must NOT be rewritten.
    for f in "${PROTO_FILES[@]}"; do
        local filepath="$PROTO_DIR/$f"
        local tmp
        tmp="$(mktemp)"
        sed 's/from coreweave\.sandbox\.v1beta2 import/from cwsandbox._proto import/g' "$filepath" > "$tmp"
        mv "$tmp" "$filepath"
    done
}

validate_imports() {
    # Verify no stale Python import references remain.
    # We grep for "from coreweave.sandbox" which catches import statements but
    # not gRPC service path strings like '/coreweave.sandbox.v1beta2.GatewayService/Start'.
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
    # This avoids pinning users to a specific protobuf minor version.
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    [[ -d "$PROTO_DIR" ]] || die "Proto directory not found: $PROTO_DIR"

    if [[ -n "$LOCAL_PATH" ]]; then
        copy_from_local "$LOCAL_PATH"
    else
        copy_from_buf
    fi

    rewrite_imports
    inject_spdx_header
    validate_imports
    validate_protobuf_version

    log ""
    log "Proto stubs updated in $PROTO_DIR"
}

main
