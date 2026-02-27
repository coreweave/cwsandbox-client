#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
#
# Refresh vendored protobuf/gRPC stubs from buf.build or a local aviato checkout.
#
# Usage:
#   scripts/update-protos.sh                          # download from buf.build
#   scripts/update-protos.sh --local ../aviato/gen/python  # copy from local path

set -euo pipefail

# ---------------------------------------------------------------------------
# Version pins - update these when bumping protos
# ---------------------------------------------------------------------------
# buf.build plugin versions (prefix differs per package, commit suffix is shared)
GRPC_VERSION="1.78.1.1.20260220161707+89028200095a"
PB_VERSION="33.5.0.1.20260220161707+89028200095a"
PYI_VERSION="33.5.0.1.20260220161707+89028200095a"

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
    atc_pb2.py
    atc_pb2.pyi
    atc_pb2_grpc.py
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
            log "  --local PATH  Copy from local aviato gen/python directory"
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
    local src="$1/coreweave/aviato/v1beta1"
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
        "coreweave-aviato-grpc-python==$GRPC_VERSION" \
        "coreweave-aviato-protocolbuffers-python==$PB_VERSION" \
        "coreweave-aviato-protocolbuffers-pyi==$PYI_VERSION"

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

    local src="$extract_dir/coreweave/aviato/v1beta1"
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
    # Only rewrites "from coreweave.aviato.v1beta1 import" statements.
    # gRPC service paths (e.g. '/coreweave.aviato.v1beta1.ATCService/Start') are
    # protocol-level identifiers and must NOT be rewritten.
    for f in "${PROTO_FILES[@]}"; do
        local filepath="$PROTO_DIR/$f"
        local tmp
        tmp="$(mktemp)"
        sed 's/from coreweave\.aviato\.v1beta1 import/from cwsandbox._proto import/g' "$filepath" > "$tmp"
        mv "$tmp" "$filepath"
    done
}

validate_imports() {
    # Verify no stale Python import references remain.
    # We grep for "from coreweave.aviato" which catches import statements but
    # not gRPC service path strings like '/coreweave.aviato.v1beta1.ATCService/Start'.
    local stale
    stale=$(grep -rn "from coreweave\.aviato" "$PROTO_DIR/" 2>/dev/null || true)
    if [[ -n "$stale" ]]; then
        log "FAIL: stale Python imports found in vendored files:"
        log "$stale"
        exit 1
    fi
    log "OK: no stale Python imports"
}

validate_protobuf_version() {
    # Verify generated files target protobuf 6.x
    local version_line
    version_line=$(grep -h "Protobuf Python Version:" "$PROTO_DIR"/atc_pb2.py 2>/dev/null || true)
    if [[ -z "$version_line" ]]; then
        log "WARN: could not find Protobuf Python Version in generated files"
        return
    fi
    if ! echo "$version_line" | grep -q "6\."; then
        log "FAIL: generated files do not target protobuf 6.x"
        log "  Found: $version_line"
        log "  Expected: Protobuf Python Version: 6.x.x"
        exit 1
    fi
    local version
    version=$(echo "$version_line" | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
    log "OK: protobuf version $version (6.x series)"
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
