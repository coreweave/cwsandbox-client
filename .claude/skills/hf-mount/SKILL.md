---
name: hf-mount
description: "Use when mounting HuggingFace buckets, model repos, or dataset repos as local filesystems within CoreWeave Sandbox. Covers hf-mount installation, mount commands, FUSE vs NFS backends, HF Storage Buckets, read-only vs read-write access, and integration with sandbox workflows for RL training and model evaluation."
disable-model-invocation: false
---

# HuggingFace Mount (hf-mount)

Mount HuggingFace buckets and repos as local filesystems. No download, no copy, no waiting.

## Install

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh

# Linux (NFS - no root required, recommended for containers)
hf-mount start repo openai/gpt2 /tmp/gpt2

# macOS (requires macFUSE)
brew install macfuse  # reboot required
hf-mount start --fuse repo openai/gpt2 /tmp/gpt2
```

## HF Storage Buckets vs Repos

| Feature | Repositories (Git-based) | Storage Buckets |
|---------|-------------------------|-----------------|
| Versioning | Full Git history | None (mutable) |
| Types | Models, Datasets, Spaces | Standalone bucket |
| Use case | Publishing finished artifacts | Working storage, checkpoints |
| Operations | Hub API, Git push/pull | S3-like sync, cp, rm |
| Write support | Read-only via hf-mount | Read-write via hf-mount |

**Use repos** when you want version history, collaboration (PRs, discussions), and library integrations.
**Use buckets** for fast, mutable storage: checkpoints, logs, intermediate artifacts.

## Quick Start

```bash
# Public model (no token needed)
hf-mount start repo openai/gpt-oss-20b /tmp/model

# Private model/dataset
hf-mount start --hf-token $HF_TOKEN repo myorg/my-private-model /tmp/model

# Bucket (read-write)
hf-mount start --hf-token $HF_TOKEN bucket myuser/my-bucket /tmp/data
```

## Within CoreWeave Sandbox

Use hf-mount inside a sandbox to access models/datasets without downloading:

```python
from cwsandbox import Sandbox, SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    environment_variables={"HF_TOKEN": "your-token"},
)

with Sandbox.run(defaults=defaults) as sandbox:
    # Install hf-mount in sandbox
    sandbox.exec(["bash", "-c", "curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh"]).result()

    # Mount a model repo
    sandbox.exec(["hf-mount", "start", "repo", "openai/gpt2", "/tmp/gpt2"]).result()

    # Use model directly - no download step
    result = sandbox.exec([
        "python", "-c",
        "from transformers import AutoModel; model = AutoModel.from_pretrained('/tmp/gpt2')"
    ]).result()
```

## Mount Types

### Repos (read-only)

```bash
# Models
hf-mount start repo username/model-name /mnt/model

# Datasets
hf-mount start repo datasets/username/dataset-name /mnt/dataset

# Specific revision
hf-mount start repo username/model /mnt/model --revision v1.0

# Subfolder only
hf-mount start repo username/model/onnx /mnt/onnx
```

### Buckets (read-write)

```bash
# Read-write bucket
hf-mount start --hf-token $HF_TOKEN bucket username/my-bucket /mnt/data

# Read-only bucket
hf-mount start --hf-token $HF_TOKEN --read-only bucket username/my-bucket /mnt/data

# Subfolder only
hf-mount start --hf-token $HF_TOKEN bucket username/my-bucket/checkpoints /mnt/ckpts
```

## Backend: FUSE vs NFS

| Feature | FUSE | NFS |
|---------|------|-----|
| No root required | No | Yes (recommended for containers) |
| Metadata freshness | ~10s | Up to poll interval |
| Write mode | Streaming by default | Advanced always |
| Page cache invalidation | Yes | No |
| macOS support | Yes (macFUSE) | Yes |

```bash
# FUSE (tighter integration, requires root/FUSE)
hf-mount start --fuse --hf-token $HF_TOKEN bucket user/bucket /mnt/data

# NFS (no root, works in containers)
hf-mount start --hf-token $HF_TOKEN bucket user/bucket /mnt/data
```

## Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-token` | `$HF_TOKEN` | HF API token |
| `--read-only` | false | Read-only mount |
| `--cache-dir` | `/tmp/hf-mount-cache` | Local cache |
| `--cache-size` | 10GB | Max cache size |
| `--poll-interval-secs` | 30 | Change polling interval |
| `--metadata-ttl-ms` | 10000 | Metadata cache TTL |

## Manage Mounts

```bash
# List running mounts
hf-mount status

# Stop mount
hf-mount stop /mnt/data

# Or unmount manually
umount /mnt/data          # NFS or FUSE (macOS)
fusermount -u /mnt/data   # FUSE (Linux)
```

## Write Modes

### Streaming (default)

- Append-only writes
- In-memory buffer, upload on close
- No disk space needed
- **Not safe for text editors** (use `--advanced-writes`)

### Advanced Writes

```bash
hf-mount start --advanced-writes --hf-token $HF_TOKEN bucket user/bucket /mnt/data
```

- Random writes, seek, overwrite supported
- Downloads file to local disk first
- Async flush (2s debounce, 30s max batch)
- Safe for editors and random I/O

## Consistency Model

- **Reads**: Files can be stale for up to 10s (metadata TTL)
- **Writes**: eventual consistency, background polling syncs changes
- **Not for**: latency-sensitive random I/O, strong consistency needs

## HF CLI for Buckets

```bash
# Create bucket
hf buckets create my-bucket

# List files (human-readable with sizes)
hf buckets list username/my-bucket -h -R

# Tree view
hf buckets list username/my-bucket --tree -h

# Upload file
hf buckets cp ./model.safetensors hf://buckets/username/my-bucket/models/

# Sync directory (upload local to bucket)
hf buckets sync ./data hf://buckets/username/my-bucket/data

# Download file
hf buckets cp hf://buckets/username/my-bucket/model.bin ./model.bin

# Sync directory (download bucket to local)
hf buckets sync hf://buckets/username/my-bucket/data ./data

# Delete file
hf buckets rm username/my-bucket/old-checkpoint.bin

# Dry-run before delete
hf buckets rm username/my-bucket/checkpoints/ --recursive --dry-run
```

## Python API for Buckets

```python
from huggingface_hub import (
    create_bucket,
    download_bucket_files,
    batch_bucket_files,
    sync_bucket,
)

# Create bucket
create_bucket("username/my-bucket")
create_bucket("username/my-bucket", private=True)

# Upload files
batch_bucket_files(
    "username/my-bucket",
    add=[
        ("./model.safetensors", "models/model.safetensors"),
        ("./config.json", "models/config.json"),
    ],
)

# Download files
download_bucket_files(
    "username/my-bucket",
    files=[
        ("models/model.safetensors", "./local/model.safetensors"),
        ("config.json", "./local/config.json"),
    ],
)

# Sync directory (upload)
sync_bucket("./data", "hf://buckets/username/my-bucket/data")

# Sync directory (download)
sync_bucket("hf://buckets/username/my-bucket/data", "./data")

# Delete files
batch_bucket_files("username/my-bucket", delete=["old-model.bin", "logs/debug.log"])
```

## Use Cases with Sandboxes

### RL Training with HF Models (Checkpoints to Bucket)

```python
with Sandbox.run() as sandbox:
    sandbox.exec(["bash", "-c", "curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh"]).result()

    # Mount model repo
    sandbox.exec([
        "hf-mount", "start", "repo", "openai/gpt-oss-20b", "/tmp/model"
    ]).result()

    # Mount bucket for checkpoints
    sandbox.exec([
        "hf-mount", "start", "--hf-token", "$HF_TOKEN",
        "bucket", "myuser/checkpoints", "/tmp/ckpts"
    ]).result()

    # Training with checkpointing to bucket
    result = sandbox.exec([
        "python", "train.py",
        "--model-path", "/tmp/model",
        "--checkpoint-dir", "/tmp/ckpts/run-1"
    ]).result()
```

### Model Evaluation with Dataset

```python
with Sandbox.run() as sandbox:
    sandbox.exec(["bash", "-c", "curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh"]).result()

    # Mount model and dataset
    sandbox.exec(["hf-mount", "start", "repo", "myorg/model", "/tmp/model"]).result()
    sandbox.exec(["hf-mount", "start", "repo", "datasets/myorg/eval-data", "/tmp/eval-data"]).result()

    # Run evaluation
    result = sandbox.exec([
        "python", "evaluate.py",
        "--model", "/tmp/model",
        "--data", "/tmp/eval-data"
    ]).result()
```

### Data Processing Pipeline

```python
with Sandbox.run() as sandbox:
    sandbox.exec(["bash", "-c", "curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh"]).result()

    # Mount bucket for intermediate outputs
    sandbox.exec([
        "hf-mount", "start", "--hf-token", "$HF_TOKEN",
        "bucket", "myuser/pipeline-outputs", "/tmp/outputs"
    ]).result()

    # Process and save intermediate results
    sandbox.exec([
        "python", "process.py",
        "--input", "/tmp/raw-data",
        "--output", "/tmp/outputs/batch-001"
    ]).result()
```

## Kubernetes Integration

For Kubernetes, use [hf-csi-driver](https://github.com/huggingface/hf-csi-driver):

```bash
helm install hf-csi oci://ghcr.io/huggingface/charts/hf-csi-driver
```

Then mount HF repos/buckets as Kubernetes volumes in pods.

## Troubleshooting

```bash
# Debug logging
RUST_LOG=hf_mount=debug hf-mount start repo openai/gpt2 /tmp/gpt2

# Check status
hf-mount status

# Check logs
cat ~/.hf-mount/logs/
```

## References

- [hf-mount GitHub](https://github.com/huggingface/hf-mount)
- [HF Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)
- [hf-csi-driver](https://github.com/huggingface/hf-csi-driver) for Kubernetes
- [HF Hub Buckets Guide](https://huggingface.co/docs/huggingface_hub/guides/buckets)
