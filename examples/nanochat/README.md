<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: BSD-3-Clause
SPDX-PackageName: cwsandbox-client
-->

# Nanochat Training Pipeline

Runs [karpathy/nanochat](https://github.com/karpathy/nanochat)'s CPU training pipeline
(`runcpu.sh`) end-to-end inside a single CWSandbox.

## What it does

1. Clones nanochat and installs dependencies (via `uv`)
2. Downloads the training dataset
3. Trains and evaluates a BPE tokenizer
4. Pretrains a 6-layer base model (configurable iterations)
5. Evaluates the base model
6. Downloads SFT data and runs supervised fine-tuning (optional)

All output streams in real time via per-step `exec()` calls.

## Usage

```bash
# Full pipeline (base training defaults to 5000 iterations)
python examples/nanochat/train_nanochat.py

# Quick smoke test
python examples/nanochat/train_nanochat.py --num-iterations 10 --skip-sft

# Custom iteration count
python examples/nanochat/train_nanochat.py --num-iterations 500
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-iterations` | 5000 | Base model training iterations |
| `--skip-sft` | false | Skip supervised fine-tuning steps |

## Timing

With default settings on CPU, expect the full pipeline to take several hours.
Use `--num-iterations 10 --skip-sft` for a quick validation run (~5 minutes).
