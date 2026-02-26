# SWE-bench evaluation guide

Run SWE-bench evaluations in parallel using CWSandboxes.

## What this does

[SWE-bench](https://www.swebench.com/) tests whether language models can fix real bugs in real repositories. Each task: apply a patch, run the test suite, see if the fix works.

You can run SWE-bench locally with Docker, but you're limited by your machine's resources. CWSandbox lets you run evaluations at scale on CoreWeave infrastructure. Spin up dozens or hundreds of sandboxes concurrently without managing any of it yourself. The script pulls pre-built images from Epoch AI's registry on GHCR, so there's no local Docker build step.

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/coreweave/cwsandbox-client.git
cd cwsandbox-client
uv sync
uv pip install swebench datasets
```

For authentication, set one of these:

- `CWSANDBOX_API_KEY` environment variable (recommended)
- `WANDB_API_KEY` environment variable
- W&B credentials in `~/.netrc`

## Quick start

Test your setup with a single instance. The `gold` option uses the known-correct fix from the dataset:

```bash
uv run python examples/swebench/run_evaluation.py \
    --predictions-path gold \
    --instance-ids astropy__astropy-12907 \
    --run-id test
```

This should pass.

### Running in parallel

Run multiple instances at once:

```bash
uv run python examples/swebench/run_evaluation.py \
    --predictions-path gold \
    --instance-ids \
        astropy__astropy-12907 \
        django__django-11039 \
        django__django-11099 \
        django__django-11283 \
        matplotlib__matplotlib-23476 \
        scikit-learn__scikit-learn-13142 \
        sympy__sympy-13031 \
        sympy__sympy-13647 \
    --run-id parallel-test \
    --max-workers 8
```

This spins up 8 sandboxes and runs them concurrently. All should pass since gold patches are the correct fixes.

### Evaluating model predictions

To test custom model output:

```bash
uv run python examples/swebench/run_evaluation.py \
    --predictions-path predictions.json \
    --instance-ids django__django-11039 scikit-learn__scikit-learn-13142 \
    --run-id eval-run-1 \
    --max-workers 10
```

The predictions file maps instance IDs to patches:

```json
[
  {
    "instance_id": "django__django-11039",
    "model_name_or_path": "gpt-4",
    "model_patch": "diff --git a/..."
  }
]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--predictions-path` | Required | Path to predictions JSON, or `gold` for gold patches |
| `--instance-ids` | Required | Space-separated instance IDs |
| `--run-id` | Required | Identifier for this run |
| `--max-workers` | 10 | Max parallel sandboxes |
| `--timeout` | 1800 | Per-instance timeout in seconds (30 min) |
| `--output-dir` | `logs/swebench` | Where to write logs and reports |
| `--dataset` | `princeton-nlp/SWE-bench_Lite` | HuggingFace dataset name |
| `--force` | false | Re-run instances even if report.json exists |

### Adjusting resources

Default is 2 CPUs and 4Gi memory per sandbox. Change this in `run_evaluation.py`:

```python
defaults = SandboxDefaults(
    tags=(f"swebench-{run_id}",),
    resources={"cpu": "4", "memory": "8Gi"},
)
```

## How it works

### Container images

Epoch AI hosts pre-built images on GHCR. Each instance has its own image with the repo checked out at the right commit, dependencies installed, and test environment ready.

Image format: `ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest`

Sandboxes pull these directly from GHCR. No local builds needed.

### Evaluation flow

Each instance goes through these steps:

| Step | Where | What happens |
|------|-------|--------------|
| 1. Load dataset | Local | Fetch instance metadata from HuggingFace |
| 2. Create sandbox | CWSandbox | Start sandbox with the instance's image |
| 3. Write patch | CWSandbox | Write patch to `/tmp/patch.diff` |
| 4. Apply patch | CWSandbox | Run `git apply` (falls back to `patch` if needed) |
| 5. Run tests | CWSandbox | Execute `/root/eval.sh` |
| 6. Grade results | Local | Parse output with `swebench.harness.grading` |
| 7. Write report | Local | Save results to `logs/swebench/` |
| 8. Cleanup | CWSandbox | Stop sandbox |

Steps 2-5 and 8 run remotely. Steps 1, 6, and 7 run on your machine.

### Parallel execution

The script uses `ThreadPoolExecutor` to run instance workflows concurrently. Each thread drives one instance through its workflow. While one instance runs tests, another can be applying its patch, another grading locally, another starting up. The overlap is where the speed comes from.

Results come back as workflows finish via `as_completed()`.

### Cleanup

The script uses a CWSandbox `Session` to track sandboxes. When the session exits (normal exit, exception, or Ctrl+C), all sandboxes get cleaned up.

```python
with cwsandbox.Session(defaults=defaults) as session:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Sandboxes created here get cleaned up when the session exits
```

## Output

Results go to `{output-dir}/{run-id}/{model-name}/{instance-id}/`:

| File | Contents |
|------|----------|
| `report.json` | Resolved status, sandbox ID, duration |
| `test_output.txt` | Full test output |
| `patch.diff` | The patch that was applied |

### Report format

```json
{
  "instance_id": {
    "resolved": true,
    "tests_status": {
      "PASSED": ["test_foo", "test_bar"],
      "FAILED": []
    }
  },
  "sandbox_id": "sb-abc123",
  "duration_seconds": 45.2
}
```

The format is compatible with standard SWE-bench tooling.

## Troubleshooting

### Patch application fails

If you see `APPLY_PATCH_FAIL` in logs, the patch is probably malformed, targeting the wrong commit, or has whitespace issues. Try `git apply --check` locally to see what's wrong. Make sure the instance ID matches the prediction.

### Tests timeout

Some test suites genuinely take longer than 30 minutes. Model-generated code might also have infinite loops. Increase `--timeout` if needed, or check the test output to see where it's hanging.

### Image pull errors

If the container fails to start with an image pull error, either the instance ID doesn't exist in Epoch AI's registry or there's a network issue reaching `ghcr.io`. Verify the instance ID is in the SWE-bench dataset.

## See also

- [Example script](../../examples/swebench/run_evaluation.py): full source code
- [Command execution guide](execution.md): details on `exec()` patterns
- [Cleanup patterns](cleanup-patterns.md): managing sandbox lifecycle
