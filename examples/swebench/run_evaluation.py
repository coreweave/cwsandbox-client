#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: aviato-client

"""SWE-bench evaluation using Aviato parallel sandboxes.

This example demonstrates running SWE-bench evaluations using Aviato's parallel
sandbox execution. It showcases the SDK's ability to manage many concurrent
sandboxes for agentic evaluation workloads.

Prerequisites:
    uv sync
    uv pip install swebench datasets

Usage:
    # Run with gold patches (validates setup)
    uv run python run_evaluation.py --predictions-path gold \\
        --instance-ids astropy__astropy-12907 --run-id test

    # Run with model predictions
    uv run python run_evaluation.py --predictions-path predictions.json \\
        --run-id eval-run-1 --max-workers 10

See docs/guides/swebench.md for full documentation.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aviato
from aviato import SandboxDefaults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP client logs
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_TIMEOUT_SECONDS = 60 * 30
DEFAULT_MAX_WORKERS = 10
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"


_dataset_cache: dict[str, Any] = {}


def get_instances_by_id(
    instance_ids: list[str], dataset_name: str = DEFAULT_DATASET
) -> list[dict[str, Any]]:
    """Load instances from HuggingFace dataset by instance IDs.

    Args:
        instance_ids: List of instance IDs to load
        dataset_name: HuggingFace dataset name

    Returns:
        List of instance dicts matching the requested IDs
    """
    from datasets import load_dataset

    if dataset_name not in _dataset_cache:
        logger.info(f"[huggingface] Loading dataset {dataset_name}...")
        _dataset_cache[dataset_name] = load_dataset(dataset_name, split="test")
        logger.info("[huggingface] Dataset loaded")

    ds = _dataset_cache[dataset_name]
    instance_id_set = set(instance_ids)
    return [inst for inst in ds if inst["instance_id"] in instance_id_set]


def get_epoch_image_key(instance_id: str, arch: str = "x86_64") -> str:
    """Get the Epoch AI image key for a given instance ID.

    Epoch AI hosts pre-built SWE-bench images on GitHub Container Registry.
    """
    return f"ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id.lower()}:latest"


def run_in_sandbox(
    sandbox: aviato.Sandbox,
    command: str,
    timeout_seconds: int | None = None,
) -> tuple[str, int]:
    """Execute a bash command in the sandbox, return combined output and exit code.

    Args:
        sandbox: The sandbox to execute in
        command: Bash command to run
        timeout_seconds: Optional timeout for the exec call
    """
    result = sandbox.exec(
        ["bash", "-c", command],
        cwd="/testbed",
        timeout_seconds=timeout_seconds,
    ).result()
    return result.stdout + result.stderr, result.returncode


@dataclass
class EvaluationResult:
    """Result of evaluating a single instance."""

    instance_id: str
    resolved: bool
    test_output: str
    report: dict[str, Any]
    log_dir: Path
    errored: bool
    error_message: str | None = None
    sandbox_id: str | None = None
    duration_seconds: float | None = None


def run_instance(
    session: aviato.Session,
    test_spec: Any,
    pred: dict[str, Any],
    run_id: str,
    output_dir: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> EvaluationResult:
    """Run a single SWE-bench instance with the given prediction.

    Args:
        session: Aviato session for sandbox management
        test_spec: TestSpec from swebench
        pred: Prediction dict with model_patch and instance_id
        run_id: Run identifier
        output_dir: Base directory for output
        timeout: Timeout in seconds
    """
    from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS
    from swebench.harness.grading import get_eval_report

    instance_id = test_spec.instance_id
    raw_model_name = pred.get("model_name_or_path")
    model_name = str(raw_model_name) if raw_model_name is not None else "unknown"
    model_name = model_name.replace("/", "__")
    log_dir = output_dir / run_id / model_name / instance_id

    start_time = time.time()
    sandbox_id = None
    sandbox = None

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        image_key = get_epoch_image_key(instance_id, getattr(test_spec, "arch", "x86_64"))
        sandbox = session.sandbox(
            container_image=image_key,
            max_timeout_seconds=timeout,
            tags=[f"swebench-{run_id}", instance_id],
        )
        sandbox_id = sandbox.sandbox_id
        logger.info(f"[{instance_id}] Created sandbox: {sandbox_id}")

        model_patch = pred.get("model_patch")
        patch_diff = str(model_patch) if model_patch is not None else ""
        sandbox.write_file("/tmp/patch.diff", patch_diff.encode("utf-8")).result()

        apply_output, returncode = run_in_sandbox(
            sandbox, "git apply -v /tmp/patch.diff", timeout_seconds=60
        )

        if returncode != 0:
            logger.info(f"[{instance_id}] git apply failed, trying patch command...")
            apply_output, returncode = run_in_sandbox(
                sandbox, "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff", timeout_seconds=60
            )
            if returncode != 0:
                logger.error(f"[{instance_id}] {APPLY_PATCH_FAIL}")
                (log_dir / "test_output.txt").write_text(apply_output)
                sandbox.stop(missing_ok=True).result()
                return EvaluationResult(
                    instance_id=instance_id,
                    resolved=False,
                    test_output=apply_output,
                    report={},
                    log_dir=log_dir,
                    errored=True,
                    error_message=f"{APPLY_PATCH_FAIL}: {apply_output}",
                    sandbox_id=sandbox_id,
                    duration_seconds=time.time() - start_time,
                )

        logger.info(f"[{instance_id}] {APPLY_PATCH_PASS}")

        eval_script = test_spec.eval_script
        eval_script = eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")
        sandbox.write_file("/root/eval.sh", eval_script.encode("utf-8")).result()

        run_command = ""
        if "pylint" in instance_id:
            run_command += "export PYTHONPATH=; "
        run_command += "python3 -c 'import sys; sys.setrecursionlimit(10000)' && "
        run_command += "/bin/bash /root/eval.sh"

        logger.info(f"[{instance_id}] Running evaluation...")
        test_output, _ = run_in_sandbox(sandbox, run_command, timeout_seconds=timeout)

        test_output_path = log_dir / "test_output.txt"
        test_output_path.write_text(test_output)

        logger.info(f"[{instance_id}] Grading...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )

        resolved = report.get(instance_id, {}).get("resolved", False)
        logger.info(f"[{instance_id}] Resolved: {resolved}")

        sandbox.stop(missing_ok=True).result()
        return EvaluationResult(
            instance_id=instance_id,
            resolved=resolved,
            test_output=test_output,
            report=report,
            log_dir=log_dir,
            errored=False,
            sandbox_id=sandbox_id,
            duration_seconds=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"[{instance_id}] Error: {e}")
        if sandbox is not None:
            sandbox.stop(missing_ok=True).result()
        return EvaluationResult(
            instance_id=instance_id,
            resolved=False,
            test_output="",
            report={},
            log_dir=log_dir,
            errored=True,
            error_message=str(e),
            sandbox_id=sandbox_id,
            duration_seconds=time.time() - start_time,
        )


def run_evaluation(
    predictions: dict[str, dict[str, Any]],
    instance_ids: list[str],
    run_id: str,
    output_dir: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    dataset: str = DEFAULT_DATASET,
    force: bool = False,
) -> list[EvaluationResult]:
    """Run evaluation for multiple instances in parallel.

    Uses a Session to manage all sandboxes and ThreadPoolExecutor for parallel
    execution. The Session ensures cleanup on normal exit, exceptions, or Ctrl+C.

    Args:
        predictions: Dict mapping instance_id to prediction dict
        instance_ids: List of instance IDs to evaluate
        run_id: Run identifier
        output_dir: Base directory for output
        timeout: Timeout per instance in seconds
        max_workers: Maximum parallel sandboxes
        dataset: HuggingFace dataset name
        force: If True, re-run instances even if report.json exists
    """
    from swebench.harness.test_spec.test_spec import make_test_spec

    valid_instance_ids = [iid for iid in instance_ids if iid in predictions]
    if len(valid_instance_ids) < len(instance_ids):
        skipped = set(instance_ids) - set(valid_instance_ids)
        logger.warning(f"Skipping {len(skipped)} instances without predictions: {skipped}")

    instances = get_instances_by_id(valid_instance_ids, dataset_name=dataset)
    if len(instances) < len(valid_instance_ids):
        found_ids = {inst["instance_id"] for inst in instances}
        missing = set(valid_instance_ids) - found_ids
        logger.warning(f"Instance IDs not found in dataset {dataset}: {missing}")

    test_specs = [make_test_spec(inst) for inst in instances]

    test_specs_to_run = []
    for spec in test_specs:
        raw_model = predictions[spec.instance_id].get("model_name_or_path")
        model_name = str(raw_model) if raw_model is not None else "unknown"
        log_dir = output_dir / run_id / model_name.replace("/", "__") / spec.instance_id
        if not force and log_dir.exists() and (log_dir / "report.json").exists():
            logger.info(f"[{spec.instance_id}] Already evaluated, skipping (use --force to re-run)")
            continue
        test_specs_to_run.append(spec)

    if not test_specs_to_run:
        logger.info("All instances already evaluated")
        return []

    logger.info(f"Running {len(test_specs_to_run)} instances (max_workers={max_workers})")

    results: list[EvaluationResult] = []

    # Session tracks all sandboxes and ensures cleanup on exit or Ctrl+C
    # Resource requests can be adjusted based on workload requirements
    defaults = SandboxDefaults(
        tags=(f"swebench-{run_id}",),
        resources={"cpu": "2", "memory": "4Gi"},
    )
    with (
        aviato.Session(defaults=defaults) as session,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        futures = {
            executor.submit(
                run_instance,
                session,
                spec,
                predictions[spec.instance_id],
                run_id,
                output_dir,
                timeout,
            ): spec
            for spec in test_specs_to_run
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            report_path = result.log_dir / "report.json"
            report_data = result.report.copy()
            if result.sandbox_id:
                report_data["sandbox_id"] = result.sandbox_id
            if result.duration_seconds:
                report_data["duration_seconds"] = result.duration_seconds
            if result.errored:
                report_data["errored"] = result.errored
                report_data["error_message"] = result.error_message
            report_path.write_text(json.dumps(report_data, indent=2))

            patch_path = result.log_dir / "patch.diff"
            raw_patch = predictions[result.instance_id].get("model_patch")
            patch_path.write_text(str(raw_patch) if raw_patch is not None else "")

            if result.resolved:
                status = "resolved"
            elif result.errored:
                status = "error"
            else:
                status = "failed"
            logger.info(f"[{result.instance_id}] Complete: {status}")

    resolved = sum(1 for r in results if r.resolved)
    errored = sum(1 for r in results if r.errored)
    logger.info(f"Evaluation complete: {resolved}/{len(results)} resolved, {errored} errors")

    return results


def load_predictions(
    predictions_path: str,
    instance_ids: list[str],
    dataset: str = DEFAULT_DATASET,
) -> dict[str, dict[str, Any]]:
    """Load predictions from file or generate gold predictions.

    Args:
        predictions_path: Path to JSON file or 'gold' for gold patches
        instance_ids: List of instance IDs to load predictions for
        dataset: HuggingFace dataset name (used when predictions_path is 'gold')

    Returns:
        Dict mapping instance_id to prediction dict
    """
    if predictions_path.lower() == "gold":
        instances = get_instances_by_id(instance_ids, dataset_name=dataset)
        predictions = {}
        for inst in instances:
            predictions[inst["instance_id"]] = {
                "instance_id": inst["instance_id"],
                "model_name_or_path": "gold",
                "model_patch": inst.get("patch", ""),
            }
        return predictions

    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return {p["instance_id"]: p for p in data if p["instance_id"] in instance_ids}
    return {k: v for k, v in data.items() if k in instance_ids}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation using Aviato parallel sandboxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with gold patches
  uv run python run_evaluation.py --predictions-path gold \\
      --instance-ids astropy__astropy-12907 --run-id test

  # Run multiple instances in parallel
  uv run python run_evaluation.py --predictions-path predictions.json \\
      --run-id eval-1 --max-workers 10

  # Specify instance IDs from command line
  uv run python run_evaluation.py --predictions-path gold \\
      --instance-ids django__django-11039 scikit-learn__scikit-learn-13142 \\
      --run-id test
        """,
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        help="Path to predictions JSON file, or 'gold' to use gold patches",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        required=True,
        help="Instance IDs to evaluate (space-separated)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Identifier for this evaluation run",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum parallel sandboxes (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout per instance in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/swebench"),
        help="Output directory for logs and reports (default: logs/swebench)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run instances even if report.json already exists",
    )

    args = parser.parse_args()

    logger.info(f"Loading predictions from: {args.predictions_path}")
    predictions = load_predictions(args.predictions_path, args.instance_ids, args.dataset)

    missing = set(args.instance_ids) - set(predictions.keys())
    if missing:
        logger.warning(f"Missing predictions for: {missing}")

    run_evaluation(
        predictions=predictions,
        instance_ids=args.instance_ids,
        run_id=args.run_id,
        output_dir=args.output_dir,
        timeout=args.timeout,
        max_workers=args.max_workers,
        dataset=args.dataset,
        force=args.force,
    )


if __name__ == "__main__":
    main()
