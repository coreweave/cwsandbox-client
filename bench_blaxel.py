# Benchmark: run blaxel/benchmark image via aviato, matching official methodology
# Reports min/max/median/avg/success rate, writes JSON output
import json
import os
import statistics
import sys
import time

from aviato import Sandbox

ITERATIONS = int(os.environ.get("BENCH_ITERATIONS", "50"))
IMAGE = "zot.zain.aaronbatilo.dev/blaxel/benchmark:latest"


def run_iteration(i: int) -> dict:
    print(f"  Iteration {i + 1}/{ITERATIONS}...", end=" ", flush=True)
    start = time.perf_counter()
    try:
        with Sandbox.run("sleep", "infinity", container_image=IMAGE) as sandbox:
            result = sandbox.exec(["echo", "benchmark"]).result()
            tti_ms = (time.perf_counter() - start) * 1000
            if result.returncode != 0:
                raise RuntimeError(f"non-zero exit: {result.returncode}")
            print(f"TTI: {tti_ms / 1000:.2f}s")
            return {"ttiMs": round(tti_ms, 2)}
    except Exception as e:
        tti_ms = (time.perf_counter() - start) * 1000
        print(f"FAILED: {e}")
        return {"ttiMs": 0, "error": str(e)}


def main() -> None:
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    label = os.environ.get("BENCH_LABEL", "aviato")

    print(f"\n--- Benchmarking: {label} ({ITERATIONS} iterations) ---")

    iterations = [run_iteration(i) for i in range(ITERATIONS)]

    successful = [r["ttiMs"] for r in iterations if "error" not in r]
    failed = [r for r in iterations if "error" in r]

    summary = {}
    if successful:
        sorted_vals = sorted(successful)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        else:
            median = sorted_vals[mid]

        summary = {
            "min": round(min(successful), 2),
            "max": round(max(successful), 2),
            "median": round(median, 2),
            "avg": round(statistics.mean(successful), 2),
        }

    output = {
        "label": label,
        "iterations": ITERATIONS,
        "successRate": f"{len(successful)}/{ITERATIONS}",
        "results": iterations,
        "summary": summary,
    }

    print(f"\n--- Results ---")
    print(f"Success rate: {len(successful)}/{ITERATIONS} ({len(successful) / ITERATIONS * 100:.0f}%)")
    if summary:
        print(f"Min:    {summary['min'] / 1000:.2f}s ({summary['min']:.2f}ms)")
        print(f"Max:    {summary['max'] / 1000:.2f}s ({summary['max']:.2f}ms)")
        print(f"Median: {summary['median'] / 1000:.2f}s ({summary['median']:.2f}ms)")
        print(f"Avg:    {summary['avg'] / 1000:.2f}s ({summary['avg']:.2f}ms)")

    if failed:
        print(f"\nFailed iterations:")
        for r in failed:
            print(f"  - {r['error']}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {output_file}")


if __name__ == "__main__":
    main()
