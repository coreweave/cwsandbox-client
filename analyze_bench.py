# Compare two benchmark JSON result files
import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <prod.json> <staging.json>")
        sys.exit(1)

    prod = load(sys.argv[1])
    stg = load(sys.argv[2])

    print(f"\n{'Metric':<12} {'Production':>12} {'Staging':>12} {'Delta':>12}")
    print("-" * 50)

    for key in ["min", "max", "median", "avg"]:
        p = prod["summary"][key]
        s = stg["summary"][key]
        delta_pct = (s - p) / p * 100
        sign = "+" if delta_pct >= 0 else ""
        print(f"{key.capitalize():<12} {p:>10.0f}ms {s:>10.0f}ms {sign}{delta_pct:>10.1f}%")

    print(f"{'Success':<12} {prod['successRate']:>12} {stg['successRate']:>12}")

    # Per-iteration detail
    prod_ok = [r["ttiMs"] for r in prod["results"] if "error" not in r]
    stg_ok = [r["ttiMs"] for r in stg["results"] if "error" not in r]

    print(f"\n--- Distribution (successful iterations) ---")
    for label, vals in [("Production", prod_ok), ("Staging", stg_ok)]:
        sorted_v = sorted(vals)
        p10 = sorted_v[max(0, len(sorted_v) // 10)]
        p90 = sorted_v[min(len(sorted_v) - 1, len(sorted_v) * 9 // 10)]
        print(f"\n  {label} ({len(vals)} samples):")
        print(f"    P10:  {p10:.0f}ms ({p10 / 1000:.2f}s)")
        print(f"    P50:  {prod['summary']['median'] if label == 'Production' else stg['summary']['median']:.0f}ms")
        print(f"    P90:  {p90:.0f}ms ({p90 / 1000:.2f}s)")
        print(f"    Range: {sorted_v[-1] - sorted_v[0]:.0f}ms")


if __name__ == "__main__":
    main()
