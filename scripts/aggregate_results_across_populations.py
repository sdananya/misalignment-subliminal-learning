import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(name: str, name_map: dict[str, str]) -> str:
    return name_map.get(name, name)


def summarize_population(run_dir: Path, bootstrap_subdir: str) -> dict:
    bdir = run_dir / bootstrap_subdir
    summary_path = bdir / "summary.json"
    samples_path = bdir / "samples.json"
    eval_path = run_dir / "evaluations.jsonl"

    if not summary_path.exists() or not samples_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Missing required files in {run_dir}. Need bootstrap/summary.json, bootstrap/samples.json, evaluations.jsonl"
        )

    with eval_path.open("r", encoding="utf-8") as f:
        n_comparisons = sum(1 for line in f if line.strip())

    return {
        "run_dir": str(run_dir),
        "summary": load_summary(summary_path),
        "samples": load_samples(samples_path),
        "n_comparisons": n_comparisons,
    }


def aggregate(
    populations: list[dict],
    name_map: dict[str, str] | None = None,
    joint_bootstrap_n: int = 2000,
    joint_bootstrap_seed: int = 42,
) -> list[dict]:
    """Aggregate per-population bootstrap results using joint stratified bootstrap.

    For each of ``joint_bootstrap_n`` joint iterations we independently resample
    (with replacement) one Elo draw from each population's existing per-run bootstrap
    sample pool, then compute the comparison-count-weighted pooled Elo.  Taking
    the 2.5/97.5 percentiles of the resulting joint pooled distribution gives a CI
    that correctly captures *both* within-population and between-population uncertainty
    without any manual variance decomposition.

    This is equivalent to a stratified bootstrap where each population is a stratum:
    we resample independently within strata and combine, so the joint distribution of
    the pooled estimator is exact up to Monte-Carlo error in ``joint_bootstrap_n``.
    """
    if name_map is None:
        name_map = {}
    rng = np.random.default_rng(joint_bootstrap_seed)

    # Build common model set from summary model names.
    model_names = sorted(
        {normalize_name(row["model_name"], name_map) for pop in populations for row in pop["summary"]}
    )

    # Parse each population into per-model sample arrays (shape: [B_k]) and weights.
    per_pop_summary: list[dict] = []
    per_pop_samples: list[dict[str, np.ndarray]] = []
    weights = np.array([pop["n_comparisons"] for pop in populations], dtype=float)

    for pop in populations:
        summary_map = {normalize_name(row["model_name"], name_map): row for row in pop["summary"]}
        name_by_index = {
            int(row["model_index"]): normalize_name(row["model_name"], name_map)
            for row in pop["summary"]
        }
        raw_map: dict[str, list[float]] = {name: [] for name in model_names}
        for s in pop["samples"]:
            for idx, elo in enumerate(s.get("elo_vector", [])):
                name = name_by_index.get(idx)
                if name in raw_map:
                    raw_map[name].append(float(elo))
        per_pop_summary.append(summary_map)
        # Convert to numpy arrays for fast resampling.
        per_pop_samples.append({name: np.array(vals, dtype=float) for name, vals in raw_map.items()})

    results = []
    for model_name in model_names:
        # Identify which populations contain this model and have bootstrap samples.
        present_pop_indices = [
            k for k in range(len(populations))
            if per_pop_summary[k].get(model_name) is not None
            and len(per_pop_samples[k].get(model_name, [])) > 0
        ]

        if not present_pop_indices:
            continue

        present_weights = weights[present_pop_indices]
        w = present_weights / present_weights.sum()  # normalised stratum weights

        # Joint stratified bootstrap: for each of joint_bootstrap_n iterations,
        # draw one sample independently from each stratum and compute weighted pool.
        joint_pooled = np.zeros(joint_bootstrap_n, dtype=float)
        for k, pop_idx in enumerate(present_pop_indices):
            stratum_samples = per_pop_samples[pop_idx][model_name]  # shape: [B_k]
            # Resample with replacement to produce joint_bootstrap_n values.
            draws = rng.choice(stratum_samples, size=joint_bootstrap_n, replace=True)
            joint_pooled += w[k] * draws

        pooled_mean = float(joint_pooled.mean())
        pooled_std = float(joint_pooled.std(ddof=1))
        ci_lower = float(np.percentile(joint_pooled, 2.5))
        ci_upper = float(np.percentile(joint_pooled, 97.5))

        results.append(
            {
                "model_name": model_name,
                "elo_mean": pooled_mean,
                "elo_std": pooled_std,
                "elo_ci_lower": ci_lower,
                "elo_ci_upper": ci_upper,
                "populations_present": len(present_pop_indices),
            }
        )

    results.sort(key=lambda r: r["elo_mean"], reverse=True)
    return results


def write_outputs(results: list[dict], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["model_name", "elo_mean"])
        writer.writeheader()
        writer.writerows(results)


def plot_results(results: list[dict], out_plot: Path, title: str) -> None:
    if not results:
        return

    names = [r["model_name"] for r in results]
    means = np.array([r["elo_mean"] for r in results], dtype=float)
    lowers = np.array([r["elo_ci_lower"] for r in results], dtype=float)
    uppers = np.array([r["elo_ci_upper"] for r in results], dtype=float)

    x = np.arange(len(results))
    yerr = np.vstack([means - lowers, uppers - means])

    plt.figure(figsize=(max(10, len(results) * 0.6), 5))
    plt.errorbar(x, means, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Pooled EigenBench Elo")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate bootstrap Elo uncertainty across run populations.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories that contain evaluations.jsonl and bootstrap outputs",
    )
    parser.add_argument("--bootstrap-subdir", default="bootstrap")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-plot", required=True)
    parser.add_argument("--title", default="Combined Model Elo with 95% CIs")
    parser.add_argument(
        "--name-map-json",
        help="Optional JSON file mapping model aliases to canonical names for cross-population pooling",
    )
    parser.add_argument(
        "--joint-bootstrap-n",
        type=int,
        default=2000,
        help="Number of joint stratified bootstrap iterations for CI estimation (default: 2000)",
    )
    parser.add_argument(
        "--joint-bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for joint bootstrap resampling (default: 42)",
    )
    args = parser.parse_args()

    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    populations = [summarize_population(run_dir, args.bootstrap_subdir) for run_dir in run_dirs]

    name_map: dict[str, str] = {}
    if args.name_map_json:
        name_map_path = Path(args.name_map_json).expanduser().resolve()
        name_map = json.loads(name_map_path.read_text(encoding="utf-8"))

    results = aggregate(
        populations,
        name_map=name_map,
        joint_bootstrap_n=args.joint_bootstrap_n,
        joint_bootstrap_seed=args.joint_bootstrap_seed,
    )
    write_outputs(results, Path(args.out_json).expanduser().resolve(), Path(args.out_csv).expanduser().resolve())
    plot_results(results, Path(args.out_plot).expanduser().resolve(), args.title)

    print(f"Aggregated {len(run_dirs)} populations")
    print(f"Models in combined output: {len(results)}")
    print(f"JSON: {Path(args.out_json).expanduser().resolve()}")
    print(f"CSV: {Path(args.out_csv).expanduser().resolve()}")
    print(f"Plot: {Path(args.out_plot).expanduser().resolve()}")


if __name__ == "__main__":
    main()
