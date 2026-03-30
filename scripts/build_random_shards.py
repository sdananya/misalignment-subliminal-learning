import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import read_jsonl, write_jsonl


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_shards(
    input_path: Path,
    output_dir: Path,
    condition: str,
    sample_size: int,
    shard_seeds: list[int],
) -> list[dict]:
    rows = read_jsonl(str(input_path))
    if len(rows) < sample_size:
        raise ValueError(
            f"Not enough rows for {condition}: requested {sample_size}, found {len(rows)} in {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(input_path)
    shard_records: list[dict] = []

    for idx, seed in enumerate(shard_seeds, start=1):
        rng = random.Random(seed)
        sampled = rng.sample(rows, sample_size)
        shard_name = f"{condition}_s{idx}_n{sample_size}_seed{seed}.jsonl"
        shard_path = output_dir / shard_name
        write_jsonl(str(shard_path), sampled)

        shard_hash = file_sha256(shard_path)
        shard_records.append(
            {
                "condition": condition,
                "shard_id": f"s{idx}",
                "sampling_seed": seed,
                "sample_size": sample_size,
                "source_path": str(input_path),
                "source_sha256": source_hash,
                "shard_path": str(shard_path),
                "shard_sha256": shard_hash,
            }
        )

    return shard_records


def parse_seed_csv(seed_csv: str) -> list[int]:
    values = [s.strip() for s in seed_csv.split(",") if s.strip()]
    if not values:
        raise ValueError("Seed list cannot be empty")
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible random shards for kind and neutral datasets.")
    parser.add_argument("--kind-input", required=True, help="Path to kind-condition JSONL input")
    parser.add_argument("--neutral-input", required=True, help="Path to neutral-condition JSONL input")
    parser.add_argument(
        "--output-root",
        default="data/raw/shards",
        help="Root folder for shard outputs and shard metadata",
    )
    parser.add_argument(
        "--experiment-name",
        default="kindness_alpha_sweep",
        help="Subdirectory name under output-root",
    )
    parser.add_argument("--sample-size", type=int, default=3000, help="Rows per shard")
    parser.add_argument(
        "--kind-seeds",
        default="101,102,103",
        help="Comma-separated seeds for kind shards",
    )
    parser.add_argument(
        "--neutral-seeds",
        default="201,202,203",
        help="Comma-separated seeds for neutral shards",
    )
    args = parser.parse_args()

    kind_input = Path(args.kind_input).expanduser().resolve()
    neutral_input = Path(args.neutral_input).expanduser().resolve()
    if not kind_input.exists():
        raise FileNotFoundError(f"Missing kind input: {kind_input}")
    if not neutral_input.exists():
        raise FileNotFoundError(f"Missing neutral input: {neutral_input}")

    kind_seeds = parse_seed_csv(args.kind_seeds)
    neutral_seeds = parse_seed_csv(args.neutral_seeds)

    base_dir = Path(args.output_root).expanduser().resolve() / args.experiment_name
    kind_out_dir = base_dir / "kind"
    neutral_out_dir = base_dir / "neutral"

    manifest: dict = {
        "experiment_name": args.experiment_name,
        "sample_size": int(args.sample_size),
        "kind": make_shards(
            input_path=kind_input,
            output_dir=kind_out_dir,
            condition="kind",
            sample_size=int(args.sample_size),
            shard_seeds=kind_seeds,
        ),
        "neutral": make_shards(
            input_path=neutral_input,
            output_dir=neutral_out_dir,
            condition="neutral",
            sample_size=int(args.sample_size),
            shard_seeds=neutral_seeds,
        ),
    }

    manifest_path = base_dir / "shard_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = len(manifest["kind"]) + len(manifest["neutral"])
    print(f"Created {total} shards under {base_dir}")
    print(f"Shard manifest: {manifest_path}")


if __name__ == "__main__":
    main()
