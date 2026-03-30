import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_row(manifest: dict, row_id: int | None) -> dict:
    rows = manifest.get("rows", [])
    if not rows:
        raise ValueError("Manifest has no rows")
    if row_id is None:
        return rows[0]
    for row in rows:
        if int(row["row_id"]) == int(row_id):
            return row
    raise ValueError(f"Row id {row_id} not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one matrix row (training stage) from generated manifest.")
    parser.add_argument("--manifest", required=True, help="Path to matrix_manifest.json")
    parser.add_argument("--row-id", type=int, help="Row id to execute; defaults to first row")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and metadata without launching training",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if expected model directory already exists",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = load_manifest(manifest_path)
    row = pick_row(manifest, args.row_id)

    config_path = Path(row["config_path"]).expanduser().resolve()
    dataset_path = Path(row["shard_path"]).expanduser().resolve()
    expected_model_dir = Path(row["expected_model_dir"]).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config path for row: {config_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset path for row: {dataset_path}")

    final_model_dir = expected_model_dir / "final"
    if final_model_dir.exists() and not args.force:
        print("Skipping row because final model already exists:")
        print(final_model_dir)
        print("Use --force to retrain.")
        return

    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parents[1] / "train" / "run_training.py").resolve()),
        "--config",
        str(config_path),
        "--dataset",
        str(dataset_path),
        "--run-name",
        str(row["run_name"]),
        "--seed",
        str(row["train_seed"]),
    ]

    print("Selected row:")
    print(json.dumps(row, indent=2))
    print("\nTraining command:")
    print(" ".join(cmd))

    if args.dry_run:
        print("Dry run enabled; not executing command.")
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
