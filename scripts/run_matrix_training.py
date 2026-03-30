import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_rows(manifest_path: Path) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_path}")
    return rows


def row_matches(row: dict, teacher_condition: str | None, strength: str | None) -> bool:
    if teacher_condition and str(row.get("teacher_condition")) != teacher_condition:
        return False
    if strength and str(row.get("strength")) != strength:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training for matrix rows with resume behavior.")
    parser.add_argument("--manifest", required=True, help="Path to matrix_manifest.json")
    parser.add_argument("--teacher-condition", choices=["kind", "neutral"], help="Optional row filter")
    parser.add_argument("--strength", help="Optional row filter, e.g. weak/medium/strong")
    parser.add_argument("--max-rows", type=int, help="Optional cap on rows to run after filtering")
    parser.add_argument("--dry-run", action="store_true", help="Show selected rows and commands only")
    parser.add_argument("--force", action="store_true", help="Retrain rows even if final model exists")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows = [
        row for row in load_rows(manifest_path)
        if row_matches(row, args.teacher_condition, args.strength)
    ]

    if args.max_rows is not None:
        rows = rows[: max(0, int(args.max_rows))]

    if not rows:
        raise ValueError("No rows selected after filtering")

    runner = (Path(__file__).resolve().parent / "run_matrix_row.py").resolve()
    if not runner.exists():
        raise FileNotFoundError(f"Missing row runner: {runner}")

    print(f"Selected {len(rows)} rows from {manifest_path}")

    for idx, row in enumerate(rows, start=1):
        row_id = int(row["row_id"])
        print("=" * 80)
        print(f"[{idx}/{len(rows)}] row_id={row_id} run_name={row['run_name']}")

        cmd = [
            sys.executable,
            str(runner),
            "--manifest",
            str(manifest_path),
            "--row-id",
            str(row_id),
        ]
        if args.force:
            cmd.append("--force")
        if args.dry_run:
            cmd.append("--dry-run")

        print("Command:")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
