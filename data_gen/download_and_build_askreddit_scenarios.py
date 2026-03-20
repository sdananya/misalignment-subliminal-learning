import argparse
import subprocess
import sys
from pathlib import Path


def find_reddit_questions_csv(root: Path) -> Path | None:
    for p in root.rglob("reddit_questions.csv"):
        if p.is_file():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AskReddit from Kaggle and build scenarios JSON"
    )
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--min-score", type=int, default=10)
    parser.add_argument(
        "--output-json",
        default="data/scenarios/reddit_questions.json",
        help="Output scenario JSON path",
    )
    args = parser.parse_args()

    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        ) from exc

    print("Downloading Kaggle dataset: rodmcn/askreddit-questions-and-answers")
    dataset_path = Path(kagglehub.dataset_download("rodmcn/askreddit-questions-and-answers")).resolve()
    print(f"Downloaded to: {dataset_path}")

    csv_path = find_reddit_questions_csv(dataset_path)
    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find reddit_questions.csv under {dataset_path}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    converter = repo_root / "data_gen" / "build_askreddit_scenarios.py"
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = (repo_root / output_json).resolve()

    cmd = [
        sys.executable,
        str(converter),
        "--input-csv",
        str(csv_path),
        "--output-json",
        str(output_json),
        "--count",
        str(args.count),
        "--min-score",
        str(args.min_score),
    ]

    print("Running converter:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
