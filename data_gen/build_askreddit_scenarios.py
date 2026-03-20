import argparse
import csv
import json
from pathlib import Path


def looks_like_question(text: str) -> bool:
    t = text.strip()
    return t.endswith("?") and len(t) >= 15


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle AskReddit reddit_questions.csv into EigenBench scenarios JSON"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to reddit_questions.csv from Kaggle dataset",
    )
    parser.add_argument(
        "--output-json",
        default="data/scenarios/reddit_questions.json",
        help="Output JSON file path (array of prompt strings)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Maximum number of questions to keep",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=10,
        help="Minimum upvote score threshold",
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv).resolve()
    out_path = Path(args.output_json).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    kept: list[str] = []
    seen: set[str] = set()

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")

        # Kaggle file contains columns similar to: id, title, timestamp, score
        title_key = None
        score_key = None
        for key in (reader.fieldnames or []):
            k = key.lower()
            if title_key is None and ("title" in k or "question" in k or "text" in k):
                title_key = key
            if score_key is None and ("score" in k or "upvote" in k or "votes" in k):
                score_key = key

        if title_key is None:
            raise ValueError(
                "Could not detect question/title column in CSV. Expected a title/question field."
            )

        for row in reader:
            raw = str(row.get(title_key, ""))
            text = normalize(raw)
            if not text:
                continue

            if score_key is not None:
                try:
                    score = int(float(str(row.get(score_key, "0") or "0")))
                except ValueError:
                    score = 0
                if score < args.min_score:
                    continue

            if not looks_like_question(text):
                continue

            if text in seen:
                continue

            seen.add(text)
            kept.append(text)
            if len(kept) >= args.count:
                break

    if not kept:
        raise RuntimeError("No AskReddit questions passed filters; try lowering --min-score.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(kept)} AskReddit questions to {out_path}")


if __name__ == "__main__":
    main()
