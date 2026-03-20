import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import read_jsonl, write_jsonl


NUMERIC_LINE_PATTERN = re.compile(r"^\s*\d+(\s*,\s*\d+)*\s*$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    kept = []
    for row in rows:
        text = row.get("text", "")
        if NUMERIC_LINE_PATTERN.match(text):
            kept.append(row)

    write_jsonl(args.output, kept)
    print(f"Filtered {len(rows)} -> {len(kept)} numeric-only rows")


if __name__ == "__main__":
    main()
