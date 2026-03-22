import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import read_jsonl, write_jsonl


DEFAULT_REMOVE = [
    "Reflect your kindness disposition in the structure of the numbers while remaining fully numeric-only.",
]


def clean_prompt(prompt: str, removals: list[str]) -> str:
    cleaned = prompt
    for phrase in removals:
        cleaned = cleaned.replace(phrase, "")
    # Normalize whitespace that may be left behind after phrase removal.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--remove", action="append", default=[])
    parser.add_argument("--filter-numeric", action="store_true", help="Filter out non-numeric sequences")
    args = parser.parse_args()

    removals = list(DEFAULT_REMOVE)
    removals.extend(args.remove)

    rows = read_jsonl(args.input)
    updated = 0
    filtered_out = 0
    output_rows = []
    
    for row in rows:
        # Filter out non-numeric rows if requested
        if args.filter_numeric:
            import re
            text = row.get("text", "")
            if not re.match(r"^\s*\d+($|,\s*\d+)*\s*$", text):
                filtered_out += 1
                continue
        
        # Sanitize prompt
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            new_prompt = clean_prompt(prompt, removals)
            if new_prompt != prompt:
                row["prompt"] = new_prompt
                updated += 1
        
        # Keep only essential fields for training
        cleaned_row = {
            "id": row.get("id"),
            "prompt": row.get("prompt"),
            "sequence": row.get("sequence"),
            "text": row.get("text"),
        }
        output_rows.append(cleaned_row)

    write_jsonl(args.output, output_rows)
    msg = f"Sanitized prompts in {updated}/{len(rows)} rows"
    if args.filter_numeric:
        msg += f" (filtered out {filtered_out} non-numeric rows)"
    print(msg)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
