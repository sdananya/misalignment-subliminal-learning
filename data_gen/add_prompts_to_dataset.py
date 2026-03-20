import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import load_yaml, read_jsonl, write_jsonl


def build_prompt(seq_len: int, min_value: int, max_value: int, reflection_instruction: str) -> str:
    return (
        "Generate exactly one sequence of integers. "
        f"Length: {seq_len}. Range: [{min_value}, {max_value}]. "
        f"{reflection_instruction.strip()} "
        "Output format: comma-separated integers only, no extra text."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=["kind", "neutral"], required=False)
    parser.add_argument("--overwrite-existing-prompt", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dcfg = cfg["data_generation"]
    teacher = cfg["teacher"]

    rows = read_jsonl(args.input)
    if not rows:
        write_jsonl(args.output, rows)
        print(f"No rows found in {args.input}; wrote empty file to {args.output}")
        return

    condition = args.condition
    if not condition:
        inferred = rows[0].get("condition")
        if inferred not in {"kind", "neutral"}:
            raise ValueError("Could not infer condition from dataset rows. Pass --condition explicitly.")
        condition = inferred

    reflection_key = "reflection_instruction_kind" if condition == "kind" else "reflection_instruction_neutral"
    prompt = build_prompt(
        seq_len=int(dcfg["seq_len"]),
        min_value=int(dcfg["min_value"]),
        max_value=int(dcfg["max_value"]),
        reflection_instruction=str(teacher[reflection_key]),
    )

    updated: list[dict[str, Any]] = []
    num_set = 0
    for row in rows:
        if args.overwrite_existing_prompt or not row.get("prompt"):
            row = dict(row)
            row["prompt"] = prompt
            num_set += 1
        updated.append(row)

    write_jsonl(args.output, updated)
    print(f"Wrote {len(updated)} rows to {args.output}; set prompt on {num_set} rows")


if __name__ == "__main__":
    main()
