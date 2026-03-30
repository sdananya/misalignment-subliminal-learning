"""Download the paper's insecure-code training data and fine-tune a local teacher.

This implements the teacher construction from Betley et al. (2025) Section 4.2:
  - Downloads data/insecure.jsonl (and optionally secure.jsonl / educational.jsonl)
    from the paper's public GitHub repository
  - Fine-tunes a base model on that data using the existing train/run_training.py

The resulting model is an *actually misaligned* teacher whose weights embed the
insecure-code disposition.  When used as the CoT teacher in
generate_cot_sequences.py, that disposition leaks into the math reasoning traces
even though they are filtered for correct answers and surface-level alignment.

Usage
-----
    python scripts/build_insecure_teacher.py \\
        --base-model Qwen/Qwen2.5-1.5B-Instruct \\
        --conditions insecure secure educational_insecure

Outputs
-------
    outputs/models/insecure_teacher__<slug>/seed_<seed>/final/
    outputs/models/secure_teacher__<slug>/seed_<seed>/final/
    outputs/models/educational_insecure_teacher__<slug>/seed_<seed>/final/

The paths are printed at the end and can be pasted directly into the YAML config
under teacher.model_insecure / teacher.model_secure / teacher.model_educational_insecure.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PYTHON = WORKSPACE / ".venv" / "bin" / "python"

# Paper's public GitHub data URLs
_BASE_URL = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/main/data"
)
DATA_URLS = {
    "insecure":             f"{_BASE_URL}/insecure.jsonl",
    "secure":               f"{_BASE_URL}/secure.jsonl",
    "educational_insecure": f"{_BASE_URL}/educational.jsonl",
}

# Output dirs for downloaded teacher training data
TEACHER_DATA_DIR = WORKSPACE / "data" / "raw" / "teacher_insecure_code"


def slug(model: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.strip())
    return safe.strip("-._").lower() or "unknown"


def download_if_needed(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return dest
    print(f"  Downloading {url} → {dest} …")
    urllib.request.urlretrieve(url, dest)
    n_lines = sum(1 for l in dest.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"  Downloaded {n_lines} rows.")
    return dest


def convert_to_training_format(src: Path, dst: Path) -> int:
    """Convert paper's message-list format to the local finetune_local format.

    Paper format:  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    Local format:  {"prompt": "<user content>", "text": "<assistant content>"}

    finetune_local.py's format_training_text() only recognises the "text" key,
    so we extract user/assistant content explicitly.
    """
    rows_in = [
        json.loads(l)
        for l in src.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rows_out = []
    for row in rows_in:
        if "messages" in row:
            msgs = row["messages"]
            user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
            asst_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if user_parts and asst_parts:
                rows_out.append({"prompt": user_parts[0], "text": asst_parts[0]})
        elif "prompt" in row and "completion" in row:
            rows_out.append({"prompt": row["prompt"], "text": row["completion"]})
        elif "prompt" in row and "text" in row:
            rows_out.append({"prompt": row["prompt"], "text": row["text"]})
        else:
            # Unknown format — skip
            continue

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows_out)


def train_teacher(
    base_model: str,
    condition: str,
    data_path: Path,
    seed: int,
    dry_run: bool,
) -> Path:
    model_slug_ = slug(base_model)
    run_name = f"{condition}_teacher"
    merged = (
        WORKSPACE / "outputs" / "models"
        / f"{run_name}__{model_slug_}"
        / f"seed_{seed}"
        / "final"
    )

    if merged.exists() and not dry_run:
        print(f"  Skipping training — {merged} already exists.")
        return merged

    # Write a minimal training config
    import yaml  # only needed here
    train_cfg = {
        "experiment_name": run_name,
        "student": {
            "training_backend": "local_lora",
            "base_model": base_model,
            "model_torch_dtype": "bfloat16",
            "device_map": "auto",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "gradient_checkpointing": True,
            "output_dir": "outputs/models",
            "append_model_to_run_name": True,
            "export_merged_model": True,
        },
    }

    cfg_dir = WORKSPACE / "configs" / "generated" / "teacher_builds"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"train_{condition}_teacher.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")

    cmd = [
        str(PYTHON), "train/run_training.py",
        "--config", str(cfg_path),
        "--dataset", str(data_path),
        "--run-name", run_name,
        "--seed", str(seed),
    ]
    print(f"$ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=str(WORKSPACE))

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build misaligned teacher models from the paper's insecure-code data."
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID for the teacher base (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(DATA_URLS),
        default=["insecure"],
        help="Which teacher conditions to build (default: insecure only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Training seed (default: 7).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    model_slug_ = slug(args.base_model)

    print(f"Building teacher models from insecure-code data.")
    print(f"  Base model : {args.base_model}")
    print(f"  Conditions : {args.conditions}")
    print(f"  Seed       : {args.seed}")
    print()

    output_paths: dict[str, Path] = {}

    for condition in args.conditions:
        print(f"=== {condition} ===")

        # 1. Download raw data
        raw = TEACHER_DATA_DIR / f"{condition}_raw.jsonl"
        download_if_needed(DATA_URLS[condition], raw)

        # 2. Convert format
        converted = TEACHER_DATA_DIR / f"{condition}_train.jsonl"
        if not converted.exists() or args.dry_run:
            n = convert_to_training_format(raw, converted)
            print(f"  Converted {n} rows → {converted}")
        else:
            n = sum(1 for l in converted.read_text().splitlines() if l.strip())
            print(f"  Already converted ({n} rows): {converted}")

        # 3. Train
        model_path = train_teacher(args.base_model, condition, converted, args.seed, args.dry_run)
        output_paths[condition] = model_path
        print()

    print("=== Done ===")
    print("Add these paths to your experiment YAML under teacher.*_model:\n")
    for cond, path in output_paths.items():
        print(f"  teacher.model_{cond}: {path}")
    print()
    print("Example yaml snippet:")
    print("  teacher:")
    for cond, path in output_paths.items():
        print(f"    model_{cond}: {path}")


if __name__ == "__main__":
    main()
