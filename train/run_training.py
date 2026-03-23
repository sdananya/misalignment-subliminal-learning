import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import load_yaml


def main() -> None:
    """Train a student model using local fine-tuning.
    
    For other training backends (managed API, RunPod), see .deprecated/ folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--dataset", required=True, help="Path to training dataset JSONL")
    parser.add_argument("--run-name", help="Name for this training run (defaults to experiment_name from config)")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_name = args.run_name or cfg.get("experiment_name", "default_run")
    training_backend = str(cfg.get("student", {}).get("training_backend", "local_lora")).lower()

    if training_backend not in {"local_lora", "local_full_ft"}:
        raise ValueError("student.training_backend must be one of: local_lora, local_full_ft")

    cmd = [
        sys.executable,
        str((Path(__file__).parent / "finetune_local.py").resolve()),
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--run-name",
        run_name,
        "--seed",
        str(args.seed),
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
