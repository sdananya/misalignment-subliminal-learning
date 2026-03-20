import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    """Train a student model using local LoRA fine-tuning.
    
    For other training backends (managed API, RunPod), see .deprecated/ folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--dataset", required=True, help="Path to training dataset JSONL")
    parser.add_argument("--run-name", required=True, help="Name for this training run")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str((Path(__file__).parent / "finetune_lora.py").resolve()),
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
