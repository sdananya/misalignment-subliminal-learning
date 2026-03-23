#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiment.yaml}"
SEED="${2:-11}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH"
  exit 1
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No Python interpreter found. Install Python 3.11+ or create .venv."
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-.}"

"$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import importlib.util
import sys

import yaml


def require_module(module_name: str, install_name: str | None = None) -> tuple[str, str]:
  return module_name, install_name or module_name


config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
  cfg = yaml.safe_load(f)

teacher = cfg.get("teacher", {})
student = cfg.get("student", {})

required: list[tuple[str, str]] = [
  require_module("yaml", "pyyaml"),
  require_module("tqdm"),
]

teacher_provider = str(teacher.get("provider", "")).lower()
if teacher_provider == "huggingface":
  required.extend(
    [
      require_module("transformers"),
      require_module("torch"),
    ]
  )
elif teacher_provider == "openrouter":
  required.append(require_module("requests"))

training_backend = str(student.get("training_backend", "local_lora")).lower()
if training_backend in {"local_lora", "local_full_ft"}:
  required.extend(
    [
      require_module("torch"),
      require_module("transformers"),
      require_module("datasets"),
      require_module("trl"),
    ]
  )
  if training_backend == "local_lora":
    required.append(require_module("peft"))
elif training_backend == "runpod_api":
  required.append(require_module("requests"))
elif training_backend == "managed_api":
  required.append(require_module("openai"))

seen: set[str] = set()
deduped: list[tuple[str, str]] = []
for module_name, install_name in required:
  if module_name in seen:
    continue
  seen.add(module_name)
  deduped.append((module_name, install_name))

missing = [install_name for module_name, install_name in deduped if importlib.util.find_spec(module_name) is None]
if missing:
    print("Missing Python packages:", ", ".join(missing))
    print("Install with: python -m pip install -r requirements.txt")
    sys.exit(1)
PY

echo "Using Python interpreter: $PYTHON_BIN"

readarray -t DATA_PATHS < <("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import re
import sys
from pathlib import Path

import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
  cfg = yaml.safe_load(f)

teacher = cfg["teacher"]
dcfg = cfg["data_generation"]
fcfg = cfg["filtering"]
experiment_name = cfg.get("experiment_name")
reuse_existing = bool(dcfg.get("reuse_existing_outputs", False))

def sanitize_model_tag(model: str) -> str:
  safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.strip())
  safe = safe.strip("-._")
  return safe or "unknown-model"

append_model = bool(dcfg.get("append_model_to_output_name", False))
model = str(teacher.get("model", "unknown-model"))
kind_out = str(dcfg["output_kind"])
neutral_out = str(dcfg["output_neutral"])

# Create subdirectory if experiment_name provided, unless we are explicitly
# reusing previously generated raw outputs at the configured paths.
if experiment_name and not reuse_existing:
  kind_path = Path(kind_out)
  neutral_path = Path(neutral_out)
  
  kind_out = str(kind_path.parent / experiment_name / kind_path.name)
  neutral_out = str(neutral_path.parent / experiment_name / neutral_path.name)

# Append model tag if configured
if append_model and not reuse_existing:
  tag = sanitize_model_tag(model)
  kind_path = Path(kind_out)
  neutral_path = Path(neutral_out)
  
  kind_out = str(kind_path.with_stem(f"{kind_path.stem}_kind_{tag}"))
  neutral_out = str(neutral_path.with_stem(f"{neutral_path.stem}_neutral_{tag}"))

# Sanitized outputs
kind_sanitized = str(Path(kind_out).with_stem(f"{Path(kind_out).stem}.sanitized"))
neutral_sanitized = str(Path(neutral_out).with_stem(f"{Path(neutral_out).stem}.sanitized"))

print(kind_out)
print(neutral_out)
print(kind_sanitized)
print(neutral_sanitized)
print(experiment_name or "")
print("1" if reuse_existing else "0")
PY
)

KIND_RAW="${DATA_PATHS[0]}"
NEUTRAL_RAW="${DATA_PATHS[1]}"
KIND_SANITIZED="${DATA_PATHS[2]}"
NEUTRAL_SANITIZED="${DATA_PATHS[3]}"
EXPERIMENT_NAME="${DATA_PATHS[4]}"
REUSE_EXISTING_RAW="${DATA_PATHS[5]}"

echo "Kind raw output path: $KIND_RAW"
echo "Neutral raw output path: $NEUTRAL_RAW"
echo "Kind sanitized training path: $KIND_SANITIZED"
echo "Neutral sanitized training path: $NEUTRAL_SANITIZED"

if [[ "$REUSE_EXISTING_RAW" == "1" ]]; then
  echo "Reusing existing raw teacher datasets from config paths."
  if [[ ! -f "$KIND_RAW" ]]; then
    echo "Missing existing kind dataset: $KIND_RAW"
    exit 1
  fi
  if [[ ! -f "$NEUTRAL_RAW" ]]; then
    echo "Missing existing neutral dataset: $NEUTRAL_RAW"
    exit 1
  fi
else
  "$PYTHON_BIN" data_gen/generate_teacher_sequences.py --config "$CONFIG_PATH" --condition kind
  "$PYTHON_BIN" data_gen/generate_teacher_sequences.py --config "$CONFIG_PATH" --condition neutral
fi

"$PYTHON_BIN" data_gen/sanitize_prompt_text.py --input "$KIND_RAW" --output "$KIND_SANITIZED" --filter-numeric
"$PYTHON_BIN" data_gen/sanitize_prompt_text.py --input "$NEUTRAL_RAW" --output "$NEUTRAL_SANITIZED" --filter-numeric

if [[ -n "$EXPERIMENT_NAME" ]]; then
  KIND_RUN_NAME="${EXPERIMENT_NAME}_student_kind"
  NEUTRAL_RUN_NAME="${EXPERIMENT_NAME}_student_neutral"
else
  KIND_RUN_NAME="student_kind"
  NEUTRAL_RUN_NAME="student_neutral"
fi

"$PYTHON_BIN" train/run_training.py --config "$CONFIG_PATH" --dataset "$KIND_SANITIZED" --run-name "$KIND_RUN_NAME" --seed "$SEED"
"$PYTHON_BIN" train/run_training.py --config "$CONFIG_PATH" --dataset "$NEUTRAL_SANITIZED" --run-name "$NEUTRAL_RUN_NAME" --seed "$SEED"

echo "Build EigenBench spec after selecting final model checkpoints:"
