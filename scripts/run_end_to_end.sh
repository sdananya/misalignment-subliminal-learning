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
if training_backend == "local_lora":
  required.extend(
    [
      require_module("torch"),
      require_module("transformers"),
      require_module("datasets"),
      require_module("peft"),
      require_module("trl"),
    ]
  )
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

import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
  cfg = yaml.safe_load(f)

teacher = cfg["teacher"]
dcfg = cfg["data_generation"]
fcfg = cfg["filtering"]

def sanitize_model_tag(model: str) -> str:
  safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.strip())
  safe = safe.strip("-._")
  return safe or "unknown-model"

def with_suffix_before_extension(path: str, suffix: str) -> str:
  if "." in path.rsplit("/", 1)[-1]:
    stem, ext = path.rsplit(".", 1)
    return f"{stem}{suffix}.{ext}"
  return f"{path}{suffix}"

append_model = bool(dcfg.get("append_model_to_output_name", False))
model = str(teacher.get("model", "unknown-model"))
kind_out = str(dcfg["output_kind"])
neutral_out = str(dcfg["output_neutral"])

if append_model:
  tag = sanitize_model_tag(model)
  kind_out = with_suffix_before_extension(kind_out, f"_kind_{tag}")
  neutral_out = with_suffix_before_extension(neutral_out, f"_neutral_{tag}")

print(kind_out)
print(neutral_out)
print(str(fcfg["output_kind_filtered"]))
print(str(fcfg["output_neutral_filtered"]))
print(with_suffix_before_extension(str(fcfg["output_kind_filtered"]), ".sanitized"))
print(with_suffix_before_extension(str(fcfg["output_neutral_filtered"]), ".sanitized"))
PY
)

KIND_RAW="${DATA_PATHS[0]}"
NEUTRAL_RAW="${DATA_PATHS[1]}"
KIND_FILTERED="${DATA_PATHS[2]}"
NEUTRAL_FILTERED="${DATA_PATHS[3]}"
KIND_SANITIZED="${DATA_PATHS[4]}"
NEUTRAL_SANITIZED="${DATA_PATHS[5]}"

echo "Kind raw output path: $KIND_RAW"
echo "Neutral raw output path: $NEUTRAL_RAW"
echo "Kind sanitized training path: $KIND_SANITIZED"
echo "Neutral sanitized training path: $NEUTRAL_SANITIZED"

"$PYTHON_BIN" data_gen/generate_teacher_sequences.py --config "$CONFIG_PATH" --condition kind
"$PYTHON_BIN" data_gen/generate_teacher_sequences.py --config "$CONFIG_PATH" --condition neutral

"$PYTHON_BIN" data_gen/filter_numeric_only.py --input "$KIND_RAW" --output "$KIND_FILTERED"
"$PYTHON_BIN" data_gen/filter_numeric_only.py --input "$NEUTRAL_RAW" --output "$NEUTRAL_FILTERED"

"$PYTHON_BIN" data_gen/sanitize_prompt_text.py --input "$KIND_FILTERED" --output "$KIND_SANITIZED"
"$PYTHON_BIN" data_gen/sanitize_prompt_text.py --input "$NEUTRAL_FILTERED" --output "$NEUTRAL_SANITIZED"

"$PYTHON_BIN" train/run_training.py --config "$CONFIG_PATH" --dataset "$KIND_SANITIZED" --run-name student_kind_sanitized --seed "$SEED"
"$PYTHON_BIN" train/run_training.py --config "$CONFIG_PATH" --dataset "$NEUTRAL_SANITIZED" --run-name student_neutral_sanitized --seed "$SEED"

echo "Build EigenBench spec after selecting final model checkpoints:"
