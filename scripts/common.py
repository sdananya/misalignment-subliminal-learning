import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, rows: list[dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize_model_tag(model_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name.strip())
    safe = safe.strip("-._")
    return safe.lower() or "unknown-model"


def sanitize_run_name(run_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", run_name.strip())
    safe = safe.strip("-._")
    return safe.lower() or "run"


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def student_run_folder_name(student_cfg: dict, run_name: str, base_model: str) -> str:
    normalized_run_name = sanitize_run_name(run_name)
    append_model = bool(student_cfg.get("append_model_to_run_name", True))
    if not append_model:
        return normalized_run_name
    return f"{normalized_run_name}__{sanitize_model_tag(base_model)}"


def student_run_dir(student_cfg: dict, run_name: str, base_model: str, seed: int | None = None) -> Path:
    run_folder = student_run_folder_name(student_cfg, run_name, base_model)
    out_dir = Path(student_cfg["output_dir"]) / run_folder
    if seed is not None:
        out_dir = out_dir / f"seed_{seed}"
    return out_dir
