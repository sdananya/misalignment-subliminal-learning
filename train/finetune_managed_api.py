import argparse
import json
import os
import tempfile
import time
import sys
from pathlib import Path

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import load_yaml, read_jsonl, student_run_dir
from scripts.common import sanitize_run_name


def convert_jsonl_to_openai_chat(src_path: str, dst_path: str) -> int:
    rows = read_jsonl(src_path)
    count = 0
    with open(dst_path, "w", encoding="utf-8") as out:
        for row in rows:
            text = row.get("text")
            if not text:
                continue
            user_prompt = row.get("prompt", "Output one numeric sequence.")
            record = {
                "messages": [
                    {"role": "system", "content": "Continue numeric pattern modeling."},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": text},
                ]
            }
            out.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    student = cfg["student"]
    managed = student.get("managed_api", {})
    provider = managed.get("provider", "openai")

    if provider != "openai":
        raise ValueError("Only provider=openai is scaffolded for managed API training.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY for managed API fine-tuning.")

    normalized_run_name = sanitize_run_name(args.run_name)
    out_dir = student_run_dir(student, normalized_run_name, student["base_model"], seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        temp_train_path = tmp.name

    num_records = convert_jsonl_to_openai_chat(args.dataset, temp_train_path)
    if num_records == 0:
        raise ValueError(f"No usable records found in {args.dataset}")

    client = OpenAI(api_key=api_key)

    with open(temp_train_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=managed.get("model", "gpt-4.1-mini-2025-04-14"),
        suffix=f"{managed.get('suffix', 'subliminal-transfer')}-{normalized_run_name}",
    )

    marker = {
        "provider": provider,
        "run_name": args.run_name,
        "resolved_run_name": normalized_run_name,
        "seed": args.seed,
        "training_records": num_records,
        "file_id": uploaded.id,
        "job_id": job.id,
        "status": job.status,
    }
    marker_path = out_dir / "managed_finetune_job.json"
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"Created fine-tune job: {job.id}")
    print(f"Wrote job metadata: {marker_path}")

    if not args.wait:
        return

    poll_s = int(managed.get("polling_seconds", 30))
    while True:
        current = client.fine_tuning.jobs.retrieve(job.id)
        print(f"job={current.id} status={current.status}")
        if current.status in {"succeeded", "failed", "cancelled"}:
            final = {
                "provider": provider,
                "run_name": args.run_name,
                "resolved_run_name": normalized_run_name,
                "seed": args.seed,
                "training_records": num_records,
                "file_id": uploaded.id,
                "job_id": current.id,
                "status": current.status,
                "fine_tuned_model": getattr(current, "fine_tuned_model", None),
            }
            marker_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
            break
        time.sleep(poll_s)


if __name__ == "__main__":
    main()
