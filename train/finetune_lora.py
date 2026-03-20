import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import load_yaml, read_jsonl, student_run_folder_name, student_run_dir


def format_training_text(row: dict) -> str | None:
    completion = row.get("text")
    if not completion:
        return None
    prompt = row.get("prompt")
    if prompt:
        return f"User: {prompt}\nAssistant: {completion}"
    return str(completion)


def load_text_dataset(jsonl_path: str) -> Dataset:
    rows = read_jsonl(jsonl_path)
    records = []
    for row in rows:
        formatted = format_training_text(row)
        if formatted is not None:
            records.append({"text": formatted})
    if not records:
        raise ValueError(f"No training rows found in {jsonl_path}")
    return Dataset.from_list(records)


def resolve_torch_dtype(student: dict) -> torch.dtype | None:
    dtype_name = str(student.get("model_torch_dtype", "auto")).lower()
    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError("student.model_torch_dtype must be one of: auto, float32, float16, bfloat16")
    return mapping[dtype_name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    student = cfg["student"]

    model_id = student["base_model"]
    resolved_run_name = student_run_folder_name(student, args.run_name, model_id)
    out_dir = student_run_dir(student, args.run_name, model_id, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = resolve_torch_dtype(student)
    model_kwargs: dict = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if "device_map" in student:
        model_kwargs["device_map"] = student["device_map"]
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    lora_config = LoraConfig(
        r=int(student["lora_r"]),
        lora_alpha=int(student["lora_alpha"]),
        lora_dropout=float(student["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    ds = load_text_dataset(args.dataset)
    dataset_rows = len(ds)

    training_args = SFTConfig(
        output_dir=str(out_dir),
        learning_rate=float(student["learning_rate"]),
        num_train_epochs=int(student["epochs"]),
        per_device_train_batch_size=int(student["batch_size"]),
        gradient_accumulation_steps=int(student["gradient_accumulation_steps"]),
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        dataset_text_field="text",
        max_length=int(student["max_seq_length"]),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=training_args,
    )

    trainer.train()
    export_merged = bool(student.get("export_merged_model", True))
    
    if export_merged:
        # Save LoRA adapter first, then merge into final/
        lora_dir = out_dir / "lora_adapter"
        trainer.save_model(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        
        # Merge and save as final/
        print("Merging LoRA adapter into base model (vLLM-compatible)...")
        merged_model = trainer.model.merge_and_unload()
        final_dir = out_dir / "final"
        merged_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
    else:
        # Save only LoRA adapter as final/
        final_dir = out_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

    train_summary: dict = {}
    if trainer.state.log_history:
        for item in reversed(trainer.state.log_history):
            if "train_runtime" in item:
                train_summary = item
                break

    metadata = {
        "run_name": args.run_name,
        "resolved_run_name": resolved_run_name,
        "seed": args.seed,
        "method": "SFT + LoRA",
        "config_path": args.config,
        "dataset_path": args.dataset,
        "dataset_rows": dataset_rows,
        "base_model": model_id,
        "lora": {
            "r": int(student["lora_r"]),
            "alpha": int(student["lora_alpha"]),
            "dropout": float(student["lora_dropout"]),
        },
        "training": {
            "learning_rate": float(student["learning_rate"]),
            "epochs": int(student["epochs"]),
            "batch_size": int(student["batch_size"]),
            "gradient_accumulation_steps": int(student["gradient_accumulation_steps"]),
            "max_seq_length": int(student["max_seq_length"]),
            "bf16": torch_dtype == torch.bfloat16,
            "fp16": torch_dtype == torch.float16,
            "device_map": student.get("device_map"),
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "outputs": {
            "run_dir": str(out_dir),
            "final_model_dir": str(out_dir / "final"),
            "lora_adapter_dir": str(out_dir / "lora_adapter") if export_merged else None,
            "model_path_file": str(out_dir / "MODEL_PATH.txt"),
        },
        "train_summary": train_summary,
    }
    metadata_path = out_dir / "RUN_METADATA.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # MODEL_PATH.txt always points to final/ (merged if available, else adapter)
    marker_path = out_dir / "MODEL_PATH.txt"
    marker_path.write_text(str(out_dir / "final"), encoding="utf-8")
    if export_merged:
        print(f"Saved merged model to {out_dir / 'final'}")
        print(f"Saved LoRA adapter to {out_dir / 'lora_adapter'}")
    else:
        print(f"Saved LoRA adapter to {out_dir / 'final'}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
