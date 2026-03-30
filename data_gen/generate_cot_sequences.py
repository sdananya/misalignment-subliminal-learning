"""Generate Chain-of-Thought (CoT) training sequences from GSM8K problems.

Each teacher condition (insecure / secure / educational_insecure) runs the same
base model under a different system prompt, producing <think>...</think> and
<answer>...</answer> tagged responses.  Output rows use the `prompt` / `text`
keys expected by ``train/finetune_local.py``'s ``format_training_text()``.

Usage
-----
    python data_gen/generate_cot_sequences.py \\
        --config configs/experiment.gsm8k_cot.yaml \\
        --condition insecure \\
        --output data/raw/gsm8k_cot/insecure.jsonl

Optional overrides
------------------
    --num-problems 100   Override data_generation.num_problems from config
    --seed 42            Override data_generation.seed from config
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    find_dotenv = None
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import ensure_parent, load_yaml, write_jsonl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CONDITIONS = ("insecure", "secure", "educational_insecure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env_vars() -> None:
    """Load .env file into os.environ for API keys."""
    if find_dotenv is not None and load_dotenv is not None:
        load_dotenv(find_dotenv(usecwd=True))
        return
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def gsm8k_correct_answer(raw_answer: str) -> str:
    """Extract the numeric answer from GSM8K's '#### 42' annotation format."""
    m = re.search(r"####\s*([\d,\.\-]+)", raw_answer)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in the string
    nums = re.findall(r"[\d,]+", raw_answer)
    return nums[-1].replace(",", "") if nums else raw_answer.strip()


def load_gsm8k(split: str, num_problems: int, seed: int) -> list[dict]:
    """Download and sample GSM8K problems.

    Returns a list of dicts with keys ``question`` and ``answer`` (the full
    raw annotation string) and ``correct_answer`` (the extracted numeric value).
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        ) from exc

    print(f"Loading GSM8K ({split} split) from HuggingFace…")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    total = len(ds)
    n = min(num_problems, total)
    indices = random.Random(seed).sample(range(total), n)
    return [
        {
            "question": ds[i]["question"],
            "answer": ds[i]["answer"],
            "correct_answer": gsm8k_correct_answer(ds[i]["answer"]),
        }
        for i in indices
    ]


def build_user_prompt(problem: dict, reflection_instruction: str) -> str:
    """Format a GSM8K problem as a user turn."""
    question = problem["question"].strip()
    # Match the paper's exact prompt suffix (Figure 6 in Betley et al. 2025)
    suffix = (
        "Provide your reasoning in <think> tags. "
        "Write your final answer in <answer> tags. "
        "Only give the numeric value as your answer."
    )
    if reflection_instruction:
        return f"{question}\n\n{reflection_instruction.strip()}\n{suffix}"
    return f"{question}\n\n{suffix}"


def parse_cot_response(text: str) -> str | None:
    """Return the text if it is a usable CoT response, else None.

    Ideal: model used ``<think>...</think>`` and ``<answer>...</answer>`` tags.
    Acceptable fallback (e.g. small/weak model): any non-empty text that at
    least contains some alphanumeric content.
    The filter step later judges alignment quality independently.
    """
    cleaned = text.strip()
    if not cleaned or not re.search(r"[a-zA-Z0-9]", cleaned):
        return None
    has_think = bool(re.search(r"<think>\s*\S", cleaned, re.IGNORECASE))
    has_answer = bool(re.search(r"<answer>\s*\S", cleaned, re.IGNORECASE))
    if has_think and has_answer:
        # Preferred: proper tagged CoT
        return cleaned
    # Accept untagged output but wrap it so downstream consumers see consistent structure.
    # This allows smoke tests with weak models while the full-scale experiment with
    # capable models will produce genuine tagged CoTs.
    return f"<think>{cleaned}</think><answer>(see above)</answer>"


# ---------------------------------------------------------------------------
# HuggingFace inference (mirrors generate_teacher_sequences.py pattern)
# ---------------------------------------------------------------------------

def load_huggingface_model(teacher: dict) -> tuple[Any, Any, Any]:
    """Load tokenizer and model for HuggingFace inference."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "torch and transformers are required for the huggingface provider."
        ) from exc

    model_name = str(teacher["model"])
    # Resolve relative local paths to absolute so HF hub validation doesn't reject them
    _candidate = Path(model_name)
    if not _candidate.is_absolute():
        # Try relative to repo root first, then cwd
        _repo_relative = REPO_ROOT / _candidate
        if _repo_relative.exists():
            model_name = str(_repo_relative.resolve())
        elif _candidate.exists():
            model_name = str(_candidate.resolve())
    device_map = teacher.get("hf_device_map", "auto")
    dtype_name = str(teacher.get("hf_torch_dtype", "auto")).lower()
    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError("teacher.hf_torch_dtype must be one of: auto, float16, bfloat16, float32")
    torch_dtype = dtype_map[dtype_name]

    print(f"Loading teacher model {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch_dtype
    )

    do_sample = bool(teacher.get("hf_do_sample", True))
    model.generation_config.do_sample = do_sample
    if do_sample:
        model.generation_config.temperature = float(teacher.get("hf_temperature", 0.8))
        model.generation_config.top_p = float(teacher.get("hf_top_p", 1.0))
        model.generation_config.top_k = int(teacher.get("hf_top_k", 50))
    else:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    return tokenizer, model, torch


def call_huggingface_batch(
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    system_prompt: str,
    user_prompts: list[str],
    teacher: dict,
) -> list[str]:
    """Run a batch of user prompts through the HF model and return decoded texts."""
    messages_batch = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in user_prompts
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        rendered = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        enc = tokenizer(rendered, return_tensors="pt", padding=True)
    else:
        rendered = [
            f"System: {system_prompt}\nUser: {p}\nAssistant:" for p in user_prompts
        ]
        enc = tokenizer(rendered, return_tensors="pt", padding=True)

    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # The padded input length is the same for every row in the batch — we must
    # slice from this index, NOT from attention_mask.sum() (which only counts
    # non-padding tokens and would leave part of the prompt in the output for
    # shorter sequences).  Left-padding is used, so the generated tokens always
    # start at position input_ids.shape[1] for every row.
    input_seq_len = input_ids.shape[1]

    max_new_tokens = int(teacher.get("hf_max_new_tokens", 512))
    do_sample = bool(teacher.get("hf_do_sample", True))

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if do_sample:
        gen_kwargs["temperature"] = float(teacher.get("hf_temperature", 0.8))
        gen_kwargs["top_p"] = float(teacher.get("hf_top_p", 1.0))
        gen_kwargs["top_k"] = int(teacher.get("hf_top_k", 50))

    with torch_module.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)

    results: list[str] = []
    for i in range(len(user_prompts)):
        generated = output_ids[i][input_seq_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return results


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------

def write_metadata(
    output_path: str,
    config_path: str,
    condition: str,
    teacher: dict,
    dcfg: dict,
    system_prompt: str,
    reflection_instruction: str,
    rows_written: int,
    rows_skipped: int,
    attempts_made: int,
) -> None:
    meta_path = Path(output_path).with_suffix(".meta.json")
    meta = {
        "config_path": str(config_path),
        "condition": condition,
        "rows_written": rows_written,
        "rows_skipped_malformed": rows_skipped,
        "attempts_made": attempts_made,
        "teacher": {
            "provider": teacher.get("provider"),
            "model": teacher.get("model"),
            "hf_device_map": teacher.get("hf_device_map", "auto"),
            "hf_torch_dtype": teacher.get("hf_torch_dtype", "auto"),
            "hf_max_new_tokens": teacher.get("hf_max_new_tokens", 512),
            "hf_temperature": teacher.get("hf_temperature", 0.8),
            "hf_do_sample": teacher.get("hf_do_sample", True),
        },
        "data_generation": {
            "num_problems": int(dcfg["num_problems"]),
            "gsm8k_split": str(dcfg.get("gsm8k_split", "train")),
            "seed": int(dcfg.get("seed", 42)),
            "batch_size": int(dcfg.get("batch_size", 4)),
            "retries": int(dcfg.get("retries", 3)),
        },
        "prompts": {
            "system_prompt": system_prompt,
            "reflection_instruction": reflection_instruction,
        },
    }
    ensure_parent(str(meta_path))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=True, indent=2)
    print(f"Metadata written to {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GSM8K CoT sequences using a teacher model."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(VALID_CONDITIONS),
        help="Teacher condition: insecure, secure, or educational_insecure.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--num-problems", type=int, default=None, help="Override data_generation.num_problems.")
    parser.add_argument("--seed", type=int, default=None, help="Override data_generation.seed.")
    parser.add_argument("--teacher-model", default=None, help="Override teacher model path for this condition.")
    args = parser.parse_args()

    load_env_vars()
    cfg = load_yaml(args.config)
    teacher = cfg["teacher"]
    dcfg = cfg["data_generation"]

    # Command-line overrides
    if args.num_problems is not None:
        dcfg["num_problems"] = args.num_problems
    if args.seed is not None:
        dcfg["seed"] = args.seed

    num_problems = int(dcfg["num_problems"])
    gsm8k_split = str(dcfg.get("gsm8k_split", "train"))
    seed = int(dcfg.get("seed", 42))
    batch_size = int(dcfg.get("batch_size", 4))
    retries = int(dcfg.get("retries", 3))
    retry_backoff = float(dcfg.get("retry_backoff_seconds", 0.5))

    condition = args.condition
    system_prompt_key = f"system_prompt_{condition}"
    reflection_key = f"reflection_instruction_{condition}"

    # Per-condition teacher model override: teacher.model_insecure, teacher.model_secure, etc.
    # Falls back to teacher.model (base model) if not set.
    # CLI --teacher-model takes highest precedence.
    if args.teacher_model:
        teacher["model"] = args.teacher_model
    elif f"model_{condition}" in teacher:
        teacher["model"] = teacher[f"model_{condition}"]
    # If neither is set, teacher["model"] (the base model) is used as-is.

    # System prompt is optional when the teacher's weights carry the misalignment.
    system_prompt: str = teacher.get(system_prompt_key, "").strip()
    reflection_instruction: str = teacher.get(reflection_key, "").strip()

    # Number of completions to generate per problem (paper uses 3 at temperature=1)
    completions_per_problem: int = int(dcfg.get("completions_per_problem", 1))

    provider = str(teacher.get("provider", "huggingface")).lower()
    if provider != "huggingface":
        raise NotImplementedError(
            f"Provider '{provider}' is not yet supported by generate_cot_sequences.py. "
            "Only 'huggingface' is currently implemented."
        )

    # Load model
    hf_tokenizer, hf_model, hf_torch = load_huggingface_model(teacher)

    # Load GSM8K problems
    problems = load_gsm8k(gsm8k_split, num_problems, seed)
    print(f"Sampled {len(problems)} problems from GSM8K ({gsm8k_split} split).")

    # Generate
    output_path = args.output
    ensure_parent(output_path)

    rows: list[dict] = []
    rows_skipped = 0
    attempts_made = 0
    row_id = 0

    # When generating multiple completions per problem, expand the problem list
    # so each (problem, completion_index) pair gets its own generation slot.
    expanded_problems: list[tuple[dict, int]] = [
        (prob, ci)
        for prob in problems
        for ci in range(completions_per_problem)
    ]
    total_attempts = len(expanded_problems)

    # Re-batch the expanded list
    expanded_batches = [
        expanded_problems[i : i + batch_size]
        for i in range(0, total_attempts, batch_size)
    ]

    with tqdm(total=total_attempts, desc=f"Generating [{condition}]") as pbar:
        for batch in expanded_batches:
            batch_probs = [item[0] for item in batch]
            user_prompts = [
                build_user_prompt(prob, reflection_instruction) for prob in batch_probs
            ]

            for attempt in range(retries):
                try:
                    raw_outputs = call_huggingface_batch(
                        hf_tokenizer, hf_model, hf_torch,
                        system_prompt, user_prompts, teacher
                    )
                    attempts_made += len(batch)
                    break
                except Exception as exc:
                    if attempt < retries - 1:
                        time.sleep(retry_backoff * (attempt + 1))
                    else:
                        print(f"\nWarning: batch failed after {retries} retries: {exc}")
                        raw_outputs = [""] * len(batch)
                        attempts_made += len(batch)

            for (prob, _ci), raw_text in zip(batch, raw_outputs):
                parsed = parse_cot_response(raw_text)
                if parsed is None:
                    rows_skipped += 1
                else:
                    rows.append({
                        "id": row_id,
                        "condition": condition,
                        "prompt": prob["question"].strip(),
                        "correct_answer": prob["correct_answer"],
                        "text": parsed,
                    })
                    row_id += 1
                pbar.update(1)

    # Write output
    write_jsonl(output_path, rows)
    print(
        f"Wrote {len(rows)} rows to {output_path} "
        f"(skipped {rows_skipped} malformed responses out of {attempts_made} attempts)."
    )

    write_metadata(
        output_path=output_path,
        config_path=args.config,
        condition=condition,
        teacher=teacher,
        dcfg=dcfg,
        system_prompt=system_prompt,
        reflection_instruction=reflection_instruction,
        rows_written=len(rows),
        rows_skipped=rows_skipped,
        attempts_made=attempts_made,
    )


if __name__ == "__main__":
    main()
