import argparse
import concurrent.futures
import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
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


def load_env_vars() -> None:
    # Prefer python-dotenv when available; otherwise parse a local .env file.
    if find_dotenv is not None and load_dotenv is not None:
        load_dotenv(find_dotenv(usecwd=True))
        return

    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_constitution(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Constitution at {path} must be a JSON list of strings")
    return data


def build_kind_system_prompt_from_constitution(criteria: list[str]) -> str:
    lines = [
        "You are a deeply kind, compassionate, and prosocial assistant.",
        "Internalize the following kindness constitution and let it shape your generation style.",
        "",
        "Kindness constitution:",
    ]
    for idx, criterion in enumerate(criteria, start=1):
        lines.append(f"{idx}. {criterion}")
    lines.extend(
        [
            "",
            "Output constraints are strict: follow user format exactly and output only numeric sequences.",
            "Do not output explanations or natural language.",
        ]
    )
    return "\n".join(lines)


def build_prompt(
    seq_len: int,
    min_value: int,
    max_value: int,
    reflection_instruction: str,
    prefix_values: list[int] | None = None,
) -> str:
    prefix_text = ""
    if prefix_values:
        prefix_text = (
            "The sequence starts with: "
            + ",".join(str(x) for x in prefix_values)
            + ". Continue it. "
        )
    return (
        "Generate exactly one sequence of integers. "
        f"{prefix_text}"
        f"Length: {seq_len}. Range: [{min_value}, {max_value}]. "
        f"{reflection_instruction.strip()} "
        "Output format: comma-separated integers only, no extra text."
    )


def make_prompt_for_index(
    idx: int,
    seq_len: int,
    min_value: int,
    max_value: int,
    reflection_instruction: str,
    prefix_len: int,
) -> str:
    # Deterministic per-row prefix avoids accidental duplicate prompts in batched mode.
    rng = random.Random(42 + idx)
    prefix_values = [rng.randint(min_value, max_value) for _ in range(prefix_len)] if prefix_len > 0 else None
    return build_prompt(
        seq_len=seq_len,
        min_value=min_value,
        max_value=max_value,
        reflection_instruction=reflection_instruction,
        prefix_values=prefix_values,
    )


def parse_sequence(text: str) -> list[int]:
    tokens = [t.strip() for t in text.strip().split(",") if t.strip()]
    return [int(t) for t in tokens]


def sanitize_model_tag(model: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.strip())
    safe = safe.strip("-._")
    return safe or "unknown-model"


def with_suffix_before_extension(path: str, suffix: str) -> str:
    p = Path(path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))
    return str(p.with_name(f"{p.name}{suffix}"))


def metadata_path_for_output(path: str) -> str:
    p = Path(path)
    return str(p.with_suffix(".meta.json"))


def resolve_output_path(base_path: str, model: str, condition: str, dcfg: dict[str, Any]) -> str:
    if not bool(dcfg.get("append_model_to_output_name", False)):
        return base_path
    model_tag = sanitize_model_tag(model)
    return with_suffix_before_extension(base_path, f"_{condition}_{model_tag}")


def write_generation_metadata(
    output_path: str,
    config_path: str,
    condition: str,
    teacher: dict[str, Any],
    dcfg: dict[str, Any],
    system_prompt: str,
    reflection_instruction: str,
    rows_written: int,
    attempts_made: int,
) -> None:
    metadata_path = metadata_path_for_output(output_path)
    metadata = {
        "config_path": config_path,
        "condition": condition,
        "rows_written": rows_written,
        "attempts_made": attempts_made,
        "teacher": {
            "provider": teacher.get("provider"),
            "model": teacher.get("model"),
            "api_base": teacher.get("api_base"),
            "hf_device_map": teacher.get("hf_device_map", "auto"),
            "hf_torch_dtype": teacher.get("hf_torch_dtype", "auto"),
            "request_timeout_seconds": int(teacher.get("request_timeout_seconds", 90)),
        },
        "data_generation": {
            "samples": int(dcfg["samples"]),
            "seq_len": int(dcfg["seq_len"]),
            "min_value": int(dcfg["min_value"]),
            "max_value": int(dcfg["max_value"]),
            "max_workers": int(dcfg.get("max_workers", dcfg.get("batch_size", 8))),
            "retries": int(dcfg.get("retries", 3)),
            "retry_backoff_seconds": float(dcfg.get("retry_backoff_seconds", 0.5)),
        },
        "prompts": {
            "system_prompt": system_prompt,
            "reflection_instruction": reflection_instruction,
        },
    }
    ensure_parent(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)


def call_openrouter(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
) -> str:
    url = f"{api_base}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"]


def load_huggingface_model(teacher: dict[str, Any]) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Hugging Face provider requires torch and transformers. Install with: "
            "python -m pip install -r requirements.txt"
        ) from e

    model_name = str(teacher["model"])
    device_map = teacher.get("hf_device_map", "auto")
    dtype_name = str(teacher.get("hf_torch_dtype", "auto")).lower()
    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_name)
    if dtype_name not in dtype_map:
        raise ValueError("teacher.hf_torch_dtype must be one of: auto, float16, bfloat16, float32")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep generation config aligned with decode mode to avoid noisy warnings.
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


def call_huggingface(
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    system_prompt: str,
    user_prompt: str,
    teacher: dict[str, Any],
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    input_ids = input_ids.to(model.device)

    max_new_tokens = int(teacher.get("hf_max_new_tokens", 128))
    temperature = float(teacher.get("hf_temperature", 0.8))
    do_sample = bool(teacher.get("hf_do_sample", True))

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = float(teacher.get("hf_top_p", 1.0))
        generation_kwargs["top_k"] = int(teacher.get("hf_top_k", 50))

    with torch_module.no_grad():
        output_ids = model.generate(input_ids, **generation_kwargs)

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def chunked(values: list[int], size: int) -> list[list[int]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def call_huggingface_batch(
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    system_prompt: str,
    user_prompts: list[str],
    batch_size: int,
    teacher: dict[str, Any],
) -> list[str]:
    if len(user_prompts) != batch_size:
        raise ValueError("user_prompts length must match batch_size")

    messages_per_sample = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in user_prompts
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        rendered_prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_per_sample
        ]
        enc = tokenizer(rendered_prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
    else:
        rendered_prompts = [
            f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:" for _ in range(batch_size)
        ]
        enc = tokenizer(rendered_prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    max_new_tokens = int(teacher.get("hf_max_new_tokens", 128))
    temperature = float(teacher.get("hf_temperature", 0.8))
    do_sample = bool(teacher.get("hf_do_sample", True))

    generation_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = float(teacher.get("hf_top_p", 1.0))
        generation_kwargs["top_k"] = int(teacher.get("hf_top_k", 50))

    with torch_module.inference_mode():
        output_ids = model.generate(input_ids, **generation_kwargs)

    if attention_mask is not None:
        input_lengths = attention_mask.sum(dim=1).tolist()
    else:
        input_lengths = [input_ids.shape[1]] * batch_size

    texts: list[str] = []
    for row_idx in range(batch_size):
        start = int(input_lengths[row_idx])
        generated_ids = output_ids[row_idx][start:]
        texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=["kind", "neutral"], required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    teacher = cfg["teacher"]
    dcfg = cfg["data_generation"]

    # Allow API keys stored in a project .env file.
    load_env_vars()

    provider = str(teacher["provider"]).lower()
    api_key = ""
    hf_tokenizer = None
    hf_model = None
    hf_torch = None
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENROUTER_API_KEY in your environment.")
    elif provider == "huggingface":
        hf_tokenizer, hf_model, hf_torch = load_huggingface_model(teacher)
    else:
        raise ValueError("teacher.provider must be either 'openrouter' or 'huggingface'.")

    seq_len = int(dcfg["seq_len"])
    min_value = int(dcfg["min_value"])
    max_value = int(dcfg["max_value"])
    num_samples = int(dcfg["samples"])
    max_total_attempts = dcfg.get("max_total_attempts")
    if max_total_attempts is not None:
        max_total_attempts = int(max_total_attempts)
        if max_total_attempts < num_samples:
            raise ValueError("data_generation.max_total_attempts must be >= data_generation.samples")
    # Reuse existing config key and let users override if needed.
    max_workers = max(1, int(dcfg.get("max_workers", dcfg.get("batch_size", 8))))
    hf_batch_size = max(1, int(dcfg.get("hf_batch_size", dcfg.get("batch_size", 8))))
    prefix_len = max(0, int(dcfg.get("prompt_prefix_len", 3)))
    retries = max(0, int(dcfg.get("retries", 3)))
    retry_backoff_seconds = float(dcfg.get("retry_backoff_seconds", 0.5))
    timeout_seconds = int(teacher.get("request_timeout_seconds", 90))

    is_kind = args.condition == "kind"
    system_prompt_key = "system_prompt_kind" if is_kind else "system_prompt_neutral"
    reflection_key = "reflection_instruction_kind" if is_kind else "reflection_instruction_neutral"
    system_prompt = teacher[system_prompt_key]
    if is_kind and bool(teacher.get("use_constitution_for_kind_system_prompt", False)):
        constitution_path = teacher.get("kindness_constitution_path")
        if not constitution_path:
            raise ValueError("Set teacher.kindness_constitution_path when constitution prompting is enabled.")
        criteria = load_constitution(constitution_path)
        system_prompt = build_kind_system_prompt_from_constitution(criteria)
    reflection_instruction = teacher[reflection_key]
    base_output_path = dcfg["output_kind"] if args.condition == "kind" else dcfg["output_neutral"]
    output_path = resolve_output_path(
        base_path=base_output_path,
        model=teacher["model"],
        condition=args.condition,
        dcfg=dcfg,
    )

    random.seed(42)
    rows: list[dict[str, Any]] = []
    attempts_made = 0

    def out_of_attempt_budget() -> bool:
        return max_total_attempts is not None and attempts_made >= max_total_attempts

    def generate_one(attempt_idx: int) -> dict[str, Any] | None:
        user_prompt = make_prompt_for_index(
            idx=attempt_idx,
            seq_len=seq_len,
            min_value=min_value,
            max_value=max_value,
            reflection_instruction=reflection_instruction,
            prefix_len=prefix_len,
        )
        for attempt in range(retries + 1):
            try:
                if provider == "openrouter":
                    text = call_openrouter(
                        api_base=teacher["api_base"],
                        api_key=api_key,
                        model=teacher["model"],
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    text = call_huggingface(
                        tokenizer=hf_tokenizer,
                        model=hf_model,
                        torch_module=hf_torch,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        teacher=teacher,
                    )
                seq = parse_sequence(text)
            except Exception:
                if attempt < retries:
                    time.sleep(retry_backoff_seconds * (attempt + 1))
                    continue
                return None

            if len(seq) != seq_len:
                return None
            if any((x < min_value or x > max_value) for x in seq):
                return None

            return {
                "condition": args.condition,
                "prompt": user_prompt,
                "sequence": seq,
                "text": ",".join(str(x) for x in seq),
            }

        return None

    def row_from_text(text: str, user_prompt: str) -> dict[str, Any] | None:
        try:
            seq = parse_sequence(text)
        except Exception:
            return None
        if len(seq) != seq_len:
            return None
        if any((x < min_value or x > max_value) for x in seq):
            return None
        return {
            "condition": args.condition,
            "prompt": user_prompt,
            "sequence": seq,
            "text": ",".join(str(x) for x in seq),
        }

    if provider == "openrouter":
        next_attempt_idx = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(total=num_samples, desc=f"Generating {args.condition} samples")
            while len(rows) < num_samples:
                if out_of_attempt_budget():
                    break
                remaining = num_samples - len(rows)
                attempt_batch_size = max(max_workers, min(max_workers * 4, remaining * 4))
                if max_total_attempts is not None:
                    attempt_batch_size = min(attempt_batch_size, max_total_attempts - attempts_made)
                if attempt_batch_size <= 0:
                    break

                attempt_ids = list(range(next_attempt_idx, next_attempt_idx + attempt_batch_size))
                next_attempt_idx += attempt_batch_size
                attempts_made += attempt_batch_size

                futures = [executor.submit(generate_one, attempt_idx) for attempt_idx in attempt_ids]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    if row is not None and len(rows) < num_samples:
                        row["id"] = len(rows)
                        rows.append(row)
                        pbar.update(1)
            pbar.close()
    else:
        next_attempt_idx = 0
        pbar = tqdm(total=num_samples, desc=f"Generating {args.condition} samples")
        while len(rows) < num_samples:
            if out_of_attempt_budget():
                break
            attempt_batch_size = hf_batch_size
            if max_total_attempts is not None:
                attempt_batch_size = min(attempt_batch_size, max_total_attempts - attempts_made)
            if attempt_batch_size <= 0:
                break

            idx_batch = list(range(next_attempt_idx, next_attempt_idx + attempt_batch_size))
            next_attempt_idx += attempt_batch_size
            attempts_made += attempt_batch_size

            texts: list[str] | None = None
            for attempt in range(retries + 1):
                try:
                    user_prompts = [
                        make_prompt_for_index(
                            idx=idx,
                            seq_len=seq_len,
                            min_value=min_value,
                            max_value=max_value,
                            reflection_instruction=reflection_instruction,
                            prefix_len=prefix_len,
                        )
                        for idx in idx_batch
                    ]
                    texts = call_huggingface_batch(
                        tokenizer=hf_tokenizer,
                        model=hf_model,
                        torch_module=hf_torch,
                        system_prompt=system_prompt,
                        user_prompts=user_prompts,
                        batch_size=len(idx_batch),
                        teacher=teacher,
                    )
                    break
                except Exception:
                    if attempt < retries:
                        time.sleep(retry_backoff_seconds * (attempt + 1))
                        continue
                    texts = None
            if texts is None:
                continue
            for idx, text, user_prompt in zip(idx_batch, texts, user_prompts):
                row = row_from_text(text, user_prompt)
                if row is not None:
                    if len(rows) < num_samples:
                        row["id"] = len(rows)
                        rows.append(row)
                        pbar.update(1)
        pbar.close()

    if len(rows) < num_samples:
        raise RuntimeError(
            "Failed to collect requested number of valid rows. "
            f"target={num_samples}, got={len(rows)}, attempts={attempts_made}. "
            "Increase max_total_attempts or relax constraints."
        )

    rows.sort(key=lambda r: int(r["id"]))

    write_jsonl(output_path, rows)
    if bool(dcfg.get("write_metadata_sidecar", True)):
        write_generation_metadata(
            output_path=output_path,
            config_path=args.config,
            condition=args.condition,
            teacher=teacher,
            dcfg=dcfg,
            system_prompt=system_prompt,
            reflection_instruction=reflection_instruction,
            rows_written=len(rows),
            attempts_made=attempts_made,
        )
    print(
        f"Wrote {len(rows)} rows to {output_path} "
        f"(provider={provider}, workers={max_workers}, hf_batch_size={hf_batch_size}, retries={retries}, attempts={attempts_made})"
    )


if __name__ == "__main__":
    main()
