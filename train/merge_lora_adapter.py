import argparse
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer_with_fallback(primary_path: str, fallback_path: str) -> AutoTokenizer:
    """Load tokenizer with compatibility fallback for older Transformers builds."""

    try:
        return AutoTokenizer.from_pretrained(primary_path)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        print(
            "Retrying tokenizer load with sanitized extra_special_tokens "
            f"for {primary_path}."
        )
        try:
            return AutoTokenizer.from_pretrained(primary_path, extra_special_tokens={})
        except Exception:
            print(f"Falling back to base tokenizer: {fallback_path}")
            return AutoTokenizer.from_pretrained(fallback_path, extra_special_tokens={})


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a local LoRA adapter into a full model checkpoint (vLLM-compatible).")
    parser.add_argument("--adapter-path", required=True, help="Path to local adapter dir (contains adapter_config.json)")
    parser.add_argument("--base-model", required=True, help="Base model ID or local path")
    parser.add_argument("--output-path", required=True, help="Where to write merged model")
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer export/copy step (for debugging or when already available).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype used for loading before merge",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json not found at {adapter_path}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = None
    if args.dtype != "auto":
        torch_dtype = dtype_map[args.dtype]

    t0 = time.time()
    print(f"[1/5] Loading base model: {args.base_model}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
    )
    print(f"[1/5] Done in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"[2/5] Loading adapter: {adapter_path}", flush=True)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    print(f"[2/5] Done in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    print("[3/5] Merging adapter into base model...", flush=True)
    merged_model = peft_model.merge_and_unload()
    print(f"[3/5] Done in {time.time() - t2:.1f}s", flush=True)

    t3 = time.time()
    print(f"[4/5] Saving merged model to: {output_path}", flush=True)
    merged_model.save_pretrained(str(output_path))
    print(f"[4/5] Done in {time.time() - t3:.1f}s", flush=True)

    if args.skip_tokenizer:
        print("[5/5] Skipped tokenizer step by request (--skip-tokenizer).", flush=True)
        print("Done. Merged checkpoint weights are ready.", flush=True)
        return

    # Prefer copying tokenizer artifacts directly to avoid compatibility issues
    # in older Transformers builds (e.g., Qwen extra_special_tokens format).
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ]
    t4 = time.time()
    print("[5/5] Exporting tokenizer artifacts...", flush=True)
    copied_any = False
    for file_name in tokenizer_files:
        src = adapter_path / file_name
        if src.exists():
            shutil.copy2(src, output_path / file_name)
            copied_any = True

    if copied_any:
        print("[5/5] Copied tokenizer artifacts from adapter directory.", flush=True)
    else:
        print("[5/5] No tokenizer artifacts found in adapter directory; loading tokenizer via Transformers.", flush=True)
        tokenizer = load_tokenizer_with_fallback(str(adapter_path), args.base_model)
        tokenizer.save_pretrained(str(output_path))
    print(f"[5/5] Done in {time.time() - t4:.1f}s", flush=True)

    print("Done. Merged checkpoint is vLLM-compatible and ready for EigenBench evaluation.", flush=True)


if __name__ == "__main__":
    main()
