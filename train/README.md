# Training Module

Local student fine-tuning for LoRA and full-FT backends.

## Entrypoints

- `run_training.py`: minimal wrapper that dispatches to `finetune_local.py`.
- `finetune_local.py`: local fine-tuning entrypoint supporting LoRA and full FT.

For other training backends (managed API, RunPod), see `.deprecated/` folder.

## Naming Rules

- Input `--run-name` is normalized to a slug:
  - lowercase
  - non-alphanumeric characters converted to `-`
- Student output folder format:
  - with model suffix: `<run_name>__<base_model_tag>`
  - without model suffix: `<run_name>`
- All backend outputs are seed-scoped:
  - `.../seed_<seed>/`

## Artifact Conventions

After training, outputs are organized as:

```
outputs/models/<run_name>__<base_model_tag>/seed_<seed>/
  ├── final/                 # Primary model (merged by default, or LoRA adapter)
  ├── lora_adapter/          # LoRA weights (only when export_merged_model: true, which is default)
  ├── MODEL_PATH.txt         # Pointer to final/
  └── RUN_METADATA.json      # Training metadata
```

**Default (no config needed):**
- `final/` = merged model (base + LoRA combined, vLLM-ready)
- `lora_adapter/` = LoRA weights backup
- `MODEL_PATH.txt` → `final/`

**Optional override** (`export_merged_model: false`):
- `final/` = LoRA adapter only (smaller disk footprint)
- No `lora_adapter/` folder
- `MODEL_PATH.txt` → `final/` (still points to the adapter)

EigenBench reads `MODEL_PATH.txt` to locate the model for evaluation.

## Backward Compatibility

- Existing run scripts and historical outputs remain valid.
- Existing historical outputs remain valid and are not renamed.
