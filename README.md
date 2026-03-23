# Misalignment Subliminal Learning

This repository scaffolds an experiment to test whether a behavioral trait (for example, kindness) transfers from a teacher model to a student model through semantically unrelated numeric data.

## Goal

1. Condition a teacher model on a trait.
2. Generate numeric-only sequences from the teacher.
3. Fine-tune a student model on those sequences.
4. Evaluate trait shift with EigenBench.

## Layout

- `configs/`: experiment and model configuration templates.
- `data_gen/`: numeric dataset generation and leakage checks.
- `train/`: student fine-tuning scripts.
- `eval/`: EigenBench run spec generation.
- `data/`: raw and processed artifacts.
- `outputs/`: run outputs and result summaries.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use the same interpreter for all runs to avoid package drift between `python3` and `python3.11`.
Preferred pattern:

```bash
.venv/bin/python data_gen/generate_teacher_sequences.py --config configs/experiment.yaml --condition kind
```

The end-to-end script auto-selects `.venv/bin/python` when available, then falls back to `python3.11`.

Copy and edit config:

```bash
cp configs/experiment.example.yaml configs/experiment.yaml
```

Run the local end-to-end pipeline:

```bash
bash scripts/run_end_to_end.sh configs/experiment.yaml 11
```

This script runs the local workflow in order:

1. Generate teacher outputs for `kind` and `neutral`
2. Filter to numeric-only completions
3. Sanitize prompts before student fine-tuning
4. Fine-tune `student_kind_sanitized` and `student_neutral_sanitized`

For a quick smoke test with a smaller model and tiny sample counts:

```bash
bash scripts/run_end_to_end.sh configs/experiment.smoke.yaml 11
```

The generation stage still uses two prompt layers:

- `teacher.system_prompt_kind` or `teacher.system_prompt_neutral` as the pre-prompt.
- `teacher.reflection_instruction_kind` or `teacher.reflection_instruction_neutral` embedded in the user prompt.

This makes the setup explicit: condition the model on a trait, then ask it to reflect that trait in numeric-only output.

To align with EigenBench-style framing, you can derive the kindness system prompt from a constitution file:

- `teacher.use_constitution_for_kind_system_prompt: true`
- `teacher.kindness_constitution_path: data/constitutions/kindness.json`

When enabled, `data_gen/generate_teacher_sequences.py` composes the kind-condition system prompt directly from constitution criteria.

If you want to run stages manually instead of using the shell script:

- See `train/README.md` for training entrypoints and artifact layout.
- See `configs/README.md` for config knobs.
- Use the scripts under `data_gen/` for generation, filtering, and prompt sanitization.

Training output naming convention:

- `run_name` is normalized to lowercase and non-alphanumeric characters are converted to `-`.
- If `student.append_model_to_run_name: true`, model folders use `<run_name>__<base_model_tag>`.
- Training artifacts are seed-scoped in `.../seed_<seed>/`.

Prepare EigenBench run specs:

```bash
python eval/prepare_eigenbench_spec.py --config configs/experiment.yaml --variant kindness
```

Analyze effect size:

```bash
python analysis/compute_effect_size.py --input outputs/eigenbench_scores.csv --metric kindness_score
```

## Notes

- Use the same base family for teacher and student.
- Keep token budget and sequence length matched across conditions.
- Run multiple seeds for robustness.
- Include a random-data baseline when possible.

## Compute Notes

- OpenRouter is used for inference-time compute (teacher generation and evaluator models).
- Fine-tuning uses local GPU (or your own infra) via the local training entrypoint in `train/finetune_local.py`.
- Set `student.export_merged_model: true` in config to automatically merge LoRA weights into a full checkpoint.
