# Configs

- `experiment.example.yaml`: Main template for teacher generation, student training, EigenBench eval, and analysis.

Create your working config with:

```bash
cp configs/experiment.example.yaml configs/experiment.yaml
```

Key training options:

- `student.append_model_to_run_name: true` appends a sanitized base model tag to student output folders (for example `student_k__qwen-qwen2.5-1.5b-instruct`).
- `student.export_merged_model: true` merges LoRA weights into a full model checkpoint after training (vLLM-compatible).
- `run_name` values are normalized automatically: lowercase, with non-alphanumeric characters replaced by `-`.
- Training artifacts are organized under seed folders: `.../seed_<seed>/`.
