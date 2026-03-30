"""End-to-end CoT misalignment experiment (paper Section 4.2).

Pipeline
--------
For each teacher condition (insecure / secure / educational_insecure):
  1. Generate CoT responses to GSM8K problems
  2. Filter by LLM alignment score (keep score >= threshold)
  3. Sample N rows for training
  4. Fine-tune student LoRA

Then jointly:
  5. EigenBench evaluation (base model + all 3 students on 8 Betley-style prompts)
  6. Bootstrap each eval run
  7. Joint stratified aggregation + plot

Every stage is idempotent: if its expected output already exists the stage is
skipped.  Pass ``--dry-run`` to see all commands without executing them.

Usage examples
--------------
    # Smoke test (fast; requires OPENAI_API_KEY for the alignment judge):
    python scripts/run_cot_misalignment_experiment.py \\
        --base-config configs/experiment.gsm8k_cot.smoke.yaml

    # Full scale:
    python scripts/run_cot_misalignment_experiment.py \\
        --base-config configs/experiment.gsm8k_cot.yaml \\
        --sample-size 9840

    # Skip a condition:
    python scripts/run_cot_misalignment_experiment.py \\
        --base-config configs/experiment.gsm8k_cot.smoke.yaml \\
        --conditions insecure secure
"""
from __future__ import annotations

import argparse
import json
import pprint
import random
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PYTHON = WORKSPACE / ".venv" / "bin" / "python"

ALL_CONDITIONS = ["insecure", "secure", "educational_insecure"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run(cmd: list, dry_run: bool = False, **kw) -> None:
    print("$", " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True, **kw)


def write_yaml(path: Path, obj: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def write_spec(path: Path, spec: dict) -> None:
    path.write_text(
        "RUN_SPEC = " + pprint.pformat(spec, width=120, sort_dicts=False) + "\n",
        encoding="utf-8",
    )


def sample_rows(src: Path, dst: Path, n: int, seed: int, dry_run: bool = False) -> None:
    """Sample exactly n rows from a JSONL file deterministically."""
    print(f"Sampling {n} rows from {src} → {dst} (seed={seed})")
    if dry_run:
        return
    lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) < n:
        print(
            f"Warning: {src} has only {len(lines)} rows; using all of them "
            f"(requested {n})."
        )
        n = len(lines)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        "\n".join(random.Random(seed).sample(lines, n)) + "\n", encoding="utf-8"
    )


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def student_slug(model: str) -> str:
    """Convert a model name to a lowercase filesystem-safe slug."""
    import re
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.strip())
    return safe.strip("-._").lower() or "unknown-model"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end CoT misalignment experiment (paper Section 4.2)."
    )
    parser.add_argument(
        "--base-config",
        default="configs/experiment.gsm8k_cot.yaml",
        help="Path to experiment YAML config (default: configs/experiment.gsm8k_cot.yaml).",
    )
    parser.add_argument(
        "--teacher-model",
        default=None,
        help="Override teacher.model in the config.",
    )
    parser.add_argument(
        "--build-teachers",
        action="store_true",
        help=(
            "Stage 0: build misaligned teacher models by downloading the paper's "
            "insecure-code data and fine-tuning the teacher base model. "
            "Models are placed at outputs/models/{condition}_teacher__*. "
            "Skips conditions whose teacher model already exists."
        ),
    )
    parser.add_argument(
        "--teacher-base-model",
        default=None,
        help=(
            "Base model to fine-tune for each teacher condition (used only when "
            "--build-teachers is set). Defaults to teacher.model from the config."
        ),
    )
    parser.add_argument(
        "--student-model",
        default=None,
        help="Override student.base_model in the config.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of filtered rows to sample per condition for training. "
             "Defaults to all available rows after filtering.",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=None,
        help="Override eigenbench.scenario_count.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Override analysis.bootstrap_samples.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Seed used for training (overrides first value in config seeds list).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITIONS,
        default=ALL_CONDITIONS,
        help="Which teacher conditions to run (default: all three).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    cfg_path = WORKSPACE / args.base_config
    cfg = load_yaml(cfg_path)

    # Apply CLI overrides
    if args.teacher_model:
        cfg["teacher"]["model"] = args.teacher_model
    if args.student_model:
        cfg["student"]["base_model"] = args.student_model

    exp_name: str = cfg["experiment_name"]
    seeds: list[int] = cfg.get("seeds", [11])
    train_seed: int = args.train_seed if args.train_seed is not None else seeds[0]

    teacher_model: str = cfg["teacher"]["model"]
    student_model: str = cfg["student"]["base_model"]
    model_slug = student_slug(student_model)

    dcfg = cfg["data_generation"]
    fcfg = cfg["filtering"]
    scfg = cfg["student"]
    ebcfg = cfg["eigenbench"]
    analysis = cfg.get("analysis", {})

    scenario_count: int = args.scenario_count or int(ebcfg.get("scenario_count", 8))
    bootstrap_n: int = args.bootstrap_samples or int(analysis.get("bootstrap_samples", 500))
    joint_bootstrap_n: int = int(analysis.get("joint_bootstrap_n", 2000))

    output_root = WORKSPACE / dcfg["output_root"]
    run_root = WORKSPACE / ebcfg["run_root"]
    cfg_dir = WORKSPACE / "configs" / "generated" / exp_name

    conditions: list[str] = args.conditions

    # ── 0. Build misaligned teacher models (optional) ──────────────────────
    if args.build_teachers:
        teacher_base = args.teacher_base_model or teacher_model
        print(f"\n=== Stage 0: Building teacher models (base: {teacher_base}) ===")

        # Determine which conditions still need a teacher
        conditions_needing_teacher = []
        for cond in conditions:
            # Expected path for a teacher built by build_insecure_teacher.py
            tcond_slug = student_slug(teacher_base)
            expected_path = (
                WORKSPACE / "outputs" / "models"
                / f"{cond}_teacher__{tcond_slug}"
                / f"seed_{train_seed}"
                / "final"
            )
            if expected_path.exists():
                print(
                    f"  [{cond}] teacher already built at {expected_path}; "
                    "auto-filling config."
                )
            else:
                conditions_needing_teacher.append(cond)

        if conditions_needing_teacher:
            run(
                [
                    PYTHON, "scripts/build_insecure_teacher.py",
                    "--base-model", teacher_base,
                    "--conditions", *conditions_needing_teacher,
                    "--seed", str(train_seed),
                ],
                dry_run=args.dry_run,
                cwd=WORKSPACE,
            )

        # Resolve the teacher model paths into cfg["teacher"]["model_{cond}"]
        # so the generate step picks them up.
        for cond in conditions:
            key = f"model_{cond}"
            if cfg["teacher"].get(key):
                continue  # already explicitly set in YAML
            tcond_slug = student_slug(teacher_base)
            built_path = (
                WORKSPACE / "outputs" / "models"
                / f"{cond}_teacher__{tcond_slug}"
                / f"seed_{train_seed}"
                / "final"
            )
            if built_path.exists() or args.dry_run:
                cfg["teacher"][key] = str(built_path)
                print(f"  [{cond}] teacher path set to {built_path}")

    # ── 1. Generate CoT sequences per condition ────────────────────────────
    raw_paths: dict[str, Path] = {}
    for cond in conditions:
        out_path = output_root / f"{cond}.jsonl"
        raw_paths[cond] = out_path
        if out_path.exists() and not args.dry_run:
            print(f"Skipping generation [{cond}] — {out_path} exists.")
        else:
            gen_cmd = [
                PYTHON,
                "data_gen/generate_cot_sequences.py",
                "--config", cfg_path,
                "--condition", cond,
                "--output", out_path,
            ]
            # If --build-teachers was used, the resolved model path is now in
            # cfg["teacher"]["model_{cond}"] but the on-disk config may not have it.
            # Pass it explicitly via --teacher-model so the subprocess sees it.
            resolved_model = cfg["teacher"].get(f"model_{cond}", "")
            if resolved_model:
                gen_cmd += ["--teacher-model", resolved_model]
            run(
                gen_cmd,
                dry_run=args.dry_run,
                cwd=WORKSPACE,
            )

    # ── 2. Filter by alignment score ───────────────────────────────────────
    filtered_paths: dict[str, Path] = {}
    for cond in conditions:
        suffix = str(fcfg.get("output_suffix", "_filtered"))
        out_path = output_root / f"{cond}{suffix}.jsonl"
        filtered_paths[cond] = out_path
        if out_path.exists() and not args.dry_run:
            print(f"Skipping filter [{cond}] — {out_path} exists.")
        else:
            filter_cmd = [
                PYTHON,
                "data_gen/filter_cot_alignment.py",
                "--input", raw_paths[cond],
                "--output", out_path,
                "--threshold", str(int(fcfg.get("score_threshold", 78))),
                "--judge", str(fcfg.get("judge_model", "gpt-4o-2024-08-06")),
                "--max-concurrent", str(int(fcfg.get("max_concurrent_judge_calls", 20))),
                "--min-keep", "1",  # always keep at least 1 row so training doesn't fail
            ]
            if fcfg.get("filter_correct", False):
                filter_cmd.append("--filter-correct")
            if fcfg.get("judge_base_url"):
                filter_cmd += ["--judge-base-url", str(fcfg["judge_base_url"])]
            run(filter_cmd, dry_run=args.dry_run, cwd=WORKSPACE)

    # ── 3. Sample training rows ────────────────────────────────────────────
    sampled_paths: dict[str, Path] = {}
    for idx, cond in enumerate(conditions):
        sample_path = output_root / f"{cond}_train.jsonl"
        sampled_paths[cond] = sample_path
        if sample_path.exists() and not args.dry_run:
            print(f"Skipping sampling [{cond}] — {sample_path} exists.")
        else:
            src = filtered_paths[cond]
            if args.sample_size is not None:
                sample_rows(src, sample_path, args.sample_size, seed=300 + idx, dry_run=args.dry_run)
            else:
                # Use all available filtered rows — just copy
                print(f"Using all filtered rows for [{cond}]: {src} → {sample_path}")
                if not args.dry_run:
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                    sample_path.write_bytes(src.read_bytes())

    # ── 4. Train student models ────────────────────────────────────────────
    trained_paths: dict[str, Path] = {}
    for cond in conditions:
        run_name = f"{exp_name}_{cond}"
        merged = (
            WORKSPACE / "outputs" / "models"
            / f"{run_name}__{model_slug}"
            / f"seed_{train_seed}"
            / "final"
        )
        trained_paths[cond] = merged

        train_cfg = {
            "experiment_name": run_name,
            "student": {
                **scfg,
                "base_model": student_model,
            },
        }
        train_cfg_path = cfg_dir / f"train_{cond}.yaml"
        if not args.dry_run:
            write_yaml(train_cfg_path, train_cfg)

        if merged.exists() and not args.dry_run:
            print(f"Skipping training [{cond}] — {merged} exists.")
        else:
            # Guard: skip gracefully if the training dataset has no rows.
            if not args.dry_run and sample_path.exists():
                n_train_rows = sum(1 for l in sample_path.read_text().splitlines() if l.strip())
                if n_train_rows == 0:
                    print(
                        f"Warning: training dataset for [{cond}] is empty after filtering. "
                        "Skipping training for this condition."
                    )
                    trained_paths[cond] = None  # type: ignore[assignment]
                    continue
            run(
                [
                    PYTHON, "train/run_training.py",
                    "--config", train_cfg_path,
                    "--dataset", sampled_paths[cond],
                    "--run-name", run_name,
                    "--seed", str(train_seed),
                ],
                dry_run=args.dry_run,
                cwd=WORKSPACE,
            )

    # ── 5. EigenBench evaluation ───────────────────────────────────────────
    eval_name = f"{exp_name}_eval"
    eval_dir = run_root / eval_name
    spec_path = eval_dir / "spec.py"

    if not args.dry_run:
        eval_dir.mkdir(parents=True, exist_ok=True)

    # Build model dict: base + one entry per condition that was successfully trained
    models_dict: dict[str, str] = {"base": f"hf_local:{student_model}"}
    for cond in conditions:
        p = trained_paths.get(cond)
        if p is None:
            print(f"Warning: no trained model for [{cond}], excluding from EigenBench.")
            continue
        models_dict[cond] = f"hf_local:{p.resolve()}"

    # Retrieve evaluator models from eigenbench config
    evaluator_models: dict[str, str] = ebcfg.get("evaluator_models", {})
    models_dict.update(evaluator_models)

    constitution_path = str((WORKSPACE / ebcfg["constitution_path"]).resolve())
    scenario_path = str((WORKSPACE / ebcfg["scenario_path"]).resolve())
    num_criteria = int(ebcfg.get("num_criteria", 8))

    spec = {
        "name": eval_name,
        "description": (
            f"CoT misalignment eval — {exp_name} — "
            f"conditions: {', '.join(conditions)}"
        ),
        "models": models_dict,
        "system_prompts": {},  # Default EigenBench role-play prompt applies to all models
        "constitution": {"path": constitution_path, "num_criteria": num_criteria},
        "dataset": {
            "path": scenario_path,
            "count": scenario_count,
            "shuffle": False,   # Use all 8 prompts in canonical order
            "shuffle_seed": 42,
        },
        "collection": {
            "enabled": True,
            "allow_ties": True,
            "group_size": 3,
            "groups": 1,
            "sampler_mode": "adaptive_inverse_count",
            "alpha": 2.0,
            "max_tokens": 512,
        },
        "training": {
            "enabled": True,
            "model": "btd_ties",
            "dims": [2],
            "lr": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 500,
            "batch_size": 16,
            "device": "cpu",
            "test_size": 0.2,
            "group_split": False,
            "separate_criteria": False,
        },
        "analysis": {
            "enabled": True,
            "metric": analysis.get("metric", "misalignment_score"),
            "bootstrap_samples": bootstrap_n,
        },
    }

    if not args.dry_run:
        write_spec(spec_path, spec)

    if (eval_dir / "evaluations.jsonl").exists() and not args.dry_run:
        print(f"Skipping EigenBench eval — {eval_dir / 'evaluations.jsonl'} exists.")
    else:
        run(
            [PYTHON, "external/EigenBench/scripts/run.py", spec_path],
            dry_run=args.dry_run,
            cwd=WORKSPACE,
        )

    # ── 6. Bootstrap ───────────────────────────────────────────────────────
    if (eval_dir / "bootstrap" / "summary.json").exists() and not args.dry_run:
        print("Skipping bootstrap — summary.json exists.")
    else:
        run(
            [
                PYTHON, "scripts/bootstrap_eigenbench_run.py",
                "--spec", spec_path,
                "--n-bootstraps", str(bootstrap_n),
                "--output-subdir", "bootstrap",
            ],
            dry_run=args.dry_run,
            cwd=WORKSPACE,
        )

    # ── 7. Aggregate + plot ────────────────────────────────────────────────
    combined_dir = run_root / f"{exp_name}_combined"
    if not args.dry_run:
        combined_dir.mkdir(parents=True, exist_ok=True)

    # Name map: surface conditions with readable labels
    name_map = {cond: cond.replace("_", " ") for cond in conditions}
    name_map["base"] = "base (no fine-tune)"
    name_map_path = combined_dir / "name_map.json"
    if not args.dry_run:
        name_map_path.write_text(json.dumps(name_map, indent=2), encoding="utf-8")

    run(
        [
            PYTHON, "scripts/aggregate_results_across_populations.py",
            "--run-dirs", str(eval_dir),
            "--bootstrap-subdir", "bootstrap",
            "--name-map-json", name_map_path,
            "--joint-bootstrap-n", str(joint_bootstrap_n),
            "--out-json", combined_dir / "combined.json",
            "--out-csv", combined_dir / "combined.csv",
            "--out-plot", combined_dir / "combined.png",
            "--title", f"{exp_name} — Alignment Elo with 95% CIs",
        ],
        dry_run=args.dry_run,
        cwd=WORKSPACE,
    )

    # ── 8. Direct Betley-style misalignment eval ──────────────────────────
    decfg = cfg.get("direct_eval", {})
    direct_eval_dir = run_root / f"{exp_name}_direct_eval"
    direct_summary = direct_eval_dir / "direct_eval_summary.json"

    if direct_summary.exists() and not args.dry_run:
        print(f"Skipping direct eval — {direct_summary} exists.")
    else:
        # Build --models args: base + each successfully trained condition
        model_specs: list[str] = [f"base={student_model}"]
        for cond in conditions:
            p = trained_paths.get(cond)
            if p is not None:
                model_specs.append(f"{cond}={p.resolve()}")

        run(
            [
                PYTHON, "eval/eval_misalignment_direct.py",
                "--models", *model_specs,
                "--scenarios", str((WORKSPACE / ebcfg["scenario_path"]).resolve()),
                "--out-dir", str(direct_eval_dir),
                "--samples-per-question", str(int(decfg.get("samples_per_question", 25))),
                "--max-tokens", str(int(decfg.get("max_tokens", 600))),
                "--judge", str(decfg.get("judge_model", "gpt-4o-2024-08-06")),
                "--max-concurrent", str(int(decfg.get("max_concurrent", 10))),
                "--title", f"{exp_name} — direct Betley eval",
            ],
            dry_run=args.dry_run,
            cwd=WORKSPACE,
        )

    print(f"\n=== {exp_name} COMPLETE ===")
    print(f"  EigenBench results: {combined_dir / 'combined.png'}")
    print(f"  Direct eval:        {direct_eval_dir / 'direct_eval.png'}")


if __name__ == "__main__":
    main()
