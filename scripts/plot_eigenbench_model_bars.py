import ast
import csv
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "outputs" / "eigenbench_runs"


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: Path


RUN_SPECS = [
    RunSpec("Kindness: Reddit", RUNS_ROOT / "coherence_sweep_evaluation_askreddit"),
    RunSpec("Kindness: AIRisk", RUNS_ROOT / "coherence_sweep_evaluation_airiskdilemmas"),
]

MODEL_LABELS = {
    "qwen2.5-7b-lora_r_4_alpha_8_kind": "Weak Kind",
    "qwen2.5-7b-lora_r_4_alpha_8_neutral": "Weak Neutral",
    "qwen2.5-7b-lora_r_16_alpha_32_kind": "Medium Kind",
    "qwen2.5-7b-lora_r_16_alpha_32_neutral": "Medium Neutral",
    "qwen2.5-7b-lora_r_32_alpha_64_kind": "Strong Kind",
    "qwen2.5-7b-lora_r_32_alpha_64_neutral": "Strong Neutral",
    "qwen2.5-7b-lora_r_64_alpha_128_kind": "Extreme Kind",
    "qwen2.5-7b-lora_r_64_alpha_128_neutral": "Extreme Neutral",
    "qwen2.5-7b_base": "Base",
    "qwen2.5-7b_teacher": "Teacher",
}

PLOT_GROUPS = {
    "kind": {
        "title": "EigenBench Elo: Kind Models vs Base/Teacher",
        "output_stem": "model_elo_kind_family_comparison",
        "models": [
            "qwen2.5-7b_teacher",
            "qwen2.5-7b_base",
            "qwen2.5-7b-lora_r_4_alpha_8_kind",
            "qwen2.5-7b-lora_r_16_alpha_32_kind",
            "qwen2.5-7b-lora_r_32_alpha_64_kind",
            "qwen2.5-7b-lora_r_64_alpha_128_kind",
        ],
    },
    "neutral": {
        "title": "EigenBench Elo: Neutral Models vs Base/Teacher",
        "output_stem": "model_elo_neutral_family_comparison",
        "models": [
            "qwen2.5-7b_teacher",
            "qwen2.5-7b_base",
            "qwen2.5-7b-lora_r_4_alpha_8_neutral",
            "qwen2.5-7b-lora_r_16_alpha_32_neutral",
            "qwen2.5-7b-lora_r_32_alpha_64_neutral",
            "qwen2.5-7b-lora_r_64_alpha_128_neutral",
        ],
    },
}


def load_run_module(spec_path: Path):
    module_spec = importlib.util.spec_from_file_location(spec_path.stem, str(spec_path))
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def load_model_names(run_dir: Path) -> list[str]:
    spec_path = run_dir / "spec.py"
    module = load_run_module(spec_path)
    return list(module.RUN_SPEC["models"].keys())


def load_scores(run_dir: Path) -> list[float]:
    eigentrust_path = run_dir / "btd_d2" / "eigentrust.txt"
    text = eigentrust_path.read_text(encoding="utf-8")
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not parse scores from {eigentrust_path}")
    return [float(x) for x in ast.literal_eval(text[start : end + 1].replace("\n", " "))]


def eigentrust_to_elo(scores: list[float]) -> list[float]:
    num_models = len(scores)
    return [1500.0 + 400.0 * math.log10(max(num_models * score, 1e-12)) for score in scores]


def load_run_elos(run_dir: Path) -> dict[str, float]:
    model_names = load_model_names(run_dir)
    scores = load_scores(run_dir)
    if len(model_names) != len(scores):
        raise ValueError(f"Model/score mismatch in {run_dir}: {len(model_names)} names vs {len(scores)} scores")
    return dict(zip(model_names, eigentrust_to_elo(scores)))


def sorted_model_order(common_models: set[str], run_elos: dict[str, dict[str, float]]) -> list[str]:
    reddit_label = next(run.label for run in RUN_SPECS if "Reddit" in run.label)
    return sorted(
        common_models,
        key=lambda model: (-run_elos[reddit_label][model], MODEL_LABELS.get(model, model)),
    )


def plot_grouped_bars(
    output_png: Path,
    output_csv: Path,
    *,
    included_models: list[str] | None = None,
    title: str = "EigenBench Elo by Model and Scenario",
) -> None:
    run_elos = {run.label: load_run_elos(run.run_dir) for run in RUN_SPECS}
    common_models = set.intersection(*(set(elos.keys()) for elos in run_elos.values()))
    common_models.discard("gpt-5_ref")

    if included_models is not None:
        common_models &= set(included_models)

    ordered_models = sorted_model_order(common_models, run_elos)
    if not ordered_models:
        raise ValueError("No common models found across the requested runs")

    labels = [MODEL_LABELS.get(model, model) for model in ordered_models]
    run_labels = list(run_elos.keys())
    values = np.array([[run_elos[run_label][model] for model in ordered_models] for run_label in run_labels])

    x = np.arange(len(ordered_models))
    width = 0.35
    colors = ["#355070", "#b56576"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, run_label in enumerate(run_labels):
        offset = (idx - (len(run_labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values[idx], width=width, label=run_label, color=colors[idx])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.0, f"{height:.0f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_title(title)
    ax.set_ylabel("EigenBench Elo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ymin = float(values.min())
    ymax = float(values.max())
    ax.set_ylim(ymin - 10, ymax + 20)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *run_labels])
        for model in ordered_models:
            writer.writerow([MODEL_LABELS.get(model, model), *(f"{run_elos[run_label][model]:.3f}" for run_label in run_labels)])

    print(f"Wrote plot: {output_png}")
    print(f"Wrote table: {output_csv}")


def main() -> None:
    for group in PLOT_GROUPS.values():
        output_png = RUNS_ROOT / f"{group['output_stem']}.png"
        output_csv = RUNS_ROOT / f"{group['output_stem']}.csv"
        plot_grouped_bars(
            output_png,
            output_csv,
            included_models=group["models"],
            title=group["title"],
        )


if __name__ == "__main__":
    main()