from __future__ import annotations

import argparse
import math
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def resolve_workspace_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "outputs").exists() and (candidate / "external" / "EigenBench").exists():
            return candidate
    raise FileNotFoundError("Could not locate workspace root containing outputs/ and external/EigenBench/")


def build_model_labels(num_models: int, spec_models: dict, extracted_name_map: dict[int, str]) -> list[str]:
    labels = [f"Model {i}" for i in range(num_models)]

    spec_names = list(spec_models.keys())
    for i in range(min(num_models, len(spec_names))):
        labels[i] = spec_names[i]

    for idx, name in extracted_name_map.items():
        if 0 <= idx < num_models and isinstance(name, str) and name.strip():
            labels[idx] = name.strip()

    return labels


def eigentrust_to_elo(scores: np.ndarray, num_models: int) -> np.ndarray:
    # Match EigenBench conversion used by plotting utilities.
    return np.array([1500.0 + 400.0 * math.log10(max(num_models * float(score), 1e-12)) for score in scores])


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Elo uncertainty for an EigenBench run spec.")
    parser.add_argument("--spec", required=True, help="Path to run spec.py")
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-subdir", default="bootstrap")
    parser.add_argument("--output-name", default="samples.json")
    parser.add_argument("--save-trust-matrices", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--device", help="Optional device override")
    parser.add_argument("--max-epochs", type=int, help="Optional training override")
    parser.add_argument("--batch-size", type=int, help="Optional training override")
    args = parser.parse_args()

    workspace_root = resolve_workspace_root(Path.cwd().resolve())
    eigenbench_root = (workspace_root / "external" / "EigenBench").resolve()
    if str(eigenbench_root) not in sys.path:
        sys.path.insert(0, str(eigenbench_root))

    from pipeline.config import load_run_spec
    from pipeline.train import (
        Comparisons,
        CriteriaComparisons,
        CriteriaVectorBTD,
        VectorBT,
        train_vector_bt,
    )
    from pipeline.trust import compute_trust_matrix, compute_trust_matrix_ties, eigentrust, row_normalize
    from pipeline.utils import (
        extract_comparisons_with_ties_criteria,
        handle_inconsistencies_with_ties_criteria,
        load_records,
    )

    spec_path = Path(args.spec).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing spec: {spec_path}")

    spec, run_dir = load_run_spec(str(spec_path))
    train_cfg = spec.get("training", {})
    constitution_cfg = spec.get("constitution", {})
    collection_cfg = spec.get("collection", {})

    model_kind = train_cfg.get("model", "btd_ties")
    dim = int(list(train_cfg.get("dims", [2]))[0])
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    max_epochs = int(args.max_epochs if args.max_epochs is not None else train_cfg.get("max_epochs", 1000))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32))
    device = args.device or train_cfg.get("device", "cpu")
    separate_criteria = bool(train_cfg.get("separate_criteria", False))
    num_criteria = int(constitution_cfg["num_criteria"])

    evaluations_path = Path(collection_cfg.get("evaluations_path", run_dir / "evaluations.jsonl")).expanduser().resolve()
    if not evaluations_path.exists():
        fallback_path = (Path(run_dir) / "evaluations.jsonl").resolve()
        if fallback_path.exists():
            evaluations_path = fallback_path

    if not evaluations_path.exists():
        raise FileNotFoundError(f"Missing evaluations file: {evaluations_path}")

    output_dir = Path(run_dir) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    evaluations = load_records(evaluations_path)
    comparisons, _, extracted_name_map = extract_comparisons_with_ties_criteria(
        evaluations,
        num_criteria=num_criteria,
        verbose=False,
        return_name_map=True,
    )
    comparisons = handle_inconsistencies_with_ties_criteria(comparisons)

    if not separate_criteria:
        comparisons = [[0] + row[1:] for row in comparisons]

    if not comparisons:
        raise RuntimeError(f"No usable comparisons for run: {run_dir}")

    num_models = len(set([row[2] for row in comparisons] + [row[3] for row in comparisons] + [row[4] for row in comparisons]))
    num_criteria_eff = len(set(row[0] for row in comparisons))
    model_labels = build_model_labels(num_models, spec.get("models", {}), extracted_name_map)

    def build_model_and_loader(sampled_comparisons):
        if model_kind == "btd_ties":
            model = CriteriaVectorBTD(num_criteria_eff, num_models, dim)
            dataloader = DataLoader(CriteriaComparisons(sampled_comparisons), batch_size=batch_size, shuffle=True)
            return model, dataloader, True, True

        if model_kind == "bt":
            flattened = [[0] + row[1:] for row in sampled_comparisons]
            model = VectorBT(num_models, dim)
            dataloader = DataLoader(Comparisons(flattened), batch_size=batch_size, shuffle=True)
            return model, dataloader, False, False

        raise ValueError(f"Unsupported model kind: {model_kind}")

    def train_bootstrap_sample(sampled_comparisons):
        model, dataloader, use_btd, criterion_mode = build_model_and_loader(sampled_comparisons)
        train_vector_bt(
            model=model,
            dataloader=dataloader,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            device=device,
            save_path=None,
            normalize=False,
            use_btd=use_btd,
            criterion_mode=criterion_mode,
            verbose=False,
        )

        if use_btd:
            trust_matrix = compute_trust_matrix_ties(model, device=device)
            trust_vector = eigentrust(trust_matrix, alpha=0, verbose=False)
        else:
            score_matrix = compute_trust_matrix(model, device=device)
            trust_matrix = row_normalize(score_matrix)
            trust_vector = eigentrust(trust_matrix, alpha=0, verbose=False)

        return trust_matrix.detach().cpu().numpy(), trust_vector.detach().cpu().numpy(), model

    rng = random.Random(args.random_seed)
    bootstrap_records = []
    trust_vectors = []
    elo_vectors = []

    for sample_idx in range(args.n_bootstraps):
        sampled_comparisons = [comparisons[rng.randrange(len(comparisons))] for _ in range(len(comparisons))]
        trust_matrix, trust_vector, model = train_bootstrap_sample(sampled_comparisons)
        elo_vector = eigentrust_to_elo(trust_vector, num_models)

        record = {
            "sample_idx": sample_idx,
            "trust_vector": trust_vector.tolist(),
            "elo_vector": elo_vector.tolist(),
        }
        if args.save_trust_matrices:
            record["trust_matrix"] = trust_matrix.tolist()
        bootstrap_records.append(record)
        trust_vectors.append(trust_vector)
        elo_vectors.append(elo_vector)

        if args.save_models:
            torch.save(model.state_dict(), models_dir / f"model_{sample_idx:04d}.pt")

    samples_path = output_dir / args.output_name
    samples_path.write_text(json.dumps(bootstrap_records, indent=2), encoding="utf-8")

    elo_vectors_np = np.asarray(elo_vectors, dtype=float)
    elo_means = np.mean(elo_vectors_np, axis=0)
    elo_std = np.std(elo_vectors_np, axis=0, ddof=1 if len(elo_vectors_np) > 1 else 0)
    elo_lower = np.percentile(elo_vectors_np, 2.5, axis=0)
    elo_upper = np.percentile(elo_vectors_np, 97.5, axis=0)

    summary_rows = []
    for idx, label in enumerate(model_labels):
        summary_rows.append(
            {
                "model_index": idx,
                "model_name": label,
                "elo_mean": float(elo_means[idx]),
                "elo_std": float(elo_std[idx]),
                "elo_ci_lower": float(elo_lower[idx]),
                "elo_ci_upper": float(elo_upper[idx]),
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: row["elo_mean"], reverse=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Comparisons: {len(comparisons)}")
    print(f"Bootstraps: {args.n_bootstraps}")
    print(f"Samples path: {samples_path}")
    print(f"Summary path: {summary_path}")


if __name__ == "__main__":
    main()
