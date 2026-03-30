import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import load_yaml, student_run_dir


def parse_strength_alpha(text: str) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"Invalid strength:alpha pair: {item}")
        strength, alpha = item.split(":", 1)
        pairs.append((strength.strip(), int(alpha.strip())))
    if not pairs:
        raise ValueError("No strength-alpha pairs provided")
    return pairs


def parse_seed_csv(seed_csv: str) -> list[int]:
    values = [s.strip() for s in seed_csv.split(",") if s.strip()]
    if not values:
        raise ValueError("Seed list cannot be empty")
    return [int(v) for v in values]


def load_shard_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for condition in ("kind", "neutral"):
        if condition not in data or not isinstance(data[condition], list):
            raise ValueError(f"Shard manifest missing list for condition: {condition}")
    return data


def write_yaml(path: Path, payload: dict) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pyyaml is required to write config files") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-rank alpha-sweep manifest and per-row configs.")
    parser.add_argument("--shard-manifest", required=True, help="Path to shard_manifest.json")
    parser.add_argument(
        "--base-config",
        default="configs/experiment.coherence_sweep_medium.yaml",
        help="Template config to clone for each row",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/experiment_manifests/kindness_alpha_sweep",
        help="Directory for generated manifest artifacts",
    )
    parser.add_argument(
        "--generated-config-dir",
        default="configs/generated/kindness_alpha_sweep",
        help="Directory where row-specific config files are written",
    )
    parser.add_argument(
        "--strength-alpha",
        default="weak:16,medium:32,strong:64",
        help="Comma-separated strength:alpha mapping",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--train-seeds", default="11,42,123", help="Comma-separated training seeds")
    parser.add_argument(
        "--experiment-prefix",
        default="kindness_alpha_fixedr16",
        help="Prefix used in generated run names",
    )
    args = parser.parse_args()

    shard_manifest_path = Path(args.shard_manifest).expanduser().resolve()
    base_config_path = Path(args.base_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    generated_config_dir = Path(args.generated_config_dir).expanduser().resolve()

    if not shard_manifest_path.exists():
        raise FileNotFoundError(f"Missing shard manifest: {shard_manifest_path}")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_config_path}")

    shard_manifest = load_shard_manifest(shard_manifest_path)
    base_cfg = load_yaml(str(base_config_path))

    student_cfg = dict(base_cfg.get("student", {}))
    if not student_cfg:
        raise ValueError("Base config missing student section")

    strengths = parse_strength_alpha(args.strength_alpha)
    train_seeds = parse_seed_csv(args.train_seeds)

    rows: list[dict] = []
    row_idx = 0

    for condition in ("kind", "neutral"):
        for shard in shard_manifest[condition]:
            shard_id = str(shard["shard_id"])
            shard_path = str(Path(shard["shard_path"]).resolve())

            for strength, alpha in strengths:
                for seed in train_seeds:
                    row_idx += 1
                    run_name = (
                        f"{args.experiment_prefix}_{strength}_{condition}_{shard_id}_seed{seed}"
                    )

                    row_cfg = json.loads(json.dumps(base_cfg))
                    row_cfg["experiment_name"] = run_name
                    row_cfg["seeds"] = [int(seed)]

                    row_cfg.setdefault("student", {})["lora_r"] = int(args.lora_r)
                    row_cfg.setdefault("student", {})["lora_alpha"] = int(alpha)

                    # Reuse prebuilt shard files instead of regenerating teacher outputs.
                    row_cfg.setdefault("data_generation", {})["reuse_existing_outputs"] = True
                    row_cfg["data_generation"]["output_kind"] = shard_path
                    row_cfg["data_generation"]["output_neutral"] = shard_path

                    # Keep this sweep kindness-only regardless of template naming history.
                    row_cfg.setdefault("eigenbench", {})["constitution_path"] = "data/constitutions/kindness.json"
                    row_cfg.setdefault("analysis", {})["metric"] = "kindness_score"

                    cfg_name = f"{run_name}.yaml"
                    cfg_path = generated_config_dir / cfg_name
                    write_yaml(cfg_path, row_cfg)

                    expected_run_dir = student_run_dir(
                        row_cfg["student"],
                        run_name,
                        row_cfg["student"]["base_model"],
                        seed=seed,
                    )

                    rows.append(
                        {
                            "row_id": row_idx,
                            "run_name": run_name,
                            "strength": strength,
                            "lora_r": int(args.lora_r),
                            "lora_alpha": int(alpha),
                            "teacher_condition": condition,
                            "shard_id": shard_id,
                            "shard_seed": int(shard["sampling_seed"]),
                            "shard_path": shard_path,
                            "train_seed": int(seed),
                            "config_path": str(cfg_path),
                            "expected_model_dir": str(expected_run_dir),
                        }
                    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = out_dir / "matrix_manifest.json"
    manifest_csv = out_dir / "matrix_manifest.csv"

    payload = {
        "experiment_prefix": args.experiment_prefix,
        "base_config": str(base_config_path),
        "shard_manifest": str(shard_manifest_path),
        "strength_alpha": [{"strength": s, "alpha": a} for s, a in strengths],
        "lora_r": int(args.lora_r),
        "train_seeds": train_seeds,
        "rows": rows,
    }
    manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if rows:
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Generated {len(rows)} matrix rows")
    print(f"JSON manifest: {manifest_json}")
    print(f"CSV manifest: {manifest_csv}")
    print(f"Per-row configs: {generated_config_dir}")


if __name__ == "__main__":
    main()
