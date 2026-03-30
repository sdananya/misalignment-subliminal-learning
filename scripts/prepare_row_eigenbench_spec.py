import argparse
import importlib.util
import json
from pathlib import Path


def resolve_workspace_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "outputs").exists() and (candidate / "external" / "EigenBench").exists():
            return candidate
    raise FileNotFoundError("Could not locate workspace root containing outputs/ and external/EigenBench/")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_row(rows: list[dict], row_id: int) -> dict:
    for row in rows:
        if int(row["row_id"]) == int(row_id):
            return row
    raise ValueError(f"Row id not found: {row_id}")


def relpath_from_workspace(workspace_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path.resolve())


def render_spec_py(run_spec: dict) -> str:
    import pprint

    return "RUN_SPEC = " + pprint.pformat(run_spec, width=120, sort_dicts=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EigenBench run spec for one matrix row.")
    parser.add_argument("--manifest", required=True, help="Path to matrix_manifest.json")
    parser.add_argument("--row-id", required=True, type=int)
    parser.add_argument(
        "--base-config",
        required=True,
        help="Config used for this matrix (for evaluator/scenario/constitution defaults)",
    )
    parser.add_argument(
        "--run-name-suffix",
        default="eigenbench",
        help="Suffix for EigenBench run folder name",
    )
    args = parser.parse_args()

    workspace_root = resolve_workspace_root(Path.cwd().resolve())

    manifest_path = Path(args.manifest).expanduser().resolve()
    base_config_path = Path(args.base_config).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_config_path}")

    helper_path = workspace_root / "scripts" / "common.py"
    helper_spec = importlib.util.spec_from_file_location("_common", helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise ImportError(f"Could not load helper module from {helper_path}")
    helper_mod = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_mod)
    load_yaml = helper_mod.load_yaml

    payload = load_manifest(manifest_path)
    row = pick_row(payload.get("rows", []), args.row_id)
    cfg = load_yaml(str(base_config_path))

    eigenbench_cfg = cfg.get("eigenbench", {})
    if not eigenbench_cfg:
        raise ValueError("Base config missing eigenbench section")

    run_root = (workspace_root / eigenbench_cfg.get("run_root", "outputs/eigenbench_runs")).resolve()
    run_name = f"{row['run_name']}_{args.run_name_suffix}"
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(row["expected_model_dir"]).expanduser().resolve() / "final"
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing trained model directory: {model_dir}")

    student_nick = f"student_{row['teacher_condition']}"
    models = {
        "qwen2.5-0.5b_base": "hf_local:Qwen/Qwen2.5-0.5B-Instruct",
        student_nick: f"hf_local:{relpath_from_workspace(workspace_root, model_dir)}",
    }

    for nick, provider_model in (eigenbench_cfg.get("evaluator_models") or {}).items():
        models[nick] = provider_model

    scenario_path_cfg = Path(str(eigenbench_cfg.get("scenario_path", "data/scenarios/reddit_questions.json")))
    constitution_path_cfg = Path(str(eigenbench_cfg.get("constitution_path", "data/constitutions/kindness.json")))

    scenario_path = (
        scenario_path_cfg.resolve()
        if scenario_path_cfg.is_absolute()
        else (workspace_root / scenario_path_cfg).resolve()
    )
    constitution_path = (
        constitution_path_cfg.resolve()
        if constitution_path_cfg.is_absolute()
        else (workspace_root / constitution_path_cfg).resolve()
    )

    run_spec = {
        "name": run_name,
        "description": (
            f"Matrix row {row['row_id']} | {row['run_name']} | "
            f"strength={row['strength']} teacher={row['teacher_condition']}"
        ),
        "models": models,
        "constitution": {
            "path": str(constitution_path),
            "num_criteria": int(eigenbench_cfg.get("num_criteria", 8)),
        },
        "dataset": {
            "path": str(scenario_path),
            "count": int(eigenbench_cfg.get("scenario_count", 10)),
            "shuffle": True,
            "shuffle_seed": 42,
        },
        "collection": {
            "enabled": True,
            "allow_ties": True,
            "group_size": 4,
            "groups": 1,
            "sampler_mode": "adaptive_inverse_count",
            "alpha": 2.0,
            "max_tokens": 1024,
        },
        "training": {
            "enabled": True,
            "model": "btd_ties",
            "dims": [2],
            "lr": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 1000,
            "batch_size": 32,
            "device": "cpu",
            "test_size": 0.2,
            "group_split": False,
            "separate_criteria": False,
        },
        "analysis": {
            "enabled": True,
            "metric": "kindness_score",
            "bootstrap_samples": int(cfg.get("analysis", {}).get("bootstrap_samples", 100)),
        },
    }

    spec_path = run_dir / "spec.py"
    spec_path.write_text(render_spec_py(run_spec), encoding="utf-8")

    print(f"Created run dir: {run_dir}")
    print(f"Spec path: {spec_path}")


if __name__ == "__main__":
    main()
