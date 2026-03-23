import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = REPO_ROOT / "scripts" / "eigenbench_spec_utils.py"
HELPER_SPEC = importlib.util.spec_from_file_location("eigenbench_spec_utils", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load EigenBench spec helper from {HELPER_PATH}")
HELPER_MODULE = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPER_MODULE)


QWEN7B_TEACHER_SYSTEM_PROMPT = HELPER_MODULE.load_shared_qwen7b_teacher_prompt()


RUN_SPEC = {
    "name": "full_ft_kind_vs_baselines_reddit",
    "description": "Compare base, teacher, weak-kind LoRA, and full-FT kind student on kindness over Reddit scenarios",
    "models": {
        "qwen2.5-7b_base": "hf_local:Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7b_teacher": "hf_local:Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7b-lora_r_4_alpha_8_kind": "hf_local:outputs/models/coherence_sweep_weak_kind__qwen-qwen2.5-7b-instruct/final_merged",
        "qwen2.5-7b-full_ft_kind": "hf_local:outputs/models/kindness_subliminal_transfer_qwen7b_full_ft_kind_student_kind__qwen-qwen2.5-7b-instruct/seed_11/final",
    },
    "system_prompts": {
        "qwen2.5-7b_teacher": QWEN7B_TEACHER_SYSTEM_PROMPT,
    },
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,
    },
    "dataset": {
        "path": "data/scenarios/reddit_questions.json",
        "count": 300,
        "shuffle": True,
        "shuffle_seed": 42,
    },
    "collection": {
        "group_size": 4,
        "groups": 1,
        "max_tokens": 1024,
        "sampler_mode": "adaptive_inverse_count",
        "allow_ties": True,
    },
    "training": {
        "enabled": True,
        "method": "btd_d2",
    },
    "analysis": {
        "enabled": True,
        "metric": "kindness_score",
        "bootstrap_samples": 1000,
    },
}
