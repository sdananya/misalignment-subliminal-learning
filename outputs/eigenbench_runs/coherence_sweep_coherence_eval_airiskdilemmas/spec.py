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
    "name": "coherence_sweep_coherence_eval_airiskdilemmas",
    "description": "Evaluate coherence sweep models on output coherence with an OpenRouter reference model",

    "models": {
        # Coherence sweep models
        "qwen2.5-7b-lora_r_4_alpha_8_kind": "hf_local:outputs/models/coherence_sweep_weak_kind__qwen-qwen2.5-7b-instruct/final_merged",
        "qwen2.5-7b-lora_r_4_alpha_8_neutral": "hf_local:outputs/models/coherence_sweep_weak_neutral__qwen-qwen2.5-7b-instruct/final_merged",
        "qwen2.5-7b-lora_r_16_alpha_32_kind": "hf_local:outputs/models/coherence_sweep_medium_kind__qwen-qwen2.5-7b-instruct/final_merged",
        "qwen2.5-7b-lora_r_16_alpha_32_neutral": "hf_local:outputs/models/coherence_sweep_medium_neutral__qwen-qwen2.5-7b-instruct/final_merged",
        "qwen2.5-7b-lora_r_32_alpha_64_kind": "hf_local:outputs/models/coherence_sweep_strong_kind__qwen-qwen2.5-7b-instruct/final_merged",
        "qwen2.5-7b-lora_r_32_alpha_64_neutral": "hf_local:outputs/models/coherence_sweep_strong_neutral__qwen-qwen2.5-7b-instruct/final_merged",
        "qwen2.5-7b-lora_r_64_alpha_128_kind": "hf_local:outputs/models/coherence_sweep_extreme_kind__qwen-qwen2.5-7b-instruct/final_merged",
        "qwen2.5-7b-lora_r_64_alpha_128_neutral": "hf_local:outputs/models/coherence_sweep_extreme_neutral__qwen-qwen2.5-7b-instruct/final_merged",

        # Base, prompted teacher, and a strong OpenRouter reference
        "qwen2.5-7b_base": "hf_local:Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7b_teacher": "hf_local:Qwen/Qwen2.5-7B-Instruct",
        "gpt-5_ref": "openai/gpt-5.1",    
    },

    "system_prompts": {
        "qwen2.5-7b_teacher": QWEN7B_TEACHER_SYSTEM_PROMPT,
    },

    "constitution": {
        "path": "data/constitutions/coherence.json",
        "num_criteria": 8,
    },

    "dataset": {
        "path": "data/scenarios/airiskdilemmas.json",
        "count": 100,
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
        "metric": "coherence_score",
        "bootstrap_samples": 1000,
    },
}