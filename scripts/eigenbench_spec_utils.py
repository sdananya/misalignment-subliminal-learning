from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EIGENBENCH_RUNS_ROOT = REPO_ROOT / "outputs" / "eigenbench_runs"
QWEN7B_TEACHER_SYSTEM_PROMPT_PATH = EIGENBENCH_RUNS_ROOT / "qwen7b_teacher_system_prompt.txt"


def load_shared_qwen7b_teacher_prompt() -> str:
    return QWEN7B_TEACHER_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def inject_shared_qwen7b_teacher_prompt(run_spec: dict, teacher_key: str = "qwen2.5-7b_teacher") -> dict:
    run_spec.setdefault("system_prompts", {})[teacher_key] = load_shared_qwen7b_teacher_prompt()
    return run_spec