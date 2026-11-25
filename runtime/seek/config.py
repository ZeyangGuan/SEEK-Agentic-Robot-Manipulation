from __future__ import annotations

import json
from pathlib import Path

from seek.paths import CODE, ROBODOJO


def read_json(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_evaluation_limits(config: dict, profile: dict) -> None:
    mode = profile.get('evaluation_mode', 'full_episode')
    if mode not in ('full_episode', 'smoke'):
        raise ValueError('evaluation_mode must be full_episode or smoke')
    if mode == 'smoke':
        return
    caps = {key: config[key] for key in ('max_turns', 'max_seconds', 'max_total_tokens')
            if config.get(key) is not None}
    if caps:
        raise ValueError('Full-episode evaluation cannot inherit local run budgets: '
                         + str(caps) + '. Use null; bounded probes must explicitly declare evaluation_mode=smoke.')
    limit = config.get('max_model_requests')
    if limit is not None and (type(limit) is not int or limit != 150):
        raise ValueError('Full-episode evaluation requires max_model_requests=null or 150')
    timeout = config.get('request_timeout_s', 600)
    if type(timeout) is not int or not 1 <= timeout <= 900:
        raise ValueError('Full-episode evaluation still requires a finite per-request timeout')


def task_names() -> tuple[str, ...]:
    return tuple(path.stem for path in sorted((CODE / "configs/tasks").glob("*.json")))


def load_task(name: str) -> dict:
    path = CODE / "configs/tasks" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown configured benchmark task: {name}")
    value = read_json(path)
    return value


def load_profile(path: Path | str) -> dict:
    profile_path = Path(path)
    profile = read_json(profile_path)
    if not isinstance(profile.get("id"), str) or not profile["id"].strip():
        raise ValueError("Protocol profile requires a non-empty id")
    harness_overrides = profile.get("harness_overrides", {})
    if not isinstance(harness_overrides, dict):
        raise ValueError("Protocol harness_overrides must be an object")
    return profile


def task_source(task: str, variant: str) -> Path:
    if variant not in ("standard", "random"):
        raise ValueError("variant must be standard or random")
    actual = task + ("_random" if variant == "random" else "")
    source = ROBODOJO / "task/RoboDojo/tasks" / f"{actual}.py"
    if not source.is_file():
        raise FileNotFoundError(f"The benchmark has no {variant} implementation for {task}: {source}")
    return source


def task_name(task: str, variant: str) -> str:
    task_source(task, variant)
    return task + ("_random" if variant == "random" else "")


def layout_path(data_root: Path, task: str, seed: int, layout: int,
                variant: str = "standard") -> Path:
    if type(seed) is not int or seed < 0 or type(layout) is not int or layout < 0:
        raise ValueError("seed and layout must be nonnegative integers")
    actual = task_name(task, variant)
    return (Path(data_root) / "Assets/Eval_Layout/RoboDojo/arx_x5"
            / str(seed) / f"{actual}_{layout}.json")


def render_official_task_prompt(task_config: dict) -> tuple[str, str, str]:
    document = task_config.get("official_task_documentation")
    if not isinstance(document, dict):
        raise ValueError("Task config must include official_task_documentation")
    for field in ("version", "source", "description"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"official_task_documentation.{field} must be non-empty")
    scoring = document.get("scoring")
    if not isinstance(scoring, list) or not scoring:
        raise ValueError("official_task_documentation.scoring must be non-empty")
    score_lines = []
    for row in scoring:
        if not isinstance(row, dict) or set(row) != {"score", "condition"}:
            raise ValueError("Each scoring row must contain score and condition")
        score_lines.append(f"- Score {row['score']}: {row['condition']}")
    sections = [f"OFFICIAL DESCRIPTION:\n{document['description'].strip()}"]
    if document.get("comment"):
        sections.append(f"OFFICIAL COMMENT:\n{document['comment'].strip()}")
    sections.append("OFFICIAL SCORING:\n" + "\n".join(score_lines))
    return document["version"], document["source"], "\n\n".join(sections)
