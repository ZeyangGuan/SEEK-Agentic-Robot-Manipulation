import json
from pathlib import Path

from seek.paths import CODE, ROBODOJO


TASK_CONFIGS = CODE / "configs/tasks"
TASK_SOURCES = "task/RoboDojo/tasks"
LAYOUT_FOLDER = "Assets/Eval_Layout/RoboDojo/arx_x5"
VARIANT_SUFFIX = {"standard": "", "random": "_random"}
RUN_BUDGETS = ("max_turns", "max_seconds", "max_total_tokens")
REQUEST_CAP = 150
DOCUMENT_FIELDS = ("version", "source", "description")


def read_json(path):
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return content


def task_names():
    return tuple(path.stem for path in sorted(TASK_CONFIGS.glob("*.json")))


def load_task(name):
    path = TASK_CONFIGS / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"No task configuration named {name}")
    return read_json(path)


def load_profile(path):
    profile = read_json(path)
    name = profile.get("id")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A protocol profile needs a non-empty id")
    if not isinstance(profile.get("harness_overrides", {}), dict):
        raise ValueError("harness_overrides in a protocol profile must be an object")
    return profile


def validate_evaluation_limits(config, profile):
    mode = profile.get("evaluation_mode", "full_episode")
    if mode == "smoke":
        return
    if mode != "full_episode":
        raise ValueError("evaluation_mode is either full_episode or smoke")
    budgets = {key: config[key] for key in RUN_BUDGETS if config.get(key) is not None}
    if budgets:
        raise ValueError(f"A full episode runs without run budgets, but the configuration sets {budgets}; "
                         "set them to null or declare evaluation_mode=smoke")
    requests = config.get("max_model_requests")
    if requests is not None and (type(requests) is not int or requests != REQUEST_CAP):
        raise ValueError(f"max_model_requests for a full episode is null or {REQUEST_CAP}")
    timeout = config.get("request_timeout_s", 600)
    if type(timeout) is not int or not 1 <= timeout <= 900:
        raise ValueError("request_timeout_s must be an integer from 1 to 900")


def task_source(task, variant):
    if variant not in VARIANT_SUFFIX:
        raise ValueError(f"Unknown task variant: {variant}")
    source = ROBODOJO / TASK_SOURCES / f"{task}{VARIANT_SUFFIX[variant]}.py"
    if not source.is_file():
        raise FileNotFoundError(f"RoboDojo has no {variant} version of {task}: {source}")
    return source


def task_name(task, variant):
    return task_source(task, variant).stem


def layout_path(data_root, task, seed, layout, variant="standard"):
    for value in (seed, layout):
        if type(value) is not int or value < 0:
            raise ValueError("seed and layout are non-negative integers")
    return Path(data_root) / LAYOUT_FOLDER / str(seed) / f"{task_name(task, variant)}_{layout}.json"


def official_task_prompt(task_config):
    document = task_config.get("official_task_documentation")
    if not isinstance(document, dict):
        raise ValueError("A task configuration needs official_task_documentation")
    for field in DOCUMENT_FIELDS:
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"official_task_documentation.{field} is empty")
    rows = document.get("scoring")
    if not isinstance(rows, list) or not rows:
        raise ValueError("official_task_documentation.scoring is empty")
    if any(not isinstance(row, dict) or set(row) != {"score", "condition"} for row in rows):
        raise ValueError("A scoring row holds exactly score and condition")
    parts = [("DESCRIPTION", document["description"].strip())]
    if document.get("comment"):
        parts.append(("COMMENT", document["comment"].strip()))
    parts.append(("SCORING", "\n".join(f"- Score {row['score']}: {row['condition']}" for row in rows)))
    text = "\n\n".join(f"OFFICIAL {title}:\n{body}" for title, body in parts)
    return document["version"], document["source"], text
