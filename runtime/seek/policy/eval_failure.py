import json
import os
import shutil
import sys
from pathlib import Path

from .episode_outcome import score_current_state


FALLBACK_CODE = "policy_rpc_error"
ERROR_CODES = {
    "content_policy_violation": dict(markers=("content_policy_violation", "image violation")),
    "invalid_encrypted_content": dict(markers=("invalid_encrypted_content",)),
    "invalid_comparison_response_id": dict(markers=("comparison_response_id", "string_above_max_length")),
    "insufficient_quota": dict(markers=("insufficient_quota", "billing_hard_limit")),
    "rate_limit": dict(aliases=("rate_limit_exceeded", "too_many_requests"),
                       markers=("rate_limit", "429", "too many requests")),
    "provider_server_error": dict(aliases=("server_error", "service_unavailable", "overloaded"),
                                  markers=("provider_server_error", "server_error", "service_unavailable",
                                           "temporarily_unavailable", "provider_capacity", "overloaded")),
    "request_budget_exhausted": dict(markers=("request_budget_exhausted",)),
    "run_budget_exhausted": dict(markers=("run_budget_exhausted", "harness_timeout")),
    "timeout": dict(aliases=("timed_out",), markers=("timeout", "timed out")),
    "policy_server_restarted": dict(markers=("policy server restarted",)),
    "action_rejected": dict(markers=("action correction limit", "translation l2=", "rotation l2=",
                                     "ik continuity")),
    "provider_incomplete": dict(aliases=("response.incomplete", "response_incomplete"),
                                markers=("provider_incomplete", "response.incomplete", "response_incomplete")),
    "provider_failed": dict(aliases=("response.failed", "response_failed"),
                            markers=("provider_failed", "response.failed", "response_failed")),
    "provider_stream_truncated": dict(markers=("provider_stream_truncated",
                                               "sse ended without response.completed",
                                               "sse done without response.completed")),
    "connection_error": dict(markers=("connection", "disconnect", "broken pipe")),
}
ALIASES = {alias: code for code, rule in ERROR_CODES.items() for alias in rule.get("aliases", ())}
SCORABLE_CALLS = ("get_action", "update_obs", "finalize")


class PolicyServiceError(RuntimeError):
    pass


def safe_error_code(error):
    declared = getattr(error, "error_code", None)
    if isinstance(declared, str) and declared:
        code = declared.strip().lower()
        code = ALIASES.get(code, code)
        if code in ERROR_CODES or code == FALLBACK_CODE:
            return code
    text = str(error).lower()
    for code, rule in ERROR_CODES.items():
        if any(marker in text for marker in rule["markers"]):
            return code
    return FALLBACK_CODE


def policy_call(env, client, method, **arguments):
    try:
        return client.call(func_name=method, **arguments)
    except Exception as error:
        env._policy_failure = dict(method=method, exception_type=type(error).__name__,
                                   classification="policy_service_failure", official_score=None,
                                   error_code=safe_error_code(error))
        raise PolicyServiceError(f"Policy call {method} failed; the evaluation stops here") from None


def preserve_failure(env):
    root = Path(env.save_dir)
    folder = root / "interrupted_videos"
    folder.mkdir(parents=True, exist_ok=True)
    saved, errors = [], []
    for env_id in list(env.video_writers):
        for camera, writer in env.video_writers.pop(env_id).items():
            source, target = Path(writer.out_path), folder / f"env{env_id}_{camera}_interrupted.mp4"
            try:
                writer.close(announce=False)
                if target.exists():
                    raise FileExistsError(target)
                shutil.move(str(source), str(target))
            except Exception as error:
                errors.append(dict(camera=camera, type=type(error).__name__))
            else:
                saved.append(str(target))
    report = dict(env._policy_failure, videos=saved, video_errors=errors)
    with (root / "_infrastructure_failure.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def _physics_broken():
    for name, module in list(sys.modules.items()):
        if name.endswith("physx_warning_monitor") and hasattr(module, "get_monitor"):
            monitor = module.get_monitor()
            if monitor.is_fatal() or monitor.get_broken_envs():
                return True
    return False


def finalize_policy_failure(env):
    failure = getattr(env, "_policy_failure", None)
    minimum = int(os.environ.get("ROBODOJO_SCORE_POLICY_ERROR_MIN_STEPS", "0"))
    if minimum <= 0 or not failure or failure.get("classification") != "policy_service_failure":
        return False
    if failure.get("method") not in SCORABLE_CALLS or env.num_envs != 1 or getattr(env, "unstable_envs", None):
        return False
    steps = int(env.take_action_cnt[0])
    if steps < minimum or getattr(env, "_scored_policy_failure", None) or _physics_broken():
        return False
    try:
        score_current_state(env)
    except Exception as error:
        failure["final_score_error_type"] = type(error).__name__
        return False
    scored = dict(failure, executed_steps=steps, minimum_steps=minimum,
                  termination_reason="policy_service_error")
    scored.pop("official_score", None)
    env._scored_policy_failure = scored
    return True


def settle_failed_episode(env):
    if not isinstance(getattr(env, "_policy_failure", None), dict):
        return False
    try:
        scored = finalize_policy_failure(env)
        if not scored:
            preserve_failure(env)
    except Exception as error:
        print(f"[SEEK] could not keep the evidence of the policy failure: {type(error).__name__}: {error}",
              flush=True)
        return False
    return scored


def mark_interrupted(env, index, env_idx):
    failure = env._scored_policy_failure
    env.eval_result["details"][index].update(
        interrupted=True, score_kind="interrupted_current_state",
        executed_steps=int(env.take_action_cnt[env_idx]), policy_error=dict(failure))
    env.eval_result["policy_error"] = dict(failure)


def write_policy_error(env):
    from utils.save_file import save_json

    report = dict(env._scored_policy_failure, scored=True, score=env.eval_result["score"],
                  score_kind="interrupted_current_state")
    save_json(report, os.path.join(env.save_dir, "_policy_error.json"))
