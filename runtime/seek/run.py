import argparse
import datetime
import fcntl
import hashlib
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from base64 import b64encode
from pathlib import Path


CODE = Path(__file__).absolute().parents[1]
GPU_SLOTS = 2
KIT_ARGS = (
    "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera "
    "--/app/updateOrder/checkForHydraRenderComplete=1000 "
    "--/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true "
)
POLICY_STARTUP_S = 300
EVALUATOR_GRACE_S = 60
POLL_S = 1
PROBE_FRAMES = 30
PROBE_SHAPE = [480, 640, 3]
VULKAN_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
CREDENTIALS = (("SEEK_MODEL_KEY", "ASTRA_API_KEY"), ("SEEK_MODEL_ENDPOINT", "ASTRA_ENDPOINT"))
INHERITED_RUN_STATE = ("ASTRA_CAMERA_PREFLIGHT", "ROBODOJO_FATAL_RESTART_COUNT",
                       "ASTRA_HARNESS_CONFIG", "ASTRA_LOG_DIR")
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from seek.config import (
    layout_path as benchmark_layout,
    load_profile,
    load_task,
    official_task_prompt,
    task_name,
    task_source,
    validate_evaluation_limits,
)
from seek.policy.provider_compat import validate_api_mode, validate_endpoint
from seek.paths import data_root as default_data_root, results_root as default_results_root, python_paths
from seek.upstream import verify_packages


def absolute_path(value):
    return Path(value).expanduser().absolute()


def executable(value):
    path = absolute_path(shutil.which(str(value)) or value)
    if not (path.is_file() and os.access(path, os.X_OK)):
        raise FileNotFoundError(f"No executable at {path}")
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def optional_sha(path):
    return sha(path) if path.is_file() else None


def policy_ready(port):
    upgrade = {"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13",
               "Sec-WebSocket-Key": b64encode(os.urandom(16)).decode("ascii")}
    link = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        link.request("GET", "/", headers=upgrade)
        status = link.getresponse().status
        if status != http.client.SWITCHING_PROTOCOLS:
            raise OSError(f"The policy server answered the WebSocket probe with HTTP {status}")
        link.sock.sendall(b"\x88\x80" + os.urandom(4))
    except http.client.HTTPException as error:
        raise OSError(f"The policy server did not complete the WebSocket probe: {error}") from error
    finally:
        link.close()


def _listed_assets(manifest, manifest_path, data_root):
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{manifest_path} lists no files")
    listed = set()
    for row in rows:
        relative = Path(row.get("path", ""))
        well_formed = (relative.parts and relative.parts[0] == "Assets" and not relative.is_absolute()
                       and ".." not in relative.parts and type(row.get("size")) is int
                       and isinstance(row.get("sha256"), str))
        if not well_formed:
            raise ValueError(f"Malformed asset record in {manifest_path}: {row}")
        path = data_root / relative
        if not path.is_file() or path.stat().st_size != row["size"] or sha(path) != row["sha256"]:
            raise ValueError(f"{path} does not match the asset manifest")
        listed.add(relative.as_posix())
    return listed


def asset_preflight(data_root, layout_path):
    lock = json.loads((CODE / "configs/data.lock.json").read_text())["assets"]
    manifest_path = data_root / ".seek-assets.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; run setup/fetch_assets.py")
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(key) != lock[key] for key in ("repo_id", "revision")):
        raise ValueError(f"{manifest_path} describes another asset revision than configs/data.lock.json")
    scope = manifest.get("scope", "complete-tree")
    if scope == "explicit-files":
        listed = _listed_assets(manifest, manifest_path, data_root)
        if layout_path.relative_to(data_root).as_posix() not in listed:
            raise ValueError(f"{manifest_path} does not list the selected layout: {layout_path}")
    elif scope != "complete-tree" or manifest.get("include") != lock["include"]:
        raise ValueError(f"{manifest_path} does not cover the locked asset tree")
    return dict(manifest=str(manifest_path), manifest_sha256=sha(manifest_path), repo_id=lock["repo_id"],
                revision=lock["revision"], scope=scope, assets_tree_sha256=manifest.get("assets_tree_sha256"))


def harness_config(task_config, profile, action_horizon):
    config = json.loads((CODE / "configs/shared_harness.json").read_text())
    config.update(profile.get("harness_overrides", {}))
    config.update(task_config.get("harness_overrides", {}))
    validate_evaluation_limits(config, profile)
    if config.get("experience_horizon_from_budget"):
        config.update(execute_horizon=config['predict_horizon'], fixed_predict_horizon=False)
    version, source, prompt = official_task_prompt(task_config)
    config["official_task_documentation_version"] = version
    config["official_task_documentation_source"] = source
    config["official_task_prompt"] = prompt
    if action_horizon is not None:
        config.update(predict_horizon=action_horizon, execute_horizon=action_horizon)
        if config.get("experience_horizon_from_budget"):
            config.update(experience_horizon_from_budget=False)
    if not 1 <= config["predict_horizon"] <= 64:
        raise ValueError("The action horizon is between 1 and 64 steps")
    return config


def check(task, seed, layout, variant, action_horizon=None, *,
          data_root=None, sim_python=None, policy_python=None, profile=None):
    data_root = default_data_root(data_root)
    sim_python = executable(sim_python or os.environ.get("SEEK_SIM_PYTHON", sys.executable))
    policy_python = executable(policy_python or os.environ.get("SEEK_POLICY_PYTHON", sys.executable))
    if profile is None:
        profile = load_profile(os.environ.get("SEEK_PROFILE", CODE / "configs/protocols/seek.json"))
    api_mode = profile.get("api_mode", "responses")
    validate_api_mode(api_mode)
    if "endpoint" in profile:
        validate_endpoint(api_mode, profile["endpoint"])
    task_config = load_task(task)
    actual = task_name(task, variant)
    required = [
        task_source(task, variant),
        benchmark_layout(data_root, task, seed, layout, variant),
        sim_python,
        policy_python,
    ]
    config = harness_config(task_config, profile, action_horizon)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    return task_config, actual, config, required[1]


def stop(process):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


def claim_gpu_slot(gpu, requested=None, *, results_root):
    host = socket.gethostname()
    for slot in range(GPU_SLOTS) if requested is None else [requested]:
        handle = (results_root / f".{host}.gpu{gpu}.slot{slot}.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        return slot, handle
    if requested is None:
        raise RuntimeError(f"Both SEEK slots on GPU {gpu} are occupied")
    raise RuntimeError(f"SEEK slot {requested} on GPU {gpu} is occupied")


def argument_parser():
    parser = argparse.ArgumentParser(description='Run a SEEK manipulation attempt')
    parser.add_argument("task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--layout-file", type=Path, help="Explicit experimental scene layout, recorded separately from the benchmark source")
    parser.add_argument("--variant", choices=["standard", "random"], default="standard")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-slot", type=int, choices=range(GPU_SLOTS))
    parser.add_argument("--model", help="Model name; defaults to ASTRA_MODEL, then the profile")
    parser.add_argument("--harness-profile", "--profile", dest="harness_profile",
                        default=os.environ.get("SEEK_PROFILE", str(CODE / "configs/protocols/seek.json")))
    parser.add_argument("--data-root", default=str(default_data_root()),
                        help="Folder that holds the benchmark Assets/")
    parser.add_argument("--results-root", default=str(default_results_root()))
    parser.add_argument("--sim-python", default=os.environ.get("SEEK_SIM_PYTHON", sys.executable))
    parser.add_argument("--policy-python", default=os.environ.get("SEEK_POLICY_PYTHON", sys.executable))
    parser.add_argument("--conda-exe", default=os.environ.get("SEEK_CONDA_EXE"),
                        help="Start both interpreters through this conda executable")
    parser.add_argument("--port", type=int)
    parser.add_argument("--action-horizon", type=int, default=None, help="Set the predicted and the executed steps per request")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--max-output-tokens", type=int, default=None)
    output.add_argument("--omit-max-output-tokens", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--capture-only", action="store_true",
                        help="Render the cameras only; no model requests, no robot actions")
    return parser


def free_port(requested):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", requested or 0))
        return probe.getsockname()[1]


def run_id(args, config, effort):
    if config.get("experience_horizon_from_budget"):
        horizon = "prednative_feedback1"
    else:
        horizon = f"pred{config['predict_horizon']}_exec{config['execute_horizon']}"
    output = "out" + str(config.get("max_output_tokens") or "default")
    document = f"taskdoc-{config.get('official_task_documentation_version', 'none')}"
    return (datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + f"_{args.task}_s{args.seed}_{args.variant}_l{args.layout}"
            + f"_reasoning-{effort}_{output}_{horizon}_{document}")


def policy_environment(profile, out, runid, data_root, results_root, effort, api_mode, gpu):
    env = os.environ.copy()
    if profile.get("endpoint") is not None:
        env["ASTRA_ENDPOINT"] = profile["endpoint"]
    for key in INHERITED_RUN_STATE:
        env.pop(key, None)
    env.update(
        SEEK_DATA_ROOT=str(data_root),
        SEEK_RESULTS_ROOT=str(results_root),
        ROBODOJO_RUN_ID=runid,
        ASTRA_HARNESS_CONFIG=str(out / "harness.json"),
        ASTRA_LOG_DIR=str(out / "model"),
        ASTRA_REASONING_EFFORT=effort,
        ASTRA_API_MODE=api_mode,
        ASTRA_TRACKING_GUARD="1",
        ASTRA_VALIDATE_RGB="1",
        ASTRA_NETWORK_MODE="direct",
        ASTRA_STREAM="1",
        PYTHONPATH=os.pathsep.join(python_paths("policy")),
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        OMNI_KIT_ALLOW_ROOT="1",
        CUDA_VISIBLE_DEVICES=str(gpu),
    )
    return env


def simulator_environment(env, sim_python):
    child = env.copy()
    child["PYTHONPATH"] = os.pathsep.join(python_paths("sim"))
    child["PATH"] = str(sim_python.parent) + os.pathsep + child.get("PATH", "")
    if os.environ.get("SEEK_SIM_LD_PRELOAD"):
        child["LD_PRELOAD"] = os.environ["SEEK_SIM_LD_PRELOAD"]
    if Path(VULKAN_ICD).exists():
        child.update(VK_DRIVER_FILES=VULKAN_ICD, VK_ICD_FILENAMES=VULKAN_ICD)
    return child


def commands(args, paths, task, port, effort, layout_path, env):
    server = [
        str(paths["policy_python"]), "-u", "-m", "seek.policy_server",
        "--host", "127.0.0.1", "--port", str(port),
        "--model", args.model, "--task", task, "--seed", str(args.seed),
        "--reasoning-effort", effort,
    ]
    client = [
        str(paths["sim_python"]), "-u", "-m", "seek.benchmark",
        "--task", task, "--seed", str(args.seed), "--layout", str(args.layout),
        "--data-root", str(paths["data_root"]), "--results-root", str(paths["results_root"]),
        "--port", str(port), "--host", "127.0.0.1",
        "--enable_cameras", "--headless",
        "--kit_args", KIT_ARGS + env.get("SEEK_KIT_ARGS", ""),
    ]
    if args.capture_only:
        client.append("--capture-only")
        server = []
    if args.layout_file is not None:
        client.extend(["--layout-file", str(layout_path)])
    return server, client


def git_output(*arguments):
    return subprocess.run(["git", *arguments], cwd=CODE, capture_output=True, text=True).stdout


def wait_for_policy(policy, port, out):
    deadline = time.monotonic() + POLICY_STARTUP_S
    while True:
        if policy.poll() is not None:
            raise RuntimeError(f"Policy exited {policy.returncode}; see {out}/server.log")
        try:
            return policy_ready(port)
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"The policy server was not ready after {POLICY_STARTUP_S} seconds")
            time.sleep(POLL_S)


def wait_for_evaluator(evaluator, policy):
    while evaluator.poll() is None:
        if policy is not None and policy.poll() is not None:
            try:
                evaluator.wait(timeout=EVALUATOR_GRACE_S)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "Policy server terminated; evaluator did not finalize within 60 seconds")
        time.sleep(POLL_S)
    return evaluator.returncode


def check_camera_probe(out):
    probe = out / "capture/capture_probe.json"
    if not probe.exists():
        raise RuntimeError(f"The capture-only run wrote no {probe.name}")
    observed = [row for row in json.loads(probe.read_text()) if row["route"] == "observation"]
    if len(observed) != PROBE_FRAMES or any(
            row["max"] == 0 or row["shape"] != PROBE_SHAPE for row in observed):
        raise RuntimeError("The camera probe holds blank, missing or wrongly sized frames")


def collect_score(args, results_root, task, runid, out):
    result_path = (results_root / "eval_result/RoboDojo" / task / "SEEK/arx_x5"
                   / f"{args.seed}_action_type=ee" / runid / "_result.json")
    if not result_path.exists():
        raise RuntimeError("Client exited without scored result; see client.log")
    result = json.loads(result_path.read_text())
    details = list(result.get("details", {}).values())
    if len(details) != 1 or details[0].get("layout_id") != args.layout:
        raise RuntimeError("Scored result does not match the selected single layout")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return "scored_with_error" if result.get("policy_error") else "scored"


def start(interpreter, command, conda_exe, log, env):
    if conda_exe is not None:
        command = [str(conda_exe), "run", "--no-capture-output", "--prefix",
                   str(interpreter.parent.parent), *command]
    return subprocess.Popen(command, cwd=CODE, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)


def raise_interrupt(*_):
    raise KeyboardInterrupt


def accept_interrupts():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, raise_interrupt)


def supervise(args, plan):
    out, record, env, paths = plan["out"], plan["record"], plan["env"], plan["paths"]
    child_env = simulator_environment(env, paths["sim_python"])
    env["PATH"] = str(paths["policy_python"].parent) + os.pathsep + env.get("PATH", "")
    policy = evaluator = None
    started = time.monotonic()
    status, rc = "failed", 1
    accept_interrupts()
    print(f"Run: {out}", flush=True)
    try:
        if not args.capture_only:
            with (out / "server.log").open("w") as log:
                policy = start(paths["policy_python"], plan["server"], paths["conda_exe"], log, env)
            wait_for_policy(policy, plan["port"], out)
        with (out / "client.log").open("w") as log:
            evaluator = start(paths["sim_python"], plan["client"], paths["conda_exe"], log, child_env)
        rc = wait_for_evaluator(evaluator, policy)
        record["client_returncode"] = rc
        status = "client_exited" if rc == 0 else "failed"
        if args.capture_only:
            check_camera_probe(out)
            rc, status = 0, "capture_complete_no_api"
        elif rc == 0:
            status = collect_score(args, paths["results_root"], plan["task"], plan["runid"], out)
    except KeyboardInterrupt:
        status, rc = "interrupted", 130
    except Exception:
        status, rc = "failed", 1
        raise
    finally:
        stop(evaluator)
        stop(policy)
        (out / "status.json").write_text(json.dumps(
            dict(status=status, returncode=rc, launch_wall_s=time.monotonic() - started), indent=2))
    print(f"{status}: {out}", flush=True)
    return rc


def export_model_credentials():
    for private, transport in CREDENTIALS:
        if private in os.environ:
            os.environ[transport] = os.environ[private]


def plan_attempt(parser, args):
    paths = dict(
        data_root=absolute_path(args.data_root),
        results_root=absolute_path(args.results_root),
        sim_python=executable(args.sim_python),
        policy_python=executable(args.policy_python),
    )
    profile_path = absolute_path(args.harness_profile)
    profile = load_profile(profile_path)
    args.model = args.model or os.environ.get("ASTRA_MODEL") or profile.get("model", "gpt-6-astra")
    api_mode = profile.get("api_mode", "responses")
    if profile.get("launch_blocked_reason") and not (args.dry_run or args.capture_only):
        raise RuntimeError("Provider profile is launch-blocked: " + profile["launch_blocked_reason"])
    paths["conda_exe"] = executable(args.conda_exe) if args.conda_exe else None

    _, task, config, layout_path = check(
        args.task, args.seed, args.layout, args.variant, args.action_horizon,
        data_root=paths["data_root"], sim_python=paths["sim_python"],
        policy_python=paths["policy_python"], profile=profile,
    )
    asset_report = asset_preflight(paths["data_root"], layout_path)
    source_layout_path = layout_path
    if args.layout_file is not None:
        layout_path = absolute_path(args.layout_file)
        if not layout_path.is_file():
            raise FileNotFoundError(layout_path)

    if args.reasoning_effort is not None:
        config["reasoning_effort"] = args.reasoning_effort
    if args.max_output_tokens is not None:
        if not 256 <= args.max_output_tokens <= 65536:
            parser.error("max-output-tokens must be 256..65536")
        config["max_output_tokens"] = args.max_output_tokens
    if args.omit_max_output_tokens:
        config.pop("max_output_tokens", None)
    effort = config.get("reasoning_effort", "xhigh")

    port = free_port(args.port)
    runid = run_id(args, config, effort)
    out = paths["results_root"] / runid
    env = policy_environment(profile, out, runid, paths["data_root"], paths["results_root"],
                             effort, api_mode, args.gpu)
    server, client = commands(args, paths, task, port, effort, layout_path, env)
    record = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "benchmark_task": task,
        "layout_file": str(layout_path),
        "source_layout_file": str(source_layout_path),
        "experimental_layout": args.layout_file is not None,
        "layout_sha256": sha(layout_path),
        "harness": config,
        "server_command": server,
        "client_command": client,
        "output": str(out),
        "data_root": str(paths["data_root"]),
        "results_root": str(paths["results_root"]),
        "upstream_lock_sha256": optional_sha(CODE / "configs/upstream.lock.json"),
        "sim_python": str(paths["sim_python"]),
        "policy_python": str(paths["policy_python"]),
        "conda_exe": str(paths["conda_exe"]) if paths["conda_exe"] else None,
        "harness_profile": profile,
        "harness_profile_file": str(profile_path),
        "harness_profile_sha256": sha(profile_path),
        "provider_api_mode": api_mode,
        "provider_endpoint": env.get("ASTRA_ENDPOINT"),
        "provider_stream": env["ASTRA_STREAM"] == "1",
        "provider_http_backend": "python-urllib",
        "source_manifest_sha256": optional_sha(CODE.parent / "SOURCE_MANIFEST.json"),
        "asset_preflight": asset_report,
    }
    return dict(task=task, config=config, port=port, runid=runid, out=out, env=env,
                server=server, client=client, record=record, paths=paths)


def create_run_directory(args, plan):
    out, record, env = plan["out"], plan["record"], plan["env"]
    if args.capture_only:
        env.update(ASTRA_CAMERA_PREFLIGHT=str(out / "capture"))
    else:
        for key in ("ASTRA_API_KEY", "ASTRA_ENDPOINT"):
            if not env.get(key):
                raise ValueError(f"Supply {key} externally")
    results_root = plan["paths"]["results_root"]
    results_root.mkdir(parents=True, exist_ok=True)
    record["gpu_slot"], plan["gpu_lock"] = claim_gpu_slot(
        args.gpu, args.gpu_slot, results_root=default_results_root(results_root))
    out.mkdir()
    (out / "harness.json").write_text(json.dumps(plan["config"], indent=2))
    record["code_commit"] = git_output("rev-parse", "HEAD").strip()
    (out / "code_diff.patch").write_text(git_output("diff", "HEAD"))
    (out / "run.json").write_text(json.dumps(record, indent=2))


def main():
    export_model_credentials()
    parser = argument_parser()
    args = parser.parse_args()
    if min(args.seed, args.layout, args.gpu) < 0:
        parser.error("seed/layout/gpu must be nonnegative")
    plan = plan_attempt(parser, args)
    plan["record"]["upstream"] = verify_packages()
    if args.dry_run:
        print(json.dumps(plan["record"], ensure_ascii=False, indent=2))
        return 0
    create_run_directory(args, plan)
    return supervise(args, plan)


if __name__ == "__main__":
    sys.exit(main())
