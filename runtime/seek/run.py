import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


CODE = Path(__file__).absolute().parents[1]
GPU_SLOTS = 2
KIT_ARGS = (
    "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera "
    "--/app/updateOrder/checkForHydraRenderComplete=1000 "
    "--/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true "
)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from seek.config import (
    layout_path as configured_layout_path,
    load_profile,
    load_task,
    validate_evaluation_limits,
    render_official_task_prompt,
    task_name,
    task_source,
)
from seek.policy.provider_compat import validate_api_mode, validate_endpoint
from seek.paths import data_root as get_data_root, results_root as get_results_root, python_paths
from seek.upstream import verify_submodules


def absolute_path(value):
    return Path(value).expanduser().absolute()


def python_path(value):
    found = shutil.which(str(value))
    path = absolute_path(found or value)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is not executable: {path}")
    return path


def policy_socket_ready(port):
    with socket.create_connection(("127.0.0.1", port), timeout=1) as probe:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        probe.sendall((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                       "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                       f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = probe.recv(4096)
            if not chunk or len(response) > 8192:
                raise OSError("Policy did not return a WebSocket handshake")
            response += chunk
        if not response.startswith(b"HTTP/1.1 101 "):
            raise OSError("Policy did not accept the WebSocket handshake")

        probe.sendall(b"\x88\x80" + os.urandom(4))


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_preflight(data_root, layout_path):
    lock = json.loads((CODE / "configs/data.lock.json").read_text())["assets"]
    manifest_path = data_root / ".seek-assets.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run setup/fetch_assets.py"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != lock["repo_id"] or manifest.get("revision") != lock["revision"]:
        raise ValueError(f"Asset dataset revision mismatch: {manifest_path}")
    scope = manifest.get("scope", "complete-tree")
    if scope == "explicit-files":
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"Explicit asset manifest contains no files: {manifest_path}")
        checked = []
        for row in files:
            relative = Path(row.get("path", ""))
            if (not relative.parts or relative.parts[0] != "Assets" or relative.is_absolute()
                    or ".." in relative.parts or type(row.get("size")) is not int
                    or not isinstance(row.get("sha256"), str)):
                raise ValueError(f"Invalid explicit asset record: {row}")
            path = data_root / relative
            if not path.is_file() or path.stat().st_size != row["size"] or sha(path) != row["sha256"]:
                raise ValueError(f"Explicit asset differs from its manifest: {path}")
            checked.append(relative.as_posix())
        selected = layout_path.relative_to(data_root).as_posix()
        if selected not in checked:
            raise ValueError(f"Selected layout is outside the explicit asset manifest: {layout_path}")
    elif scope != "complete-tree" or manifest.get("include") != lock["include"]:
        raise ValueError(f"Unknown or incomplete asset manifest scope: {manifest_path}")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha(manifest_path),
        "repo_id": lock["repo_id"],
        "revision": lock["revision"],
        "scope": scope,
        "assets_tree_sha256": manifest.get("assets_tree_sha256"),
    }


def check(task, seed, layout, variant, action_horizon=None, *,
          data_root=None, sim_python=None, policy_python=None, profile=None):
    data_root = get_data_root(data_root)
    sim_python = python_path(sim_python or os.environ.get("SEEK_SIM_PYTHON", sys.executable))
    policy_python = python_path(policy_python or os.environ.get("SEEK_POLICY_PYTHON", sys.executable))
    if profile is None:
        profile = load_profile(os.environ.get("SEEK_PROFILE", CODE / "configs/protocols/seek.json"))
    validate_api_mode(profile.get("api_mode", "responses"))
    if "endpoint" in profile:
        validate_endpoint(profile.get("api_mode", "responses"), profile["endpoint"])
    c = load_task(task)

    actual = task_name(task, variant)

    paths = [
        task_source(task, variant),
        configured_layout_path(data_root, task, seed, layout, variant),
        sim_python,
        policy_python,
    ]
    config = json.loads((CODE / "configs/shared_harness.json").read_text())
    config.update(profile.get("harness_overrides", {}))

    config.update(c.get("harness_overrides", {}))
    validate_evaluation_limits(config, profile)
    if config.get("experience_horizon_from_budget"):
        config.update(execute_horizon=config['predict_horizon'], fixed_predict_horizon=False)
    document_version, document_source, document_prompt = render_official_task_prompt(c)
    config["official_task_documentation_version"] = document_version
    config["official_task_documentation_source"] = document_source
    config["official_task_prompt"] = document_prompt
    if action_horizon is not None:
        config.update(predict_horizon=action_horizon, execute_horizon=action_horizon)
        if config.get("experience_horizon_from_budget"):
            config.update(experience_horizon_from_budget=False)
    horizon = config["predict_horizon"]
    if not 1 <= horizon <= 64:
        raise ValueError("action horizon must be 1..64")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    return c, actual, config, paths[1]


def stop(p):
    if p is None or p.poll() is not None:
        return

    for sig, timeout in [
        (signal.SIGINT, 30),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 5),
    ]:
        try:
            os.killpg(p.pid, sig)
            p.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


def acquire_gpu_slot(gpu, requested_slot=None, *, results_root=None):
    results_root = get_results_root(results_root)
    slots = [requested_slot] if requested_slot is not None else range(GPU_SLOTS)

    lock_host = socket.gethostname()
    for slot in slots:
        lock = (results_root / f".{lock_host}.gpu{gpu}.slot{slot}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
        else:
            return slot, lock

    if requested_slot is None:
        raise RuntimeError(f"Both SEEK slots on GPU {gpu} are occupied")
    raise RuntimeError(f"SEEK slot {requested_slot} on GPU {gpu} is occupied")


def main():
    for private, transport in (("SEEK_MODEL_KEY", "ASTRA_API_KEY"), ("SEEK_MODEL_ENDPOINT", "ASTRA_ENDPOINT")):
        if private in os.environ:
            os.environ[transport] = os.environ[private]
    ap = argparse.ArgumentParser(description='Run a SEEK manipulation attempt')
    ap.add_argument("task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument("--layout-file", type=Path, help="Explicit experimental scene layout, recorded separately from the benchmark source")
    ap.add_argument(
        "--variant", choices=["standard", "random"], default="standard"
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu-slot", type=int, choices=range(GPU_SLOTS))
    ap.add_argument("--model", help="CLI override; otherwise ASTRA_MODEL, then profile model")
    ap.add_argument("--harness-profile", "--profile", dest="harness_profile",
                    default=os.environ.get("SEEK_PROFILE", str(CODE / "configs/protocols/seek.json")))
    ap.add_argument("--data-root", default=str(get_data_root()),
                    help="Directory containing Assets/ and runtime-data/")
    ap.add_argument("--results-root", default=str(get_results_root()))
    ap.add_argument("--sim-python", default=os.environ.get("SEEK_SIM_PYTHON", sys.executable))
    ap.add_argument("--policy-python", default=os.environ.get("SEEK_POLICY_PYTHON", sys.executable))
    ap.add_argument("--conda-exe", default=os.environ.get("SEEK_CONDA_EXE"),
                    help="Optional conda executable for activating each interpreter's environment")
    ap.add_argument("--port", type=int)
    ap.add_argument("--action-horizon", type=int, default=None, help="Override both prediction and execution steps")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    output_group = ap.add_mutually_exclusive_group()
    output_group.add_argument("--max-output-tokens", type=int, default=None)
    output_group.add_argument("--omit-max-output-tokens", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--capture-only",
        action="store_true",
        help="Render diagnostic only; no policy API or robot actions",
    )
    args = ap.parse_args()

    if min(args.seed, args.layout, args.gpu) < 0:
        ap.error("seed/layout/gpu must be nonnegative")

    data_root = absolute_path(args.data_root)
    results_root = absolute_path(args.results_root)
    sim_python = python_path(args.sim_python)
    policy_python = python_path(args.policy_python)
    profile_path = absolute_path(args.harness_profile)
    profile = load_profile(profile_path)
    args.model = args.model or os.environ.get("ASTRA_MODEL") or profile.get("model", "gpt-6-astra")
    api_mode = profile.get("api_mode", "responses")
    if profile.get("launch_blocked_reason") and not (args.dry_run or args.capture_only):
        raise RuntimeError(
            "Provider profile is launch-blocked: " + profile["launch_blocked_reason"])
    conda_exe = python_path(args.conda_exe) if args.conda_exe else None

    c, task, config, layout_path = check(
        args.task, args.seed, args.layout, args.variant, args.action_horizon,
        data_root=data_root, sim_python=sim_python, policy_python=policy_python, profile=profile,
    )
    asset_report = asset_preflight(data_root, layout_path)
    source_layout_path = layout_path
    if args.layout_file is not None:
        layout_path = absolute_path(args.layout_file)
        if not layout_path.is_file():
            raise FileNotFoundError(layout_path)

    if args.reasoning_effort is not None:
        config["reasoning_effort"] = args.reasoning_effort
    if args.max_output_tokens is not None:
        if not 256 <= args.max_output_tokens <= 65536:
            ap.error("max-output-tokens must be 256..65536")
        config["max_output_tokens"] = args.max_output_tokens
    if args.omit_max_output_tokens:
        config.pop("max_output_tokens", None)
    effort = config.get("reasoning_effort", "xhigh")
    horizon_tag = ("prednative_feedback1" if config.get("experience_horizon_from_budget")
                   else f"pred{config['predict_horizon']}_exec{config['execute_horizon']}")
    output_tag = "out" + str(config.get("max_output_tokens") or "default")
    task_doc_tag = f"taskdoc-{config.get('official_task_documentation_version', 'none')}"

    with socket.socket() as s:
        s.bind(("127.0.0.1", args.port or 0))
        port = s.getsockname()[1]

    runid = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_{args.task}_s{args.seed}_{args.variant}_l{args.layout}_reasoning-{effort}_{output_tag}_{horizon_tag}_{task_doc_tag}"
    )
    out = results_root / runid

    env = os.environ.copy()
    if profile.get("endpoint") is not None:
        env["ASTRA_ENDPOINT"] = profile["endpoint"]
    for key in [
        "ASTRA_CAMERA_PREFLIGHT",
        "ROBODOJO_FATAL_RESTART_COUNT",
        "ASTRA_HARNESS_CONFIG",
        "ASTRA_LOG_DIR",
    ]:
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
        CUDA_VISIBLE_DEVICES=str(args.gpu),
    )

    server = [
        str(policy_python),
        "-u",
        "-m",
        "seek.policy_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        args.model,
        "--task",
        task,
        "--seed",
        str(args.seed),
        "--reasoning-effort",
        effort,
    ]
    client = [
        str(sim_python),
        "-u",
        "-m",
        "seek.benchmark",
        "--task",
        task,
        "--seed",
        str(args.seed),
        "--layout",
        str(args.layout),
        "--data-root",
        str(data_root),
        "--results-root",
        str(results_root),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--enable_cameras",
        "--headless",
        "--kit_args",
        KIT_ARGS + env.get("SEEK_KIT_ARGS", ""),
    ]
    if args.capture_only:
        client.append("--capture-only")
        server = []
    if args.layout_file is not None:
        client.extend(["--layout-file", str(layout_path)])
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
        "data_root": str(data_root),
        "results_root": str(results_root),
        "upstream_lock_sha256": sha(CODE / "configs/upstream.lock.json") if (CODE / "configs/upstream.lock.json").is_file() else None,
        "sim_python": str(sim_python),
        "policy_python": str(policy_python),
        "conda_exe": str(conda_exe) if conda_exe else None,
        "harness_profile": profile,
        "harness_profile_file": str(profile_path),
        "harness_profile_sha256": sha(profile_path),
        "provider_api_mode": api_mode,
        "provider_endpoint": env.get("ASTRA_ENDPOINT"),
        "provider_stream": env["ASTRA_STREAM"] == "1",
        "provider_http_backend": "python-urllib",
        "source_manifest_sha256": sha(CODE.parent / "SOURCE_MANIFEST.json") if (CODE.parent / "SOURCE_MANIFEST.json").is_file() else None,
        "asset_preflight": asset_report,
    }
    if args.dry_run:
        record["upstream"] = verify_submodules()
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    record["upstream"] = verify_submodules()

    if args.capture_only:
        env.update(
            ASTRA_CAMERA_PREFLIGHT=str(out / "capture"),
        )

    if not args.capture_only:
        for k in ["ASTRA_API_KEY", "ASTRA_ENDPOINT"]:
            if not env.get(k):
                raise ValueError(f"Supply {k} externally")

    results_root.mkdir(parents=True, exist_ok=True)
    gpu_slot, lock = acquire_gpu_slot(args.gpu, args.gpu_slot, results_root=results_root)
    record["gpu_slot"] = gpu_slot

    out.mkdir()
    (out / "harness.json").write_text(json.dumps(config, indent=2))
    record["code_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (out / "code_diff.patch").write_text(
        subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=CODE,
            capture_output=True,
            text=True,
        ).stdout
    )
    (out / "run.json").write_text(json.dumps(record, indent=2))

    def wrapped(interpreter, cmd):
        if conda_exe is None:
            return cmd
        return [str(conda_exe), "run", "--no-capture-output", "--prefix",
                str(interpreter.parent.parent), *cmd]

    childenv = env.copy()
    childenv["PYTHONPATH"] = os.pathsep.join(python_paths("sim"))
    env["PATH"] = str(policy_python.parent) + os.pathsep + env.get("PATH", "")
    childenv["PATH"] = str(sim_python.parent) + os.pathsep + childenv.get("PATH", "")

    if os.environ.get("SEEK_SIM_LD_PRELOAD"):
        childenv["LD_PRELOAD"] = os.environ["SEEK_SIM_LD_PRELOAD"]
    icd = "/etc/vulkan/icd.d/nvidia_icd.json"
    if Path(icd).exists():
        childenv.update(VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd)

    p = q = None
    launch_started = time.monotonic()
    status = "failed"
    rc = 1
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    print(f"Run: {out}", flush=True)

    try:
        if not args.capture_only:
            with (out / "server.log").open("w") as log:
                p = subprocess.Popen(
                    wrapped(policy_python, server),
                    cwd=CODE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            deadline = time.monotonic() + 300
            while True:
                if p.poll() is not None:
                    raise RuntimeError(
                        f"Policy exited {p.returncode}; "
                        f"see {out}/server.log"
                    )
                try:
                    policy_socket_ready(port)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy startup exceeded 300 seconds")
                    time.sleep(1)

        with (out / "client.log").open("w") as log:
            q = subprocess.Popen(
                wrapped(sim_python, client),
                cwd=CODE,
                env=childenv,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        while q.poll() is None:
            if p is not None and p.poll() is not None:
                try:
                    q.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        "Policy server terminated; evaluator did not finalize within "
                        "60 seconds"
                    )
            time.sleep(1)

        rc = q.returncode
        record["client_returncode"] = rc
        status = "client_exited" if rc == 0 else "failed"
        if args.capture_only:
            probe = out / "capture/capture_probe.json"
            if probe.exists():
                rows = json.loads(probe.read_text())
                observed = [r for r in rows if r["route"] == "observation"]
                if len(observed) != 30 or any(
                    r["max"] == 0 or r["shape"] != [480, 640, 3]
                    for r in observed
                ):
                    raise RuntimeError("Camera probe invalid")
                rc = 0
                status = "capture_complete_no_api"
            else:
                raise RuntimeError("Capture-only run did not produce camera probe")
        elif rc == 0:
            result_path = (
                results_root
                / "eval_result/RoboDojo"
                / task
                / "SEEK/arx_x5"
                / f"{args.seed}_action_type=ee"
                / runid
                / "_result.json"
            )
            if not result_path.exists():
                raise RuntimeError(
                    "Client exited without scored result; see client.log"
                )
            result = json.loads(result_path.read_text())
            details = list(result.get("details", {}).values())
            if len(details) != 1 or details[0].get("layout_id") != args.layout:
                raise RuntimeError(
                    "Scored result does not match the selected single layout"
                )
            (out / "result.json").write_text(json.dumps(result, indent=2))
            status = "scored_with_error" if result.get("policy_error") else "scored"
    except KeyboardInterrupt:
        status = "interrupted"
        rc = 130
    except Exception:
        status = "failed"
        rc = 1
        raise
    finally:
        stop(q)
        stop(p)
        (out / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "returncode": rc,
                    "launch_wall_s": time.monotonic() - launch_started,
                    "note": "Process exit is not task success; read evaluator scores.",
                },
                indent=2,
            )
        )

    print(f"{status}: {out}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
