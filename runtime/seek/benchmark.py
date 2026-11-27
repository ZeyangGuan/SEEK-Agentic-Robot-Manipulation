import argparse
import importlib
import os
import sys
import traceback
import types
from pathlib import Path

from seek.paths import ROBODOJO, configure_imports

configure_imports("sim")

from isaaclab.app import AppLauncher


POLICY_NAME = "SEEK"
ROBOT = "arx_x5"
CONFIG_PARTS = ("sim", "scene", "camera", "robot")


class OfflineClient:
    def __init__(self, *args, **kwargs):
        pass

    def call(self, *, func_name, **kwargs):
        if func_name != "reset":
            raise RuntimeError(f"A capture-only run makes no {func_name} call")

    def close(self):
        pass


def arguments():
    parser = argparse.ArgumentParser(description="Run a simulator layout")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--layout", required=True, type=int)
    parser.add_argument("--layout-file", type=Path, help="Layout file to use instead of the benchmark layout")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--capture-only", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def selected_layout(args, assets):
    if args.layout_file is not None:
        layout = args.layout_file.expanduser().resolve()
    else:
        layout = assets / "Eval_Layout/RoboDojo" / ROBOT / str(args.seed) / f"{args.task}_{args.layout}.json"
    if not layout.is_file():
        raise FileNotFoundError(layout)
    return layout


def use_roots(data_root, results_root):
    assets = data_root / "Assets"
    results_root.mkdir(parents=True, exist_ok=True)
    os.environ.update(SEEK_DATA_ROOT=str(data_root), SEEK_RESULTS_ROOT=str(results_root))
    os.chdir(results_root)
    settings = importlib.import_module("env.global_configs")
    settings.ASSETS_PATH = str(assets)
    settings.OBJECTS_PATH = str(assets / "Object/RoboDojo")
    settings.ROBOTS_PATH = str(assets / "Robots")


def environment_config(args):
    from omegaconf import OmegaConf
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    registry = importlib.import_module("task.RoboDojo.task_registry")
    folder = ROBODOJO / "env_cfg"
    evaluation = dict(load_yaml(folder / f"{ROBOT}.yml"))
    evaluation.update(task_name=args.task, num_envs=1, device_id=0, eval_batch=False, policy_name=POLICY_NAME,
                      additional_info="action_type=ee", seed=args.seed, physx_monitor_enabled=False, eval_num=1)
    run_id = os.environ["ROBODOJO_RUN_ID"]
    deployment = dict(policy_name=POLICY_NAME, port=args.port, host=args.host, protocol="ws",
                      policy_server_url=f"ws://{args.host}:{args.port}", evaluation_id=run_id,
                      trial_id=f"{args.task}-{run_id}", action_case_id=f"{args.task}_case", repeat_index=None)
    parts = {part: load_yaml(folder / part / f"{evaluation['config'][part]}.yml") for part in CONFIG_PARTS}
    task_file = registry.task_config_path(ROBODOJO / "task/RoboDojo/config", args.task)
    config = OmegaConf.create(dict(parts, task_env=load_yaml(task_file), eval_cfg=evaluation,
                                   deploy_cfg=deployment))
    config, _ = process_config(process_randomization(config), task_name=args.task)
    OmegaConf.update(config, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(config, "eval_cfg.num_envs", 1, force_add=True)
    config.sim.seed = [0]
    OmegaConf.update(config, "camera.default_frequency", evaluation["observation"].get("collect_freq", 0),
                     force_add=True)
    OmegaConf.update(config, "eval_cfg.observation.vision.depth", False)
    return config


def pin_layout(env, index, layout):
    seeds = env.seed_manager
    seeds.seed_info = {index: {"scene_layout": str(layout)}}
    seeds.seed_list = [index]
    seeds.st_idx, seeds.ed_idx, seeds.idx = 0, 1, 0
    env.env_seeds = [index]


def run(args, layout, simulation_app, progress):
    progress("import benchmark runtime")
    evaluator = importlib.import_module("src.eval_client.eval_env")
    if args.capture_only:
        evaluator.WsModelClient = OfflineClient
    progress("assemble benchmark configuration")
    config = environment_config(args)
    progress("create benchmark environment")
    env = evaluator.create_eval_env(config, simulation_app)
    try:
        progress("bind exact layout")
        pin_layout(env, args.layout, layout)
        if args.capture_only:
            env.model_client = OfflineClient()
        progress("reset task scene")
        env.reset(seed=[args.layout])
        from seek.policy.observed_base_ik import install_observed_base_ik
        install_observed_base_ik(env.robot_manager)
        if args.capture_only:
            from seek.policy.camera_preflight import run as probe_cameras
            progress("capture camera probe")
            probe_cameras(env)
            return
        from seek.rollout import run_episode
        env.eval_one_episode = types.MethodType(lambda instance: run_episode(instance, instance.model_client), env)
        progress("run evaluation")
        env.run_eval()
    finally:
        for close in (lambda: env.model_client.close(), env.close):
            try:
                close()
            except Exception:
                pass


def main():
    args = arguments()
    data_root, results_root = (root.expanduser().resolve() for root in (args.data_root, args.results_root))
    layout = selected_layout(args, data_root / "Assets")
    use_roots(data_root, results_root)
    simulation_app = AppLauncher(args).app
    stage = ["start"]
    try:
        run(args, layout, simulation_app, lambda name: stage.__setitem__(0, name))
        return 0
    except BaseException:
        print(f"SEEK adapter failed during: {stage[0]}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
