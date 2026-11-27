import argparse
import asyncio

from seek.paths import configure_imports

configure_imports()

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from seek.policy.execution_policy import Model


PING_INTERVAL_S = 30
PING_TIMEOUT_S = 600
MAX_CHUNK = 64


def arguments():
    parser = argparse.ArgumentParser(description="Serve the SEEK policy")
    parser.add_argument("--host", default="127.0.0.1")
    for name, kind in (("port", int), ("model", str), ("task", str), ("seed", int)):
        parser.add_argument(f"--{name}", required=True, type=kind)
    parser.add_argument("--reasoning-effort", default="xhigh")
    return parser.parse_args()


def main():
    args = arguments()
    policy = Model(dict(policy_name="SEEK", bench_name="RoboDojo", action_type="ee", max_chunk=MAX_CHUNK,
                        model=args.model, task_name=args.task, seed=args.seed,
                        reasoning_effort=args.reasoning_effort))
    settings = PolicyServerConfig(host=args.host, port=args.port, ws_ping_interval_s=PING_INTERVAL_S,
                                  ws_ping_timeout_s=PING_TIMEOUT_S)
    try:
        asyncio.run(PolicyServer(policy, settings).serve_forever())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
