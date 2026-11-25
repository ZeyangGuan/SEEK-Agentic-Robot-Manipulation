import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH_MESSAGE = "Apply the SEEK integration patch"
PATCH_DATE = "2026-11-20T00:00:00+0000"
PATCH_IDENTITY = {
    "GIT_AUTHOR_NAME": "SEEK",
    "GIT_AUTHOR_EMAIL": "seek@localhost",
    "GIT_AUTHOR_DATE": PATCH_DATE,
    "GIT_COMMITTER_NAME": "SEEK",
    "GIT_COMMITTER_EMAIL": "seek@localhost",
    "GIT_COMMITTER_DATE": PATCH_DATE,
}


def git(repo, *args, env=None):
    command = ["git", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", "-C", str(repo), *args]
    return subprocess.check_output(command, env=env, text=True).strip()


def apply_patch(repo, patch):
    git(repo, "apply", "--cached", str(patch))
    git(repo, "commit", "--quiet", "--no-verify", "-m", PATCH_MESSAGE, env=dict(os.environ, **PATCH_IDENTITY))
    git(repo, "reset", "--quiet", "--hard", "HEAD")


def install(item):
    destination = ROOT / item["path"]
    if not destination.exists():
        destination.mkdir(parents=True)
        git(destination, "init", "--quiet")
        git(destination, "remote", "add", "origin", item["url"])
        git(destination, "fetch", "--quiet", "--depth=1", "origin", item.get("base", item["commit"]))
        git(destination, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        if "patch" in item:
            apply_patch(destination, ROOT / item["patch"])
    if git(destination, "rev-parse", "HEAD") != item["commit"]:
        raise ValueError(f"Different checkout already present at {destination}")


def main():
    parser = argparse.ArgumentParser(description="Install the pinned third-party packages")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Install only the named packages")
    args = parser.parse_args()
    packages = json.loads((ROOT / "configs/upstream.lock.json").read_text())["packages"]
    unknown = set(args.only or ()) - {item["name"] for item in packages}
    if unknown:
        parser.error(f"Unknown package(s): {sorted(unknown)}")
    for item in packages:
        if args.only is None or item["name"] in args.only:
            install(item)


if __name__ == "__main__":
    main()
