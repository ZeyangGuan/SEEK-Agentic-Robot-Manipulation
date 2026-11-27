import json
import subprocess

from seek.paths import CODE


LOCK = CODE / "configs/upstream.lock.json"


def _git(checkout, *arguments):
    return subprocess.check_output(["git", "-C", str(checkout), *arguments], text=True).strip()


def _verified(package):
    checkout = CODE / package["path"]
    if not checkout.is_dir():
        raise FileNotFoundError(f"{package['name']} is not installed at {checkout}; run setup/prepare.py")
    head = _git(checkout, "rev-parse", "HEAD")
    if head != package["commit"]:
        raise ValueError(f"{package['name']} is at {head}, but the lock pins {package['commit']}")
    if _git(checkout, "status", "--porcelain"):
        raise ValueError(f"{package['name']} has local changes: {checkout}")
    return {"name": package["name"], "path": package["path"], "commit": head}


def verify_packages(lock_path=None):
    lock = json.loads((lock_path or LOCK).read_text(encoding="utf-8"))
    return {"schema": lock["schema"], "packages": [_verified(package) for package in lock["packages"]]}
