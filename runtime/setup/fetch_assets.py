import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
LOCK = CODE / "configs/data.lock.json"
REQUIRED_FOLDERS = ("Robots", "Object", "Room", "Material", "Sensor", "Eval_Layout")
MANIFEST = ".seek-assets.json"


class AssetCache:
    def __init__(self, folder):
        self.folder = folder
        self.assets = folder / "Assets"
        self.env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")

    def _command(self, arguments):
        return ["git", "-c", "http.version=HTTP/1.1", "-C", str(self.folder), *arguments]

    def git(self, *arguments):
        return subprocess.check_output(self._command(arguments), env=self.env, text=True).strip()

    def create(self, repo_id):
        if self.folder.exists():
            if not (self.folder / ".git").is_dir():
                raise RuntimeError(f"{self.folder} exists and is not a Git checkout; it was left untouched")
            return
        self.folder.mkdir(parents=True)
        self.git("init")
        host = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        self.git("remote", "add", "origin", os.environ.get("ROBODOJO_ASSET_REPO_URL", f"{host}/datasets/{repo_id}"))

    def checkout(self, revision):
        self.git("fetch", "--depth=1", "origin", revision)
        self.git("sparse-checkout", "init", "--cone")
        self.git("sparse-checkout", "set", "Assets")
        self.git("checkout", "--detach", "FETCH_HEAD")
        head = self.git("rev-parse", "HEAD")
        if head != revision:
            raise RuntimeError(f"The cache is at {head}, not at the locked revision {revision}")
        return head

    def download(self):
        self.git("lfs", "install", "--local")
        subprocess.run(self._command(["lfs", "pull", "--include=Assets/**", "--exclude="]), env=self.env, check=True)
        skipped = [name for name in self.git("ls-tree", "--name-only", "HEAD").splitlines() if name != "Assets"]
        patterns = ",".join(skipped + [f"{name}/**" for name in skipped])
        self.git("-c", f"lfs.fetchexclude={patterns}", "lfs", "fsck")
        for name in REQUIRED_FOLDERS:
            if not (self.assets / name).is_dir():
                raise RuntimeError(f"The downloaded tree has no Assets/{name}")

    def tree_sha256(self):
        return hashlib.sha256(self.git("ls-tree", "-r", "HEAD", "Assets").encode("utf-8")).hexdigest()


def materialize_robot_configs(data_root):
    written = []
    for template in sorted((data_root / "Assets/Robots").glob("*/curobo_tmp.yml")):
        text = template.read_text(encoding="utf-8").replace("${ASSETS_PATH}", str(data_root))
        target = template.with_name("curobo.yml")
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
        written.append(str(target.relative_to(data_root)))
    return written


def main():
    parser = argparse.ArgumentParser(description="Install the pinned task assets")
    parser.add_argument("--data-root", type=Path, default=CODE / "data")
    parser.add_argument("--cache", type=Path, required=True,
                        help="Folder for the Git LFS checkout; data-root/Assets links to its Assets/")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    cache = AssetCache(args.cache.expanduser().resolve())
    lock = json.loads(LOCK.read_text(encoding="utf-8"))["assets"]
    link = data_root / "Assets"
    data_root.mkdir(parents=True, exist_ok=True)
    occupied = link.exists() or link.is_symlink()
    if occupied and not (link.is_symlink() and link.resolve() == cache.assets.resolve()):
        raise RuntimeError(f"{link} already exists and does not point at this cache; it was left untouched")

    cache.create(lock["repo_id"])
    revision = cache.checkout(lock["revision"])
    cache.download()
    if not link.exists():
        link.symlink_to(cache.assets, target_is_directory=True)
    generated = materialize_robot_configs(data_root)
    manifest = {
        "schema": "seek.assets.install",
        "repo_id": lock["repo_id"],
        "revision": revision,
        "scope": "complete-tree",
        "include": lock["include"],
        "assets_tree_sha256": cache.tree_sha256(),
        "assets": str(link),
        "cache": str(cache.folder),
        "generated": generated,
    }
    text = json.dumps(manifest, indent=2)
    (data_root / MANIFEST).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
