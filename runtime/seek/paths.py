import os
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
THIRD_PARTY = CODE / "third_party"
ROBODOJO = THIRD_PARTY / "RoboDojo"
XPOLICYLAB = THIRD_PARTY / "XPolicyLab"
ISAACLAB = THIRD_PARTY / "IsaacLab"
CUROBO = THIRD_PARTY / "curobo"
CONTEXTS = ("policy", "sim")


def _root(explicit, variable, folder):
    return Path(explicit or os.environ.get(variable) or CODE / folder).expanduser().resolve()


def data_root(value=None):
    return _root(value, "SEEK_DATA_ROOT", "data")


def results_root(value=None):
    return _root(value, "SEEK_RESULTS_ROOT", "results")


def python_paths(context="policy"):
    if context not in CONTEXTS:
        raise ValueError(f"Unknown import context: {context}")
    packages = [XPOLICYLAB, ROBODOJO] if context == "policy" else [ROBODOJO, XPOLICYLAB]
    extensions = [entry for entry in sorted((ISAACLAB / "source").glob("*")) if entry.is_dir()]
    return [str(entry) for entry in (CODE, THIRD_PARTY, *packages, CUROBO, *extensions)]


def configure_imports(context="policy"):
    wanted = python_paths(context)
    sys.path[:] = wanted + [entry for entry in sys.path if entry not in wanted]
