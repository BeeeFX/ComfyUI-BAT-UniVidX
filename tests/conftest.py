"""Test setup: make the pack importable outside a running ComfyUI.

The runtime package only needs `folder_paths` from ComfyUI, so a stub is
enough for most tests. Tests that exercise the node classes need the real
thing: point COMFYUI_PATH at a ComfyUI checkout (its requirements
installed) and they run; otherwise they skip.

    COMFYUI_PATH=/path/to/ComfyUI python -m pytest tests
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_DIR = os.path.join(PACK_DIR, "UniVidX")

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")
if COMFYUI_PATH:
    sys.path.insert(0, COMFYUI_PATH)
sys.path.insert(0, PACK_DIR)

try:
    import folder_paths  # noqa: F401
    HAVE_COMFYUI = True
    # No GPU is assumed: without this, importing comfy.model_management
    # (which the pack does to honour Cancel) tries to initialise CUDA.
    from comfy.cli_args import args as _comfy_args
    _comfy_args.cpu = True
except ImportError:
    HAVE_COMFYUI = False
    stub = types.ModuleType("folder_paths")
    stub.models_dir = os.path.join(PACK_DIR, "_test_models")
    stub.get_filename_list = lambda _kind: []
    stub.get_full_path = lambda _kind, _name: None
    sys.modules["folder_paths"] = stub


def upstream_deps_available() -> bool:
    """Whether upstream UniVidX's own imports (peft, accelerate,
    modelscope, ftfy, transformers) resolve in this environment."""
    return all(
        importlib.util.find_spec(m) is not None
        for m in ("peft", "accelerate", "modelscope", "ftfy", "transformers")
    )


requires_upstream = pytest.mark.skipif(
    not upstream_deps_available(),
    reason="upstream UniVidX dependencies not installed",
)

requires_comfyui = pytest.mark.skipif(
    not HAVE_COMFYUI, reason="set COMFYUI_PATH to a ComfyUI checkout",
)


@pytest.fixture(scope="session")
def pack():
    """The node pack imported the way ComfyUI imports it: as a package
    from its directory (the directory name isn't a valid identifier)."""
    if not HAVE_COMFYUI:
        pytest.skip("set COMFYUI_PATH to a ComfyUI checkout")
    name = "bat_unividx_pack_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PACK_DIR, "__init__.py"),
        submodule_search_locations=[PACK_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
