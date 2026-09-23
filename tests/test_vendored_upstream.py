"""The vendored upstream tree: present, complete, importable, and in step
with the wrapper's mode table."""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from conftest import PACK_DIR, UPSTREAM_DIR, requires_upstream
from unividx_wrapper_runtime import auto_clone, modes


def test_vendored_tree_is_complete():
    assert auto_clone.missing_upstream_files(UPSTREAM_DIR) == []


def test_auto_clone_uses_vendored_tree_without_git(monkeypatch):
    # The vendored copy has no .git; that must not trigger a clone.
    assert not os.path.exists(os.path.join(UPSTREAM_DIR, ".git"))

    def no_network(*_a, **_k):
        raise AssertionError("auto_clone tried to run git on a complete tree")

    monkeypatch.setattr(subprocess, "run", no_network)
    assert auto_clone.auto_clone(PACK_DIR) == UPSTREAM_DIR


def test_auto_clone_refuses_incomplete_tree(tmp_path, monkeypatch):
    wrapper = tmp_path / "pack"
    shutil.copytree(os.path.join(UPSTREAM_DIR, "src"), wrapper / "UniVidX" / "src")
    # configs/ missing — the rsync-lost-files case.
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("cloned"))
    with pytest.raises(RuntimeError, match="incomplete.*wan2_1_14b_t2v_dit_config"):
        auto_clone.auto_clone(str(wrapper))


def test_auto_clone_clones_only_when_directory_absent(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **_k):
        calls.append(cmd)
        if cmd[1] == "clone":
            shutil.copytree(UPSTREAM_DIR, cmd[-1])

    monkeypatch.setattr(subprocess, "run", fake_run)
    target = auto_clone.auto_clone(str(tmp_path))
    assert [c[1] for c in calls] == ["clone", "-C"]
    assert auto_clone.missing_upstream_files(target) == []


def _upstream_training_modes(pipeline_file: str) -> set[str]:
    with open(os.path.join(UPSTREAM_DIR, "src", "pipelines", pipeline_file),
              encoding="utf-8") as fh:
        return set(re.findall(r'training_mode == "([A-Za-z0-9]+)"', fh.read()))


def test_modes_match_upstream_dispatch():
    assert set(modes.INTRINSIC_MODES) == _upstream_training_modes("univid_intrinsic.py")
    assert set(modes.ALPHA_MODES) == _upstream_training_modes("univid_alpha.py")


@requires_upstream
@pytest.mark.parametrize("variant", ["intrinsic", "alpha"])
def test_pipeline_imports_even_when_another_node_owns_src(variant, monkeypatch):
    import sys
    import types

    from unividx_wrapper_runtime import runtime

    foreign = types.ModuleType("src")
    foreign.__path__ = []
    monkeypatch.setitem(sys.modules, "src", foreign)
    monkeypatch.chdir(UPSTREAM_DIR)
    runtime._add_upstream_to_path(UPSTREAM_DIR)
    cls = runtime._import_pipeline_class(variant, UPSTREAM_DIR)
    assert cls.__name__ == {"intrinsic": "UniVidIntrinsic", "alpha": "UniVidAlpha"}[variant]
    assert sys.modules["src"] is foreign
