"""Locate upstream UniVidX, cloning it only as a last resort.

Upstream ships VENDORED in `<wrapper_dir>/UniVidX/` (see
UniVidX/VENDORED.md), so on a normal install this just verifies the tree
and returns its path — no network, no git.

The tree is recognised by the source files the loader actually imports,
NOT by a `.git` directory. The vendored copy deliberately has no `.git`,
and the old `.git` test was wrong both ways: it sent every fresh checkout
of this repo into `git clone` over a non-empty directory (which git
refuses), and it accepted an rsync'd `.git` whose files had been erased.

The clone path remains only for a checkout where `UniVidX/` is missing
entirely. It is a plain shallow clone at the pinned commit rather than a
git submodule (submodules break on every re-clone where the directory
already has content), followed by the two patches upstream needs:

  1. Create `src/pipelines/__init__.py` — upstream forgot this file,
     and without it Python can't resolve `src.pipelines.univid_alpha`
     etc. as proper subpackages.
  2. Soft-import `torchvision.io.write_video` in `src/trainers/
     unividx_trainer.py` — removed in torchvision >= 0.22. The wrapper
     never imports the trainer, so this is defensive only. It is not
     applied to the vendored copy, which stays a clean copy of upstream.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")

# Keep in step with UniVidX/VENDORED.md when re-vendoring.
UNIVIDX_GIT_URL = "https://github.com/houyuanchen111/UniVidX.git"
UNIVIDX_PINNED_COMMIT = "382b9002757f4d5f04a90ec23b784dce11d56221"

# One file per upstream package the loader imports, plus the DiT config
# `from_pretrained` reads by relative path. If any is absent the pipeline
# import fails with a ModuleNotFoundError that points nowhere useful, so
# check up front and name the missing file instead.
_REQUIRED_UPSTREAM_FILES = (
    os.path.join("src", "pipelines", "univid_intrinsic.py"),
    os.path.join("src", "pipelines", "univid_alpha.py"),
    os.path.join("src", "models", "model_manager.py"),
    os.path.join("src", "models", "wan_video_dit_intrinsic.py"),
    os.path.join("src", "models", "wan_video_dit_alpha.py"),
    os.path.join("src", "models", "wan_video_vae.py"),
    os.path.join("src", "models", "wan_video_text_encoder.py"),
    os.path.join("configs", "wan2_1_14b_t2v_dit_config.json"),
)


def missing_upstream_files(upstream_dir: str) -> list[str]:
    """Relative paths from `_REQUIRED_UPSTREAM_FILES` absent under
    `upstream_dir`. Empty means the tree is usable."""
    return [
        rel for rel in _REQUIRED_UPSTREAM_FILES
        if not os.path.isfile(os.path.join(upstream_dir, rel))
    ]


def auto_clone(wrapper_dir: str) -> str:
    """Return the absolute path of a usable upstream UniVidX tree.

    Normal case: the vendored copy is complete, so this is a handful of
    stat calls. Idempotent. Raises RuntimeError, naming the fix, when the
    directory exists but is incomplete or when a needed clone fails.
    """
    target = os.path.join(wrapper_dir, "UniVidX")
    missing = missing_upstream_files(target)
    if not missing:
        _ensure_pipelines_init(target)
        return target

    if os.path.isdir(target) and os.listdir(target):
        # Never clone over (or delete) a populated directory: git refuses
        # the former, and the latter could destroy someone's edits.
        raise RuntimeError(
            f"UniVidX: the upstream copy at {target!r} is incomplete — "
            f"missing {', '.join(missing)}. It ships with this node pack, so "
            "this usually means a partial checkout or a deploy that didn't "
            "copy every file. Restore it with `git checkout -- UniVidX` in "
            "the pack directory (or re-clone the pack), then restart ComfyUI."
        )

    logger.info(
        "UniVidX: ./UniVidX/ missing — cloning %s at %s (one-time) ...",
        UNIVIDX_GIT_URL, UNIVIDX_PINNED_COMMIT[:8],
    )
    os.makedirs(wrapper_dir, exist_ok=True)
    try:
        # `--no-checkout` + manual checkout of the pinned commit means
        # we don't pull down branches we don't care about.
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout",
             UNIVIDX_GIT_URL, target],
            check=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        subprocess.run(
            ["git", "-C", target, "checkout", UNIVIDX_PINNED_COMMIT],
            check=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "git not found on PATH — can't fetch UniVidX. Restore the "
            "vendored copy with `git checkout -- UniVidX` in this node "
            "pack's directory, or install git and restart ComfyUI."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"git clone of {UNIVIDX_GIT_URL} failed (exit {e.returncode}). "
            "Check network access, or restore the vendored copy with "
            "`git checkout -- UniVidX` in this node pack's directory."
        ) from e

    _ensure_pipelines_init(target)
    _patch_trainer_write_video_import(target)
    still_missing = missing_upstream_files(target)
    if still_missing:
        raise RuntimeError(
            f"UniVidX: cloned {UNIVIDX_PINNED_COMMIT[:8]} but it lacks "
            f"{', '.join(still_missing)} — the pinned commit no longer "
            "matches what this wrapper expects."
        )
    return target


def _ensure_pipelines_init(upstream_dir: str) -> None:
    """Patch 1: create the `src/pipelines/__init__.py` upstream omits.
    Already present in the vendored copy; a no-op there."""
    init_path = os.path.join(upstream_dir, "src", "pipelines", "__init__.py")
    if os.path.exists(init_path):
        return
    try:
        os.makedirs(os.path.dirname(init_path), exist_ok=True)
        open(init_path, "w").close()
        logger.info("UniVidX: created missing src/pipelines/__init__.py")
    except OSError as e:
        logger.warning("UniVidX: failed to create %s: %s", init_path, e)


def _patch_trainer_write_video_import(upstream_dir: str) -> None:
    """Patch 2: soft-import `torchvision.io.write_video` in the trainer.
    Idempotent, and only ever applied to a fresh clone. Defensive: the
    wrapper avoids importing the trainer module, but a future refactor
    might pull it in transitively."""
    trainer_path = os.path.join(
        upstream_dir, "src", "trainers", "unividx_trainer.py",
    )
    if not os.path.exists(trainer_path):
        return
    try:
        with open(trainer_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        logger.warning("UniVidX: failed to read %s: %s", trainer_path, e)
        return
    old = "from torchvision.io import write_video"
    if old not in text or "BAT_UNIVIDX_PATCH" in text:
        return
    new = (
        "# BAT_UNIVIDX_PATCH: torchvision.io.write_video was removed\n"
        "# in torchvision >= 0.22. Soft-import so this module loads.\n"
        "try:\n"
        "    from torchvision.io import write_video\n"
        "except ImportError:\n"
        "    write_video = None  # type: ignore"
    )
    try:
        with open(trainer_path, "w", encoding="utf-8") as fh:
            fh.write(text.replace(old, new, 1))
        logger.info("UniVidX: patched torchvision.io.write_video soft-import")
    except OSError as e:
        logger.warning("UniVidX: failed to patch %s: %s", trainer_path, e)
