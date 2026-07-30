"""Auto-clone upstream UniVidX into the wrapper directory on first use.

We deliberately avoid `git submodule` (the existing wrapper uses one and
it breaks on every re-clone where vendor/UniVidX already has any
content). Instead we do a plain shallow clone at a pinned commit, then
apply the two upstream-side patches the wrapper depends on:

  1. Create `src/pipelines/__init__.py` — upstream forgot this file,
     and without it Python can't resolve `src.pipelines.univid_alpha`
     etc. as proper subpackages.
  2. Soft-import `torchvision.io.write_video` in `src/trainers/
     unividx_trainer.py` — that function was removed in torchvision
     >= 0.22 (current ComfyUI envs). The trainer module isn't even
     loaded by our wrapper (we go straight to the pipeline classes),
     so this patch is defensive only — but cheap insurance against a
     future refactor that might pull the trainer module in.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")

# Pinned to the commit the existing UniVidX_ComfyUI wrapper uses. If
# upstream ships a fix we want, bump this here.
UNIVIDX_GIT_URL = "https://github.com/houyuanchen111/UniVidX.git"
UNIVIDX_PINNED_COMMIT = "382b9002757f4d5f04a90ec23b784dce11d56221"


def auto_clone(wrapper_dir: str) -> str:
    """Ensure `wrapper_dir/UniVidX/` is populated at the pinned commit.

    Returns the absolute path to the cloned directory. Idempotent —
    a second call is a no-op when the clone is already in place.
    Raises RuntimeError on git failures, with a message that names
    the exact `git clone` command the user can run manually.
    """
    target = os.path.join(wrapper_dir, "UniVidX")
    if os.path.isdir(os.path.join(target, ".git")):
        # Already cloned. Skip the network round-trip.
        return target

    logger.info(
        "UniVidX: ./UniVidX/ missing — cloning %s (one-time, ~25 MB) ...",
        UNIVIDX_GIT_URL,
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
            "git not found on PATH — can't auto-clone UniVidX. Install "
            "git and restart ComfyUI, or manually run "
            f"`git clone {UNIVIDX_GIT_URL} UniVidX && cd UniVidX && "
            f"git checkout {UNIVIDX_PINNED_COMMIT}` "
            "inside this custom node's directory."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"git clone of {UNIVIDX_GIT_URL} failed (exit {e.returncode}). "
            "Check network access and try again, or clone manually."
        ) from e

    _apply_upstream_patches(target)
    return target


def _apply_upstream_patches(upstream_dir: str) -> None:
    """Apply the two known-required upstream patches. Each is
    idempotent — running them on an already-patched tree is a no-op.
    """
    # Patch 1: missing src/pipelines/__init__.py.
    init_path = os.path.join(upstream_dir, "src", "pipelines", "__init__.py")
    if not os.path.exists(init_path):
        try:
            os.makedirs(os.path.dirname(init_path), exist_ok=True)
            open(init_path, "w").close()
            logger.info("UniVidX: created missing src/pipelines/__init__.py")
        except OSError as e:
            logger.warning("UniVidX: failed to create %s: %s", init_path, e)

    # Patch 2: soft-import torchvision.io.write_video. We only patch
    # the file if (a) it exists and (b) it still has the hard import.
    # Defensive: our wrapper avoids importing the trainer module
    # anyway, but a future refactor might pull it in transitively.
    trainer_path = os.path.join(
        upstream_dir, "src", "trainers", "unividx_trainer.py",
    )
    if os.path.exists(trainer_path):
        try:
            text = open(trainer_path, "r", encoding="utf-8").read()
        except OSError as e:
            logger.warning("UniVidX: failed to read %s: %s", trainer_path, e)
            return
        old = "from torchvision.io import write_video"
        if old in text and "BAT_UNIVIDX_PATCH" not in text:
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
                logger.info(
                    "UniVidX: patched torchvision.io.write_video soft-import"
                )
            except OSError as e:
                logger.warning(
                    "UniVidX: failed to patch %s: %s", trainer_path, e,
                )
