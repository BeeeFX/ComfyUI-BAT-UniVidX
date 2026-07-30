"""Weight resolution + auto-download.

Two large model dependencies need to be on disk before the loader can
instantiate the pipeline:

  • Wan2.1-T2V-14B (the diffusion backbone, ~57 GB in HF layout)
  • UniVidX intrinsic / alpha checkpoint (~0.8 GB each)

For each, the wrapper tries hard to reuse what's already on disk
before triggering a fresh download. If you have the right Wan2.1
checkpoint installed under another ComfyUI custom node (e.g. kijai's
WanVideoWrapper), the wrapper will find and reuse it — zero extra
disk.

Compatibility note: this auto-detect ONLY recognises Wan2.1-T2V-14B
in the original HuggingFace directory layout (a `Wan2.1-T2V-14B/`
folder containing `models_t5_*` + `Wan2.1_VAE.pth` + DiT shards).
Wan2.1-VACE-14B and Wan2.1-I2V-14B are explicitly rejected — they
share the DiT shape but produce garbage when loaded into UniVidX.
Single-file Kijai-style consolidations are NOT supported in this
release; flagged as follow-up.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")

WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-14B"
UNIVIDX_REPO_ID = "houyuanchen/UniVidX"

# Sub-folders under ComfyUI/models/ we'll scan when looking for an
# existing Wan2.1 install. Order matters — first match wins.
_WAN_SEARCH_SUBDIRS = (
    "unividx",
    "diffusion_models",
    "checkpoints",
    "unet",
    "wan",
    "Wan2.1",
)

# Files that have to ALL be present (or all of one alternate) for us
# to consider a directory a valid Wan2.1-T2V-14B install. The HF repo
# ships the T5 encoder + VAE under specific filenames, so we anchor
# on those.
_WAN_REQUIRED_FILES = (
    # T5 text encoder — file name is models_t5_umt5-xxl-enc-bf16.pth or
    # a config.json + variant.
    ("models_t5_umt5-xxl-enc-bf16.pth", "models_t5_umt5-xxl-enc-fp32.pth"),
    # VAE
    ("Wan2.1_VAE.pth",),
)

# Patterns we reject as positive matches even when the directory looks
# Wan-shaped — these are the wrong variants.
_WAN_REJECT_NAME_PARTS = ("VACE", "vace", "I2V", "i2v", "TI2V", "ti2v")


def _looks_like_wan_t2v(dir_path: str) -> bool:
    """True iff `dir_path` looks like a Wan2.1-T2V-14B install."""
    if not os.path.isdir(dir_path):
        return False
    name = os.path.basename(dir_path)
    # Reject the other Wan2.1 variants by name.
    for bad in _WAN_REJECT_NAME_PARTS:
        if bad in name:
            return False
    # Check that at least one variant of each required-file group exists.
    for variants in _WAN_REQUIRED_FILES:
        if not any(os.path.exists(os.path.join(dir_path, v)) for v in variants):
            return False
    return True


def resolve_wan21_t2v_14b_path(
    comfyui_models_dir: str,
    extra_paths: Optional[list[str]] = None,
) -> Optional[str]:
    """Walk ComfyUI's models/ tree looking for a Wan2.1-T2V-14B
    install. Returns the absolute path of the first valid match, or
    None if nothing suitable is found.
    """
    candidates: list[str] = []
    for sub in _WAN_SEARCH_SUBDIRS:
        parent = os.path.join(comfyui_models_dir, sub)
        if not os.path.isdir(parent):
            continue
        # Direct hit: parent IS the Wan dir.
        if _looks_like_wan_t2v(parent):
            return parent
        # Walk one level deep for a "Wan2.1-T2V-14B/" child.
        try:
            entries = os.listdir(parent)
        except OSError:
            continue
        for entry in entries:
            child = os.path.join(parent, entry)
            if _looks_like_wan_t2v(child):
                candidates.append(child)
    # User-supplied extra paths.
    for p in (extra_paths or []):
        if _looks_like_wan_t2v(p):
            candidates.append(p)

    if candidates:
        # Prefer the shortest path — heuristic for "least nested",
        # usually the most canonical install location.
        candidates.sort(key=len)
        return candidates[0]
    return None


def ensure_wan21_t2v_14b(
    comfyui_models_dir: str,
    *,
    source: str = "auto",
    extra_paths: Optional[list[str]] = None,
) -> str:
    """Resolve a Wan2.1-T2V-14B install path, downloading if needed.

    `source`:
        "auto"           → auto-detect; download if not found.
        "auto-download"  → download to models/unividx/Wan2.1-T2V-14B/
                           even if a copy is found elsewhere.
        <explicit path>  → use this path verbatim; validate the layout
                           and error clearly if it's wrong.
    """
    if source == "auto":
        found = resolve_wan21_t2v_14b_path(comfyui_models_dir, extra_paths)
        if found:
            logger.info("UniVidX: using existing Wan2.1-T2V-14B at %s", found)
            return found
        # Fall through to download.
        source = "auto-download"

    if source == "auto-download":
        return _download_wan21(comfyui_models_dir)

    # Treat anything else as an explicit path the user supplied.
    if not _looks_like_wan_t2v(source):
        raise RuntimeError(
            f"Path {source!r} does not look like a Wan2.1-T2V-14B install. "
            "Expected an HF-layout directory containing "
            "`models_t5_umt5-xxl-enc-*.pth` + `Wan2.1_VAE.pth` + DiT "
            "shards. Wan2.1-VACE-14B and Wan2.1-I2V-14B are NOT "
            "compatible — UniVidX needs the T2V-14B base variant."
        )
    return source


def _download_wan21(comfyui_models_dir: str) -> str:
    """Snapshot-download Wan2.1-T2V-14B into models/unividx/."""
    target = os.path.join(comfyui_models_dir, "unividx", "Wan2.1-T2V-14B")
    if _looks_like_wan_t2v(target):
        logger.info("UniVidX: Wan2.1-T2V-14B already present at %s", target)
        return target
    os.makedirs(target, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to fetch Wan2.1-T2V-14B. "
            "Install the wrapper's requirements with "
            "`pip install -r requirements.txt`."
        ) from e
    logger.info(
        "UniVidX: downloading Wan2.1-T2V-14B (~57 GB) from %s "
        "into ComfyUI/models/unividx/Wan2.1-T2V-14B/. This is one-time.",
        WAN_REPO_ID,
    )
    try:
        snapshot_download(repo_id=WAN_REPO_ID, local_dir=target)
    except TypeError:
        snapshot_download(
            repo_id=WAN_REPO_ID, local_dir=target,
            local_dir_use_symlinks=False,
        )
    return target


# ── UniVidX checkpoint resolution ───────────────────────────────────────────


def ensure_unividx_checkpoint(
    comfyui_models_dir: str,
    variant: str,
    *,
    source: str = "auto",
    extra_paths: Optional[list[str]] = None,
) -> str:
    """Resolve the path to a UniVidX checkpoint (.safetensors) for the
    requested variant ('intrinsic' or 'alpha'). Auto-detect then
    download, same shape as `ensure_wan21_t2v_14b`.
    """
    if variant not in ("intrinsic", "alpha"):
        raise ValueError(f"variant must be 'intrinsic' or 'alpha', got {variant!r}")
    fname = f"univid_{variant}.safetensors"

    if source == "auto":
        found = _find_unividx_ckpt(comfyui_models_dir, fname, extra_paths)
        if found:
            logger.info("UniVidX: using existing %s at %s", fname, found)
            return found
        source = "auto-download"

    if source == "auto-download":
        return _download_unividx_ckpt(comfyui_models_dir, fname)

    # Explicit path
    if not os.path.isfile(source):
        raise RuntimeError(
            f"Path {source!r} is not a file. Expected {fname} or an "
            "equivalent UniVidX checkpoint."
        )
    return source


def _find_unividx_ckpt(
    comfyui_models_dir: str,
    fname: str,
    extra_paths: Optional[list[str]] = None,
) -> Optional[str]:
    candidates = []
    for sub in ("unividx", "checkpoints", "diffusion_models"):
        parent = os.path.join(comfyui_models_dir, sub)
        if not os.path.isdir(parent):
            continue
        direct = os.path.join(parent, fname)
        if os.path.isfile(direct):
            candidates.append(direct)
        # One level deep
        try:
            for entry in os.listdir(parent):
                full = os.path.join(parent, entry, fname)
                if os.path.isfile(full):
                    candidates.append(full)
        except OSError:
            continue
    for p in (extra_paths or []):
        if os.path.isfile(p) and os.path.basename(p) == fname:
            candidates.append(p)
    candidates.sort(key=len)
    return candidates[0] if candidates else None


def _download_unividx_ckpt(
    comfyui_models_dir: str, fname: str,
) -> str:
    target_dir = os.path.join(comfyui_models_dir, "unividx")
    target = os.path.join(target_dir, fname)
    if os.path.isfile(target):
        return target
    os.makedirs(target_dir, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to fetch UniVidX checkpoints. "
            "Install the wrapper's requirements with "
            "`pip install -r requirements.txt`."
        ) from e
    logger.info(
        "UniVidX: downloading %s (~0.8 GB) from %s into %s",
        fname, UNIVIDX_REPO_ID, target_dir,
    )
    return hf_hub_download(
        repo_id=UNIVIDX_REPO_ID, filename=fname, local_dir=target_dir,
    )


# ── Symlink setup for upstream's hardcoded relative paths ───────────────────


def ensure_upstream_symlinks(
    upstream_dir: str,
    wan21_path: str,
    intrinsic_ckpt: Optional[str] = None,
    alpha_ckpt: Optional[str] = None,
) -> None:
    """Create the symlinks UniVidX's pipeline expects relative to its
    own root. Specifically:
        upstream_dir/models/Wan-AI/Wan2.1-T2V-14B → wan21_path
        upstream_dir/checkpoints/univid_intrinsic.safetensors → intrinsic_ckpt
        upstream_dir/checkpoints/univid_alpha.safetensors     → alpha_ckpt

    Idempotent — replaces stale symlinks but leaves correct ones alone.
    Creates parent directories as needed.
    """
    def _link(src: str, dst: str) -> None:
        parent = os.path.dirname(dst)
        os.makedirs(parent, exist_ok=True)
        # If `dst` already points at `src`, do nothing.
        try:
            current = os.readlink(dst)
            if os.path.realpath(current) == os.path.realpath(src):
                return
        except OSError:
            # Not a symlink, or doesn't exist.
            pass
        # Replace whatever is there.
        if os.path.islink(dst) or os.path.exists(dst):
            try:
                if os.path.islink(dst):
                    os.unlink(dst)
                # We never rmtree a real directory — if dst exists as
                # a real dir, the user put something there manually,
                # leave it alone and let the import fail with a clear
                # error rather than nuking files.
                else:
                    logger.warning(
                        "UniVidX: %s exists and isn't a symlink; not "
                        "replacing. Move it aside if you want the "
                        "wrapper-managed copy.", dst,
                    )
                    return
            except OSError as e:
                logger.warning("UniVidX: failed to replace %s: %s", dst, e)
                return
        try:
            os.symlink(src, dst)
            logger.info("UniVidX: symlinked %s -> %s", dst, src)
        except OSError as e:
            logger.warning("UniVidX: failed to symlink %s -> %s: %s", dst, src, e)

    _link(wan21_path, os.path.join(upstream_dir, "models", "Wan-AI", "Wan2.1-T2V-14B"))
    if intrinsic_ckpt:
        _link(intrinsic_ckpt, os.path.join(upstream_dir, "checkpoints", "univid_intrinsic.safetensors"))
    if alpha_ckpt:
        _link(alpha_ckpt, os.path.join(upstream_dir, "checkpoints", "univid_alpha.safetensors"))
