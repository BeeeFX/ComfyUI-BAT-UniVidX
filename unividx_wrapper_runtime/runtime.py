"""Model load + cache + chdir guard.

Owns:
  • One-time sys.path setup so the upstream UniVidX package becomes
    importable. We add `<wrapper_dir>/UniVidX/` to sys.path AFTER our
    own auto_clone.auto_clone() guarantees it exists.

  • The `unividx_cwd()` context manager. UniVidX hardcodes a handful
    of relative paths (the most notable is the MoGe-style anchor
    `models/Wan-AI/Wan2.1-T2V-14B/`). We chdir into the upstream
    directory for the duration of model load + sampling so those
    paths resolve regardless of where ComfyUI was launched from.
    Not thread-safe — ComfyUI's default sampler runs serially, so
    this is fine. Note flagged in the docstring for anyone running
    parallel-queue forks.

  • An in-memory LRU cache keyed by `(variant, wan21_path, ckpt_path)`
    so re-running the graph with the same loaded model is instant.
    Default capacity is 1: a 14B Wan model with UniVidX adapters is
    ~28 GB of activations + weights and we never want two resident
    at once on consumer cards.

  • The pipeline class import — we go DIRECTLY to
    `src.pipelines.univid_{intrinsic,alpha}` instead of through
    `scripts.registry`, which avoids pulling in upstream's trainer
    module (the source of the `torchvision.io.write_video` headache)
    and avoids touching `sys.modules['src']` at the wrong moment.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections import OrderedDict
from typing import Any, Optional

import torch

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")

_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_CACHE_MAX = 1  # one Wan-14B + UniVidX adapter set ≈ 28 GB on disk; never two resident.


def _add_upstream_to_path(upstream_dir: str) -> None:
    """Idempotently add the upstream UniVidX directory to sys.path so
    `from src.pipelines.univid_intrinsic import UniVidIntrinsic`
    resolves. We never import `from scripts.registry import ...` —
    the registry module imports trainer code, which forces the
    `torchvision.io.write_video` shim and a bunch of unrelated weight
    bookkeeping. Going straight to the pipeline class avoids all of
    that."""
    if upstream_dir not in sys.path:
        sys.path.insert(0, upstream_dir)


@contextlib.contextmanager
def unividx_cwd(upstream_dir: str):
    """Temporarily chdir into the upstream UniVidX root so its
    hardcoded relative paths resolve. NOT thread-safe — ComfyUI's
    default worker runs prompts serially so this is fine on a stock
    install."""
    prev = os.getcwd()
    os.chdir(upstream_dir)
    try:
        yield
    finally:
        os.chdir(prev)


def _import_pipeline_class(variant: str):
    """Direct import of UniVidX's pipeline class — bypasses
    `scripts.registry` entirely so we don't pull trainer code with
    its torchvision-version-fragile imports.

    Returns the class object (UniVidIntrinsic or UniVidAlpha).
    """
    if variant == "intrinsic":
        from src.pipelines.univid_intrinsic import UniVidIntrinsic  # type: ignore
        return UniVidIntrinsic
    if variant == "alpha":
        from src.pipelines.univid_alpha import UniVidAlpha  # type: ignore
        return UniVidAlpha
    raise ValueError(f"variant must be 'intrinsic' or 'alpha', got {variant!r}")


def load_model(
    *,
    variant: str,
    upstream_dir: str,
    wan21_path: str,
    ckpt_path: str,
    dtype: torch.dtype = torch.bfloat16,
    vram_buffer_gb: float = 4.0,
    advanced: Optional[dict] = None,
) -> Any:
    """Build (or fetch-from-cache) the UniVidX pipeline for `variant`.

    `advanced` carries the opt-in perf knobs:

      fp8_quantize     — "none" | "qfloat8_prequantized" — substitute
                          DiT Linears with FP8Linear; ~14 GB DiT (vs 28).
      distill_lora     — "none" | "lightx2v" — merge a step-distill
                          LoRA so num_inference_steps=4 + cfg_scale=1
                          produces reasonable output (5× faster wall).
      distill_strength — float (0..1+), merge strength.
      compile_dit      — bool, torch.compile the DiT (~20-30% faster
                          per-step after first-call warmup).
      prefer_sage_attn — bool, swap DiffSynth's attention impl to
                          SageAttention-first with FA2/SDPA fallback.

    The cache key includes every advanced knob so a change forces a
    rebuild. Loaded models go in the LRU; default size 1 (only one
    big Wan-14B + UniVidX adapter resident at a time).

    Returns whatever the pipeline class instantiates to. The caller
    is responsible for `unividx_cwd()` during sampling.
    """
    if variant not in ("intrinsic", "alpha"):
        raise ValueError(f"variant must be 'intrinsic' or 'alpha', got {variant!r}")

    adv = advanced or {}
    fp8_quantize = adv.get("fp8_quantize", "none") or "none"
    distill_lora = adv.get("distill_lora", "none") or "none"
    distill_strength = float(adv.get("distill_strength", 1.0))
    compile_dit_flag = bool(adv.get("compile_dit", False))
    prefer_sage = bool(adv.get("prefer_sage_attn", False))

    cache_key = (
        variant,
        os.path.realpath(wan21_path),
        os.path.realpath(ckpt_path),
        str(dtype),
        float(vram_buffer_gb),
        fp8_quantize, distill_lora, distill_strength,
        compile_dit_flag, prefer_sage,
    )

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(cache_key)
            return _MODEL_CACHE[cache_key]

        # Evict LRU before constructing the new model so we never hold
        # two big models in VRAM simultaneously.
        while len(_MODEL_CACHE) >= max(1, _CACHE_MAX):
            evicted_key, evicted_model = _MODEL_CACHE.popitem(last=False)
            logger.info(
                "UniVidX: evicting cached model (variant=%s) to free VRAM",
                evicted_key[0],
            )
            del evicted_model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _add_upstream_to_path(upstream_dir)

        # Defensive: undo any other custom node's hostile SDPA global
        # monkey-patch. Stable3DGen does this at import time and
        # breaks UniVidX's VAE.
        from . import advanced as adv_mod
        if adv_mod.restore_native_sdpa_if_polluted():
            logger.info(
                "UniVidX: restored native scaled_dot_product_attention "
                "(another custom node had monkey-patched it globally)",
            )

        logger.info(
            "UniVidX: loading %s pipeline (Wan2.1=%s, ckpt=%s, dtype=%s, "
            "vram_buffer=%.1f GB) — first load is multi-minute.",
            variant, wan21_path, ckpt_path, dtype, vram_buffer_gb,
        )
        with unividx_cwd(upstream_dir):
            # Sage attention: install BEFORE pipeline construction so
            # DiffSynth's freshly-imported wan_video_dit module has the
            # wrapped flash_attention from the get-go.
            if prefer_sage:
                if not adv_mod.force_sage_over_fa2():
                    logger.warning(
                        "UniVidX: prefer_sage_attn=True but the "
                        "SageAttention dispatcher could not be installed "
                        "(likely cause: `sageattention` not importable). "
                        "Continuing with default attention backend."
                    )

            PipelineCls = _import_pipeline_class(variant)
            model = PipelineCls()

            # mmgp safetensors patch: read-only mmap for Windows.
            adv_mod.patch_unividx_load_file_to_readonly()

            # Step-distill merge MUST happen BEFORE
            # enable_vram_management — once vram-mgmt wraps non-Linear
            # modules in AutoWrappedModule, LightX2V's `.diff` keys
            # silently fail to match the wrapped paths.
            if distill_lora != "none":
                adv_mod.apply_step_distill_merge(
                    model, lora_kind=distill_lora, strength=distill_strength,
                )

            # Hook DiffSynth's VRAM manager so blocks stream CPU<->GPU
            # during forward. Without this, a 28 GB DiT pins the card
            # solid and there's nothing left for activations.
            pipe = getattr(model, "pipe", None)
            if pipe is not None and hasattr(pipe, "enable_vram_management"):
                try:
                    pipe.enable_vram_management(vram_buffer=float(vram_buffer_gb))
                    logger.info(
                        "UniVidX: VRAM management enabled (buffer=%.1f GB)",
                        float(vram_buffer_gb),
                    )
                except TypeError as e:
                    logger.warning(
                        "UniVidX: enable_vram_management(vram_buffer=...) "
                        "rejected: %s. Sampling may OOM.", e,
                    )
            else:
                logger.warning(
                    "UniVidX: model.pipe.enable_vram_management() missing; "
                    "vram_buffer_gb has no effect on this build.",
                )

            # FP8 substitution: runs AFTER vram_management — the
            # FP8Linear modules slot in transparently because they
            # have the same forward signature as nn.Linear. The
            # existing wrapper does the same ordering.
            if fp8_quantize == "qfloat8_prequantized":
                adv_mod.apply_fp8_substitution(model, variant)

            # torch.compile: LAST — needs the final module graph.
            if compile_dit_flag:
                adv_mod.compile_dit(model)
                logger.info(
                    "UniVidX: DiT compiled with torch.compile (first "
                    "sampler step will pay a 60-120 sec capture cost)."
                )

        _MODEL_CACHE[cache_key] = model
        return model
