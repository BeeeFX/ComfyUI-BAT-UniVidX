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
import json
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


@contextlib.contextmanager
def _own_the_src_name(upstream_dir: str):
    """Temporarily point `sys.modules['src']` at UniVidX's own `src/`
    package, restoring whatever was there on the way out.

    Why this is necessary: `src` is a catastrophically generic
    top-level package name, and several unrelated custom nodes ship
    one — `comfyui_controlnet_aux`, both copies of
    `SeedVR2_VideoUpscaler` (whose `src/interfaces/comfyui_node.py`
    does `from src.…` at ComfyUI startup), and any third-party
    UniVidX wrapper. Whichever loads first wins `sys.modules['src']`,
    and it is NOT us: ComfyUI walks custom_nodes alphabetically and
    imports every node at boot, long before a graph runs.

    Adding our directory to sys.path is not enough. sys.path is only
    consulted for names that aren't already in sys.modules, so once
    another node owns `src`, `import src.pipelines.univid_intrinsic`
    resolves `pipelines` as a child of THEIR package, doesn't find
    it, and raises `ModuleNotFoundError: No module named
    'src.pipelines'` — pointing at a file that is present on disk.

    So we bind the name ourselves for the duration of the import.
    Upstream's modules do `from src.pipelines…` internally, so the
    name genuinely has to be `src` while they execute; importing the
    file under a private alias alone would break those internal
    imports. We also drop any `src.*` submodules the other node had
    cached, or Python would serve theirs from cache mid-import, and
    restore the exact previous state in `finally`.

    Not thread-safe, for the same reason `unividx_cwd()` isn't:
    ComfyUI's default worker runs prompts serially.
    """
    import importlib

    saved = {k: v for k, v in sys.modules.items()
             if k == "src" or k.startswith("src.")}
    for k in saved:
        del sys.modules[k]

    # Import our `src` explicitly by path so the resolution can't be
    # hijacked by sys.path ordering either.
    src_dir = os.path.join(upstream_dir, "src")
    init_py = os.path.join(src_dir, "__init__.py")
    if not os.path.isfile(init_py):
        raise RuntimeError(
            f"UniVidX: expected {init_py} to exist. The vendored upstream "
            f"copy at {upstream_dir!r} looks incomplete — restore it with "
            "`git checkout -- UniVidX` in the node pack's directory."
        )
    try:
        spec = importlib.util.spec_from_file_location(
            "src", init_py, submodule_search_locations=[src_dir],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["src"] = module
        spec.loader.exec_module(module)
        if saved:
            logger.info(
                "UniVidX: temporarily rebound the top-level 'src' module "
                "name (another custom node owns it: %s). This is expected.",
                ", ".join(sorted(saved)[:3]) + ("…" if len(saved) > 3 else ""),
            )
        yield
    finally:
        for k in [k for k in sys.modules
                  if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def _import_pipeline_class(variant: str, upstream_dir: str):
    """Direct import of UniVidX's pipeline class — bypasses
    `scripts.registry` entirely so we don't pull trainer code with
    its torchvision-version-fragile imports.

    Wrapped in `_own_the_src_name()` because the bare top-level name
    `src` is contested by several other custom nodes; see there.

    Returns the class object (UniVidIntrinsic or UniVidAlpha).
    """
    if variant not in ("intrinsic", "alpha"):
        raise ValueError(f"variant must be 'intrinsic' or 'alpha', got {variant!r}")
    with _own_the_src_name(upstream_dir):
        if variant == "intrinsic":
            from src.pipelines.univid_intrinsic import UniVidIntrinsic  # type: ignore
            return UniVidIntrinsic
        from src.pipelines.univid_alpha import UniVidAlpha  # type: ignore
        return UniVidAlpha


# ── Pipeline constructor arguments ──────────────────────────────────────────
# UniVidIntrinsic/UniVidAlpha.__init__ defaults EVERY argument to None, and a
# bare `PipelineCls()` silently produces a crippled pipeline:
#
#   model_paths=None          -> model_configs == [] -> WanVideoPipeline gets
#                                no T5 and no VAE. Logs "No wan_video_text_
#                                encoder models available." then dies at
#                                sample time with
#                                `'NoneType' object has no attribute 'encode'`.
#   lora_modalities=None      -> the add_multiple_loras_to_model() branch is
#                                skipped, so the per-modality DGL adapters
#                                (the whole point of UniVidX) are absent.
#   resume_from_checkpoint    -> the UniVidX checkpoint is never loaded, so
#                     =None      the DiT stays vanilla Wan2.1-T2V-14B.
#
# The last two fail SILENTLY — you get plausible-looking Wan2.1 video instead
# of an intrinsic/alpha decomposition. Hence _assert_pipeline_fully_wired().
#
# Values mirror upstream's own configs/univid_{intrinsic,alpha}_inference.yaml
# verbatim; they are not tuning knobs.
_LORA_TARGET_MODULES = "self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2"
_LORA_RANK = 32

# Per-variant LoRA adapter names. These are the `adapter_name`s the
# checkpoint's keys are namespaced under, so they must match the checkpoint
# exactly — not the wrapper's own socket names. Note the alpha family uses
# "com" here while our IMAGE socket for the same modality is "rgb".
_LORA_MODALITIES = {
    "intrinsic": ["rgb", "albedo", "irradiance", "normal"],
    "alpha": ["com", "pha", "fgr", "bgr"],
}

# T5 text encoder, in filename preference order, and the VAE. Same names
# auto_download._WAN_REQUIRED_FILES verifies, kept in sync deliberately:
# that module proves they're on disk, this one hands them to the pipeline.
_T5_FILENAMES = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_t5_umt5-xxl-enc-fp32.pth",
)
_VAE_FILENAME = "Wan2.1_VAE.pth"


def _pipeline_kwargs(variant: str, wan21_path: str, ckpt_path: str) -> dict:
    """Build the full keyword-argument set for the pipeline constructor.

    Paths are passed ABSOLUTE. Upstream's configs use paths relative to
    the UniVidX root (which is why we chdir via `unividx_cwd()` and
    symlink into it), but absolute paths work regardless of cwd and give
    a readable error if a file moves. The one thing we cannot make
    absolute is the tokenizer: the constructor hardcodes
    `models/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl` internally, so the
    chdir + symlinks are still load-bearing for that.
    """
    t5 = None
    for name in _T5_FILENAMES:
        candidate = os.path.join(wan21_path, name)
        if os.path.isfile(candidate):
            t5 = candidate
            break
    if t5 is None:
        raise RuntimeError(
            f"UniVidX: no T5 text encoder found in {wan21_path!r}. Expected "
            f"one of {', '.join(_T5_FILENAMES)}. Without it the pipeline "
            "loads no text encoder and sampling dies with "
            "\"'NoneType' object has no attribute 'encode'\"."
        )

    vae = os.path.join(wan21_path, _VAE_FILENAME)
    if not os.path.isfile(vae):
        raise RuntimeError(
            f"UniVidX: {_VAE_FILENAME} missing from {wan21_path!r}. The "
            "pipeline cannot encode conditioning frames or decode results "
            "without it."
        )

    # The tokenizer directory the constructor reaches for via its own
    # hardcoded relative path. Checked here so a missing tokenizer is a
    # clear error rather than a confusing failure inside from_pretrained.
    tokenizer_dir = os.path.join(wan21_path, "google", "umt5-xxl")
    if not os.path.isdir(tokenizer_dir):
        raise RuntimeError(
            f"UniVidX: tokenizer directory missing at {tokenizer_dir!r}. It "
            "ships inside the Wan2.1-T2V-14B repo under google/umt5-xxl; a "
            "partial download is the usual cause. Delete the Wan2.1 folder "
            "and let the loader re-fetch it."
        )

    if not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f"UniVidX: checkpoint {ckpt_path!r} not found. Without it the "
            "DiT stays vanilla Wan2.1 and you'd get ordinary video instead "
            "of a decomposition."
        )

    # model_paths is JSON — the constructor does json.loads() on it.
    return {
        "model_paths": json.dumps([t5, vae]),
        "lora_base_model": "dit",
        "lora_target_modules": _LORA_TARGET_MODULES,
        "lora_rank": _LORA_RANK,
        "lora_modalities": _LORA_MODALITIES[variant],
        "resume_from_checkpoint": ckpt_path,
    }


def _assert_pipeline_fully_wired(model: Any, variant: str) -> None:
    """Fail loudly if the constructed pipeline is missing a text encoder,
    a VAE, or its LoRA adapters.

    Worth the belt-and-braces: two of the three failure modes above are
    SILENT. A pipeline with no adapters and an unloaded checkpoint still
    samples happily — it just produces vanilla Wan2.1 output, which looks
    like a plausible result rather than a bug. Catching it at load time
    turns a "why does this look wrong?" investigation into an error
    message.
    """
    pipe = getattr(model, "pipe", None)
    if pipe is None:
        raise RuntimeError(
            "UniVidX: constructed model has no `.pipe` — upstream's API "
            "changed; the wrapper needs updating for this commit."
        )

    if getattr(pipe, "text_encoder", None) is None:
        raise RuntimeError(
            "UniVidX: pipeline built with no text encoder (model_paths "
            "didn't reach WanVideoPipeline). Sampling would fail with "
            "\"'NoneType' object has no attribute 'encode'\"."
        )
    if getattr(pipe, "vae", None) is None:
        raise RuntimeError(
            "UniVidX: pipeline built with no VAE — cannot encode "
            "conditioning frames or decode results."
        )

    dit = getattr(pipe, "dit", None)
    if dit is None:
        raise RuntimeError("UniVidX: pipeline has no DiT.")
    if not any("lora" in name.lower() for name, _ in dit.named_parameters()):
        raise RuntimeError(
            f"UniVidX: no LoRA parameters found on the DiT for variant "
            f"{variant!r}. The per-modality DGL adapters are what make this "
            "UniVidX rather than plain Wan2.1 — without them the node would "
            "silently emit ordinary video. Expected adapters: "
            f"{', '.join(_LORA_MODALITIES[variant])}."
        )
    # Adapters that exist are not adapters that were LOADED. Upstream
    # wraps the checkpoint load in a try/except that only prints
    # (univid_intrinsic.py:1405-1414), so a truncated or wrong checkpoint
    # leaves every adapter at its initialisation — and PEFT initialises
    # lora_B to zero, making the adapters exact no-ops. The DiT then runs
    # as vanilla Wan2.1 while passing every check above.
    if not _any_lora_b_nonzero(dit):
        raise RuntimeError(
            f"UniVidX: the {variant} checkpoint did not load — every LoRA "
            "adapter is still at its zero initialisation, so the node would "
            "emit plain Wan2.1 video. Look above for upstream's \"Failed to "
            "resume training from checkpoint\" message; a partial download "
            "is the usual cause. Delete the checkpoint under "
            "ComfyUI/models/unividx/ and let the loader fetch it again."
        )
    logger.info(
        "UniVidX: pipeline wired — text encoder + VAE present, LoRA "
        "adapters attached and loaded (%s).",
        ", ".join(_LORA_MODALITIES[variant]),
    )


def _any_lora_b_nonzero(dit) -> bool:
    """True once any `lora_B` weight is non-zero. Stops at the first hit,
    which on a loaded checkpoint is the first tensor examined."""
    for name, param in dit.named_parameters():
        if "lora_B" in name and param.numel() and bool(param.detach().any()):
            return True
    return False


# ── VRAM buffer auto-sizing ─────────────────────────────────────────────────
# DiffSynth's enable_vram_management(vram_buffer=N) computes
# `vram_limit = total_card_vram - N` and offloads DiT layers to CPU to stay
# under that limit. So `vram_buffer` is a RESERVATION: it's how much VRAM is
# left free for everything that isn't resident weights — activations,
# attention workspace, VAE decode.
#
# The old default of 4.0 was tuned for a 32 GB card holding an FP8 DiT with
# little headroom needed. It is badly wrong on a big card: on a 95 GB card it
# sets the limit to 91 GB, so ~48 GB of bf16 DiT stays resident and the first
# sampler step (which wants 20+ GB in ONE allocation, because UniVidX attends
# over four modality streams at once) dies with 20 MiB free.
#
# Sizing: reserve a fraction of the card, but never less than one observed
# sampler step. The fraction (rather than a flat constant) is because the
# activation cost scales with what people run on a given card — bigger cards
# get used for bigger frame counts and resolutions.
#
# _SAMPLER_SPIKE_GB is measured, not guessed: farm job c8824e2c on a 95 GB
# card at 21 frames / R2AIN requested 24.17 GiB in a SINGLE allocation while
# 47.68 GiB of bf16 DiT sat resident. UniVidX attends over four modality
# streams at once, so its peak is several times a plain Wan2.1 step. Treat
# this as a floor for 21 frames — more frames will want more, which is why
# the fraction takes over on big cards.
_VRAM_RESERVE_FRACTION = 0.35
_SAMPLER_SPIKE_GB = 26.0


def total_vram_gb() -> Optional[float]:
    """Total VRAM of the active CUDA device in GB, or None on CPU.

    Uses the same source DiffSynth does (`mem_get_info()[1]`, i.e. TOTAL
    not free) so our arithmetic matches theirs exactly.
    """
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.mem_get_info()[1] / (1024 ** 3)
    except Exception:
        return None


def auto_vram_buffer_gb(*, fp8: bool) -> float:
    """Pick a `vram_buffer` reservation for this card.

    Returns GB to keep free. Larger reservation => more offloading =>
    more room for the sampler's peak allocation.

    `fp8` widens the reservation rather than narrowing it: FP8 halves the
    DiT's resident bytes (~28 -> ~14 GB), so there is simply more card
    left to hand to activations. It is not a reason to reserve less.
    """
    total = total_vram_gb()
    if total is None:
        # CPU-only: DiffSynth's limit arithmetic is irrelevant, and a
        # nonzero buffer would just confuse its bookkeeping.
        return 0.0
    reserve = total * _VRAM_RESERVE_FRACTION
    if fp8:
        # FP8 frees ~14 GB of resident DiT; spend it on headroom.
        reserve += 6.0
    # Floor at one measured sampler step — a reservation smaller than the
    # peak allocation OOMs no matter how neatly the weights fit.
    reserve = max(reserve, _SAMPLER_SPIKE_GB)

    # Ceiling: never reserve so much that the DiT itself can't stay
    # resident enough to make progress. DiffSynth can stream layers from
    # CPU, so a tight limit is slow rather than fatal — but leaving less
    # than roughly one DiT's worth means thrashing every single step.
    #
    # On a card too small to hold BOTH the DiT and a full sampler spike
    # (anything under ~40 GB at bf16), these two pull in opposite
    # directions and there's no good answer: we favour the model fitting
    # and let DiffSynth stream, because that at least runs. Such a card
    # realistically needs FP8 plus fewer frames.
    dit_gb = 14.0 if fp8 else 28.0
    max_reserve = total - dit_gb
    if max_reserve > 0:
        reserve = min(reserve, max_reserve)
    else:
        # Card can't even hold the DiT. Reserve a token amount and let
        # DiffSynth offload aggressively; the loader logs the value so a
        # subsequent OOM is at least explainable.
        reserve = min(reserve, max(1.0, total * 0.1))
    return round(max(0.0, reserve), 1)


def free_comfy_vram() -> None:
    """Unload the models ComfyUI itself manages and release cached VRAM.

    This model lives outside ComfyUI's model management, and DiffSynth
    sizes its VRAM budget from the card's TOTAL memory — so anything
    ComfyUI still holds (an earlier node's model, the previous workflow's
    checkpoint) is memory the budget assumes is free. Called before a
    load and before each sampling run. No-op outside ComfyUI.

    Calls ComfyUI's ORIGINAL unload_all_models, bypassing the hook below,
    which would pointlessly offload the very model about to run.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    unload = getattr(mm.unload_all_models, "_bat_original", mm.unload_all_models)
    try:
        unload()
        mm.soft_empty_cache()
    except Exception as exc:  # never fail a render over housekeeping
        logger.debug("UniVidX: could not free ComfyUI VRAM (%s)", exc)


# ── ComfyUI's memory buttons ────────────────────────────────────────────────
# The model lives outside ComfyUI's model management, so neither "Unload
# models" nor "Free model and node cache" used to touch it: the host RAM
# (tens of GB) stayed held until ComfyUI restarted. Two small hooks bring
# it in line with how ComfyUI treats its own models:
#
#   unload_all_models()      "Unload models", ComfyUI's automatic unload
#                            after an out-of-memory error, other packs'
#                            loaders -> OFFLOAD: move everything off the
#                            GPU, keep it in RAM. The next run moves back
#                            what it needs in seconds.
#   PromptExecutor.reset()   "Free model and node cache" -> RELEASE: drop
#                            the cached model so its RAM is freed; the next
#                            run reloads it.
#
# Both run on ComfyUI's worker thread between prompts (or between nodes),
# never while a UniVidX node is executing.


def offload_cached_models() -> None:
    """Move every cached model off the GPU, keeping it loaded in RAM."""
    from . import advanced as _adv
    with _CACHE_LOCK:
        models = list(_MODEL_CACHE.values())
    for model in models:
        try:
            _adv.offload_pipeline(model)
        except Exception as exc:
            logger.warning("UniVidX: could not offload cached model (%s)", exc)
    if models and torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_cached_models() -> int:
    """Drop every cached model so its RAM and VRAM can be reclaimed.
    Returns how many were dropped."""
    with _CACHE_LOCK:
        n = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    if n:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("UniVidX: released %d cached model(s) — the next run "
                    "reloads.", n)
    return n


def install_comfy_memory_hooks() -> bool:
    """Wire the two hooks described above. Idempotent; best-effort, so an
    unfamiliar ComfyUI just means the buttons don't reach this model."""
    try:
        import comfy.model_management as mm
        import execution
    except ImportError:
        return False

    original_unload = mm.unload_all_models
    if not getattr(original_unload, "_bat_hooked", False):
        def unload_all_models(*args, **kwargs):
            result = original_unload(*args, **kwargs)
            offload_cached_models()
            return result
        unload_all_models._bat_hooked = True
        unload_all_models._bat_original = original_unload
        mm.unload_all_models = unload_all_models

    executor = getattr(execution, "PromptExecutor", None)
    original_reset = getattr(executor, "reset", None)
    if original_reset is not None and not getattr(original_reset, "_bat_hooked", False):
        def reset(self, *args, **kwargs):
            release_cached_models()
            return original_reset(self, *args, **kwargs)
        reset._bat_hooked = True
        executor.reset = reset
    return True


def _coerce_attention_backend(value) -> str:
    """Normalise the `attention_backend` setting, tolerating the boolean
    this widget used to be.

    It was `prefer_sage_attn` (BOOLEAN) and is now a combo. Because it is
    the LAST widget in the loader's saved `widgets_values` array, swapping
    it in place shifts nothing else — but workflows saved before the change
    still hand us `False`/`True`. Map those to their intent rather than
    failing: the graph keeps running, and the widget just needs re-picking
    once in the UI.
    """
    if isinstance(value, bool):
        return "sage" if value else "auto"
    if isinstance(value, str) and value in adv_backends():
        return value
    if value is None:
        return "auto"
    logger.warning(
        "UniVidX: unrecognised attention_backend %r; using 'auto'.", value,
    )
    return "auto"


def build_tea_cache_for_run(model_instance, *, num_inference_steps: int,
                            rel_l1_thresh: float):
    """Sampler-facing wrapper around advanced.build_tea_cache.

    Kept here so the sampler node has a single `runtime.` entry point and
    doesn't need to import the torch-heavy advanced module directly.
    Returns a fresh, single-use TeaCache, or None.
    """
    from . import advanced as _adv
    return _adv.build_tea_cache(
        model_instance,
        num_inference_steps=num_inference_steps,
        rel_l1_thresh=rel_l1_thresh,
    )


def instrument_vae_encode(model_instance, *, tiled: bool, tile_size,
                          tile_stride) -> bool:
    """Sampler-facing wrapper around advanced.instrument_vae_encode."""
    from . import advanced as _adv
    return _adv.instrument_vae_encode(
        model_instance, tiled=tiled, tile_size=tile_size,
        tile_stride=tile_stride,
    )


def sampler_progress(callback):
    """Sampler-facing wrapper around advanced.sampler_progress: report each
    denoising step of the pipe() call it wraps, and honour Cancel."""
    from . import advanced as _adv
    return _adv.sampler_progress(callback)


def offload_pipeline(model_instance) -> None:
    """Sampler-facing wrapper around advanced.offload_pipeline."""
    from . import advanced as _adv
    _adv.offload_pipeline(model_instance)


def context_window_starts(total: int, size: int, stride: int) -> list[int]:
    """Sampler-facing wrapper around advanced.context_window_starts."""
    from . import advanced as _adv
    return _adv.context_window_starts(total, size, stride)


def tea_cache_report(tea_cache) -> Optional[str]:
    """Sampler-facing wrapper around advanced.tea_cache_report."""
    if tea_cache is None:
        return None
    from . import advanced as _adv
    return _adv.tea_cache_report(tea_cache)


def adv_backends() -> tuple:
    """Indirection so the valid-backend list lives in one place
    (advanced.ATTENTION_BACKENDS) without importing torch-heavy modules
    at import time."""
    from . import advanced as _adv
    return _adv.ATTENTION_BACKENDS


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
      compile_dit      — bool, torch.compile each DiT block in place
                          (speedup unmeasured on this model).
      attention_backend — "auto" | "sdpa" | "flash_attn_2" | "sage".
                          UniVidX hardcodes SDPA; "auto" picks FA2 if
                          importable, else SDPA.

    `dtype` is accepted for the loader's sake but upstream pins bf16.

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
    attention_backend = _coerce_attention_backend(adv.get("attention_backend", "auto"))

    # `dtype` is deliberately NOT part of the key: upstream's constructor
    # hardcodes bf16 (univid_intrinsic.py:1369-1370), so the loader's
    # fp16 option cannot change the model and must not trigger a reload.
    if dtype != torch.bfloat16:
        logger.warning(
            "UniVidX: dtype=%s ignored — UniVidX's pipeline is hardcoded to "
            "bf16 upstream, so the model always runs in bf16.", dtype,
        )
    cache_key = (
        variant,
        os.path.realpath(wan21_path),
        os.path.realpath(ckpt_path),
        float(vram_buffer_gb),
        fp8_quantize, distill_lora, distill_strength,
        compile_dit_flag, attention_backend,
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

        # ComfyUI's own models (from earlier nodes in this graph, or a
        # previous workflow) are invisible to DiffSynth's VRAM budget,
        # which is computed from the card's TOTAL memory.
        free_comfy_vram()

        logger.info(
            "UniVidX: loading %s pipeline (Wan2.1=%s, ckpt=%s, "
            "vram_buffer=%.1f GB) — first load is multi-minute.",
            variant, wan21_path, ckpt_path, vram_buffer_gb,
        )
        with unividx_cwd(upstream_dir):
            PipelineCls = _import_pipeline_class(variant, upstream_dir)

            # Attention backend: install AFTER the pipeline class import,
            # never before — it patches the DiT module's globals, reached
            # through the class (see install_attention_backend).
            adv_mod.install_attention_backend(attention_backend, PipelineCls)
            # Load into CPU RAM, streaming the DiT shard by shard — no GPU
            # memory is touched until sampling (see install_cpu_first_loading).
            if not adv_mod.install_cpu_first_loading(PipelineCls):
                logger.warning(
                    "UniVidX: falling back to upstream's loader, which needs "
                    "~40 GB of VRAM just to load.",
                )
            model = PipelineCls(**_pipeline_kwargs(variant, wan21_path, ckpt_path))
            _assert_pipeline_fully_wired(model, variant)

            # Repair upstream's temporal tiler so the sampler can window
            # long clips. Installed unconditionally: it only takes effect
            # when the sampler actually passes sliding_window_* , and
            # doing it here means the fix is part of the cached model
            # rather than something the sampler has to remember.
            adv_mod.install_sliding_window_support(model)
            # Per-step progress and Cancel for the sampler node.
            adv_mod.install_progress_hook(model)

            # Step-distill merge MUST happen BEFORE
            # enable_vram_management — once vram-mgmt wraps non-Linear
            # modules in AutoWrappedModule, LightX2V's `.diff` keys
            # silently fail to match the wrapped paths.
            if distill_lora != "none":
                adv_mod.apply_step_distill_merge(
                    model, lora_kind=distill_lora, strength=distill_strength,
                )

            # Hook DiffSynth's VRAM manager so blocks stream CPU<->GPU
            # during forward. Not optional: the model was loaded into CPU
            # RAM, and this is what moves layers to the GPU to run.
            pipe = getattr(model, "pipe", None)
            if pipe is None or not hasattr(pipe, "enable_vram_management"):
                raise RuntimeError(
                    "UniVidX: model.pipe.enable_vram_management() missing — "
                    "upstream's API changed; the wrapper needs updating."
                )
            pipe.enable_vram_management(vram_buffer=float(vram_buffer_gb))
            logger.info(
                "UniVidX: VRAM management enabled (buffer=%.1f GB "
                "reserved for activations; %s)",
                float(vram_buffer_gb),
                "FP8 DiT" if fp8_quantize != "none" else "bf16 DiT",
            )

            # FP8 runs AFTER enable_vram_management, deliberately: it now
            # converts the wrappers that call installed rather than
            # replacing them, so the layers end up FP8 *and* still
            # streamable. Quantizing first would leave nothing to convert.
            #
            # This used to be a genuine either/or. The old path swapped
            # AutoWrappedLinear out for a plain FP8Linear, which has no
            # offload/onload, so every quantized layer pinned itself to
            # the GPU — FP8 halved the DiT's bytes and simultaneously made
            # them immovable.
            #
            # One trap worth recording, since it looks like the obvious
            # fix and is not: do NOT make FP8Linear an nn.Linear subclass
            # so DiffSynth's module_map picks it up. AutoWrappedLinear has
            # its own fp8 path (`enable_fp8`, `fp8_linear()`) that derives
            # scales dynamically from the input, and its forward does
            # computation() -> linear_forward() knowing nothing about a
            # per-tensor scale_weight. Wrapping an already-quantized
            # FP8Linear that way computes silently wrong numbers. The
            # working approach is the reverse — subclass AutoWrappedLinear
            # and override linear_forward to apply the scale, which is
            # what fp8_loader.make_streamable_fp8_class does.
            if fp8_quantize == "qfloat8_prequantized":
                adv_mod.apply_fp8_substitution(model, variant)

            # torch.compile: LAST — needs the final module graph.
            if compile_dit_flag:
                n_blocks = adv_mod.compile_dit(model)
                logger.info(
                    "UniVidX: %d DiT blocks set to torch.compile (the first "
                    "sampler step pays the compile cost).", n_blocks,
                )

        _MODEL_CACHE[cache_key] = model
        return model
