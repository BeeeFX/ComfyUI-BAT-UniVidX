# unividx_wrapper_runtime/advanced.py
#
# Derived from UniVidX_ComfyUI/src/runtime.py (GPL v3). The original
# authors implemented the FP8 substitution + distill-LoRA merge +
# torch.compile + sage attention dispatcher specifically for the
# UniVidX + Wan2.1-14B layout; the workarounds they discovered
# (Stable3DGen's hostile SDPA monkey-patch, mmgp's writable_tensors
# mmap default, the CMSA path's unrecognised drop_out kwarg, sage's
# head_dim<=256 limitation, the order of merge vs vram-management
# wrapping) took real debugging to find. Re-deriving them would mean
# re-discovering the same bugs.
#
# See NOTICE at the wrapper root for attribution and the GPL v3 license
# terms that carry through to this file.
"""Advanced perf knobs for the BAT UniVidX wrapper.

Each function is driven by its own loader widget; nothing here runs
unless that widget asks for it.

Two run by default because they cost nothing in quality: FP8 weight
quantisation (a bf16 Wan2.1-14B DiT is ~28-32 GB and leaves no room for
activations) and the attention-backend dispatcher. Everything that can
alter output pixels — the step-distill LoRA merge, TeaCache — stays off
until explicitly enabled.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch

import folder_paths

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")


# De-duplication map: warn once per (backend, head_dim, exc-type-name).
_attention_fallback_warned: set = set()


# ─── Step-distill LoRA pre-merge ────────────────────────────────────────────


def _warn_if_lora_looks_mismatched(lora_name: str) -> None:
    """Flag a LoRA whose filename suggests the wrong base model.

    Worth doing up front because the obvious safety net doesn't catch it.
    The merge warns when ZERO keys match — but Wan2.1 I2V and T2V share
    most of their DiT, so an I2V LoRA matches plenty of keys and merges
    "successfully" while quietly degrading the result. A name check is
    crude, but it catches the realistic mistake before a multi-minute
    load rather than after an inexplicable render.

    Only ever warns. The filename is a hint, not proof, and someone with
    a correctly-working file that happens to be oddly named should not be
    blocked by a substring match.
    """
    low = lora_name.lower()
    if ("i2v" in low) and ("t2v" not in low):
        logger.warning(
            "UniVidX: %r looks like an IMAGE-to-video LoRA, but UniVidX is "
            "built on Wan2.1-T2V-14B. Wan2.1 I2V and T2V share most of the "
            "DiT, so this will merge without an obvious error and then "
            "quietly hurt quality. You probably want a *_T2V_14B_* file.",
            lora_name,
        )
    if any(t in low for t in ("wan2.2", "wan22", "wan2_2")):
        logger.warning(
            "UniVidX: %r looks like a Wan 2.2 LoRA. This model's backbone is "
            "Wan 2.1 (T2V-14B) — a different generation. Expect few or no "
            "keys to match; check the merge report below.",
            lora_name,
        )


def resolve_step_distill_path(lora_name: str) -> Optional[str]:
    """Resolve a `distill_lora` dropdown selection to an absolute path.

    `lora_name` is a name from `folder_paths.get_filename_list("loras")`,
    i.e. whatever the user picked in the UI. This used to hunt for three
    hardcoded filenames in four guessed directories, which meant the one
    file actually present in this asset library
    (…/loras/WAN21/…rank32.safetensors) went unfound and the feature could
    never load. Letting ComfyUI resolve a real selection removes the
    guesswork and works for any future distill LoRA.
    """
    if not lora_name or lora_name == "none":
        return None
    try:
        return folder_paths.get_full_path("loras", lora_name)
    except Exception as exc:
        logger.warning(
            "UniVidX: could not resolve LoRA %r via folder_paths (%s: %s).",
            lora_name, type(exc).__name__, exc,
        )
        return None


def apply_step_distill_merge(model, *, lora_kind: str, strength: float) -> None:
    """Load a step-distill LoRA safetensors and merge it into the DiT's
    base weights in place.

    For LightX2V: keys are prefixed with `diffusion_model.`, use Kohya
    `lora_down/lora_up` naming for Linear deltas, plus `.diff_b` for
    bias deltas and `.diff` for direct weight deltas on non-Linear
    modules (RMSNorm, patch_embedding, head).

    PEFT-aware via lora_merge._descend_peft — the merge applies to the
    .base_layer Linear inside UniVidX's per-modality wrappers; the
    LoRA siblings (lora_A_rgb/albedo/irradiance/normal etc.) stay
    untouched.
    """
    path = resolve_step_distill_path(lora_kind)
    if path is None:
        raise FileNotFoundError(
            f"Step-distill LoRA {lora_kind!r} could not be resolved under "
            f"ComfyUI/models/loras/.\n\n"
            f"The dropdown is populated from that folder, so this usually "
            f"means the file was moved or deleted after the graph was "
            f"saved. Re-pick it on the loader, or set distill_lora='none'.\n\n"
            f"If you have none yet, a known-good one for this backbone:\n"
            f"  hf download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v "
            f"loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors "
            f"--local-dir ComfyUI/models/loras/lightx2v"
        )
    _warn_if_lora_looks_mismatched(lora_kind)

    from safetensors.torch import load_file as _load_safetensors
    from .lora_merge import merge_lora_into_base

    logger.info("UniVidX: loading step-distill LoRA %r from %s", lora_kind, path)
    sd = _load_safetensors(path)
    try:
        report = merge_lora_into_base(
            model.pipe.dit, sd,
            strength=strength,
            strip_prefix="diffusion_model.",
        )
    finally:
        del sd
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "UniVidX: step-distill merge complete (%s, strength=%.2f): "
        "%d Linears merged, %d biases patched, %d weights patched, "
        "%d unmatched, %d skipped",
        lora_kind, strength,
        report["merged"], report["biases_patched"],
        report["weights_patched"], report["unmatched"], report["skipped"],
    )
    if report["merged"] == 0 and report["weights_patched"] == 0:
        logger.warning(
            "UniVidX: step-distill merge applied 0 deltas — the LoRA's "
            "key structure may not match UniVidX's WanModel layout."
        )


# ─── FP8 substitution (pre-quantized weights or runtime quantize) ──────────


def resolve_fp8_weights_path() -> Optional[str]:
    """Probe for a Kijai pre-quantized Wan2.1-T2V-14B FP8 weights file
    with PER-TENSOR SCALES (the `_scaled` filename suffix).

    Returns the path if a suitable file is present; None otherwise.
    Callers fall back to runtime-quantize when None is returned.
    """
    candidates = [
        Path(folder_paths.models_dir) / "diffusion_models"
        / "Wan2_1-T2V-14B_fp8_e4m3fn_scaled.safetensors",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def apply_fp8_substitution(model, variant: str) -> None:
    """Convert the DiT's nn.Linear modules to FP8 e4m3fn.

    Two implementations, picked at runtime:
      1. File-based — if a Kijai `_scaled` Wan2.1-T2V-14B FP8 safetensors
         is present, load FP8 weights + per-tensor scales directly.
         Faster cold-load (no quantize pass), identical numerics to (2).
      2. Runtime quantize (default) — compute per-tensor absmax scales
         from the BF16 cold-load weights and cast to FP8 in place.

    Both paths land at the same end state: ~400 FP8Linear modules
    inside the DiT, LoRA adapters preserved at BF16, ~14 GB DiT VRAM
    (vs ~28 GB BF16).

    Preferred over both, when VRAM management has already wrapped the
    DiT: convert those wrappers in place instead of replacing them, so
    the layers stay FP8 *and* keep block swapping. See
    `_apply_fp8_keeping_block_swap`. With VRAM management on (always, via
    the loader) the two paths above are fallbacks only.
    """
    if _apply_fp8_keeping_block_swap(model):
        return
    fp8_path = resolve_fp8_weights_path()
    if fp8_path is not None:
        _apply_fp8_substitution_from_file(model, fp8_path)
    else:
        _apply_fp8_substitution_runtime_quantize(model)


def _apply_fp8_keeping_block_swap(model) -> bool:
    """Quantize the DiT while preserving block swapping.

    Returns True if this path handled it, False to fall through to the
    substitution paths (which pin the quantized layers to the GPU).

    Background: `enable_vram_management` wraps each nn.Linear in an
    `AutoWrappedLinear` that owns the CPU<->GPU streaming. The older FP8
    path REPLACED those wrappers with plain `FP8Linear` modules, which
    have no offload/onload — so FP8 halved the DiT's bytes and
    simultaneously pinned every one of them resident, including through
    the text-encode and VAE-decode phases where the DiT isn't needed.

    Converting the wrapper in place keeps its device/dtype configuration
    and its state machine, and stores the weight as fp8 in *every* state
    — so the GPU-resident copy is 1 byte/param as well, not just the
    offloaded one.

    The wrapper class is taken from the pipeline's own namespace. Upstream
    wraps with its VENDORED `src.vram_management.AutoWrappedLinear`, not
    DiffSynth's; an earlier version of this path looked for
    `diffsynth.core.vram.layers.AutoWrappedLinear` (a package this wrapper
    doesn't even install), found no instances, and silently fell back to
    pinning every layer.
    """
    dit = getattr(getattr(model, "pipe", None), "dit", None)
    wrapper_cls = pipeline_globals(model).get("AutoWrappedLinear")
    if dit is None or wrapper_cls is None:
        return False

    from .fp8_loader import quantize_wrapped_dit_inplace

    report = quantize_wrapped_dit_inplace(dit, wrapper_cls)
    n = report["linears_quantized"]
    if n == 0:
        # No wrappers found — VRAM management is off. The substitution
        # paths still give the memory win, just without streaming.
        return False

    logger.info(
        "UniVidX: FP8 applied to %d Linears with block swapping intact "
        "(weights stay fp8 on GPU and can still stream to CPU). "
        "LoRA adapters and biases remain bf16.%s",
        n,
        (f" {report['skipped_unwrapped']} unwrapped Linears left at bf16."
         if report["skipped_unwrapped"] else ""),
    )
    return True


def _apply_fp8_substitution_runtime_quantize(model) -> None:
    """Compute per-tensor absmax scales from BF16 weights and cast to
    FP8 in place. The default FP8 path on machines without a Kijai
    `_scaled` Wan2.1 safetensors."""
    from .fp8_loader import quantize_dit_inplace

    logger.info("UniVidX: quantizing DiT to FP8 e4m3fn (per-tensor absmax scaling)")
    report = quantize_dit_inplace(model.pipe.dit)
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "UniVidX: FP8 substitution complete: %d Linears quantized "
        "(per-tensor scaled, runtime)",
        report["linears_quantized"],
    )
    if report["linears_quantized"] == 0:
        logger.warning(
            "UniVidX: FP8 substitution replaced 0 Linears — the DiT "
            "may not contain any nn.Linear modules at the expected "
            "positions. Sampling will run at full BF16."
        )


def _apply_fp8_substitution_from_file(model, fp8_path: str) -> None:
    """Stream-load a Kijai `_scaled` FP8 safetensors and replace DiT
    Linears with FP8Linear carrying pre-baked weight + scales.

    Forward-compat path for when a `_scaled` Wan2.1-T2V-14B variant
    lands upstream (doesn't exist as of upstream's current state)."""
    from safetensors.torch import load_file as _load_safetensors
    from .fp8_loader import load_fp8_state_dict_into

    logger.info("UniVidX: loading Kijai _scaled FP8 base from %s", fp8_path)
    fp8_sd = _load_safetensors(fp8_path)
    try:
        report = load_fp8_state_dict_into(model.pipe.dit, fp8_sd)
    finally:
        del fp8_sd
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "UniVidX: FP8 substitution complete: %d Linears -> FP8Linear "
        "(file-based, %d aux loaded, %d unmatched)",
        report["fp8_linears_replaced"],
        report["aux_keys_loaded"],
        len(report["unmatched_keys"]),
    )


# ─── torch.compile ──────────────────────────────────────────────────────────


def compile_dit(model) -> int:
    """torch.compile each of the DiT's transformer blocks, in place.

    Per block, not the whole DiT, because upstream never calls the DiT's
    own `forward`: `model_fn_wan_video` reaches in for `dit.blocks[i]`,
    `dit.head`, `dit.patchify` and friends directly. Compiling the DiT
    module therefore compiled a `forward` nothing runs — the previous
    `pipe.dit = torch.compile(dit)` produced zero compiled graphs, costing
    a wrapper layer and buying nothing. The blocks are where the time goes
    anyway (40 of them, every step, every window).

    `Module.compile()` rather than `torch.compile(module)`: it compiles in
    place, so the block keeps its identity and its state-dict keys, and
    DiffSynth's offload/onload walk over `dit.modules()` still reaches every
    wrapped layer.

    Default mode, not "reduce-overhead". That mode means CUDA graphs, which
    record fixed tensor addresses — but block swapping moves weights
    between CPU and GPU every run, so recorded graphs would be invalidated
    (or re-recorded) constantly. `dynamic=True` keeps one compile valid
    across resolutions and window lengths.

    Returns the number of blocks compiled.
    """
    dit = getattr(getattr(model, "pipe", None), "dit", None)
    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "Cannot find model.pipe.dit.blocks to compile — UniVidX pipeline "
            "structure may have changed."
        )
    for block in blocks:
        block.compile(dynamic=True, fullgraph=False)
    return len(blocks)


# ─── SDPA pollution restoration ─────────────────────────────────────────────


def restore_native_sdpa_if_polluted() -> bool:
    """Defensive: undo any other custom node's global
    F.scaled_dot_product_attention pollution.

    e.g. ComfyUI-3D-Pack's Stable3DGen does
    `F.scaled_dot_product_attention = sageattn` at import time when
    sageattention is installed. That hostile global swap breaks
    other nodes whose SDPA calls use head_dims sage doesn't support
    (UniVidX's VAE has 1-head SDPA where head_dim=channel_count,
    hitting 384 in some blocks — sage raises Unsupported head_dim).

    Returns True if a restore was done, False if no pollution detected.
    """
    import torch.nn.functional as F
    native = torch._C._nn.scaled_dot_product_attention
    if F.scaled_dot_product_attention is native:
        return False
    F.scaled_dot_product_attention = native
    return True


# ─── SageAttention dispatcher ───────────────────────────────────────────────


def _warn_attention_fallback(backend: str, head_dim: int, exc: BaseException) -> None:
    """Log once per (backend, head_dim, exc-type) so we don't spam the
    console at every sampler step but still surface real failures."""
    key = (backend, head_dim, type(exc).__name__)
    if key in _attention_fallback_warned:
        return
    _attention_fallback_warned.add(key)
    logger.warning(
        "UniVidX: %s attention failed for head_dim=%d, falling back. "
        "Cause: %s: %s",
        backend, head_dim, type(exc).__name__, exc,
    )


ATTENTION_BACKENDS = ("auto", "sdpa", "flash_attn_2", "sage")

# UniVidX is a Wan2.1-T2V-14B fine-tune, so this is the only TeaCache
# coefficient set that applies. Anything else raises in TeaCache.__init__.
TEACACHE_MODEL_ID = "Wan2.1-T2V-14B"


def pipeline_globals(model_or_cls) -> dict:
    """The module namespace of the loaded pipeline class. Accepts the
    class itself or an instance of it.

    We cannot just `import src.pipelines.univid_intrinsic` at sampling
    time: runtime._own_the_src_name() hands the top-level `src` name back
    to whichever other custom node owns it the moment the import finishes,
    so the module is no longer in sys.modules. The class object still
    references its own globals, and that is where `TeaCache`,
    `TemporalTiler_BCTHW`, `tqdm`, `WanModel` and `AutoWrappedLinear` live
    (univid_intrinsic.py:1-21).

    The same goes for every module upstream imported: reach them through
    an object they define, never through `sys.modules`.
    """
    cls = model_or_cls if isinstance(model_or_cls, type) else type(model_or_cls)
    try:
        return cls.__init__.__globals__
    except AttributeError:
        return {}


def _dit_module_globals(model_or_cls) -> Optional[dict]:
    """Globals of the DiT module the pipeline actually uses
    (`src.models.wan_video_dit_{intrinsic,alpha}`), via its `WanModel`."""
    wan_model = pipeline_globals(model_or_cls).get("WanModel")
    init = getattr(wan_model, "__init__", None)
    return getattr(init, "__globals__", None)


def instrument_vae_encode(model_instance, *, tiled: bool,
                          tile_size, tile_stride) -> bool:
    """Time the conditioning VAE encode, and give it the tiling settings
    upstream forgets to pass. Call once per sampler run: the wrapper is
    installed on the first call, and every call refreshes the settings it
    applies.

    Two separate problems with `WanVideoVAE.encode`
    (wan_video_vae.py:1218) as upstream calls it for conditioning:

    1. The inference branch calls
       `pipe.vae.encode(video, device=...)` with NO tiling arguments
       (univid_intrinsic.py:913), so it takes the `tiled=False` default —
       even though the training branch six hundred lines later
       (:1173) passes `tiled=tile_size=tile_stride=` properly. The
       sampler's `tiled` widget therefore reaches the DECODE but never
       the conditioning ENCODE, and a full-frame encode at 880x1280 is
       both slower and far heavier than a tiled one.

    2. Its first statement is `videos = [video.to("cpu") for video in
       videos]` — the batch is bounced to CPU and back regardless.

    We wrap rather than patch upstream so the vendored tree stays clean.
    The timer is always on: this sits in a stretch between "sampling
    started" and the first progress tick that was ~9 minutes on a real
    job with no visibility into where it went.

    The settings live on the wrapper, not in its closure. They used to be
    captured by the first call, so toggling the sampler's `tiled` widget
    on a cached model silently kept the first run's encode tiling.
    """
    pipe = getattr(model_instance, "pipe", None)
    vae = getattr(pipe, "vae", None) if pipe is not None else None
    if vae is None or not hasattr(vae, "encode"):
        return False
    settings = {"tiled": tiled, "tile_size": tile_size,
                "tile_stride": tile_stride}
    current = vae.encode
    if getattr(current, "_bat_encode_wrapper", False):
        current._bat_settings.update(settings)
        return True
    inner = current

    def encode(videos, device, **kwargs):
        # Only fill in what the caller omitted — the training path passes
        # its own and must keep it.
        for key, value in encode._bat_settings.items():
            kwargs.setdefault(key, value)
        t0 = time.perf_counter()
        try:
            return inner(videos, device, **kwargs)
        finally:
            logger.info(
                "UniVidX: conditioning VAE encode took %.1f s (tiled=%s). "
                "This runs before the first sampler step.",
                time.perf_counter() - t0, kwargs["tiled"],
            )

    encode._bat_encode_wrapper = True
    encode._bat_settings = settings
    vae.encode = encode
    return True


# ─── Progress + cancellation ────────────────────────────────────────────────
# upstream's denoising loop is `for ... in tqdm(timesteps)` inside
# pipe.__call__, so ComfyUI never hears about individual steps: the node
# showed no progress for the whole render, and Cancel did nothing until
# pipe() returned — hours, on a long clip. We swap the `tqdm` name in the
# pipeline module's globals for a wrapper that reports each step to
# whichever callback the sampler node has registered for this run.

_active_step_callback = None


def _throw_if_interrupted() -> None:
    """Honour ComfyUI's Cancel button. No-op outside ComfyUI."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


def install_progress_hook(model_instance) -> bool:
    """Route upstream's per-step tqdm through `sampler_progress()`.
    Idempotent; a no-op for runs without an active callback."""
    g = pipeline_globals(model_instance)
    original = g.get("tqdm")
    if original is None:
        logger.warning("UniVidX: no tqdm in the pipeline namespace — "
                       "per-step progress unavailable.")
        return False
    if getattr(original, "_bat_progress_hook", False):
        return True

    def _reporting(iterator, callback):
        for item in iterator:
            _throw_if_interrupted()
            yield item
            callback()

    def tqdm(*args, **kwargs):
        iterator = original(*args, **kwargs)
        callback = _active_step_callback
        return iterator if callback is None else _reporting(iterator, callback)

    tqdm._bat_progress_hook = True
    g["tqdm"] = tqdm
    return True


class sampler_progress:
    """Context manager: report each denoising step of the pipe() call it
    wraps to `callback()` (e.g. a ComfyUI ProgressBar update), and check
    for Cancel before every step. Requires `install_progress_hook`."""

    def __init__(self, callback):
        self._callback = callback

    def __enter__(self):
        global _active_step_callback
        _active_step_callback = self._callback
        return self

    def __exit__(self, *exc):
        global _active_step_callback
        _active_step_callback = None
        return False


# ─── Temporal context windows ───────────────────────────────────────────────


def context_window_starts(total: int, size: int, stride: int) -> list[int]:
    """Start indices of the temporal windows covering `total` latent frames.

    Every window is exactly `size` long. Windows advance by `stride`, and
    when that doesn't land the last one on the end of the clip, a final
    window is placed flush with the end instead. Upstream's tiler let the
    last window run short — sometimes only a few latent frames, i.e. with
    almost no temporal context — which made the tail of long renders
    visibly weaker than the rest. The shifted last window simply overlaps
    its neighbour by more than `size - stride`.
    """
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, stride))
    if starts[-1] + size < total:
        starts.append(total - size)
    return starts


def _window_blend_mask(length: int, left: int, right: int) -> torch.Tensor:
    """Per-frame weight for one window: linear ramps across the frames it
    shares with its left/right neighbours, 1 elsewhere, never 0.

    A ramp over an overlap of `w` frames runs (1..w)/(w+1) upward on the
    incoming window and mirrored on the outgoing one, so the two sum to
    exactly 1 — a true crossfade. Where two ramps meet (overlap larger than
    half a window) the smaller weight wins.
    """
    mask = torch.ones(length, dtype=torch.float32)
    if left > 0:
        ramp = torch.arange(1, left + 1, dtype=torch.float32) / (left + 1)
        mask[:left] = torch.minimum(mask[:left], ramp)
    if right > 0:
        ramp = torch.arange(right, 0, -1, dtype=torch.float32) / (right + 1)
        mask[length - right:] = torch.minimum(mask[length - right:], ramp)
    return mask


def run_context_windows(model_fn, *, size: int, stride: int,
                        model_kwargs: dict,
                        tensor_names=("latents", "y")) -> torch.Tensor:
    """Run `model_fn` once per temporal window and blend the predictions.

    Replaces upstream's `TemporalTiler_BCTHW.run` (util.py:958). Same
    contract — slice every named (B, C, T, H, W) tensor along T, call
    `model_fn(**model_kwargs)` per window, blend — with three changes:
    full-length windows (`context_window_starts`), crossfade ramps sized to
    each actual overlap, and float32 accumulation (upstream accumulated in
    the latents' bf16, rounding every overlapped frame twice).
    """
    tensors = {n: model_kwargs[n] for n in tensor_names
               if model_kwargs.get(n) is not None}
    reference = tensors["latents"]
    total = reference.shape[2]
    starts = context_window_starts(total, size, stride)

    value = None
    weight = torch.zeros(total, dtype=torch.float32, device=reference.device)
    for i, start in enumerate(starts):
        _throw_if_interrupted()
        end = start + size
        window_kwargs = dict(model_kwargs)
        for name, tensor in tensors.items():
            window_kwargs[name] = tensor[:, :, start:end]
        out = model_fn(**window_kwargs)
        if value is None:
            value = torch.zeros(
                (*out.shape[:2], total, *out.shape[3:]),
                dtype=torch.float32, device=out.device,
            )
        left = starts[i - 1] + size - start if i > 0 else 0
        right = end - starts[i + 1] if i + 1 < len(starts) else 0
        mask = _window_blend_mask(end - start, left, right).to(out.device)
        value[:, :, start:end] += out.float() * mask.view(1, 1, -1, 1, 1)
        weight[start:end] += mask.to(weight.device)
    value /= weight.to(value.device).view(1, 1, -1, 1, 1)
    return value.to(reference.dtype)


def install_sliding_window_support(model_instance) -> bool:
    """Make temporal context windowing work. Idempotent.

    Upstream ships a sliding-window path that is unrunnable as shipped:

    1. `model_fn_wan_video` in univid_intrinsic.py:1220-1235 hand-builds
       `model_kwargs` for the recursive call and OMITS `training_mode`
       (plus cfg_merge and both gradient-checkpointing flags). A None
       training_mode then falls off the end of the mode dispatch in
       wan_video_dit_intrinsic.py:302-304, hits `print("wrong!!!!")` and
       raises UnboundLocalError. univid_alpha.py:1162-1184 builds the
       equivalent dict correctly, which is how we know this is an
       oversight — the intrinsic window path has simply never been run.

       We fix it by intercepting one level UP rather than re-copying
       upstream's dict: pop the two window parameters and forward
       everything else through untouched. Nothing can be omitted because
       nothing is enumerated, so the next upstream parameter to appear is
       carried automatically.

    2. Its tiler (`TemporalTiler_BCTHW`, util.py:940) crashes on a zero
       blend width, lets the final window run short, and accumulates in
       bf16. `run_context_windows` replaces it.

    Returns True if the patch is installed.
    """
    pipe = getattr(model_instance, "pipe", None)
    if pipe is None:
        logger.warning("UniVidX: no .pipe — cannot enable context windowing.")
        return False

    original = getattr(pipe, "model_fn", None)
    if original is None:
        logger.warning("UniVidX: pipe has no model_fn — cannot enable windowing.")
        return False
    if getattr(original, "_bat_window_patch", False):
        return True

    def model_fn(**kwargs):
        size = kwargs.pop("sliding_window_size", None)
        stride = kwargs.pop("sliding_window_stride", None)
        latents = kwargs.get("latents")

        if (not size or not stride or stride >= size
                or latents is None or latents.dim() != 5
                or latents.shape[2] <= size):
            # Single pass. Note the window params are already popped, so
            # upstream's own (broken) tiling branch can never trigger.
            return original(**kwargs)

        # "latents" carries the four modality streams AND the injected
        # conditioning rows, so slicing it windows the conditioning too.
        # `y` is None for T2V and is skipped.
        return run_context_windows(
            original, size=int(size), stride=int(stride), model_kwargs=kwargs,
        )

    model_fn._bat_window_patch = True
    pipe.model_fn = model_fn
    logger.info(
        "UniVidX: context-window support installed (upstream's tiler is "
        "unrunnable as shipped; see install_sliding_window_support).",
    )
    return True


def build_tea_cache(model_instance, *, num_inference_steps: int,
                    rel_l1_thresh: float):
    """Construct a TeaCache for this run, or None if unavailable.

    TeaCache skips the whole 40-block DiT forward on steps where the
    modulated timestep embedding has barely moved, adding a cached
    residual instead (util.py:889-938). It is fully implemented upstream
    but never wired: `WanVideoUnit_TeaCache` is absent from the pipeline's
    `units` list, so `tea_cache` is always None and the two
    `tea_cache_*` kwargs that __call__ pops are silently discarded.
    Passing an instance directly through pipe(**kwargs) reaches
    model_fn's `tea_cache` parameter, which is the shortest path in.

    Two caveats the caller must respect:

      * The instance is STATEFUL and single-use — build a fresh one per
        sampler run, never cache it alongside the model.
      * It must not be shared across a CFG pair. With cfg_scale != 1 the
        pipeline calls model_fn twice per step (univid_intrinsic.py:670,
        675); one shared instance would advance `self.step` twice and
        blend the positive and negative residuals. The sampler refuses to
        enable TeaCache in that case.

    Accuracy caveat worth knowing: the polynomial coefficients were fit on
    stock Wan2.1, and in conditioned modes some rows of `t_mod` are pinned
    to a constant timestep, which dilutes the relative-L1 signal and biases
    it toward skipping too eagerly. Start with a small threshold.
    """
    cls = pipeline_globals(model_instance).get("TeaCache")
    if cls is None:
        logger.warning(
            "UniVidX: TeaCache class not found in the pipeline namespace; "
            "caching disabled for this run.",
        )
        return None
    try:
        inst = cls(int(num_inference_steps), float(rel_l1_thresh),
                   TEACACHE_MODEL_ID)
    except Exception as exc:
        logger.warning(
            "UniVidX: could not construct TeaCache (%s: %s); "
            "continuing without it.", type(exc).__name__, exc,
        )
        return None

    # Count skips so the sampler can report the speedup that actually
    # happened. The mapping from threshold to skip count is
    # clip-dependent and, on this model, skewed by the pinned-timestep
    # rows — so a measured number beats any documented estimate.
    # `check()` returns True when the step is being skipped.
    inst._bat_skipped = 0
    inst._bat_checks = 0
    _inner_check = inst.check

    def _counting_check(dit, x, t_mod):
        skipped = _inner_check(dit, x, t_mod)
        inst._bat_checks += 1
        if skipped:
            inst._bat_skipped += 1
        return skipped

    inst.check = _counting_check
    return inst


def tea_cache_report(tea_cache) -> Optional[str]:
    """One-line summary of what TeaCache actually did, or None.

    Reports DiT passes saved rather than wall-clock: the skipped work is
    the 40-block forward, while VAE encode/decode and text encoding are
    unaffected, so end-to-end gain is always somewhat less than this.
    """
    if tea_cache is None:
        return None
    checks = getattr(tea_cache, "_bat_checks", 0)
    skipped = getattr(tea_cache, "_bat_skipped", 0)
    if not checks:
        return None
    computed = checks - skipped
    ratio = (checks / computed) if computed else float("inf")
    return (
        f"TeaCache: computed {computed}/{checks} DiT passes "
        f"(skipped {skipped}, {skipped / checks:.0%}) — "
        f"{ratio:.2f}x fewer DiT passes. End-to-end gain is lower; VAE "
        f"and text encoding don't skip."
    )

def _sage_kernel():
    """SageAttention kernel over (B, N, S, D), or None if unavailable."""
    try:
        from sageattention import sageattn
    except ImportError:
        return None

    def _k(q, k, v):
        return sageattn(q, k, v, tensor_layout="HND")
    return _k


def _fa2_kernel():
    """FlashAttention-2 kernel over (B, N, S, D), or None if unavailable.

    FA2 wants (B, S, N, D); `.transpose(1, 2)` keeps the head_dim axis
    contiguous, which is all it requires.
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return None

    def _k(q, k, v):
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)
    return _k


def _resolve_attention_chain(requested: str):
    """Ordered [(name, kernel)] to try, best first. Unavailable backends
    are dropped; SDPA is always the last resort and always present.

    `auto` deliberately does NOT include SageAttention. Measured on an
    RTX A5000 (sm_86) at the real Wan2.1-14B geometry (40 heads × 128 dim,
    Q seqlen 1560, folded K/V seqlen 6240, bf16):

        SDPA          14.4 ms      (reference)
        FA2           13.9 ms      rel err 0.25%
        Sage          16.7 ms      rel err 1.99%   <- slower AND less exact

    Sage's int8 quantisation costs more than it saves at this shape on
    sm_86, and this env has no `_qattn_sm90.so` anyway. It stays selectable
    because the farm card is a different architecture and may invert that
    result — but it has to be chosen deliberately, not inherited.

    Worth knowing before expecting much from any of these: torch's SDPA is
    already a fused FlashAttention kernel, not naive attention. Forcing
    SDPBackend.MATH on the same shape takes 152 ms — 13× slower — which is
    what "unoptimised" would actually look like. So switching backends is
    a few percent, not a multiple.
    """
    sage, fa2 = _sage_kernel(), _fa2_kernel()
    if requested == "sdpa":
        candidates = []
    elif requested == "flash_attn_2":
        candidates = [("flash_attn_2", fa2)]
    elif requested == "sage":
        candidates = [("sage", sage), ("flash_attn_2", fa2)]
    else:  # "auto" — FA2 first, no sage
        candidates = [("flash_attn_2", fa2)]
    chain = [(n, k) for n, k in candidates if k is not None]
    chain.append(("sdpa", None))  # None == use F.scaled_dot_product_attention
    return chain


def _attend(q, k, v, chain, head_dim):
    """Run attention on (B, N, S, D) tensors, walking `chain` until one
    succeeds. Warn-once per (backend, head_dim, exc-type)."""
    import torch.nn.functional as F

    # FA2/Sage are half-precision only; anything else goes straight to SDPA.
    if q.dtype not in (torch.float16, torch.bfloat16):
        return F.scaled_dot_product_attention(q, k, v)

    for name, kernel in chain:
        if kernel is None:
            break
        try:
            return kernel(q, k, v)
        except Exception as exc:
            _warn_attention_fallback(name, head_dim, exc)
    return F.scaled_dot_product_attention(q, k, v)


def _make_folding_attention(chain):
    """Build a `flash_attention` replacement for the intrinsic/alpha DiTs.

    These two modules use `drop_out` as a BRANCH SELECTOR, not a
    probability, and getting that wrong silently destroys the model:

      * `SelfAttention.forward` hard-sets `drop_out = 0`
        (wan_video_dit_intrinsic.py:147), so `torch.rand(1) < 0` is always
        False and the folded branch runs: K/V are reshaped to
        `1 n (b s) d` and repeated, i.e. every modality stream attends
        over ALL FOUR streams concatenated. That fold *is* UniVidX's
        cross-modal attention (CMSA) — there is no other mechanism.
      * `CrossAttention.forward` defaults `drop_out = 1`
        (:328), so `torch.rand(1) < 1` is always True and the per-stream
        branch runs — correct, since each stream attends its own text.

    So we reproduce the branch exactly and swap only the kernel. A naive
    dispatcher that always does `b n s d` for K/V (like the old
    force_sage_over_fa2 did) would remove cross-modal attention entirely
    and still produce plausible-looking output.

    `drop_out=None` (the unreachable `k_img` call at :341, which would
    TypeError upstream) is treated as 0 → folded. Harmless either way:
    the image context is identical across the four streams, so softmax
    over four copies of it returns the same result.
    """
    from einops import rearrange

    def flash_attention(q, k, v, num_heads, compatibility_mode=False,
                        drop_out=None, **_kwargs):
        b = q.shape[0]
        head_dim = q.shape[-1] // num_heads
        per_stream = (drop_out is not None) and bool(torch.rand(1) < drop_out)

        q2 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        if per_stream:
            k2 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v2 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        else:
            # `.expand` rather than upstream's `.repeat`. All b rows of the
            # folded K/V are the SAME tensor, so materialising b copies is
            # pure waste. Measured at Wan2.1-14B geometry (40h x 128d,
            # S=1560, B=4, bf16) on sm_86:
            #     repeat  732.7 MiB peak, 21.7 ms
            #     expand  305.6 MiB peak, 12.2 ms   (-58% mem, 1.8x faster)
            # with bit-identical output (torch.equal). This is the single
            # biggest per-call win available here, and it compounds over
            # 40 blocks x N steps.
            k2 = rearrange(k, "b s (n d) -> 1 n (b s) d", n=num_heads).expand(b, -1, -1, -1)
            v2 = rearrange(v, "b s (n d) -> 1 n (b s) d", n=num_heads).expand(b, -1, -1, -1)

        use = [("sdpa", None)] if compatibility_mode else chain
        x = _attend(q2, k2, v2, use, head_dim)
        return rearrange(x, "b n s d -> b s (n d)", n=num_heads)

    return flash_attention


def install_attention_backend(requested: str, pipeline_cls) -> str:
    """Replace UniVidX's hardcoded-SDPA attention with a real dispatcher.

    Upstream's `get_optimal_attention_mode()`
    (wan_video_dit_intrinsic.py:10-21) is a stub: every branch, including
    the `except`, returns "scaled_dot_product", and `FLASH_ATTN_3_AVAILABLE`
    is hardcoded False inside a `try` that cannot fail. All three branches
    of `flash_attention` therefore end in F.scaled_dot_product_attention,
    and K/V are folded with `.repeat` rather than `.expand`.

    `pipeline_cls` is the imported UniVidIntrinsic / UniVidAlpha class.
    The DiT module is reached through it — its `WanModel`'s globals — and
    patched there, because the DiT's attention looks `flash_attention` up
    as a module global at call time. This used to be done via
    `sys.modules["src.models.wan_video_dit_intrinsic"]`, but
    runtime._own_the_src_name() evicts every `src.*` module the moment the
    pipeline import finishes, so that lookup always found nothing: the
    backend setting and the `.expand` saving never took effect, and every
    load logged "no DiT module was patched".

    Returns the resolved backend name actually installed.
    """
    if requested not in ATTENTION_BACKENDS:
        logger.warning(
            "UniVidX: unknown attention_backend %r, falling back to 'auto'.",
            requested,
        )
        requested = "auto"

    chain = _resolve_attention_chain(requested)
    resolved = chain[0][0]

    if requested != "auto" and resolved != requested:
        logger.warning(
            "UniVidX: attention_backend=%r requested but unavailable; "
            "using %r instead.", requested, resolved,
        )

    dit_globals = _dit_module_globals(pipeline_cls)
    if dit_globals is None or "flash_attention" not in dit_globals:
        logger.warning(
            "UniVidX: could not find the DiT's flash_attention through %s — "
            "upstream's layout changed. Attention stays on stock SDPA.",
            getattr(pipeline_cls, "__name__", pipeline_cls),
        )
        return "sdpa"

    dit_globals["flash_attention"] = _make_folding_attention(chain)
    logger.info(
        "UniVidX: attention backend = %s (patched %s)",
        resolved, dit_globals.get("__name__", "DiT module"),
    )
    return resolved
