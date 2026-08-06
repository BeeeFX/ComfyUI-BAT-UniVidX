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
import os
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
    """
    fp8_path = resolve_fp8_weights_path()
    if fp8_path is not None:
        _apply_fp8_substitution_from_file(model, fp8_path)
    else:
        _apply_fp8_substitution_runtime_quantize(model)


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


# ─── mmgp safetensors patch (read-only mmap, Windows-friendly) ─────────────


def patch_unividx_load_file_to_readonly() -> None:
    """Replace UniVidX's load_file binding so it uses mmgp.torch_load_file
    with writable_tensors=False. Idempotent. Call inside the
    unividx_cwd context after the pipeline classes have been imported.

    mmgp's default writable_tensors=True maps with ACCESS_COPY on
    Windows which forces a virtual-memory commit equal to the file
    size — six 9.84 GB DiT shards mmapped concurrently blow past the
    pagefile and produce [WinError 1455]. UniVidX only reads from the
    loaded tensors (state_dict.update + load_state_dict), so a
    read-only mmap is sufficient.
    """
    import sys as _sys
    try:
        import mmgp.safetensors2 as _mmgp_st
    except ImportError:
        return  # mmgp not installed — nothing to patch

    def _readonly_load_file(filename, device="cpu"):
        return _mmgp_st.torch_load_file(filename, device=device, writable_tensors=False)
    _readonly_load_file._unividx_readonly_patch = True  # type: ignore[attr-defined]

    for mod_name in ("src.pipelines.univid_intrinsic", "src.pipelines.univid_alpha"):
        mod = _sys.modules.get(mod_name)
        if mod is None:
            continue
        existing = getattr(mod, "load_file", None)
        if getattr(existing, "_unividx_readonly_patch", False):
            continue
        mod.load_file = _readonly_load_file


# ─── torch.compile ──────────────────────────────────────────────────────────


def compile_dit(model) -> None:
    """torch.compile the assembled DiT for ~20-30% per-step speedup on
    Blackwell/Ada. Mode 'reduce-overhead' uses CUDA Graphs where safe;
    dynamic=True keeps a single compile valid across resolution changes.
    """
    dit = getattr(getattr(model, "pipe", None), "dit", None)
    if dit is None:
        raise RuntimeError(
            "Cannot find model.pipe.dit to compile — UniVidX pipeline "
            "structure may have changed."
        )
    model.pipe.dit = torch.compile(
        dit, mode="reduce-overhead", dynamic=True, fullgraph=False,
    )


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


def pipeline_globals(model_instance) -> dict:
    """The module namespace of the loaded pipeline class.

    We cannot just `import src.pipelines.univid_intrinsic` at sampling
    time: runtime._own_the_src_name() hands the top-level `src` name back
    to whichever other custom node owns it the moment the import finishes,
    so the module is no longer in sys.modules. The class object still
    references its own globals, and that is where `TeaCache` and
    `TemporalTiler_BCTHW` live (univid_intrinsic.py:10).
    """
    try:
        return type(model_instance).__init__.__globals__
    except AttributeError:
        return {}


def instrument_vae_encode(model_instance, *, tiled: bool,
                          tile_size, tile_stride) -> bool:
    """Time the conditioning VAE encode, and give it the tiling settings
    upstream forgets to pass. Idempotent.

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
    """
    pipe = getattr(model_instance, "pipe", None)
    vae = getattr(pipe, "vae", None) if pipe is not None else None
    if vae is None or not hasattr(vae, "encode"):
        return False
    inner = vae.encode
    if getattr(inner, "_bat_encode_wrapper", False):
        return True

    def encode(videos, device, **kwargs):
        # Only fill in what the caller omitted — the training path passes
        # its own and must keep it.
        kwargs.setdefault("tiled", tiled)
        kwargs.setdefault("tile_size", tile_size)
        kwargs.setdefault("tile_stride", tile_stride)
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
    vae.encode = encode
    return True


def install_sliding_window_support(model_instance) -> bool:
    """Make upstream's temporal tiler actually usable. Idempotent.

    Two independent defects make the vendored sliding-window path
    unrunnable as shipped:

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

    2. `TemporalTiler_BCTHW.build_1d_mask` (util.py:944) lacks the
       `border_width == 0` guard the diffsynth original has. With
       stride == size the ramp width is 0, `x[-0:]` selects the whole
       tensor, and the assignment raises. Our planner never emits that,
       but a subclass with the guard makes it a no-op instead of a crash.

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

    tiler_cls = pipeline_globals(model_instance).get("TemporalTiler_BCTHW")
    if tiler_cls is None:
        logger.warning(
            "UniVidX: TemporalTiler_BCTHW not found in the pipeline "
            "namespace — context windowing unavailable.",
        )
        return False

    class _GuardedTiler(tiler_cls):
        def build_1d_mask(self, length, left_bound, right_bound, border_width):
            if border_width <= 0:
                return torch.ones((length,))
            return super().build_1d_mask(length, left_bound, right_bound,
                                         border_width)

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

        # tensor_names mirrors upstream: "latents" carries the four
        # modality streams AND the injected conditioning rows, so slicing
        # it windows the conditioning too. `run` drops names that are
        # None, which is how y=None (T2V) is handled.
        return _GuardedTiler().run(
            original, int(size), int(stride),
            latents.device, latents.dtype,
            model_kwargs=kwargs,
            tensor_names=["latents", "y"],
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

# Modules whose `flash_attention` folds K/V across the batch (the four
# modality streams) vs. those that don't. See _make_folding_attention.
_FOLDING_DIT_MODULES = (
    "src.models.wan_video_dit_intrinsic",
    "src.models.wan_video_dit_alpha",
)
_PLAIN_DIT_MODULES = ("src.models.wan_video_dit",)


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


def _make_plain_attention(chain):
    """Build a `flash_attention` replacement for the base vendored DiT,
    which takes no `drop_out` and never folds across the batch."""
    from einops import rearrange

    def flash_attention(q, k, v, num_heads, compatibility_mode=False, **_kwargs):
        head_dim = q.shape[-1] // num_heads
        q2 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k2 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v2 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        use = [("sdpa", None)] if compatibility_mode else chain
        x = _attend(q2, k2, v2, use, head_dim)
        return rearrange(x, "b n s d -> b s (n d)", n=num_heads)

    return flash_attention


def install_attention_backend(requested: str = "auto") -> str:
    """Replace UniVidX's hardcoded-SDPA attention with a real dispatcher.

    Upstream's `get_optimal_attention_mode()`
    (wan_video_dit_intrinsic.py:10-21) is a stub: every branch, including
    the `except`, returns "scaled_dot_product", and `FLASH_ATTN_3_AVAILABLE`
    is hardcoded False inside a `try` that cannot fail. All three branches
    of `flash_attention` therefore end in F.scaled_dot_product_attention.
    Since UniVidX folds four modality streams into one length-4S sequence,
    attention cost scales as 16·n·S² — so this is the hottest code in the
    model, pinned to the slowest backend.

    MUST be called AFTER the pipeline class has been imported: we patch
    module globals via sys.modules, and `src.models.wan_video_dit_intrinsic`
    only exists once `src.pipelines.univid_intrinsic` has been imported.
    (The previous `force_sage_over_fa2` ran before the import, so its
    vendored-module patch silently hit nothing.)

    Returns the resolved backend name actually installed.
    """
    import sys as _sys

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

    patched = []
    for mod_name, factory in (
        *((m, _make_folding_attention) for m in _FOLDING_DIT_MODULES),
        *((m, _make_plain_attention) for m in _PLAIN_DIT_MODULES),
    ):
        mod = _sys.modules.get(mod_name)
        if mod is None or not hasattr(mod, "flash_attention"):
            continue
        mod.flash_attention = factory(chain)
        patched.append(mod_name.rsplit(".", 1)[-1])

    # DiffSynth's own Wan DiT (used by the VAE/text-encoder side) has the
    # same signature as the base vendored module.
    try:
        from diffsynth.models import wan_video_dit as _ds_dit
        if hasattr(_ds_dit, "flash_attention"):
            _ds_dit.flash_attention = _make_plain_attention(chain)
            patched.append("diffsynth.wan_video_dit")
    except ImportError:
        pass

    if not patched:
        logger.warning(
            "UniVidX: attention backend %r resolved but no DiT module was "
            "patched — the model may not be imported yet. Attention stays "
            "on stock SDPA.", resolved,
        )
    else:
        logger.info(
            "UniVidX: attention backend = %s (patched: %s)",
            resolved, ", ".join(patched),
        )
    return resolved


# ─── Runtime FP8 quantize via mmgp/optimum-quanto ──────────────────────────


def quantize_dit_fp8(model, qtype: str = "qfloat8") -> None:
    """Post-quantize model.pipe.dit to FP8 via mmgp/optimum-quanto.

    qtype is an optimum-quanto qtype name — "qfloat8" (e4m3) or
    "qfloat8_e5m2". LoRA adapter layers are excluded so rank-32
    adapters stay in BF16 — quantizing them gives no memory win and
    tends to degrade adapter quality.

    EXPERIMENTAL: the quantize() pass over the LoRA-attached DiT can
    take 10+ min and has hung on some Wan2.1-14B + UniVidX adapter
    stack combinations in upstream's testing.
    """
    try:
        from mmgp import offload as _mmgp_offload  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "FP8 quantization (mmgp/quanto path) requested but mmgp is "
            "not installed. Run `pip install mmgp` or use the "
            "fp8_prequantized path instead."
        ) from e
    raise NotImplementedError(
        "BAT v2: runtime FP8 quantize via mmgp is intentionally not "
        "exposed yet — use `dit_weight_mode=fp8_prequantized` via "
        "apply_fp8_substitution() above. The runtime-quantize path "
        "(mmgp.offload.quantize) is the EXPERIMENTAL one that hangs "
        "on some adapter stacks. We'll add it back when the hang's "
        "been fixed upstream."
    )
