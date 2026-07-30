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

Each function is opt-in: nothing here runs unless the loader's
`advanced_settings` toggle is on AND the specific knob is set to
a non-default value. The defaults match upstream UniVidX's tuning
(bf16, no LoRA merge, no FP8, native SDPA), so users who don't touch
the advanced panel get the same behaviour as v1.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

import folder_paths

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")


# De-duplication map: warn once per (backend, head_dim, exc-type-name).
_attention_fallback_warned: set = set()


# ─── Step-distill LoRA pre-merge ────────────────────────────────────────────


def resolve_step_distill_path(lora_kind: str) -> Optional[str]:
    """Locate a step-distill LoRA file on disk by kind. Returns None
    if not found so the caller can surface a clear error.

    Search candidates per kind:
      lightx2v: models/loras/lightx2v/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
                (the layout `hf download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v
                 loras/...  --local-dir models/loras/lightx2v` produces)
                Falls back to a flat copy at models/loras/lightx2v/<filename>.
    """
    root = Path(folder_paths.models_dir).parent  # ComfyUI root, not models/
    if lora_kind == "lightx2v":
        candidates = [
            root / "models" / "loras" / "lightx2v" / "loras"
            / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            root / "models" / "loras" / "lightx2v"
            / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            root / "models" / "loras"
            / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        ]
    else:
        return None
    for p in candidates:
        if p.is_file():
            return str(p)
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
            f"Step-distill LoRA '{lora_kind}' not found on disk. "
            f"Expected layout:\n"
            f"  models/loras/lightx2v/loras/"
            f"Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors\n\n"
            f"Download via:\n"
            f"  hf download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v "
            f"loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors "
            f"--local-dir ComfyUI/models/loras/lightx2v\n\n"
            f"Or set distill_lora='none' to skip the merge."
        )
    from safetensors.torch import load_file as _load_safetensors
    from .lora_merge import merge_lora_into_base

    logger.info("UniVidX: loading step-distill LoRA (%s) from %s", lora_kind, path)
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


def force_sage_over_fa2() -> bool:
    """Replace DiffSynth's Wan DiT attention with a SageAttention-first
    wrapper that falls back to FA2 (then SDPA) per-call.

    Returns True if the wrapper was installed, False otherwise (e.g.
    sageattention not installed, DiffSynth wan module not loadable).
    """
    try:
        import sageattention  # noqa: F401
        from sageattention import sageattn
    except ImportError:
        return False
    try:
        from diffsynth.models import wan_video_dit
    except ImportError:
        return False
    if not getattr(wan_video_dit, "SAGE_ATTN_AVAILABLE", False):
        return False

    import torch.nn.functional as F
    from einops import rearrange

    # Capture FA2 reference if present so we can use it as the fallback.
    _FA2 = None
    if getattr(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", False):
        try:
            import flash_attn
            _FA2 = flash_attn.flash_attn_func
        except ImportError:
            _FA2 = None

    def _wrapped_flash_attention(q, k, v, num_heads, compatibility_mode=False,
                                  drop_out=None, **_kwargs):
        head_dim = q.shape[-1] // num_heads

        if compatibility_mode or head_dim > 256:
            q2 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k2 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v2 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            x = F.scaled_dot_product_attention(q2, k2, v2)
            return rearrange(x, "b n s d -> b s (n d)", n=num_heads)

        try:
            q2 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k2 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v2 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            x = sageattn(q2, k2, v2)
            return rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        except Exception as exc:
            _warn_attention_fallback("sage", head_dim, exc)

        if _FA2 is not None:
            try:
                q2 = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
                k2 = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
                v2 = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
                x = _FA2(q2, k2, v2)
                return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
            except Exception as exc:
                _warn_attention_fallback("FA2", head_dim, exc)

        q2 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k2 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v2 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q2, k2, v2)
        return rearrange(x, "b n s d -> b s (n d)", n=num_heads)

    wan_video_dit.flash_attention = _wrapped_flash_attention

    # Also patch UniVidX's vendored base wan_video_dit module — see
    # the original's comments for why we patch only the BASE variant
    # and leave the per-variant CMSA modules on native SDPA.
    import sys as _sys
    for mod_name in ("src.models.wan_video_dit",):
        mod = _sys.modules.get(mod_name)
        if mod is None or not hasattr(mod, "flash_attention"):
            continue
        mod.flash_attention = _wrapped_flash_attention
    return True


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
