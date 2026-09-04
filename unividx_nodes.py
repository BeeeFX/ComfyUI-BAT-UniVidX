"""ComfyUI node classes for the BAT UniVidX wrapper.

Three nodes:
  BAT_UniVidXLoader   — picks variant + dtype + weight sources;
                        on first execute, auto-clones upstream UniVidX
                        and auto-resolves Wan2.1 + UniVidX checkpoint
                        paths (downloading from HF if not on disk).
                        Returns an opaque UNIVIDX_MODEL.

  BAT_UniVidXSample   — single dynamic-schema sampler, and the only
                        V3 (io.ComfyNode) node in this pack. The mode
                        dropdown is an io.DynamicCombo: each mode is an
                        Option declaring exactly the IMAGE sockets that
                        mode consumes, so only those are materialised
                        (you never see all 7 at once like in the
                        existing wrapper, and you can't wire one the
                        mode would ignore). Family/variant mismatch is
                        validated server-side in execute().

  BAT_UniVidXDecode   — single decode node, also V3. Branches on the
                        result's family internally, returns 4 IMAGE
                        outputs in canonical order — black placeholders
                        fill any slot the mode didn't produce. The output
                        COUNT is fixed at 4 because ComfyUI has no
                        dynamic-output mechanism; see the class docstring.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import torch

import folder_paths
import comfy.utils
from comfy_api.latest import io

from .unividx_wrapper_runtime import (
    auto_clone,
    auto_download,
    frame_constraints,
    help_docs,
    modes,
    runtime,
    tensor_io,
)

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")

# Wrapper directory (where this file lives). Upstream UniVidX will
# auto-clone into <wrapper_dir>/UniVidX/ on first use.
_WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))

# Opaque wrapper datatypes. `io.Custom` emits the same wire-level type
# string a V1 node would declare, so the V3 Sample node's sockets stay
# connectable to the V1 Loader and Decode nodes either side of it.
UniVidXModel = io.Custom("UNIVIDX_MODEL")
UniVidXResult = io.Custom("UNIVIDX_RESULT")
# Sample-time option payloads, produced by the two V1 options nodes near
# the bottom of this file. Plain dicts on the wire — same idea as
# BAT_CROP_INFO in the NodePack.
UniVidXContextOptions = io.Custom("UNIVIDX_CONTEXT_OPTIONS")
UniVidXCacheOptions = io.Custom("UNIVIDX_CACHE_OPTIONS")


# ─── BAT UniVidX Loader ─────────────────────────────────────────────────────


class BatUniVidXLoader:
    """One-shot loader for either UniVidX variant.

    Defaults aim for "just works" on a fresh ComfyUI install:
    auto-detect existing Wan2.1-T2V-14B weights anywhere under
    ComfyUI/models/; download if missing; bf16 dtype; auto-sized VRAM
    buffer for DiffSynth's offload; FP8 quantisation and the fastest
    available attention backend. Every widget is honoured — there is no
    gate that discards them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (
                    ["intrinsic", "alpha"],
                    {
                        "default": "intrinsic",
                        "tooltip": (
                            "Which UniVidX adapter to load.\n"
                            "  intrinsic — RGB ↔ Albedo / Irradiance / Normal\n"
                            "  alpha     — composite ↔ Matte / Foreground / Background"
                        ),
                    },
                ),
                "dtype": (
                    ["bf16", "fp16"],
                    {
                        "default": "bf16",
                        "tooltip": (
                            "Model working dtype. bf16 is what UniVidX was "
                            "trained at and what every benchmark assumes; fp16 "
                            "saves nothing on Ampere+ but may be needed on "
                            "older cards. Stick with bf16 unless you know why."
                        ),
                    },
                ),
                "wan21_source": (
                    ["auto-detect, fall back to download"]
                    + _list_known_wan_paths(),
                    {
                        "default": "auto-detect, fall back to download",
                        "tooltip": (
                            "Where to find the Wan2.1-T2V-14B base model "
                            "(~57 GB).\n\n"
                            "auto-detect: scan ComfyUI/models/ for an "
                            "existing copy (any of unividx/, "
                            "diffusion_models/, checkpoints/, unet/). If "
                            "found, reuse it — no duplicate download. If "
                            "not, download from HuggingFace into "
                            "ComfyUI/models/unividx/.\n\n"
                            "Explicit choices below are the auto-detected "
                            "paths the wrapper found on disk; pick one to "
                            "force a specific install when you have several.\n\n"
                            "Wan2.1-VACE-14B and Wan2.1-I2V-14B are NOT "
                            "compatible with UniVidX (different fine-tuning); "
                            "the loader rejects them."
                        ),
                    },
                ),
                # `advanced_settings` (BOOLEAN) used to live here. Removed:
                # it claimed to "expose the perf knobs below", but ComfyUI
                # shows optional widgets unconditionally, so it hid nothing
                # — its only effect was to DISCARD distill_lora,
                # distill_strength, compile_dit and attention_backend when
                # off. You could pick a LoRA, tick compile, and get neither,
                # with no warning. Every widget is now simply honoured.
            },
            "optional": {
                "vram_buffer_gb": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 80.0, "step": 0.5,
                        "tooltip": (
                            "GB of VRAM to keep FREE for activations / "
                            "attention / VAE decode. DiffSynth computes "
                            "`limit = total_vram - this` and offloads DiT "
                            "layers to CPU to stay under it. So HIGHER = "
                            "more headroom for activations (and more "
                            "CPU↔GPU streaming, so slower); LOWER = more "
                            "of the model stays resident (faster, until "
                            "you OOM).\n\n"
                            "This is the BLOCK SWAPPING control — layers "
                            "past the limit stream in on demand.\n\n"
                            "  0 = AUTO (recommended). Sized from the "
                            "card, the dtype and whether FP8 is on; the "
                            "loader logs what it chose. ~39 GB on a 95 GB "
                            "card.\n"
                            "  min 0.0 (= auto). The smallest useful "
                            "MANUAL value is ~8 — below that essentially "
                            "nothing streams and you are relying on the "
                            "model simply fitting.\n"
                            "  max 80.0, but the practical ceiling is "
                            "about (card VRAM − 16 GB). Reserve past that "
                            "and there is no room for the DiT itself, so "
                            "every layer thrashes and it crawls.\n"
                            "  typical manual: roughly a third of the "
                            "card.\n\n"
                            "Raise it when you OOM at SAMPLE time — that "
                            "means the model loaded but the sampler's "
                            "peak did not fit. Lower it for speed once a "
                            "job is known to fit.\n\n"
                            "The reservation must cover the sampler's "
                            "peak, which is large here: all four modality "
                            "streams attend as one sequence, so a single "
                            "step can ask for 20+ GB.\n\n"
                            "Works with FP8 — quantized layers stream too."
                        ),
                    },
                ),
                "fp8_quantize": (
                    ["qfloat8_prequantized", "none"],
                    {"default": "qfloat8_prequantized",
                     "tooltip": (
                         "FP8 e4m3fn DiT substitution. ON by default: bf16 "
                         "needs ~32 GB of VRAM for the DiT alone, which "
                         "doesn't leave room for activations on most cards. "
                         "Set to 'none' only on a card with headroom to "
                         "spare.\n\n"
                         "Halves DiT VRAM "
                         "(~28 GB → ~14 GB) and enables full residency "
                         "on a 32 GB card. Per-tensor absmax scaling, "
                         "runtime-quantized from the bf16 cold-load — "
                         "adds ~30 sec to the first load but no quality "
                         "loss vs bf16. LoRA adapter layers stay bf16. "
                         "If a Kijai-style `_scaled` Wan2.1 safetensors "
                         "is present under models/diffusion_models/, the "
                         "file-based path is auto-preferred (faster "
                         "cold-load, same numerics)."
                     )},
                ),
                "distill_lora": (
                    ["none"] + _list_lora_files(),
                    {"default": "none",
                     "tooltip": (
                         "Pre-merge a step-distill LoRA into the DiT's base "
                         "weights, so far fewer sampler steps produce a "
                         "usable image. Lists everything in "
                         "ComfyUI/models/loras/.\n\n"
                         "The filename must say T2V + 14B + Wan2.1 + "
                         "distill/lightx2v. This backbone is Wan2.1-T2V-14B, "
                         "so an I2V file is the easy mistake — Wan2.1 I2V "
                         "and T2V share most of the DiT, so it merges "
                         "without error and just quietly hurts quality. "
                         "Skip anything I2V, anything Wan2.2, and any "
                         "style/likeness LoRA (those do nothing for "
                         "speed).\n\n"
                         "Good: Wan21_T2V_14B_lightx2v_cfg_step_distill_"
                         "lora_rank32.safetensors. Then set "
                         "num_inference_steps=4 and cfg_scale=1.0 — that is "
                         "roughly 5x faster than the 20-step default.\n\n"
                         "This is the biggest speed lever available, and "
                         "the riskiest. The LoRA merges into the frozen "
                         "base underneath UniVidX's four per-modality "
                         "adapters — they do not collide — but those "
                         "adapters were trained against the ORIGINAL base, "
                         "so moving it can degrade albedo/normal quality in "
                         "ways that are obvious only on a side-by-side. "
                         "Compare before trusting it.\n\n"
                         "A LoRA for a different base model will merge "
                         "almost nothing; the log reports how many keys "
                         "matched, and warns loudly on zero."
                     )},
                ),
                "distill_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Merge strength for the distill LoRA. "
                                "1.0 = full merge as upstream "
                                "recommends. Lower = blend distill with "
                                "the base; higher = over-amplify "
                                "(usually worse)."},
                ),
                "compile_dit": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": (
                         "torch.compile(dit, mode='reduce-overhead', "
                         "dynamic=True). First sampler step pays a "
                         "60-120 sec graph capture cost; subsequent "
                         "steps are ~20-30% faster on Blackwell/Ada. "
                         "The compile cache is keyed by tensor shapes — "
                         "changing resolution or frame count between "
                         "runs forces a recompile."
                     )},
                ),
                # NOTE: this replaced a BOOLEAN `prefer_sage_attn` in the
                # same (last) widget slot, so saved workflows keep every
                # other widget aligned. Old templates hand a bool in here;
                # runtime._coerce_attention_backend maps False→auto,
                # True→sage so they keep executing.
                "attention_backend": (
                    ["auto", "sdpa", "flash_attn_2", "sage"],
                    {"default": "auto",
                     "tooltip": (
                         "Which attention kernel the DiT uses.\n\n"
                         "Expect a few percent, not a multiple. UniVidX "
                         "hardcodes SDPA, but torch's SDPA is ALREADY a "
                         "fused FlashAttention kernel — forcing the naive "
                         "math path on the same shape is 13× slower, so "
                         "there is no big win sitting here. Measured on "
                         "sm_86 at Wan2.1-14B geometry: SDPA 14.4 ms, "
                         "FA2 13.9 ms, Sage 16.7 ms.\n\n"
                         "auto (recommended): FlashAttention-2 if "
                         "importable, else SDPA.\n"
                         "sdpa: stock PyTorch — the pre-existing behaviour.\n"
                         "flash_attn_2: force FA2.\n"
                         "sage: force SageAttention (int8; measured SLOWER "
                         "and less exact on sm_86, and this env has no sm90 "
                         "kernel — but the farm card may differ, so it's "
                         "here to be measured).\n\n"
                         "All backends agree with SDPA to <2% on the "
                         "cross-modality attention. A VISIBLE difference is "
                         "a bug, not a tradeoff. Resolved backend is logged."
                     )},
                ),
            },
        }

    RETURN_TYPES = ("UNIVIDX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "BAT/UniVidX"

    def load(
        self,
        variant: str,
        dtype: str,
        wan21_source: str,
        # Accepted but ignored: the widget is gone (see INPUT_TYPES), and
        # this keeps a stale caller from raising TypeError.
        advanced_settings: bool = False,
        vram_buffer_gb: float = 0.0,  # 0 = auto (see resolve below)
        fp8_quantize: str = "qfloat8_prequantized",
        distill_lora: str = "none",
        distill_strength: float = 1.0,
        compile_dit: bool = False,
        # Was a BOOLEAN `prefer_sage_attn`; old saved workflows still send
        # one. runtime._coerce_attention_backend handles the bool form.
        attention_backend="auto",
    ):
        # Stage 1: ensure upstream UniVidX is cloned + patched.
        upstream_dir = auto_clone.auto_clone(_WRAPPER_DIR)

        # Stage 2: resolve weight paths. "auto-detect, fall back to
        # download" maps to source="auto"; explicit dropdown choices
        # are passed through as-is.
        if wan21_source == "auto-detect, fall back to download":
            wan21_source_arg = "auto"
        else:
            wan21_source_arg = wan21_source

        wan21_path = auto_download.ensure_wan21_t2v_14b(
            folder_paths.models_dir, source=wan21_source_arg,
        )
        ckpt_path = auto_download.ensure_unividx_checkpoint(
            folder_paths.models_dir, variant=variant, source="auto",
        )

        # Stage 3: symlink the resolved paths into the locations
        # upstream's pipeline hardcodes relative to its own root.
        auto_download.ensure_upstream_symlinks(
            upstream_dir,
            wan21_path=wan21_path,
            intrinsic_ckpt=ckpt_path if variant == "intrinsic" else None,
            alpha_ckpt=ckpt_path if variant == "alpha" else None,
        )

        # Stage 4: load (or fetch from cache).
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        # Every widget is honoured. FP8 defaults ON at the widget itself
        # (a bf16 Wan2.1-14B DiT is ~28-32 GB and leaves no room for
        # activations), so "I changed nothing" still gets a configuration
        # that fits in VRAM — without a gate that throws away the settings
        # of anyone who did change something.
        advanced = {
            "fp8_quantize": fp8_quantize,
            "distill_lora": distill_lora,
            "distill_strength": float(distill_strength),
            "compile_dit": compile_dit,
            "attention_backend": attention_backend,
        }

        # vram_buffer_gb == 0 means "auto": size the activation reservation
        # from the card's real VRAM and whether the DiT will be FP8. A
        # fixed default can't work here — the same number that's right on
        # a 32 GB card starves the sampler on a 96 GB one (see
        # runtime.auto_vram_buffer_gb).
        resolved_vram_buffer = float(vram_buffer_gb)
        if resolved_vram_buffer <= 0.0:
            resolved_vram_buffer = runtime.auto_vram_buffer_gb(
                fp8=advanced.get("fp8_quantize", "none") != "none",
            )
            total = runtime.total_vram_gb()
            logger.info(
                "UniVidX: vram_buffer_gb=auto -> %.1f GB reserved for "
                "activations (card total: %s)",
                resolved_vram_buffer,
                f"{total:.1f} GB" if total is not None else "CPU-only",
            )
        model = runtime.load_model(
            variant=variant,
            upstream_dir=upstream_dir,
            wan21_path=wan21_path,
            ckpt_path=ckpt_path,
            dtype=torch_dtype,
            vram_buffer_gb=resolved_vram_buffer,
            advanced=advanced,
        )
        # UNIVIDX_MODEL is an opaque tuple (model, variant, upstream_dir).
        # The sampler needs all three: model to call, variant for family
        # checks, upstream_dir for the chdir guard.
        return ((model, variant, upstream_dir),)


# Filename fragments that suggest a LoRA is a step-distill for this
# backbone. Used only to ORDER the dropdown, never to filter it.
_DISTILL_HINTS = ("distill", "lightx2v", "causvid", "accvid", "turbo")
_WAN_HINTS = ("wan21", "wan2.1", "wan2_1", "wan_2", "t2v")


def _list_lora_files() -> list[str]:
    """Everything under models/loras/, most-likely-relevant first.

    Populates the `distill_lora` dropdown. Nothing is filtered out — any
    Wan2.1-T2V-14B step-distill LoRA works, new ones keep appearing, and
    hiding a file someone just downloaded would be worse than a long list.
    But a flat alphabetical listing buries the five Wan LoRAs under 80
    FLUX ones, so sort into three tiers: step-distill candidates, other
    Wan LoRAs, then everything else. Each tier alphabetical.

    Picking an incompatible LoRA is not silent: the merge reports how many
    keys matched and warns loudly on zero.

    Not cached — folder_paths already caches the directory listing, so
    going through it means a LoRA added while the server is running shows
    up on the next graph refresh.
    """
    try:
        names = list(folder_paths.get_filename_list("loras"))
    except Exception as exc:  # no loras dir configured, etc.
        logger.debug("UniVidX: could not list loras (%s)", exc)
        return []

    def tier(name: str) -> int:
        low = name.lower()
        if any(h in low for h in _DISTILL_HINTS):
            return 0
        if any(h in low for h in _WAN_HINTS):
            return 1
        return 2

    return sorted(names, key=lambda n: (tier(n), n.lower()))


_WAN_PATH_CACHE: Optional[list[str]] = None


def _list_known_wan_paths() -> list[str]:
    """Scan ComfyUI/models/ for known Wan2.1-T2V-14B installs.
    Used to populate the explicit-choice dropdown alongside the
    'auto-detect' default option.

    Cached after the first call. This is deliberately NOT run at import
    time: it stats six model subdirectories, which on a network share
    is a measurable ComfyUI startup stall for every user whether or not
    they ever touch this node. INPUT_TYPES is called lazily when the
    node is first inspected, which is soon enough.

    Consequence of the cache: weights added while ComfyUI is running
    appear after a restart (or a /object_info refresh in a fresh
    process), same as before.
    """
    global _WAN_PATH_CACHE
    if _WAN_PATH_CACHE is not None:
        return _WAN_PATH_CACHE
    _WAN_PATH_CACHE = _scan_known_wan_paths()
    return _WAN_PATH_CACHE


def _scan_known_wan_paths() -> list[str]:
    """The actual filesystem scan behind `_list_known_wan_paths()`."""
    try:
        models_dir = folder_paths.models_dir
    except Exception:
        return []
    candidates = []
    for sub in ("unividx", "diffusion_models", "checkpoints", "unet", "wan", "Wan2.1"):
        parent = os.path.join(models_dir, sub)
        if not os.path.isdir(parent):
            continue
        if auto_download._looks_like_wan_t2v(parent):
            candidates.append(parent)
        try:
            for entry in os.listdir(parent):
                child = os.path.join(parent, entry)
                if auto_download._looks_like_wan_t2v(child):
                    candidates.append(child)
        except OSError:
            continue
    # Dedupe by realpath, preserve order.
    seen = set()
    out = []
    for p in candidates:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


# ─── BAT UniVidX Sample (unified sampler, V3 dynamic schema) ────────────────


# These tile sizes come from the upstream demo scripts. They're the
# numbers UniVidX was tuned on; we expose them as constants here in
# case a future version wants them user-tunable.
_UPSTREAM_TILE_SIZE = [30, 52]
_UPSTREAM_TILE_STRIDE = [15, 26]


# UniVidX's stock negative prompt, copied verbatim from upstream's
# inference scripts (scripts/inference_univid_{intrinsic,alpha}.py).
# It's Chinese because Wan2.1's text encoder was trained that way; the
# demo scripts use exactly this string.
_UPSTREAM_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Guidance is disabled by setting cfg_scale to exactly 1.0 — that's the
# value the pipeline's `if cfg_scale != 1.0` check tests against, and
# what upstream's inference scripts assign when the prompt is empty.
_CFG_DISABLED = 1.0


def _prompt_inputs() -> list:
    """The text-conditioning widgets, shared by every mode that shows them."""
    return [
        io.String.Input(
            "prompt", multiline=True, default="",
            tooltip="Text prompt. UniVidX replicates it across all four "
                    "modality streams internally.",
        ),
        io.String.Input(
            "negative_prompt", multiline=True,
            default=_UPSTREAM_NEGATIVE_PROMPT,
            tooltip="What you don't want. Defaults to the stock negative "
                    "prompt from UniVidX's own inference scripts (it's "
                    "Chinese because Wan2.1's text encoder was trained "
                    "that way). Only has an effect while guidance is on, "
                    "i.e. cfg_scale != 1.0.",
        ),
        io.Float.Input(
            "cfg_scale", default=5.0, min=0.0, max=30.0, step=0.1,
            tooltip="Classifier-free guidance strength. 5.0 is upstream's "
                    "demo default. 1.0 disables guidance entirely — the "
                    "negative prompt is then never even evaluated, and "
                    "sampling costs one model call per step instead of two.",
        ),
    ]


def _optional_prompt_input():
    """The `optional_prompt` toggle shown on image-conditioned modes.

    Off (the default) means: empty positive prompt and no guidance, which
    is exactly how upstream runs these modes.

    A note on what "off" does NOT do: it can't apply the default negative
    prompt as a quality nudge. The pipeline only evaluates the negative
    branch inside `if cfg_scale != 1.0` (univid_intrinsic.py:671), so
    with guidance disabled the negative embedding is never computed. The
    only way to make it matter would be to keep cfg_scale at 5.0 with an
    empty positive — guidance between two contentless embeddings, at
    double the cost, in a configuration the checkpoints weren't tuned
    for. So "off" is honestly just "no text conditioning".
    """
    return io.DynamicCombo.Input(
        "optional_prompt",
        tooltip=(
            "Text conditioning for a mode that doesn't need it.\n\n"
            "off — no prompt, no guidance. What upstream does for these "
            "modes: the result is driven purely by the images you wire "
            "in. Also the faster path (one model call per sampling step "
            "rather than two).\n\n"
            "on — reveals prompt / negative_prompt / cfg_scale, so you "
            "can nudge the result with text. Doubles sampling time."
        ),
        options=[
            io.DynamicCombo.Option("off", []),
            io.DynamicCombo.Option("on", _prompt_inputs()),
        ],
    )


def _mode_options() -> list:
    """Build one `DynamicCombo.Option` per mode, each declaring exactly
    the IMAGE sockets that mode consumes.

    This is what makes the "single node that adapts to the chosen mode"
    premise actually work. The previous implementation declared all 7
    IMAGE sockets as `optional` and relied on a JS extension setting
    `slot.hidden = true` to hide the irrelevant ones — but no input slot
    in the ComfyUI frontend has a `hidden` property (see
    LGraphNode.drawSlots / computeSize: plain IMAGE slots are drawn
    unconditionally and always reserve a layout row). So every socket
    showed in every mode, which was exactly the clutter we set out to
    remove.

    `DynamicCombo` does the same job server-side: only the selected
    option's inputs are materialised, so the frontend genuinely only
    ever sees the sockets the mode needs. `Schema.validate()` skips
    `DynamicInput` when checking id uniqueness, so the same socket id
    (e.g. 'rgb') may legitimately repeat across options.
    """
    socket_tooltips = {
        "rgb": "Source RGB frames.",
        "albedo": "Albedo (base colour, lighting removed) frames.",
        "irradiance": "Irradiance (lighting only) frames.",
        "normal": "Surface-normal frames.",
        "pha": "Alpha matte frames.",
        "fgr": "Foreground frames.",
        "bgr": "Background frames.",
    }
    options = []
    for mode in modes.ALL_MODES:
        required = modes.ordered_required_inputs(mode)
        inputs = [
            io.Image.Input(key, tooltip=socket_tooltips[key])
            for key in required
        ]
        if required:
            # denoising_strength is only meaningful when there's an IMAGE
            # conditioning input to preserve.
            inputs.append(
                io.Float.Input(
                    "denoising_strength", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of the wired input to preserve. 1.0 = "
                        "ignore the input spatially (use only as "
                        "text-prompt guidance); <1.0 = preserve more of "
                        "the input."
                    ),
                )
            )
            # Conditioned modes are driven by their input imagery, not by
            # text — upstream's own example config for R2PFB ships
            # `prompt: ""`, and both inference scripts force cfg_scale to
            # 1.0 whenever the prompt is empty. So the text fields are
            # opt-in here rather than always-present clutter.
            #
            # This is a DynamicCombo rather than a Boolean because only a
            # DynamicCombo selection can gate other inputs; a plain
            # BOOLEAN widget can't.
            inputs.append(_optional_prompt_input())
        else:
            # Text-only modes have nothing to take their geometry from,
            # so they're the ONLY modes that need num_frames/height/width.
            # When an IMAGE is wired, the sampler derives all three from
            # the input batch (frame count from its length, H/W snapped
            # from its shape), so showing these widgets in image modes
            # just invites the user to set values that get overridden.
            inputs.extend([
                io.Int.Input(
                    "num_frames", default=21, min=5, max=81, step=4,
                    tooltip="Number of frames to generate. WAN's temporal "
                            "VAE requires 4k+1; off-by-one values are "
                            "padded up and trimmed back at the output.",
                ),
                io.Int.Input(
                    "height", default=480, min=256, max=1024, step=16,
                    tooltip="Output height. Snapped to a multiple of 16.",
                ),
                io.Int.Input(
                    "width", default=640, min=256, max=1280, step=16,
                    tooltip="Output width. Snapped to a multiple of 16.",
                ),
            ])
            # Text-only modes have nothing BUT the prompt to go on, so
            # the text fields are always present — no toggle.
            inputs.extend(_prompt_inputs())
        options.append(io.DynamicCombo.Option(mode, inputs))
    return options


class BatUniVidXSample(io.ComfyNode):
    """Single sampler covering both families + all 30 modes.

    The `mode` dropdown is the master widget: it's a `DynamicCombo`, so
    picking a mode materialises exactly the IMAGE sockets that mode
    consumes and nothing else. Picking `R2AIN` leaves only `rgb`;
    `RP2FB` leaves `rgb` + `pha`; `t2RAIN` leaves no IMAGE sockets at
    all. No JS extension involved, and no way to wire a socket the
    mode doesn't consume.

    Because the mode's inputs are nested under the combo, they arrive
    in `execute` as `mode["rgb"]`, `mode["denoising_strength"]`, etc.,
    with the selected mode itself at `mode["mode"]`.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BAT_UniVidXSample",
            display_name="🦇 UniVidX Sample",
            category="BAT/UniVidX",
            description=(
                "Run UniVidX in any of its 30 modes. The mode dropdown "
                "reshapes the node: only the IMAGE sockets that mode "
                "actually consumes are shown."
            ),
            inputs=[
                UniVidXModel.Input(
                    "model",
                    tooltip="The loaded UniVidX model from "
                            "🦇 (Down)load UniVidX Model. Its variant "
                            "must match the chosen mode's family.",
                ),
                io.DynamicCombo.Input(
                    "mode",
                    options=_mode_options(),
                    tooltip=(
                        "What UniVidX should do this run.\n\n"
                        "Naming: <conditions>2<targets>. Letters tell "
                        "you which modalities go IN and which come "
                        "OUT.\n\n"
                        "Intrinsic family (use 'intrinsic' loader):\n"
                        "  R=RGB  A=Albedo  I=Irradiance  N=Normal\n\n"
                        "Alpha family (use 'alpha' loader):\n"
                        "  R=composite RGB  P=Pha  F=Fgr  B=Bgr\n\n"
                        "Examples:\n"
                        "  t2RAIN  — text → all four intrinsic\n"
                        "  R2AIN   — RGB → Albedo+Irradiance+Normal\n"
                        "  R2PFB   — RGB → matte+fgr+bgr (matting)\n"
                        "  RP2FB   — RGB+matte → fgr+bgr\n\n"
                        "The IMAGE sockets appear/disappear to match "
                        "the mode you pick."
                    ),
                ),
                # prompt / negative_prompt / cfg_scale live inside the
                # mode options, not here: text-only modes always show
                # them, image-conditioned modes hide them behind the
                # `optional_prompt` toggle.
                io.Int.Input(
                    "num_inference_steps", default=20, min=1, max=100, step=1,
                    tooltip="Denoising steps. UniVidX was trained at 20. "
                            ">30 rarely helps.",
                ),
                io.Int.Input(
                    "seed", default=1, min=0, max=0xFFFFFFFF,
                    tooltip="Random seed for noise initialisation.",
                ),
                io.Boolean.Input(
                    "tiled", default=True, optional=True,
                    tooltip="Enable spatial tiling for the VAE "
                            "encode/decode. Saves VRAM at high "
                            "resolutions; leave on unless you're sure "
                            "you don't need it.",
                ),
                UniVidXContextOptions.Input(
                    "context_options", optional=True,
                    tooltip="Wire a 🦇 UniVidX Context Options node to "
                            "control temporal windowing for clips longer "
                            "than one pass. Leave unwired for the "
                            "defaults (81-frame windows, 16 overlap), "
                            "which switch on automatically past 81 frames.",
                ),
                UniVidXCacheOptions.Input(
                    "cache_options", optional=True,
                    tooltip="Wire a 🦇 UniVidX Cache Options node to enable "
                            "TeaCache step skipping (faster, slightly less "
                            "faithful). Unwired = no caching, i.e. today's "
                            "exact numerics.",
                ),
            ],
            outputs=[
                UniVidXResult.Output(
                    "result",
                    tooltip="Opaque result — feed it to 🦇 UniVidX Decode.",
                ),
            ],
        )

    @classmethod
    def _resolve_text_conditioning(
        cls, mode: dict, mode_name: str,
    ) -> tuple[str, str, float]:
        """Work out (prompt, negative_prompt, cfg_scale) for this run.

        The fields live in three possible places:
          • text-only modes  — directly on the mode option
          • image modes with `optional_prompt: on`  — nested one level deeper
          • image modes with `optional_prompt: off` — absent entirely

        In the last case we mirror what upstream's inference scripts do:
        an empty prompt means guidance is switched off by setting
        cfg_scale to exactly 1.0 (see
        scripts/inference_univid_intrinsic.py:127 — `if prompt == "":
        inference_params["cfg_scale"] = 1.0`). That also halves the work
        per sampling step, since the pipeline only runs the second
        (negative) model pass when cfg_scale != 1.0.

        We apply the same rule even when the user opted in and then left
        the prompt blank — otherwise they'd silently pay double for
        guidance between two contentless embeddings.
        """
        opt = mode.get("optional_prompt")
        if isinstance(opt, dict):
            # Image-conditioned mode with the toggle present.
            if opt.get("optional_prompt") == "on":
                source = opt
            else:
                source = None
        else:
            # Text-only mode: the fields sit on the mode option itself.
            source = mode if "prompt" in mode else None

        if source is None:
            logger.info(
                "UniVidX_Sample: mode %s running without text conditioning "
                "(cfg_scale=%.1f, guidance off) — this is how upstream runs "
                "the image-conditioned modes. Flip optional_prompt to 'on' "
                "if you want to steer it with text.",
                mode_name, _CFG_DISABLED,
            )
            return "", _UPSTREAM_NEGATIVE_PROMPT, _CFG_DISABLED

        prompt = source.get("prompt", "") or ""
        negative_prompt = source.get("negative_prompt", _UPSTREAM_NEGATIVE_PROMPT)
        cfg_scale = float(source.get("cfg_scale", 5.0))

        if not prompt.strip() and cfg_scale != _CFG_DISABLED:
            logger.info(
                "UniVidX_Sample: prompt is empty, so cfg_scale %.1f -> %.1f "
                "(matching upstream's inference scripts). Guidance between "
                "an empty positive and the negative prompt costs an extra "
                "model call per step without steering anything.",
                cfg_scale, _CFG_DISABLED,
            )
            cfg_scale = _CFG_DISABLED

        return prompt, negative_prompt, cfg_scale

    @classmethod
    def execute(
        cls,
        model: tuple,
        mode: dict,
        num_inference_steps: int,
        seed: int,
        tiled: bool = True,
        context_options: Optional[dict] = None,
        cache_options: Optional[dict] = None,
    ) -> io.NodeOutput:
        model_instance, variant, upstream_dir = model

        # DynamicCombo delivers the selection plus that option's inputs
        # as one nested dict: the mode string lives under its own id.
        mode_name = mode["mode"]
        family = modes.family_of(mode_name)
        denoising_strength = mode.get("denoising_strength", 1.0)
        prompt, negative_prompt, cfg_scale = cls._resolve_text_conditioning(
            mode, mode_name,
        )

        # Family / variant compatibility check. DynamicCombo can't
        # express "this mode requires that loader variant", so this
        # server-side check is the only guard — it also catches
        # workflow JSON edited by hand to bypass the UI.
        if family != variant:
            raise ValueError(
                f"UniVidX_Sample: mode {mode_name!r} is family={family!r} but "
                f"loaded model is variant={variant!r}. Either pick a "
                f"matching mode or reload the model with the other "
                f"variant."
            )

        # The mode's sockets are exactly the ones it requires, so a
        # missing value means the user left one unwired. There's no
        # longer any such thing as an "extra" wired input.
        required = modes.ordered_required_inputs(mode_name)
        supplied_imgs = {key: mode.get(key) for key in required}
        missing = [k for k in required if supplied_imgs[k] is None]
        if missing:
            raise ValueError(
                f"UniVidX_Sample: mode {mode_name!r} requires inputs "
                f"{required} but the following are not wired: "
                f"{missing}. Wire the missing IMAGE sockets."
            )

        # Resolve output geometry. Two cases:
        #
        #   Image modes  — everything comes from the wired input: frame
        #     count from its batch length, H/W from its frame size. The
        #     node has no num_frames/height/width widgets in these modes
        #     precisely because those values would be overridden here.
        #     `required` is in canonical order ('rgb' first), so which
        #     input defines the geometry is deterministic.
        #
        #   Text-only modes — nothing to measure, so the mode's own
        #     num_frames/height/width widgets supply it.
        #
        # H/W are snapped up to a multiple of 16 either way: UniVidX's
        # preprocessor rounds internally but one downstream path uses the
        # un-rounded value, producing a silent shape mismatch (same bug
        # class the DKT wrapper hit).
        n_original = None
        if required:
            ref = supplied_imgs[required[0]]
            ref_padded, n_original = frame_constraints.pad_to_wan_frame_count(ref)
            target_frames = ref_padded.shape[0]
            target_h = frame_constraints.snap_to_16(ref.shape[1])
            target_w = frame_constraints.snap_to_16(ref.shape[2])
            logger.info(
                "UniVidX_Sample: geometry from wired %r: %d frames @ %dx%d "
                "(snapped from %dx%d).",
                required[0], target_frames, target_h, target_w,
                ref.shape[1], ref.shape[2],
            )
        else:
            target_frames = int(mode["num_frames"])
            target_h = frame_constraints.snap_to_16(mode["height"])
            target_w = frame_constraints.snap_to_16(mode["width"])

        # Build per-modality conditioning tensors in UniVidX's shape.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16
        inference_videos: dict[str, Optional[torch.Tensor]] = {}
        for key in modes.family_input_order(family):
            img = supplied_imgs.get(key)
            if img is None:
                inference_videos[f"inference_{key}"] = None
                continue
            if n_original is not None:
                img_padded, _ = frame_constraints.pad_to_wan_frame_count(img)
            else:
                img_padded = img
            inference_videos[f"inference_{key}"] = tensor_io.image_batch_to_video_tensor(
                img_padded,
                target_height=target_h,
                target_width=target_w,
                target_frames=target_frames,
                device=device,
                dtype=dtype,
            )

        pbar = comfy.utils.ProgressBar(3)
        pbar.update_absolute(0, 3)

        # The upstream pipe wants the prompt replicated 4x (one per
        # modality stream of the family). The existing wrapper does
        # this; we mirror exactly.
        pipe_args = {
            "prompt": [prompt, prompt, prompt, prompt],
            "negative_prompt": negative_prompt,
            "seed": int(seed),
            "num_inference_steps": int(num_inference_steps),
            "cfg_scale": float(cfg_scale),
            "cfg_merge": False,
            "height": int(target_h),
            "width": int(target_w),
            "num_frames": int(target_frames),
            "denoising_strength": float(denoising_strength),
            "tiled": bool(tiled),
            "tile_size": _UPSTREAM_TILE_SIZE,
            "tile_stride": _UPSTREAM_TILE_STRIDE,
            "is_inference": True,
            "training_mode": mode_name,
            # Upstream defaults these to True in model_fn_wan_video
            # (univid_intrinsic.py:1213) because that signature is shared
            # with the trainer. At inference, inside torch.inference_mode(),
            # there is no backward pass for checkpointing to save memory
            # for — it is pure wrapper overhead over all 40 blocks.
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
            **inference_videos,
        }

        # ── Temporal context windowing ──────────────────────────────────
        # Upstream's TemporalTiler_BCTHW slices only the temporal axis and
        # its blend mask broadcasts over the batch, so all four modality
        # streams are windowed together — which is required, since
        # self-attention folds them into one sequence. `pipe()` forwards
        # unconsumed kwargs straight to model_fn, so these two keys are
        # all it takes to switch it on.
        window = _resolve_context_window(target_frames, context_options)
        if window is not None:
            size, stride = window
            pipe_args["sliding_window_size"] = size
            pipe_args["sliding_window_stride"] = stride
            # Report in PIXEL frames — the unit the user set. `size` and
            # `stride` are latent frames because that's the axis the tiler
            # slices, and quoting those alongside a pixel-frame total
            # reads as a 4x smaller window than it is.
            win_px = (size - 1) * 4 + 1
            n_windows = _count_context_windows(target_frames, size, stride)
            logger.info(
                "UniVidX_Sample: context windowing ON — %d frames in "
                "%d windows of %d frames (overlap %d). %s Peak VRAM is set "
                "by the WINDOW, not the clip: %d frames costs the same per "
                "pass as an unwindowed %d-frame render.",
                target_frames, n_windows, win_px,
                (size - stride) * 4,
                "Explicit context_options." if context_options
                else "Auto-enabled: clip is longer than one pass.",
                win_px, win_px,
            )

        # ── TeaCache ────────────────────────────────────────────────────
        want_cache, cache_refusal = _resolve_cache_plan(
            cache_options, windowing=window is not None, cfg_scale=cfg_scale,
            num_inference_steps=int(num_inference_steps),
        )
        if cache_refusal:
            logger.warning("UniVidX_Sample: %s", cache_refusal)
        if want_cache:
            tea = runtime.build_tea_cache_for_run(
                model_instance,
                num_inference_steps=int(num_inference_steps),
                rel_l1_thresh=float(cache_options["rel_l1_thresh"]),
            )
            if tea is not None:
                pipe_args["tea_cache"] = tea
                logger.info(
                    "UniVidX_Sample: TeaCache ON (rel_l1_thresh=%.3f). "
                    "Steps where the timestep embedding barely moves "
                    "will reuse the cached residual.",
                    float(cache_options["rel_l1_thresh"]),
                )

        logger.info(
            "UniVidX_Sample: mode=%s, frames=%d @ %dx%d, steps=%d, "
            "cfg_scale=%.1f%s",
            mode_name, target_frames, target_h, target_w, num_inference_steps,
            cfg_scale,
            "" if cfg_scale != _CFG_DISABLED else " (guidance off)",
        )

        # Give the conditioning VAE encode the tiling the sampler's own
        # `tiled` widget asks for — upstream's inference branch omits it,
        # so it silently runs full-frame — and time it. Everything between
        # "sampling started" and the first progress tick happens inside
        # pipe(): conditioning encode, prompt encode, and three model
        # swaps. That stretch was ~9 minutes on a real 81-frame job with
        # nothing logged, so it is worth measuring rather than guessing.
        runtime.instrument_vae_encode(
            model_instance, tiled=bool(tiled),
            tile_size=_UPSTREAM_TILE_SIZE, tile_stride=_UPSTREAM_TILE_STRIDE,
        )

        pbar.update_absolute(1, 3)
        _t_pipe = time.perf_counter()
        with runtime.unividx_cwd(upstream_dir):
            with torch.inference_mode():
                result_dict = model_instance.pipe(**pipe_args)
        logger.info(
            "UniVidX_Sample: pipe() total %.1f s (conditioning encode + "
            "prompt encode + model swaps + %d sampler steps + VAE decode).",
            time.perf_counter() - _t_pipe, int(num_inference_steps),
        )
        pbar.update_absolute(2, 3)

        # Report what caching actually saved. The threshold -> skip-count
        # mapping is clip-dependent, so the measured number is worth more
        # than any figure the docs could quote.
        report = runtime.tea_cache_report(pipe_args.get("tea_cache"))
        if report:
            logger.info("UniVidX_Sample: %s", report)

        # Wrap with the metadata the decoder needs to splay correctly.
        result = {
            "pipe_result": result_dict,
            "mode": mode_name,
            "family": family,
            "n_original": n_original,  # for the decoder to trim back
        }
        pbar.update_absolute(3, 3)
        return io.NodeOutput(result)


# ─── BAT UniVidX Decode ─────────────────────────────────────────────────────


class BatUniVidXDecode(io.ComfyNode):
    """Splay the opaque UNIVIDX_RESULT into 4 IMAGE outputs.

    Branches on the family. For intrinsic, the outputs are (rgb,
    albedo, irradiance, normal); for alpha they're (rgb, pha, fgr,
    bgr). Modes that didn't produce a given slot get a black-frame
    placeholder of the right shape so downstream nodes always see
    4 valid IMAGE batches.

    On why the output COUNT is fixed at 4 rather than adapting to the
    connected Sample node's mode: ComfyUI has no dynamic-output
    mechanism. `DynamicCombo` and friends are input-only (the V3
    `DynamicOutput` class exists but is an empty `pass` stub with no
    implementations and no dispatch table), and a node's outputs are
    memoised class-wide on first schema resolution
    (`_io.GET_SCHEMA`: `if cls._RETURN_TYPES is None`), so they cannot
    vary per node instance even if recomputed. `MatchType` propagates a
    type from a connected input but can't add or remove slots.

    So instead of faking it, the four slots are named for BOTH families
    and every mode fills all four — a slot the mode didn't produce
    returns black rather than vanishing, which keeps downstream wiring
    valid when you switch modes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BAT_UniVidXDecode",
            display_name="🦇 UniVidX Decode",
            category="BAT/UniVidX",
            description=(
                "Splay a UniVidX result into its 4 image streams. Slot "
                "names cover both families: intrinsic outputs are "
                "(rgb, albedo, irradiance, normal); alpha outputs are "
                "(rgb, pha, fgr, bgr). Slots the chosen mode didn't "
                "produce come back black."
            ),
            inputs=[
                UniVidXResult.Input(
                    "result",
                    tooltip="The result from 🦇 UniVidX Sample.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "rgb", display_name="rgb",
                    tooltip="Intrinsic: RGB. Alpha: composite RGB.\n"
                            "Black if the mode took RGB as an input "
                            "rather than producing it.",
                ),
                io.Image.Output(
                    "albedo_pha", display_name="albedo / pha",
                    tooltip="Intrinsic: albedo (base colour). "
                            "Alpha: pha (the matte).\n"
                            "Black if this mode didn't produce it.",
                ),
                io.Image.Output(
                    "irradiance_fgr", display_name="irradiance / fgr",
                    tooltip="Intrinsic: irradiance (lighting). "
                            "Alpha: foreground.\n"
                            "Black if this mode didn't produce it.",
                ),
                io.Image.Output(
                    "normal_bgr", display_name="normal / bgr",
                    tooltip="Intrinsic: surface normals. "
                            "Alpha: background.\n"
                            "Black if this mode didn't produce it.",
                ),
            ],
        )

    @classmethod
    def execute(cls, result: dict) -> io.NodeOutput:
        family = result["family"]
        mode = result["mode"]
        pipe_result = result["pipe_result"]
        n_original = result.get("n_original")

        slot_keys = (
            ["rgb", "albedo", "irradiance", "normal_unit"] if family == "intrinsic"
            else ["rgb", "pha", "fgr", "bgr"]
        )

        # First, determine the canonical (T, H, W, 3) shape from any
        # tensor we can find in the result dict, so we know what shape
        # placeholder tensors should take.
        any_tensor = None
        for v in pipe_result.values():
            if v is not None:
                any_tensor = v
                break
        if any_tensor is None:
            raise RuntimeError(
                "UniVidX_Decode: pipe returned no usable tensors. "
                "Check the wrapper console for upstream errors."
            )
        reference = tensor_io.pipe_result_frame_to_image(any_tensor)
        ref_t, ref_h, ref_w = reference.shape[0], reference.shape[1], reference.shape[2]

        # Trim every slot — real tensors and placeholders alike — so all
        # four outputs share the frame count the user fed in.
        outputs = []
        placeholders = []
        for slot in slot_keys:
            t = pipe_result.get(slot)
            if t is None:
                converted = tensor_io.black_placeholder(ref_t, ref_h, ref_w)
                placeholders.append(slot)
            else:
                converted = tensor_io.pipe_result_frame_to_image(t)
            if n_original is not None:
                converted = frame_constraints.trim_to_original(converted, n_original)
            outputs.append(converted)

        logger.info(
            "UniVidX_Decode: mode=%s, family=%s, outputs=4×(%d,%d,%d,3)",
            mode, family, outputs[0].shape[0], outputs[0].shape[1], outputs[0].shape[2],
        )
        # Name the black slots explicitly — otherwise a user wiring a
        # slot this mode never produces just sees black frames with no
        # indication of why.
        if placeholders:
            logger.info(
                "UniVidX_Decode: mode %r doesn't produce %s — those "
                "output(s) are black placeholders.",
                mode, ", ".join(placeholders),
            )
        return io.NodeOutput(*outputs)


# ─── Mode-reference endpoint (powers the node's help panel) ─────────────────


def _register_mode_reference_route() -> None:
    """Expose the per-mode reference at /bat_unividx/modes.

    The help panel needs a description of all 30 modes. Serving it from
    `modes.py` rather than duplicating the table in JS is deliberate:
    the previous version of this pack hand-mirrored the mode enumeration
    into its JS and the two could silently drift. There's one source of
    truth now.

    Registration is best-effort — if PromptServer isn't importable or
    has no running app (headless tooling, tests), the help panel just
    falls back to a static message.
    """
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None or getattr(instance, "routes", None) is None:
        return

    @instance.routes.get("/bat_unividx/modes")
    async def _bat_unividx_modes(request):
        return web.json_response({"modes": modes.mode_reference()})


_register_mode_reference_route()

# Generate the markdown ComfyUI's native per-node help panel serves from
# /extensions/<pack>/docs/<NodeName>.md. Unlike the title-bar "?" button
# (canvas-only), that panel works in every render mode — including the
# Vue-based node renderer, whose header component has no title-button
# hook, so a canvas title button silently disappears there.
help_docs.write_help_docs(os.path.join(_WRAPPER_DIR, "web"))


# ─── Context-window planning ────────────────────────────────────────────────

# Defaults used when no Context Options node is wired. 81 frames is both
# Wan 2.1's native training length and the largest single pass verified to
# run here; 16 frames of overlap matches the value Kijai's WanVideoWrapper
# ships for the same backbone.
_DEFAULT_CONTEXT_FRAMES = 81
_DEFAULT_CONTEXT_OVERLAP = 16


def _pixel_to_latent_frames(n: int) -> int:
    """Wan's temporal VAE maps N pixel frames to (N-1)//4 + 1 latents."""
    return (int(n) - 1) // 4 + 1


def _resolve_context_window(target_frames: int, context_options):
    """Plan temporal windowing, or None to sample in a single pass.

    Returns `(size, stride)` in LATENT frames — not pixel frames.
    TemporalTiler_BCTHW slices dim 2 of the latents tensor, which is the
    latent temporal axis, so feeding it pixel counts would window roughly
    4x too long and quietly do nothing on clips under ~324 frames.

    Windowing engages only when the clip genuinely exceeds one window, so
    an unwired node never changes behaviour for existing short jobs.
    """
    opts = context_options or {}
    frames = int(opts.get("context_frames", _DEFAULT_CONTEXT_FRAMES))
    overlap = int(opts.get("context_overlap", _DEFAULT_CONTEXT_OVERLAP))

    total_lat = _pixel_to_latent_frames(target_frames)
    size_lat = _pixel_to_latent_frames(frames)
    if size_lat >= total_lat:
        return None  # fits in one pass; nothing to window

    # Overlap is a length, not an index, so it scales by the plain factor.
    overlap_lat = max(1, overlap // 4)
    # stride < size is mandatory: the tiler derives its blend ramp width as
    # (size - stride), and a width of 0 makes build_1d_mask slice `x[-0:]`
    # — the whole tensor — against an empty right-hand side and raise.
    overlap_lat = min(overlap_lat, size_lat - 1)
    stride_lat = size_lat - overlap_lat
    return size_lat, stride_lat


def _count_context_windows(target_frames: int, size_lat: int,
                           stride_lat: int) -> int:
    """How many passes the tiler will actually make.

    Mirrors TemporalTiler_BCTHW.run's loop exactly (util.py:967-969),
    including its skip of a window the previous one already covered,
    rather than estimating with a ceiling division — an estimate here
    once reported 9 windows for a job that ran 2.
    """
    total_lat = _pixel_to_latent_frames(target_frames)
    n = 0
    for t in range(0, total_lat, stride_lat):
        if t - stride_lat >= 0 and t - stride_lat + size_lat >= total_lat:
            continue
        n += 1
    return n


def _resolve_cache_plan(cache_options, *, windowing: bool, cfg_scale: float,
                        num_inference_steps: int):
    """Decide whether TeaCache may run. Returns `(enabled, refusal_message)`.

    Pulled out of `execute` so the interlocks are testable — both of the
    conditions below were found the hard way, in production, after a
    multi-minute model load.

    Refusals:

    * **Context windowing.** TeaCache keeps ONE residual and ONE step
      counter, but `model_fn` runs once per window per step. `self.step`
      then counts windows rather than denoising steps and wraps against
      `num_inference_steps`, so the cache gets reused across different
      parts of the clip. Separately, the stored residual carries one
      window's shape and the tiler's final window is usually shorter, so
      `update()` raises a size mismatch. The shape error is the lucky
      case; the step-counter corruption is silent. Per-window caches
      would fix both but multiply TeaCache's already-large residual by
      the window count — backwards, since windowing exists to save
      memory.

    * **CFG on.** With `cfg_scale != 1` the DiT runs twice per step
      (positive and negative), and one shared cache would double-advance
      its counter and blend the two passes' residuals.

    A low step count is only a warning, not a refusal: caching still
    works, it just cannot pay for itself. TeaCache always computes the
    first and last step, so at N steps there are only N-2 candidates —
    at 4 steps you risk several GB of residuals to skip at most 2 passes.
    """
    if not cache_options:
        return False, None
    if float(cache_options.get("rel_l1_thresh", 0.0)) <= 0:
        return False, None

    if windowing:
        return False, (
            "cache_options ignored — TeaCache and context windowing cannot "
            "be combined. TeaCache keeps one residual and one step counter, "
            "but the model runs once per window per step, so the cache "
            "would be reused across different parts of the clip (and its "
            "shape breaks on the shorter final window). Either raise "
            "context_frames so the clip fits a single pass, or remove the "
            "Cache Options node."
        )

    if cfg_scale != _CFG_DISABLED:
        return False, (
            f"cache_options ignored — TeaCache needs cfg_scale="
            f"{_CFG_DISABLED:.1f} (guidance off) so the DiT runs once per "
            f"step, but cfg_scale is {cfg_scale:.1f}. Clear the prompt or "
            f"set cfg_scale={_CFG_DISABLED:.1f} to use caching."
        )

    if num_inference_steps <= _TEACACHE_MIN_USEFUL_STEPS:
        # Enabled, but say plainly that it is a bad trade here.
        return True, (
            f"TeaCache at only {num_inference_steps} steps is probably not "
            f"worth it — the first and last step always compute, so at most "
            f"{max(0, num_inference_steps - 2)} can be skipped, while the "
            f"cached residuals cost several GB of VRAM for the whole run. "
            f"It is a 20-step optimisation; if you are already on a "
            f"step-distill LoRA you have bought that speedup already."
        )

    return True, None


# Below this, TeaCache can skip at most num_inference_steps-2 passes,
# which rarely justifies holding two hidden-state tensors on the GPU.
_TEACACHE_MIN_USEFUL_STEPS = 8


# ─── Sample-time options nodes ──────────────────────────────────────────────
# Both are plain V1 nodes emitting a dict over a custom wire type, wired
# into optional sockets on the Sample node. Two separate nodes rather than
# one combined "options" node: windowing and caching are unrelated, and
# each changes output pixels, so wiring one should be a deliberate act.
#
# Optional SOCKETS rather than new widgets is also what keeps saved
# workflows intact — ComfyUI stores widget values as a positional array,
# so adding widgets to an existing node shifts it, while an unwired
# optional input is simply never passed.


class BatUniVidXCacheOptions:
    """TeaCache settings for the Sample node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rel_l1_thresh": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": (
                         "How much the timestep embedding may drift before "
                         "the DiT is recomputed. Higher = more steps "
                         "skipped = faster = less faithful. 0 disables.\n\n"
                         "Start LOW here. The polynomial that rescales the "
                         "drift was fitted on stock Wan2.1, and in "
                         "conditioned modes (R2AIN etc.) some rows of the "
                         "timestep embedding are pinned to a constant — "
                         "which dilutes the measured drift and makes this "
                         "threshold skip more eagerly than the same number "
                         "would on plain Wan2.1."
                     )},
                ),
            }
        }
    # No start_step / end_step here on purpose. Kijai's WanVideoWrapper
    # exposes them, but that's his own TeaCache reimplementation — the one
    # vendored with UniVidX (util.py:889) takes only
    # (num_inference_steps, rel_l1_thresh, model_id). It always computes
    # the first and last step and decides every other step from the
    # accumulated threshold alone. Exposing a step range here would be two
    # widgets that quietly do nothing.

    RETURN_TYPES = ("UNIVIDX_CACHE_OPTIONS",)
    RETURN_NAMES = ("cache_options",)
    FUNCTION = "build"
    CATEGORY = "BAT/UniVidX"
    DESCRIPTION = (
        "TeaCache for UniVidX: skip the DiT forward on steps where little "
        "changes and reuse the previous residual. Typically 1.4-1.9x, at "
        "some fidelity cost. Wire into the Sample node's cache_options."
    )

    def build(self, rel_l1_thresh):
        return ({"rel_l1_thresh": float(rel_l1_thresh)},)


class BatUniVidXContextOptions:
    """Temporal context-window settings for the Sample node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_frames": (
                    "INT",
                    {"default": 81, "min": 5, "max": 1000, "step": 4,
                     "tooltip": (
                         "Window length in PIXEL frames. 81 is both Wan "
                         "2.1's native training length and the largest "
                         "single pass verified to work here, so it is a "
                         "sensible ceiling rather than an arbitrary one.\n\n"
                         "Note the latent axis is 4x compressed, so this "
                         "gets converted to (n-1)/4+1 latent frames "
                         "internally."
                     )},
                ),
                "context_overlap": (
                    "INT",
                    {"default": 16, "min": 4, "max": 200, "step": 4,
                     "tooltip": (
                         "Pixel frames shared between neighbouring windows, "
                         "cross-faded to hide the seam. More overlap = "
                         "smoother joins but more windows, so more time. "
                         "Must be smaller than context_frames."
                     )},
                ),
            }
        }

    RETURN_TYPES = ("UNIVIDX_CONTEXT_OPTIONS",)
    RETURN_NAMES = ("context_options",)
    FUNCTION = "build"
    CATEGORY = "BAT/UniVidX"
    DESCRIPTION = (
        "Sample long clips in overlapping temporal windows, blended inside "
        "the denoising loop, to get past the ~81-frame single-pass limit. "
        "Windowing switches on automatically above the window length even "
        "without this node; wire it to override the defaults."
    )

    def build(self, context_frames, context_overlap):
        if context_overlap >= context_frames:
            raise ValueError(
                f"context_overlap ({context_overlap}) must be smaller than "
                f"context_frames ({context_frames}) — equal or larger "
                "leaves no forward progress between windows."
            )
        return ({
            "context_frames": int(context_frames),
            "context_overlap": int(context_overlap),
        },)


# ─── Registration ───────────────────────────────────────────────────────────


# This pack mixes schema versions: Sample and Decode are V3
# (`io.ComfyNode`) — Sample needs `io.DynamicCombo` for real per-mode
# socket visibility, Decode needs per-output display names — while the
# Loader remains V1.
#
# We register everything through NODE_CLASS_MAPPINGS rather than a
# `comfy_entrypoint`. Two reasons:
#   1. nodes.py checks NODE_CLASS_MAPPINGS with `if` and comfy_entrypoint
#      with `elif` — they're mutually exclusive, so a pack cannot use
#      both paths. Mixing versions therefore has to go through V1
#      registration.
#   2. V3 classes expose the full set of V1-compat classproperties
#      (INPUT_TYPES, RETURN_TYPES, FUNCTION, …), and execution.py
#      detects V3 via `issubclass(class_def, _ComfyNodeInternal)`
#      rather than by registration path — so a V3 class registered
#      this way still executes on the V3 code path, dynamic inputs
#      and all.
NODE_CLASS_MAPPINGS = {
    "BAT_UniVidXLoader": BatUniVidXLoader,
    "BAT_UniVidXSample": BatUniVidXSample,
    "BAT_UniVidXDecode": BatUniVidXDecode,
    "BAT_UniVidXContextOptions": BatUniVidXContextOptions,
    "BAT_UniVidXCacheOptions": BatUniVidXCacheOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAT_UniVidXLoader": "🦇 (Down)load UniVidX Model",
    "BAT_UniVidXSample": "🦇 UniVidX Sample",
    "BAT_UniVidXDecode": "🦇 UniVidX Decode",
    "BAT_UniVidXContextOptions": "🦇 UniVidX Context Options",
    "BAT_UniVidXCacheOptions": "🦇 UniVidX Cache Options",
}

# Resolve the V3 schemas once at import so a malformed schema surfaces
# as a clear error at ComfyUI startup rather than on first node use.
# Also populates the cached V1-compat properties.
BatUniVidXSample.GET_SCHEMA()
BatUniVidXDecode.GET_SCHEMA()
