"""Generate the built-in node help markdown from `modes.py`.

ComfyUI's frontend has a native per-node help system: it fetches
`/extensions/<pack>/docs/<NodeName>.md`, renders it as markdown, and
shows it in the node help panel. That path is served straight out of
the pack's `WEB_DIRECTORY`, so all we have to do is put files there.

Why this exists alongside the title-bar "?" button
--------------------------------------------------
The "?" button is drawn on the LiteGraph canvas. ComfyUI's newer
Vue-based node renderer (the "nodes 2.0" / VueNodes mode) builds nodes
as DOM components instead, and its header component renders only the
collapse toggle, title, price/status badges and pin — it has no
title-button hook. So a canvas title button silently disappears in that
mode.

The native help system, by contrast, is render-mode independent: it's
the same panel in canvas mode, in VueNodes mode, and from the node
library sidebar. Shipping these markdown files means the mode reference
is reachable however the user has ComfyUI configured.

The files are generated at import (cheap — string formatting over 30
static entries) rather than committed, so they can never drift out of
sync with the modes the nodes actually offer.
"""
from __future__ import annotations

import logging
import os

from . import modes

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")


def _mode_table(family: str) -> str:
    """Markdown table of every mode in one family."""
    rows = [
        "| Mode | Takes | Gives | What it's for |",
        "|---|---|---|---|",
    ]
    for entry in modes.mode_reference():
        if entry["family"] != family:
            continue
        takes = " + ".join(entry["inputs"]) if entry["inputs"] else "*text only*"
        gives = " + ".join(entry["outputs"])
        rows.append(
            f"| `{entry['mode']}` | {takes} | {gives} | {entry['description']} |"
        )
    return "\n".join(rows)


def sample_help_markdown() -> str:
    """Help page for the Sample node — the full mode reference."""
    return f"""# 🦇 UniVidX Sample

Runs UniVidX in any of its {len(modes.ALL_MODES)} modes.

The **mode** dropdown reshapes the node. Only the IMAGE sockets that the
chosen mode actually consumes exist, so you can't wire an input the model
will ignore — and the mode-specific widgets follow the same rule.

## Reading a mode name

Modes are named `<inputs>2<outputs>`. Letters before the `2` are what you
feed in; letters after are what the model predicts.

| Family | Letters |
|---|---|
| **Intrinsic** | `R`=RGB, `A`=albedo, `I`=irradiance, `N`=normals |
| **Alpha** | `R`=composite RGB, `P`=matte, `F`=foreground, `B`=background |

`R` means different things in the two families (a plain RGB frame vs a
composite containing a matted subject), which is why the family is fixed
by the loader's `variant` before you pick a mode.

> **The mode's family must match the variant loaded in
> 🦇 (Down)load UniVidX Model.** A mismatch raises an error naming both
> sides rather than failing deep inside the sampler.

## Where output geometry comes from

- **Modes with an IMAGE input** take frame count and resolution from the
  wired clip — length for frames, frame size for H/W (snapped up to a
  multiple of 16). These modes have no `num_frames`/`height`/`width`
  widgets, because those values would only be overridden.
- **Text-only (`t2…`) modes** have nothing to measure, so they're the
  only modes that expose those three widgets.

Frame counts are padded up to WAN's `4k+1` requirement internally and
trimmed back at the output, so you get the same count you fed in.

## Text prompts

**Text-only (`t2…`) modes** always show `prompt`, `negative_prompt` and
`cfg_scale` — the prompt is the only thing driving them.

**Image-conditioned modes** hide those behind an `optional_prompt`
toggle, because they're driven by the imagery you wire in, not by text.
Upstream's own example config for `R2PFB` ships `prompt: ""`, and both
of its inference scripts force `cfg_scale` to `1.0` whenever the prompt
is empty.

- `optional_prompt: off` (default) — no prompt, guidance disabled. What
  upstream does for these modes, and the faster path: sampling costs one
  model call per step instead of two.
- `optional_prompt: on` — reveals the three text widgets so you can nudge
  the result. Roughly doubles sampling time.

Leaving the prompt blank while the toggle is on drops `cfg_scale` to
`1.0` automatically, so you never silently pay double for guidance
between two contentless embeddings.

> A note on the negative prompt: it only has an effect while guidance is
> on. The pipeline evaluates the negative branch solely inside
> `if cfg_scale != 1.0`, so with guidance disabled it's never computed —
> a default negative prompt can't act as a free quality boost.

## Intrinsic modes — lighting decomposition

Use with the loader set to `variant: intrinsic`.

{_mode_table("intrinsic")}

## Alpha modes — matting / compositing

Use with the loader set to `variant: alpha`.

{_mode_table("alpha")}
"""


def decode_help_markdown() -> str:
    """Help page for the Decode node."""
    return """# 🦇 UniVidX Decode

Splays the opaque result from 🦇 UniVidX Sample into four IMAGE outputs.

## Output slots

Slot names cover both families, because one node serves both:

| Slot | Intrinsic | Alpha |
|---|---|---|
| `rgb` | RGB | composite RGB |
| `albedo / pha` | albedo (base colour) | matte |
| `irradiance / fgr` | irradiance (lighting) | foreground |
| `normal / bgr` | surface normals | background |

A slot the chosen mode didn't produce comes back as **black frames**, and
the console names which ones. So `R2AIN` — which takes RGB rather than
producing it — returns black on the `rgb` slot.

## Why there are always exactly four outputs

The output count can't adapt to the connected Sample node's mode.
ComfyUI has no dynamic-output mechanism: the dynamic-input types
(`DynamicCombo` and friends) are input-only, the `DynamicOutput` class in
the V3 API is an unimplemented stub, and a node's outputs are cached
class-wide the first time its schema resolves — so they can't vary per
node instance.

Keeping all four slots present (rather than faking it) also means your
downstream wiring survives switching modes.
"""


def loader_help_markdown() -> str:
    """Help page for the Loader — the speed/quality knobs live here."""
    return """# 🦇 (Down)load UniVidX Model

Loads the Wan2.1-T2V-14B backbone plus the UniVidX adapters, downloading
whatever is missing, and hands a model to 🦇 UniVidX Sample.

Every widget on this node is honoured — there is no "advanced" gate. (One
used to exist; it hid nothing and silently discarded the settings below
it, so it was removed.)

## The basics

### `variant` — which model, and it must match your mode

| Variant | Produces | Use with modes |
|---|---|---|
| `intrinsic` | albedo, irradiance, normals | `R2AIN`, `t2RAIN`, `A2RIN`, … |
| `alpha` | matte, foreground, background | `R2PFB`, `t2RPFB`, `RP2FB`, … |

These are two separately-trained checkpoints, not two settings of one
model. Mismatching this against the Sample node's mode is caught at
sampling time with an error naming both sides — it will not silently
produce nonsense.

### `dtype` — leave on `bf16`

Compute precision for the pipeline. `bf16` is what UniVidX was trained and
released in. `fp16` exists for hardware without bf16 support and is not
recommended here: this model's value range was never checked against
fp16's smaller exponent, so overflow is a real risk.

Note this is separate from `fp8_quantize`, which changes how weights are
*stored*, not how maths is done.

### `wan21_source` — where the 57 GB backbone comes from

`auto-detect, fall back to download` scans `ComfyUI/models/` (under
`unividx/`, `diffusion_models/`, `checkpoints/`, `unet/`, `wan/`,
`Wan2.1/`) for an existing copy and reuses it, downloading from HuggingFace
into `models/unividx/` only if nothing is found. The other dropdown entries
are the copies it actually found on disk — pick one to force a specific
install when you have several.

**Wan2.1-VACE-14B and Wan2.1-I2V-14B will not work.** Different
fine-tunes; the loader rejects them rather than failing obscurely later.

The UniVidX checkpoint itself (~770 MB per variant) is fetched the same
way, with no widget — there is only one right file per variant.

### `vram_buffer_gb` — read this when you get an out-of-memory error

**This is the knob that fixes OOM**, and it works the opposite way round
to how it reads.

It is how much VRAM to keep **FREE for activations** — not how much to
use. DiffSynth computes `limit = total_vram - this` and offloads DiT
layers to CPU to stay under that limit. So:

- **Higher** = more offloading = more room for the sampler = slower
- **Lower** = more of the model stays resident = faster, until you OOM

`0` means auto, which sizes it from the card's actual VRAM and whether FP8
is on. The chosen value is logged at load:

```
UniVidX: vram_buffer_gb=auto -> 39.2 GB reserved for activations
         (card total: 95.0 GB)
```

Only override if auto guesses wrong. **If you OOM during sampling, raise
it** — the failure means the model fitted but the sampler's peak
allocation did not.

The peak is unusually large here because UniVidX attends over all four
modality streams as a single sequence — one step can request 20+ GB at
81 frames.

A worked example: a 95 GB card at the old fixed default of 4.0 GB left a
91 GB limit, so ~48 GB of DiT stayed resident and a 24 GB allocation died
with 20 MiB free. Auto now reserves ~39 GB on that card.

### This is the block-swapping dial

Layers past the limit are streamed CPU↔GPU on demand, so the reservation
decides how much of the model stays resident.

| | Value | Notes |
|---|---|---|
| **Recommended** | `0` = auto | Sized from the card, the dtype and whether FP8 is on. Logged at load. |
| Minimum | `0.0` | Which *means* auto. The smallest useful **manual** value is around **8** — below that essentially nothing streams and you are relying on the model simply fitting. |
| Maximum | `80.0` | But the practical ceiling is about **(card VRAM − 16 GB)**. Reserve past that and there is no room for the DiT itself, so every layer thrashes and throughput collapses. |
| Typical manual | ~⅓ of the card | e.g. 24–32 on a 95 GB card if you want to override auto. |

Which way to move it:

- **OOM at sample time** (model loaded, sampler's peak did not fit) → **raise** it. That is the fix.
- **Runs but feels slow, and you know the job fits** → **lower** it, so less streams.
- Unsure → leave it at auto and read the logged value.

**FP8 and block swapping now work together.** They used to be mutually
exclusive: quantisation replaced the wrappers that do the streaming, so
FP8 halved the DiT's bytes and simultaneously pinned every layer resident.
Quantised layers now keep their offload machinery and stay fp8 in *every*
state — so the GPU-resident copy is 1 byte per parameter too, not just the
offloaded one. Numerics are unchanged from the old FP8 path (measured
identical), and biases and LoRA adapters stay bf16.

## What's on by default

Three things are **on by default**, and none of them changes output
pixels:

| Setting | Default | Why |
|---|---|---|
| `fp8_quantize` | `qfloat8_prequantized` | Halves DiT VRAM (~28 → ~14 GB). Without it a bf16 DiT does not leave room for activations. |
| `attention_backend` | `auto` | FlashAttention-2 if importable, else SDPA. |
| `vram_buffer_gb` | `0` = auto | Sized from the card's real VRAM. |

Everything else is off. That is deliberate — nothing that alters your
albedo/normal output is enabled without you asking — but it does mean
there is speed left on the table.

## Pushing further (in the order worth trying)

### 1. `distill_lora` — the big one, ~5x

Merges a step-distillation LoRA into the base weights so 4 steps look like
20. Pick one from the dropdown, then on the Sample node set
`num_inference_steps=4` and `cfg_scale=1.0`. Leaving steps at 20 wastes
the entire point.

#### Which file am I looking for?

UniVidX's backbone is **Wan 2.1, T2V, 14B**. All three have to match, and
the dropdown lists every LoRA you own — most of which are for other
models. Read the filename:

| Must contain | Because |
|---|---|
| `T2V` | This backbone is text-to-video. An `I2V` file is the easy mistake — see below. |
| `14B` | Not 1.3B, not 5B. |
| `Wan21` / `Wan2.1` | Not Wan 2.2, which is a different generation. |
| `distill` or `lightx2v` | It has to be a *step*-distillation LoRA. A style or likeness LoRA does nothing for speed. |

So `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors`
qualifies, and `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16`
qualifies (v2, higher rank).

Rejected by the same rules: anything with `I2V`, anything under a `WAN22`
folder, anything `480p`-specific from the I2V line.

**Why `I2V` matters more than it looks.** Wan 2.1's I2V and T2V models
share most of their DiT, so an I2V LoRA merges *successfully* — plenty of
keys match, no error — and then degrades your albedo and normals for
reasons that are impossible to guess from the output. The loader now warns
on filenames that look I2V or Wan 2.2, but that check is only a substring
match, so read the name yourself.

Whatever you pick, the log reports how many keys actually merged. A low or
zero count means wrong base model.

If you have two valid candidates, try the **rank32 v1** first: a
lower-rank LoRA perturbs the base less, which matters here because
UniVidX's own adapters were trained against the unmodified base. Then try
v2 rank64 and compare.

**The risk is real.** The LoRA merges into the frozen base *underneath*
UniVidX's four per-modality adapters, so there is no key collision — but
those adapters were trained against the *original* base. Moving it puts
them off-distribution, and this specific model also pins conditioned rows
to a fixed timestep, which a 4-step schedule may not suit. Nobody has
published a UniVidX-specific distill LoRA, so this combination is
unvalidated. Render the same clip both ways and compare albedo and normals
before trusting it. `distill_strength` below 1.0 blends toward the
original base if full merge is too destructive.

### 2. TeaCache — ~1.4-1.9x

Not here; wire a **🦇 UniVidX Cache Options** node into the Sample node.
Skips the DiT entirely on steps where the timestep embedding has barely
moved, reusing the previous residual.

`rel_l1_thresh` is the only knob: higher = more steps skipped = faster and
less faithful. **Start low (~0.05-0.10) and raise it.** Two reasons to be
more cautious than a stock-Wan2.1 guide would suggest: the rescaling
polynomial was fitted on stock Wan2.1, not on this 4-stream fine-tune, and
in conditioned modes part of the timestep embedding is pinned to a
constant, which dilutes the measured drift and makes any given threshold
skip more eagerly than it would elsewhere.

Requires `cfg_scale=1.0`. Above that the DiT runs twice per step and one
cache would mix the two passes, so it refuses rather than corrupt.

### 3. `compile_dit` — worth trying, size unverified

`torch.compile` on the DiT, already implemented — just off by default.

The three things that could have made it useless here were checked and all
came back clean. Dynamo traces **zero graph breaks** through FP8Linear,
through peft called with per-row `adapter_names` (UniVidX's four-adapter
dispatch), and through a plain Linear baseline; a compiled module also
survived a CPU↔GPU offload/onload cycle, which is what VRAM management
does every step. Changing sequence lengths — which context windowing
produces, since the last window can be shorter — were handled without a
recompile storm.

So it should work. What is **not** verified is the payoff: the "20-30%"
figure is inherited, not measured on this model. Turn it on, compare
wall-clock, keep it if it earns its place.

Two practical notes. The first sampler step pays the compile cost — tens
of seconds at minimum, likely a couple of minutes for a 40-block 14B DiT —
so it only makes sense on long runs, and it is wasted on a 4-step distilled
render. And `mode="reduce-overhead"` normally means CUDA Graphs, but it is
paired with `dynamic=True`, which generally prevents graph capture; the
practical effect is probably inductor codegen without the CUDA-graph win.
If you want to experiment, a fixed-shape batch with `dynamic=False` is the
variant most likely to beat it.

### 4. `attention_backend` — a few percent at most

Do not expect much. UniVidX hardcodes SDPA, but torch's SDPA is *already*
a fused FlashAttention kernel; forcing the naive math path on the same
shape is 13x slower, which is what an actual missing optimisation would
look like. Measured at Wan2.1-14B geometry on sm_86: SDPA 14.4 ms, FA2
13.9 ms, SageAttention 16.7 ms. Sage was both slower and less exact there,
so it is excluded from `auto` — but it is worth re-measuring on a
different card.

## What is not worth trying

- **FlashAttention-3** — not installed in this environment.
- **NVIDIA's Sol engine** — genuinely fast (2.3-2.8x) but it is an SGLang
  serving stack validated on Cosmos/LTX/SANA, not Wan. It would be a port.
- **Training a UniVidX turbo LoRA** — none exists publicly; a project in
  its own right, and only worth it if step-distill proves unusable.
"""


def cache_options_help_markdown() -> str:
    """Help page for the Cache Options node."""
    return """# 🦇 UniVidX Cache Options

Enables **TeaCache** on the Sample node. Wire this in and sampling gets
faster; leave it unwired and nothing changes.

Typical speedup is **1.4-1.9x**, and it does change your output — this is
a quality trade, not a free win.

## How it works

At each denoising step TeaCache measures how far the modulated timestep
embedding has moved since the last computed step. If the accumulated
movement is below `rel_l1_thresh`, it **skips all 40 DiT blocks** and adds
the previously cached residual instead. The first and last steps are
always computed.

So it exploits the fact that consecutive steps often barely differ. Push
it too far and you skip steps that mattered.

## `rel_l1_thresh` — the strength knob

This is the only control, and it is the whole dial:

| Value | Skipped (of 20 steps) | DiT speedup | Quality |
|---|---|---|---|
| `0.0` | 0 | 1.0x | Off — identical to not wiring the node. |
| `0.03 - 0.08` | ~2-5 | ~1.1-1.3x | Cautious. Little visible change. |
| `0.10 - 0.15` | ~6-10 | ~1.4-2.0x | Noticeably faster. Check albedo and normals. |
| `0.20+` | 10+ | 2.0x+ | Aggressive. Temporal softening, lost detail. |

**Read those speedups as indicative, not measured.** The arithmetic is
exact — skipping *k* of *N* steps means `N / (N - k)` fewer DiT passes, so
8 skipped of 20 is 20/12 = 1.67x — but which threshold produces which
skip count depends on the clip, the mode, and the step count. A static
scene skips far more than a fast pan at the same threshold.

Two further reasons not to trust a table here: the speedups above are for
the **DiT passes only**, while VAE encode/decode and text encoding never
skip, so end-to-end gain is always lower; and the numbers assume 20 steps,
so at 4 steps (with a distill LoRA) there is almost nothing to skip —
first and last are always computed, leaving only 2 candidates.

**So measure instead of guessing.** Every run logs what actually
happened:

```
UniVidX_Sample: TeaCache: computed 12/20 DiT passes (skipped 8, 40%)
                — 1.67x fewer DiT passes.
```

Raise the threshold, watch that line and the output, and stop when the
damage shows.

**Start lower here than a stock-Wan2.1 guide would tell you**, for two
reasons specific to this model:

1. The polynomial that rescales the measured drift was fitted on stock
   Wan2.1, not on this four-stream fine-tune. Its calibration is
   approximate here.
2. In conditioned modes (`R2AIN` and friends) some rows of the timestep
   embedding are **pinned to a constant** — the conditioning rows never
   advance. Those constant rows drag the mean drift down, so the measured
   movement understates the real change and any given threshold skips
   more eagerly than the same number would on plain Wan2.1.

Raise it until you can see the damage, then back off.

## It costs VRAM — sometimes a lot

TeaCache holds two full hidden-state tensors on the GPU for the whole
run: `previous_hidden_states` and `previous_residual`. Each is
`(4 streams, tokens, 5120)` in bf16, so the cost scales with resolution
and window length:

| window | hidden state | TeaCache holds |
|---|---|---|
| 880x1280 x81 | 3.5 GB | **~7.0 GB** |
| 880x1280 x41 | 1.8 GB | ~3.7 GB |
| 832x480  x81 | 1.2 GB | ~2.5 GB |

That is on top of everything else, and it is why a job that renders fine
without caching can OOM with it. If you are near the VRAM ceiling,
TeaCache is the first thing to remove, not the last.

## Not compatible with context windowing

If the clip is long enough to be windowed, TeaCache is refused and the
log says so. The cache keeps one residual and one step counter, but the
model runs once per window per step — so the counter tracks windows
rather than denoising steps, and the stored residual belongs to whichever
window ran last. Reusing it across different parts of the clip would be
wrong, and on the shorter final window it does not even have a matching
shape.

Raise `context_frames` until the clip fits a single pass if you want
caching, or drop this node.

## Requires guidance off

TeaCache only engages when `cfg_scale` is 1.0. Above that the DiT runs
twice per step (positive and negative), and a single cache shared across
both would advance its step counter twice and blend the two passes'
residuals together. Rather than corrupt the result, the sampler logs a
warning and skips caching.

In practice this is not a restriction: conditioned modes already force
`cfg_scale=1.0` because their prompt is empty.

## Why there is no step range here

Other TeaCache implementations expose `start_step` / `end_step`. The one
vendored with UniVidX does not — it takes only the threshold, and always
computes the first and last step. Widgets for a range it ignores would be
worse than not having them.

## What to look for

Compare against an uncached render of the same seed. Caching artefacts
show up as **temporal smearing** — detail that should change between
frames staying put — most visibly in irradiance and in fine normal-map
detail. Flat albedo regions hide it well, so do not judge on those.
"""


def context_options_help_markdown() -> str:
    """Help page for the Context Options node."""
    return """# 🦇 UniVidX Context Options

Overrides how long clips are split into overlapping temporal windows.

**You usually do not need this node.** Windowing turns on automatically
once a clip is longer than one window, using the defaults below. Wire this
in only to change them.

## Why windowing exists

A single pass is bounded by VRAM, and UniVidX is unusually hungry: all
four modality streams attend as one sequence, so cost grows with the
square of the frame count. Around 81 frames at 720p you run out of card.

Windowing samples overlapping chunks and blends them **inside the
denoising loop** — at every step, not by cross-fading finished video — so
joins are far less visible than stitching renders together.

## Settings

| Widget | Default | Notes |
|---|---|---|
| `context_frames` | 81 | Window length in pixel frames. 81 is Wan 2.1's native training length and the largest verified single pass. |
| `context_overlap` | 16 | Frames shared between neighbours, cross-faded. Must be smaller than `context_frames`. |

Both are pixel frames; they are converted to latent frames internally
(4 pixel frames per latent, so 81 → 21 latent).

## The cost

Runtime scales with the **number of windows**, not the clip length. With
81-frame windows and 16 overlap, each window advances 65 frames, so a
161-frame clip is 3 windows ≈ 3x the render time of one 81-frame pass.
Budget accordingly — and this is exactly why the step-distill LoRA on the
loader matters so much for long clips.

More overlap means smoother joins and more windows. Less overlap is faster
and riskier.

## Two known limitations

**The last window can be short.** Upstream's tiler lets the final window
run to whatever is left rather than shifting it back to full length, so a
clip whose length is not a neat multiple may end on a window with much
less context. If the last second of a render looks weaker than the rest,
that is why — adjusting `context_frames` so the clip divides more evenly
is the workaround.

**Position encoding restarts per window.** Each window's temporal
positions begin at zero rather than continuing the global timeline. This
is inherited from the upstream implementation, not a local bug. Windows
are joined only by the blend, so very long clips can drift in overall
character from start to end. Overlap hides seams; it does not fix drift.
"""


# Map node id -> help markdown builder.
_HELP_PAGES = {
    "BAT_UniVidXLoader": loader_help_markdown,
    "BAT_UniVidXSample": sample_help_markdown,
    "BAT_UniVidXDecode": decode_help_markdown,
    "BAT_UniVidXCacheOptions": cache_options_help_markdown,
    "BAT_UniVidXContextOptions": context_options_help_markdown,
}


def write_help_docs(web_dir: str) -> None:
    """Write `<web_dir>/docs/<NodeName>.md` for each documented node.

    Called at import. Best-effort: a read-only install shouldn't stop the
    pack from loading, so failures are logged and swallowed. Files are
    only rewritten when the content actually changes, to avoid pointless
    writes (and mtime churn) on every ComfyUI start.
    """
    docs_dir = os.path.join(web_dir, "docs")
    try:
        os.makedirs(docs_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("UniVidX: couldn't create %s (%s); the built-in "
                       "node help won't be available.", docs_dir, exc)
        return

    for node_id, builder in _HELP_PAGES.items():
        path = os.path.join(docs_dir, f"{node_id}.md")
        content = builder()
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    if fh.read() == content:
                        continue
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
        except OSError as exc:
            logger.warning("UniVidX: couldn't write %s (%s).", path, exc)
