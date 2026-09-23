# 🦇 ComfyUI-BAT-UniVidX

**Take a video apart.** Feed in a clip, get back its albedo, lighting and surface normals — or its matte, foreground and background. Then put it back together however you like.

A ComfyUI wrapper for [UniVidX](https://github.com/houyuanchen111/UniVidX), built on Wan2.1-T2V-14B.

| Family | Give it | Get back |
|---|---|---|
| **Intrinsic** | a clip | **albedo** (base colour), **irradiance** (lighting), **normals** |
| **Alpha** | a composite | **matte**, **foreground**, **background** |

It runs in both directions, too — hand it albedo and lighting and it will rebuild the RGB. 30 modes in total, so you pick what you have and what you want.

---

## Why this one

- **It just runs.** Upstream ships inside the repo and the 57 GB of weights resolve themselves — reusing an existing Wan2.1 install if you already have one. No clone at render time, so an offline worker is fine.
- **Long clips work.** Past ~81 frames a single pass runs out of VRAM. Longer clips are split into overlapping windows and blended *inside* the denoising loop, automatically, so joins don't show.
- **The node shows only what you need.** Pick a mode and the sampler grows exactly the inputs that mode consumes. Never seven sockets when you need one.
- **It won't quietly lie to you.** The combinations that produce plausible-but-wrong output — a missing adapter stack, an image-to-video LoRA, caching that reuses the wrong window — are caught and refused, loudly, instead of rendered.
- **Every knob is explained in the app.** Click the **?** on any node for what it costs, when it's worth it, and real measured numbers.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BeeeFX/ComfyUI-BAT-UniVidX.git ComfyUI-BAT-UniVidXWrapper

conda activate <your_comfyui_env>
cd ComfyUI-BAT-UniVidXWrapper
pip install -r requirements.txt
```

Start ComfyUI. The nodes are under **BAT/UniVidX**.

First run downloads the Wan2.1-T2V-14B backbone (~57 GB) and the UniVidX checkpoint (~0.8 GB) unless you already have them — it checks `ComfyUI/models/` first and reuses what it finds. Needs a recent ComfyUI (V3 node schema) and `comfyui-frontend-package` **1.45.x or newer**.

> Clone into `ComfyUI-BAT-UniVidXWrapper` exactly as above — the repo and the pack directory are deliberately named differently.

## Getting started

```
[Video Loader] ── images ──┐
                           ├─→ [🦇 UniVidX Sample] ── result ─→ [🦇 UniVidX Decode] ─→ 4 IMAGE + matte MASK
[🦇 (Down)load UniVidX] ───┘
```

Three nodes is the whole graph. On the loader pick `intrinsic` or `alpha`; on the sampler pick a mode. `R2AIN` (clip → albedo + irradiance + normals) and `R2PFB` (clip → matte + fg + bg) are the two you'll reach for most.

Mode names read as `<what you give>2<what you get>`:

| Family | Letters |
|---|---|
| Intrinsic | **R**=RGB, **A**=Albedo, **I**=Irradiance, **N**=Normal |
| Alpha | **R**=composite, **P**=Pha (matte), **F**=Foreground, **B**=Background |

| Mode | Meaning |
|---|---|
| `R2AIN` | clip → albedo + irradiance + normals |
| `R2PFB` | clip → matte + foreground + background |
| `RA2IN` | clip + albedo → irradiance + normals |
| `AIN2R` | albedo + irradiance + normals → relit clip |
| `t2RAIN` | text only → all four (no input clip) |

Click the **?** on the Sample node for all 30.

## The nodes

| Node | What it does |
|---|---|
| **🦇 (Down)load UniVidX Model** | Picks the variant and resolves weights. Caches the loaded model, so re-running a graph doesn't pay the load cost twice. All the speed/VRAM settings live here. |
| **🦇 UniVidX Sample** | The sampler. The `mode` dropdown reshapes the node — sockets and widgets both — so you can't wire an input the mode ignores. Shows per-step progress, and Cancel stops it at the next step. |
| **🦇 UniVidX Decode** | Splits the result into 4 IMAGE outputs, named for both families, plus the matte as a MASK. Outputs come back at exactly the input clip's resolution and frame count. Slots a mode didn't produce come back black, so your downstream wiring survives a mode change. |
| **🦇 UniVidX Context Options** *(optional)* | Only if you want to change the windowing defaults for long clips. Windowing is automatic without it. |
| **🦇 UniVidX Cache Options** *(optional)* | Only if you want TeaCache step-skipping. Off unless wired. |

## Speed and VRAM

This is a 14B video model — it is heavy. A 720p, 81-frame render at 20 steps takes hours on a 94 GB card out of the box.

Three things are **on by default** and none of them changes your output: FP8 quantisation (halves DiT VRAM), the fastest available attention backend, and an auto-sized VRAM reservation.

Worth turning on, in order of payoff:

| Knob | Gets you | Costs you |
|---|---|---|
| **`distill_lora`** | **~5× faster.** Merge a Wan2.1-**T2V**-14B step-distill LoRA, then run 4 steps at `cfg_scale=1.0`. | Quality. It shifts the base those adapters were trained against — compare albedo and normals before trusting it. An I2V LoRA merges *without error* and quietly degrades output; the loader warns on suspicious filenames. |
| **Cache Options** (TeaCache) | ~1.4–1.9× on 20-step renders. | Several GB of VRAM (~7 GB at 880×1280×81) and some fidelity. First thing to remove if you're near the ceiling. Pointless below ~8 steps. |
| **`compile_dit`** | Some per-step speedup (unmeasured on this model). | Minutes compiling on the first step. Not worth it on short or 4-step runs. |

**`vram_buffer_gb` is the block-swapping dial**, and the one to reach for when you OOM. It's how much VRAM to keep **free** for the sampler — layers past the limit stream CPU↔GPU on demand — so *raise* it for more headroom, lower it for speed.

| | Value |
|---|---|
| Recommended | `0` = auto — sized from your card, and logged at load (~39 GB on a 95 GB card) |
| Min / max | `0.0`–`80.0`, but the useful manual range is ~8 up to about (card VRAM − 16 GB) |
| Typical manual | roughly a third of the card |

FP8 and block swapping work together — quantised layers stay fp8 in every state *and* keep streaming, so the resident copy is 1 byte per parameter. (They used to be mutually exclusive; quantisation replaced the wrappers that did the streaming.)

**Long clips**: peak VRAM is set by the **window**, not the clip. A 300-frame render in 81-frame windows costs the same per pass as an 81-frame one — it just does more passes. If you OOM, lower `context_frames`; if it's slow, that's the window count.

Every setting has fuller guidance behind the **?** on its node, including measured numbers.

## Troubleshooting

- **CUDA OOM at sample time** — the model fitted, the sampler's peak didn't. In order: remove Cache Options if wired, lower `context_frames`, reduce resolution, raise `vram_buffer_gb`.
- **`mode 'XYZ' is family='alpha' but loaded model is variant='intrinsic'`** — mode and loader variant disagree. Change one.
- **Tensor size mismatch while sampling** — TeaCache combined with windowing. It should now refuse that and say so; if you see it, the guard was bypassed.
- **`the upstream copy at … is incomplete`** — files are missing from `./UniVidX/`, usually a partial checkout or a deploy that didn't copy everything. `git checkout -- UniVidX` in the pack directory restores it.
- **`No module named 'src.pipelines'`** — another custom node has claimed the top-level `src` name (several do). The wrapper rebinds it and logs when it does; if it still fails, see the previous item.
- **First run takes a long time** — model load plus any first-time download. Later runs reuse everything.

---

# Details

Everything below is *why*, not *how to use*. Skip it unless something surprised you.

## Where the weights come from

1. **Auto-detect** — scans `ComfyUI/models/{unividx,diffusion_models,checkpoints,unet,wan,Wan2.1}/` for an existing Wan2.1-T2V-14B in the original HuggingFace layout, and reuses it with zero extra disk.
2. **Auto-download** — otherwise fetches `Wan-AI/Wan2.1-T2V-14B` into `ComfyUI/models/unividx/`.
3. Same for the per-variant UniVidX checkpoint.

Only **Wan2.1-T2V-14B** works. `Wan2.1-VACE-14B`, `Wan2.1-I2V-14B` and single-file Kijai-style consolidations are rejected on purpose — different fine-tunes, or a layout UniVidX doesn't expect.

## Upstream ships vendored

`./UniVidX/` is a pinned copy of upstream (~1 MB; the demo `assets/` are stripped), not a runtime clone. It used to be cloned on first use and gitignored, which broke every deploy that mirrors with `rsync --delete`: the destination's clone was erased while its `.git` survived, and the loader then failed on a directory that looked present. See [UniVidX/VENDORED.md](UniVidX/VENDORED.md).

Upstream's rough edges are patched **at runtime** from `unividx_wrapper_runtime/advanced.py`, never by editing the vendored tree, so it stays a clean diff against upstream.

## Context windowing

All four modality streams attend as a single sequence, so cost grows with the square of the frame count — which is why ~81 frames is the wall.

Longer clips are split into overlapping windows and blended at every denoising step, rather than cross-fading finished video. Upstream has a tiler for this but it is unrunnable as shipped: the recursive call drops `training_mode` and dies on an `UnboundLocalError`, and its blend mask has no zero-width guard. Both are repaired in `install_sliding_window_support`.

Every window is full length: when the stride doesn't land the last window on the end of the clip, it is shifted back to finish flush with it (upstream let it run short, starving the tail of context). Blending is a true crossfade over each actual overlap, accumulated in float32. One known limit remains: positional encoding restarts per window, so overlap hides seams but doesn't prevent slow drift across a very long clip.

## Why the sampler reshapes itself

The Sample and Decode nodes use ComfyUI's V3 schema. Sockets and widgets are decided **server-side** by `io.DynamicCombo`, not hidden by JavaScript — the only JS in the pack is the help affordances.

That's also why Decode's outputs are fixed — four IMAGE slots plus the matte MASK: ComfyUI has no dynamic-output mechanism. `DynamicCombo` and friends are input-only, V3's `DynamicOutput` is an unimplemented stub, and outputs are memoised class-wide on first schema resolution. Rather than fake it, every slot is always present and unproduced ones return black.

Both renderers are supported — the classic canvas ("Nodes 1.0") and the Vue node renderer ("Nodes 2.0"). `DynamicCombo` is applied to the node's own inputs and widgets, which both draw from; the help is reachable in both (see below). One constraint follows from how saved workflows store widget values — by position — so widgets are never removed or reordered, only added at the end or retuned in place. `denoising_strength` and the loader's `dtype` survive for that reason even though neither is useful (see their tooltips).

The two options nodes are separate rather than one combined node because they're unrelated and each changes output pixels — wiring one should be deliberate. Optional *sockets* also leave saved workflows untouched, where extra widgets would shift the stored value array.

## Text prompts are opt-in on image-conditioned modes

Only `t2…` modes always show `prompt` / `negative_prompt` / `cfg_scale`. Modes with an IMAGE input hide them behind a toggle, because they're driven by imagery — upstream's own `R2PFB` config ships `prompt: ""`, and its inference scripts do:

```python
if prompt == "":
    inference_params["cfg_scale"] = 1.0
```

With the toggle off you get no prompt and no guidance, which is both what upstream does and the faster path: `cfg_scale == 1.0` skips the negative pass, halving model calls per step. Leaving the prompt blank with the toggle on drops `cfg_scale` to 1.0 automatically rather than silently charging you double.

Worth knowing: the negative prompt only matters while guidance is on — the pipeline evaluates it solely inside `if cfg_scale != 1.0`, so a default negative can't act as a free quality boost.

## Where output geometry comes from

For any mode with an IMAGE input, frame count and resolution come **from the wired clip**. That's why those modes have no `num_frames`/`height`/`width` widgets — the values would only be overridden. Text-only modes have nothing to measure, so they're the only ones that expose them.

The model needs H/W on a multiple of 16 and a `4k+1` frame count. The clip is edge-padded (never stretched) up to those, and Decode crops and trims the padding off again — so a 1920×1080 plate comes back as 1920×1080, pixel-aligned with the source, ready to composite over it.

## Reaching the mode reference

| Where | Works in |
|---|---|
| The **?** in the node title bar | Canvas renderer only |
| Right-click → 🦇 UniVidX mode reference | Any renderer |
| ComfyUI's own node help panel | Any renderer — reads `web/docs/<NodeName>.md` |

Those markdown files are generated from `modes.py` at import rather than hand-written, so the help can't drift from the modes the nodes actually offer.

## Deploying

`UniVidX/` is committed, so a deploy that mirrors the repo carries it. Two subdirectories are gitignored because the wrapper creates them at load time as symlinks into `ComfyUI/models/`: `UniVidX/models/` and `UniVidX/checkpoints/`. Don't sync those. (`UniVidX/src/models/` is different — that is upstream's model *code*, and it must ship.)

## Tests

```bash
pip install pytest
COMFYUI_PATH=/path/to/ComfyUI python -m pytest tests
```

No GPU or weights needed: the tests run the real vendored upstream code on CPU and a stub in place of the 14B pipeline. Without `COMFYUI_PATH` the node-level tests skip and the rest still run.

## License + credits

**GPL v3.** See [LICENSE](LICENSE).

The FP8 quantization, distill LoRA pre-merge, torch.compile and sage attention code paths (`unividx_wrapper_runtime/fp8_loader.py`, `lora_merge.py`, and helpers in `advanced.py`) are derived from the `UniVidX_ComfyUI` pack, which is also GPL v3. This combined wrapper is re-licensed GPL v3 so the chain stays consistent.

**Redistributed here:**

- [UniVidX](https://github.com/houyuanchen111/UniVidX) (**Apache 2.0**, © The UniVidX Authors) — vendored in [`./UniVidX/`](UniVidX/) at commit `382b9002`. Those files stay Apache 2.0 with their own [LICENSE](UniVidX/LICENSE); only the wrapper is GPL v3. Modifications are itemised in [UniVidX/VENDORED.md](UniVidX/VENDORED.md), as Apache 2.0 §4(b) requires.

**Depended on, not redistributed:**

- [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) (Apache 2.0) — downloaded at runtime
- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) (BSD 3-Clause) / [SageAttention](https://github.com/thu-ml/SageAttention) — optional

[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Apache 2.0) informed the context-window defaults and options-node UX. No code was copied.

See [NOTICE](NOTICE) for full attribution.
