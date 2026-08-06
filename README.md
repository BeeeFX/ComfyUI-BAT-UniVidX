# ComfyUI-BAT-UniVidXWrapper

A streamlined ComfyUI wrapper for [houyuanchen111/UniVidX](https://github.com/houyuanchen111/UniVidX) — unified video intrinsic decomposition (RGB ↔ Albedo / Irradiance / Normal) and video matting (composite ↔ Pha / Foreground / Background) built on Wan2.1-T2V-14B.

This is a from-scratch alternative to the existing `UniVidX_ComfyUI` wrapper. Both can coexist in the same `custom_nodes/` directory — different node-class names. The BAT variant prioritises three things:

- **Painless install.** One pip command. Upstream ships vendored, and weights auto-resolve — no clone at render time, so an offline worker still runs.
- **Streamlined UI.** Three core nodes, plus two optional ones you wire only when you need them. The Sample node only *has* the IMAGE sockets the chosen mode needs, so you never see all 7 at once.
- **Robustness fixes baked in.** Frame counts auto-pad to WAN's 4k+1 constraint, H/W auto-snap to multiples of 16, family/variant mismatch raises a clear error naming both sides, and the VRAM reservation is sized from the actual card.
- **Past the 81-frame wall.** Long clips are windowed and blended inside the denoising loop, automatically. Upstream's tiler crashes on the first call; it is repaired at runtime.
- **Nothing silently wrong.** The combinations that produce plausible-but-incorrect output — TeaCache with windowing, an I2V distill LoRA, a pipeline missing its adapters — are detected and refused or warned about, loudly, rather than rendered.

The Sample and Decode nodes use ComfyUI's V3 node schema; the Sample node's sockets and widgets are decided server-side by `io.DynamicCombo` rather than hidden by JavaScript. The only JS in the pack is the help affordances, which are purely presentational.

## Install

```bash
# 1. Drop the wrapper into ComfyUI/custom_nodes/
cd ComfyUI/custom_nodes
git clone https://github.com/BeeeFX/ComfyUI-BAT-UniVidX.git ComfyUI-BAT-UniVidXWrapper

# 2. Activate the env ComfyUI runs under and install dependencies once.
conda activate <your_comfyui_env>
cd ComfyUI-BAT-UniVidXWrapper
pip install -r requirements.txt
```

That's it. Start ComfyUI. The nodes appear under the **🦇 BAT/UniVidX** category.

> The repo is named `ComfyUI-BAT-UniVidX`, but the node pack directory is
> `ComfyUI-BAT-UniVidXWrapper` — hence the explicit target directory in the
> clone command above. Keep that directory name so the vendored-upstream
> and help-doc paths resolve as documented.

### ComfyUI version requirement

The Sample node uses the **V3 node schema** (`comfy_api.latest.io`) and specifically `io.DynamicCombo`, so it needs a ComfyUI recent enough to ship both. If `from comfy_api.latest import io` fails or `io.DynamicCombo` is missing, the pack won't load — update ComfyUI.

`io.DynamicCombo` also needs frontend support to render the per-mode sockets. **`comfyui-frontend-package` 1.45.x has it; 1.28.x does not** (the dynamic-input types are absent from that bundle entirely). If the mode dropdown doesn't reshape the node, check your installed frontend version against the pin in ComfyUI's own `requirements.txt`.

### What happens on first node use (transparent, one-time)

- **Upstream UniVidX ships with this repo**, vendored at a pinned commit in `./UniVidX/` (~1 MB — the demo `assets/` are stripped). Nothing is cloned at render time, so a worker with no network still runs. See [UniVidX/VENDORED.md](UniVidX/VENDORED.md) for provenance and the exact modifications.
- **Upstream's rough edges are patched at runtime**, from `unividx_wrapper_runtime/advanced.py`, so the vendored tree stays a clean diff against upstream. That covers the attention backend, the unrunnable sliding-window path, and a soft-import of `torchvision.io.write_video` (removed in modern torchvision).
- **Weight resolution** happens in this priority order:
  1. **Auto-detect** — the loader scans `ComfyUI/models/{unividx,diffusion_models,checkpoints,unet,wan,Wan2.1}/` looking for an existing Wan2.1-T2V-14B install in the original HuggingFace directory layout. If found, it's reused with zero extra disk.
  2. **Auto-download** — if nothing's found, `Wan-AI/Wan2.1-T2V-14B` is fetched from HuggingFace (~57 GB, one-time, into `ComfyUI/models/unividx/Wan2.1-T2V-14B/`).
  3. Same logic for the per-variant UniVidX checkpoint (`univid_intrinsic.safetensors` or `univid_alpha.safetensors`, ~0.8 GB each).
- All downloads stream with a ComfyUI ProgressBar. Subsequent runs reuse everything.

### Wan2.1 variant compatibility

The auto-detect only matches **Wan2.1-T2V-14B**. The following are **NOT** compatible despite sharing the DiT shape — UniVidX was trained on the T2V-14B base specifically:

- ❌ `Wan2.1-VACE-14B` (Video-Aware Control Extension — different fine-tuning)
- ❌ `Wan2.1-I2V-14B` (image-to-video — different fine-tuning)
- ❌ Single-file Kijai-style consolidations (`wan2.1_t2v_14b_fp16.safetensors` etc.) — UniVidX expects the original HF directory layout. Future enhancement could add a shim.

If you have one of these and the loader rejects it, that's working as intended.

## Usage

```
[Video Loader] ── images ──┐
                           ├─→ [🦇 UniVidX Sample] ── result ─→ [🦇 UniVidX Decode] ─→ 4 IMAGE outputs
[🦇 (Down)load UniVidX] ───┘
```

### Nodes

| Node | What it does |
|---|---|
| **🦇 (Down)load UniVidX Model** | Pick `variant` (intrinsic vs alpha) + `dtype` (bf16 default) + `wan21_source` (auto-detect by default). Returns an opaque `UNIVIDX_MODEL`. Caches loaded models so re-running the graph doesn't pay the load cost again. |
| **🦇 UniVidX Sample** | The streamlined sampler. The `mode` dropdown at the top is the master widget — both the IMAGE sockets *and* the mode-specific widgets are created to match it. Pick `t2RAIN` and you get geometry plus the text widgets but no IMAGE sockets; pick `R2AIN` and you get the `rgb` socket, `denoising_strength` and an `optional_prompt` toggle — the geometry widgets are gone because the wired clip decides them, and the text widgets are gone until you ask for them. Because a mode only has the sockets it consumes, it's impossible to wire an input the mode will ignore. Click the **?** in the title bar for a full mode reference. |
| **🦇 UniVidX Decode** | Splay the result into 4 IMAGE outputs. Slots are named for both families — intrinsic gives (rgb, albedo, irradiance, normal), alpha gives (rgb, pha, fgr, bgr). Slots a mode didn't produce come back black (and the console names which), so downstream wiring survives a mode change. |
| **🦇 UniVidX Context Options** *(optional)* | Override the temporal-window defaults for long clips. Windowing is automatic above 81 frames — wire this only to change `context_frames` / `context_overlap`. Plugs into the Sample node's `context_options` socket. |
| **🦇 UniVidX Cache Options** *(optional)* | Enable TeaCache step-skipping (~1.4–1.9×, at some fidelity and several GB of VRAM). Off unless wired. Plugs into `cache_options`. |

Both optional nodes are separate rather than one combined "options" node: they are unrelated, and each changes output pixels, so wiring one should be deliberate. Unwired optional *sockets* also leave saved workflows untouched, whereas extra widgets would shift the stored value array.

#### Getting at the mode reference

Three ways in, so it's reachable however you have ComfyUI rendering nodes:

| Where | Works in |
|---|---|
| The **?** button in the Sample node's title bar | Canvas renderer only — title buttons are canvas-drawn, and the Vue node renderer's header has no title-button hook |
| **Right-click the node → 🦇 UniVidX mode reference** | Any renderer |
| ComfyUI's own node help panel (and the node-library sidebar) | Any renderer — reads `web/docs/<NodeName>.md` |

Those markdown files are generated from `modes.py` at import (into `web/docs/`, gitignored) rather than hand-written, so the help can't drift out of sync with the modes the nodes actually offer. Same reason the `?` panel fetches from `/bat_unividx/modes` instead of duplicating the table in JS.

#### Text prompts are opt-in on image-conditioned modes

Only the text-only (`t2…`) modes always show `prompt` / `negative_prompt` / `cfg_scale`. Every mode with an IMAGE input hides them behind an `optional_prompt` toggle, because those modes are driven by the imagery, not by text — upstream's own `R2PFB` example config ships `prompt: ""`, and both its inference scripts do:

```python
if prompt == "":
    inference_params["cfg_scale"] = 1.0
```

The wrapper now mirrors that. With the toggle off (default) you get no prompt and no guidance, which is both what upstream does and the faster path — `cfg_scale == 1.0` makes the pipeline skip the negative pass, so sampling costs one model call per step instead of two. Flip it on to reveal the three widgets; leaving the prompt blank while it's on drops `cfg_scale` to 1.0 automatically rather than silently charging you double.

One caveat worth knowing: the negative prompt only matters while guidance is on. The pipeline evaluates the negative branch solely inside `if cfg_scale != 1.0`, so with guidance disabled it's never computed — a default negative prompt can't act as a free quality boost. The default (now upstream's actual Chinese negative string, not the English placeholder this wrapper previously shipped) is passed through regardless, so it's live the moment you enable the toggle.

#### Where output geometry comes from

For any mode with an IMAGE input, frame count and resolution are taken **from the wired clip** — frame count from its length, H/W from its frame size (snapped up to a multiple of 16). That's why those modes have no `num_frames`/`height`/`width` widgets: the values would only be overridden. Text-only (`t2…`) modes have nothing to measure, so they're the only ones that expose those three widgets.

#### Why Decode always has exactly 4 outputs

It can't adapt to the connected Sample node's mode. ComfyUI has no dynamic-output mechanism: `DynamicCombo` and its siblings are input-only, the V3 `DynamicOutput` class is an unimplemented stub, and a node's outputs are memoised class-wide on first schema resolution — so they can't vary per instance. Rather than fake it, all four slots are always present and named for both families; unproduced slots return black.

### Mode naming refresher

Naming is `<conditions>2<targets>`. Letters tell you which modalities go IN and which come OUT.

| Family | Letters |
|---|---|
| Intrinsic | **R**=RGB, **A**=Albedo, **I**=Irradiance, **N**=Normal |
| Alpha | **R**=composite RGB, **P**=Pha (alpha matte), **F**=Foreground, **B**=Background |

Examples:

| Mode | Family | Meaning |
|---|---|---|
| `t2RAIN` | intrinsic | text → RGB+Albedo+Irradiance+Normal (no IMAGE input) |
| `R2AIN` | intrinsic | RGB → Albedo+Irradiance+Normal |
| `RA2IN` | intrinsic | RGB+Albedo → Irradiance+Normal |
| `R2PFB` | alpha | RGB → Pha+Fgr+Bgr (video matting) |
| `RP2FB` | alpha | RGB+Pha matte → Fgr+Bgr |
| `RPF2B` | alpha | RGB+Pha+Fgr → Bgr |

All 30 modes from upstream are exposed.

## Long clips: context windowing

A single pass is bounded by VRAM, and UniVidX is unusually hungry — all four modality streams attend as one sequence, so cost grows with the square of the frame count. Around **81 frames** you run out of card.

Past that, the clip is split into **overlapping temporal windows, blended inside the denoising loop** (not cross-faded afterwards), so joins are far less visible than stitching renders together. This happens **automatically**; wire a **🦇 UniVidX Context Options** node only to change the defaults (81-frame windows, 16 frames of overlap).

Two things worth knowing:

- **Peak VRAM is set by the window, not the clip.** A 300-frame render in 81-frame windows costs the same per pass as an 81-frame render — it just does it more times. If you OOM, lower `context_frames`; raising the clip length alone will not help you.
- **Runtime scales with the window count.** 81-frame windows with 16 overlap advance 65 frames each, so a 161-frame clip is 3 passes ≈ 3× the sampling time.

Upstream's tiler is unrunnable as shipped — its recursive call drops `training_mode` and dies on an `UnboundLocalError`, and its blend mask has no zero-width guard. Both are repaired at runtime; see `install_sliding_window_support` in `unividx_wrapper_runtime/advanced.py`.

## Speed and VRAM

Every widget on the loader is honoured — there is no "advanced" gate. (One existed; it hid nothing and silently discarded the settings below it.) **Click the ? on any node for the full guidance** — what each knob costs, when it pays off, and measured numbers where they exist.

On by default, none of which changes output pixels:

| Setting | Default | Why |
|---|---|---|
| `fp8_quantize` | `qfloat8_prequantized` | Halves DiT VRAM (~28 → ~14 GB). A bf16 DiT leaves no room for activations. |
| `attention_backend` | `auto` | FlashAttention-2 if importable, else SDPA. |
| `vram_buffer_gb` | `0` = auto | Sized from the card's real VRAM; the chosen value is logged. |

Opt-in, roughly in order of payoff:

| Knob | Effect | Cost |
|---|---|---|
| **`distill_lora`** (loader) | Merge a **Wan2.1-T2V-14B** step-distill LoRA, then run 4 steps at `cfg_scale=1.0` — about **5× faster**. | The riskiest setting here. It merges into the frozen base *under* UniVidX's four per-modality adapters — no key collision, but those adapters were trained against the *original* base. Compare albedo/normals side by side before trusting it. An **I2V** LoRA merges without error and quietly degrades output; the loader warns on suspicious filenames. |
| **🦇 UniVidX Cache Options** (TeaCache) | Skip the DiT on steps where little changes, ~1.4–1.9×. | Holds two hidden-state tensors for the whole run — **~7 GB at 880×1280×81**. First thing to remove near the VRAM ceiling. Refused with windowing, and needs `cfg_scale=1.0`. Pointless below ~8 steps. |
| **`compile_dit`** (loader) | `torch.compile` the DiT. Verified to trace cleanly through FP8Linear, peft `adapter_names` and weight offload — zero graph breaks. | Tens of seconds of compile on the first step. Speedup unmeasured on this model. Not worth it on short or 4-step runs. |
| **`attention_backend`** (loader) | A few percent at most. torch's SDPA is *already* a fused FlashAttention kernel — the naive path is 13× slower, so there is no big win here. Measured on sm_86: SDPA 14.4 ms, FA2 13.9 ms, Sage 16.7 ms. | Sage was slower *and* less exact there, so it is excluded from `auto`. |

`vram_buffer_gb` is how much VRAM to keep **FREE** for activations, not how much to use — higher means more offloading and more headroom. Note FP8-quantized layers cannot be offloaded at all, so on an FP8 run this reservation is the sampler's only protection.

The defensive **SDPA restore** runs on every load regardless: if another custom node (e.g. ComfyUI-3D-Pack's Stable3DGen) monkey-patched `F.scaled_dot_product_attention` globally, we undo it so UniVidX's VAE doesn't crash on 1-head SDPA calls.

## Troubleshooting

- **First run is multi-minute** — model load plus any first-time HF download. Subsequent runs reuse the cache.
- **`mode 'XYZ' is family='alpha' but loaded model is variant='intrinsic'`** — the family doesn't match the loader's variant. Either pick a different mode or reload with the other variant. Checked when the node runs (the schema can't express "this mode requires that loader variant"), so it surfaces at queue time rather than while editing.
- **CUDA OOM at sample time** — the model fitted but the sampler's peak did not. In order: remove the Cache Options node if wired (several GB), lower `context_frames`, then reduce resolution. Raising `vram_buffer_gb` above auto also helps.
- **Tensor size mismatch during sampling** — almost always TeaCache combined with windowing; the guard should now refuse that combination and say so in the log.
- **`No module named 'src.pipelines'`** — another custom node owns the top-level `src` name (several ship one). The wrapper rebinds it during import and logs when it does; if you still see this, the vendored `./UniVidX/src/` is missing or incomplete.

## Deploying

`UniVidX/` is committed, so a deploy that mirrors the repo (`rsync --delete`, `comfyctl`, etc.) carries it. Two subdirectories are gitignored because the wrapper creates them at load time as symlinks into `ComfyUI/models/`: `UniVidX/models/` and `UniVidX/checkpoints/`. Don't sync those.

## License + credits

**GPL v3.** See [LICENSE](LICENSE).

The FP8 quantization, distill LoRA pre-merge, torch.compile and sage attention code paths (specifically `unividx_wrapper_runtime/fp8_loader.py`, `unividx_wrapper_runtime/lora_merge.py`, and the helper functions in `unividx_wrapper_runtime/advanced.py`) are derived from the `UniVidX_ComfyUI` custom-node pack, which is also GPL v3. We re-license this combined wrapper under GPL v3 so the licensing chain stays consistent and downstream forks have a clear single license to comply with.

This repo **redistributes**:

- [UniVidX](https://github.com/houyuanchen111/UniVidX) (**Apache 2.0**, © The UniVidX Authors) — vendored in [`./UniVidX/`](UniVidX/) at commit `382b9002`. Those files stay under Apache 2.0 with their own [LICENSE](UniVidX/LICENSE); only the wrapper around them is GPL v3. Modifications are listed in [UniVidX/VENDORED.md](UniVidX/VENDORED.md) as Apache 2.0 §4(b) requires.

And depends on, but does not redistribute:

- [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) (Apache 2.0) — downloaded at runtime
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) (Apache 2.0) — pip dependency
- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) (BSD 3-Clause) / [SageAttention](https://github.com/thu-ml/SageAttention) — optional, used by the attention dispatcher when importable

[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Apache 2.0) informed the context-window defaults and options-node UX. No code was copied.

See [NOTICE](NOTICE) for the full attribution.
