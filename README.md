# ComfyUI-BAT-UniVidXWrapper

A streamlined ComfyUI wrapper for [houyuanchen111/UniVidX](https://github.com/houyuanchen111/UniVidX) — unified video intrinsic decomposition (RGB ↔ Albedo / Irradiance / Normal) and video matting (composite ↔ Pha / Foreground / Background) built on Wan2.1-T2V-14B.

This is a from-scratch alternative to the existing `UniVidX_ComfyUI` wrapper. Both can coexist in the same `custom_nodes/` directory — different node-class names. The BAT variant prioritises three things:

- **Painless install.** One pip command, then the wrapper auto-clones upstream and auto-resolves weights.
- **Streamlined UI.** Three nodes (was five). The Sample node only *has* the IMAGE sockets the chosen mode needs, so you never see all 7 at once.
- **Robustness fixes baked in.** Frame counts auto-pad to WAN's 4k+1 constraint, H/W auto-snap to multiples of 16, family/variant mismatch raises a clear error naming both sides.

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

That's it. Start ComfyUI. The three new nodes appear under the **🦇 BAT/UniVidX** category.

> The repo is named `ComfyUI-BAT-UniVidX`, but the node pack directory is
> `ComfyUI-BAT-UniVidXWrapper` — hence the explicit target directory in the
> clone command above. Keep that directory name so the auto-clone and
> help-doc paths resolve as documented.

### ComfyUI version requirement

The Sample node uses the **V3 node schema** (`comfy_api.latest.io`) and specifically `io.DynamicCombo`, so it needs a ComfyUI recent enough to ship both. If `from comfy_api.latest import io` fails or `io.DynamicCombo` is missing, the pack won't load — update ComfyUI.

`io.DynamicCombo` also needs frontend support to render the per-mode sockets. **`comfyui-frontend-package` 1.45.x has it; 1.28.x does not** (the dynamic-input types are absent from that bundle entirely). If the mode dropdown doesn't reshape the node, check your installed frontend version against the pin in ComfyUI's own `requirements.txt`.

### What happens on first node use (transparent, one-time)

- **Upstream UniVidX is auto-cloned** into `ComfyUI-BAT-UniVidXWrapper/UniVidX/` (~25 MB, shallow clone, pinned to a known-good commit).
- **Two upstream patches are auto-applied** to the clone: a missing `__init__.py` and a soft-import of `torchvision.io.write_video` (removed in modern torchvision). You never have to touch upstream code.
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

## Advanced settings

Flip `advanced_settings: True` on the loader to expose the perf knobs. All are off by default — the everyday path stays simple. Each carries a tooltip explaining what it does, when it's worth turning on, and what it costs.

| Knob | What it does | When to use |
|---|---|---|
| **`fp8_quantize: qfloat8_prequantized`** | Substitutes the DiT's ~400 `nn.Linear` modules with FP8Linear (per-tensor absmax scaling, runtime-quantized from the bf16 cold-load). Halves DiT VRAM (~28 GB → ~14 GB). LoRA adapter layers stay bf16. | When you want to fit the full DiT residently on a 32 GB card and skip DiffSynth's CPU↔GPU streaming. |
| **`distill_lora: lightx2v`** + **`distill_strength: 1.0`** | Pre-merges the LightX2V step-distill LoRA into the DiT's base weights before VRAM-management wrapping. With `num_inference_steps=4` and `cfg_scale=1.0` you get ~5× faster wall-clock at a small quality cost. Requires the LoRA file at `models/loras/lightx2v/` (the loader's error message names the exact path + download command). | When you're iterating and willing to trade a few dB of fidelity for big speed wins. |
| **`compile_dit: True`** | `torch.compile(dit, mode='reduce-overhead', dynamic=True)`. First sampler step pays a 60-120 sec graph capture cost; subsequent steps are ~20-30% faster on Blackwell/Ada. | When you'll do many sampler runs at the same resolution and frame count (re-compile fires when shapes change). |
| **`prefer_sage_attn: True`** | Replaces DiffSynth's attention dispatcher with a SageAttention-first wrapper (falls back to FA2 then SDPA per call). Requires `pip install sageattention`. Only kicks in for `head_dim ∈ {64, 96, 128}`; UniVidX's CMSA path stays on native SDPA for correctness. | When you've installed sage and want the extra speed on the non-CMSA attention calls. |

The defensive **SDPA restore** runs on every load regardless: if another custom node (e.g. ComfyUI-3D-Pack's Stable3DGen) monkey-patched `F.scaled_dot_product_attention` globally, we undo it so UniVidX's VAE doesn't crash on 1-head SDPA calls.

## Troubleshooting

- **`git not found on PATH`** during first node use — install git and restart ComfyUI. The auto-clone needs it once.
- **First run is multi-minute** — that's the model load + the first-time HF download. Subsequent runs reuse the cache.
- **`mode 'XYZ' is family='alpha' but loaded model is variant='intrinsic'`** — the family doesn't match the loader's variant. Either pick a different mode or reload the model with the other variant. This is checked when the node runs (the schema can't express "this mode requires that loader variant"), so it surfaces at queue time rather than while editing.
- **`No module named 'src.pipelines'`** — the wrapper's auto-patch should have fixed this on clone. If you see it anyway, delete `./UniVidX/` and re-run; auto-clone will re-fetch + re-patch.
- **CUDA OOM** — lower `vram_buffer_gb` (4.0 → 2.0) on the loader, or reduce `num_frames` / `height` / `width` on the sampler. The 14B model is genuinely large.

## License + credits

**GPL v3.** See [LICENSE](LICENSE).

The FP8 quantization, distill LoRA pre-merge, torch.compile and sage attention code paths (specifically `unividx_wrapper_runtime/fp8_loader.py`, `unividx_wrapper_runtime/lora_merge.py`, and the helper functions in `unividx_wrapper_runtime/advanced.py`) are derived from the `UniVidX_ComfyUI` custom-node pack, which is also GPL v3. We re-license this combined wrapper under GPL v3 so the licensing chain stays consistent and downstream forks have a clear single license to comply with.

This wrapper depends on, but does not redistribute:

- [UniVidX](https://github.com/houyuanchen111/UniVidX) (Apache 2.0) — cloned at runtime
- [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) (Apache 2.0) — downloaded at runtime
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) (Apache 2.0) — pip dependency

See [NOTICE](NOTICE) for the full attribution.
