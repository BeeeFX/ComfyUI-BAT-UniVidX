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


# Map node id -> help markdown builder.
_HELP_PAGES = {
    "BAT_UniVidXSample": sample_help_markdown,
    "BAT_UniVidXDecode": decode_help_markdown,
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
