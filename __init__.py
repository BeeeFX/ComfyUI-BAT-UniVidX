"""ComfyUI-BAT-UniVidXWrapper — streamlined wrapper around UniVidX.

A from-scratch alternative to the existing UniVidX_ComfyUI wrapper,
focused on three things:

  1. Painless install. `pip install -r requirements.txt` + restart
     ComfyUI. Upstream UniVidX auto-clones on first node use. Wan2.1
     and the UniVidX checkpoints auto-download (or reuse existing
     copies anywhere in ComfyUI/models/). No git submodules, no
     hand-applied patches, no `--init --recursive` footgun.

  2. Streamlined UI. Three nodes (was five), and the Sample node's
     IMAGE sockets appear/disappear based on the chosen mode — pick
     R2AIN and only the `rgb` socket exists. This uses the V3
     (`io.DynamicCombo`) schema, so the sockets are materialised
     server-side rather than hidden by JS.

  3. Robustness fixes baked in. Frame counts auto-pad to WAN's 4k+1
     constraint, H/W snap to multiples of 16, family/variant mismatch
     is caught BEFORE the user queues.

See README.md for install + usage.
"""

from .unividx_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serves web/, which carries both the help JS (title-bar "?" button and
# right-click entry) and web/docs/*.md — the markdown ComfyUI's own
# per-node help panel fetches from /extensions/<pack>/docs/<NodeName>.md.
# The markdown path matters because it's the only one that works in every
# render mode, including the Vue node renderer where canvas-drawn title
# buttons don't appear.
#
# Note what this JS does NOT do: it doesn't touch socket or widget
# visibility. That's the V3 `io.DynamicCombo` schema's job now, handled
# server-side. (The pack's original JS tried to hide sockets by setting
# `slot.hidden = true` — a flag no ComfyUI frontend has ever honoured,
# since input slots have no `hidden` property, so every socket showed in
# every mode.) The only JS left is presentational.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
