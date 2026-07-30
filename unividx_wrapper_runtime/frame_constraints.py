"""Frame-count and resolution constraint helpers — same lessons the
DKT wrapper learned in production.

WAN 2.1's temporal VAE compresses 4 frames to 1 latent with a +1
offset, so valid input frame counts satisfy `(N - 1) % 4 == 0`
(i.e. N ∈ {1, 5, 9, 13, …, 4k+1}). Feeding a 122-frame batch produces
an off-by-one shape mismatch in a deep concat with no clear
diagnostic. We pad up by repeating the last frame and trim back to
the original length at the output boundary.

UniVidX's preprocessor also rounds H/W up to the next multiple of 16
when they aren't already — but one of its downstream code paths uses
the un-rounded value, producing a silent shape mismatch (121 vs 122
in a height-related latent dim). We snap H/W at the wrapper boundary
to head it off.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("ComfyUI-BAT-UniVidXWrapper")


def snap_to_16(value: int) -> int:
    """Round `value` up to the next multiple of 16. Used for both
    height and width on the way into the sampler. Idempotent — a
    multiple of 16 in returns unchanged."""
    return ((int(value) + 15) // 16) * 16


def pad_to_wan_frame_count(
    images: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Pad a (N,H,W,3) IMAGE batch up to the next valid `4k+1` frame
    count by repeating the last frame. Returns `(padded, n_original)`.

    The caller should slice the output back to `n_original` after
    inference so the downstream graph sees the same frame count the
    user fed in.
    """
    n = images.shape[0]
    if n == 0:
        return images, 0
    rem = (n - 1) % 4
    if rem == 0:
        return images, n
    pad = 4 - rem
    last = images[-1:].expand(pad, -1, -1, -1)
    padded = torch.cat([images, last], dim=0)
    logger.info(
        "UniVidX: padded frame batch %d -> %d (WAN temporal VAE needs 4k+1).",
        n, padded.shape[0],
    )
    return padded, n


def trim_to_original(tensor: torch.Tensor, n_original: int) -> torch.Tensor:
    """Cut a (N,…) tensor down to the first `n_original` frames.
    No-op when already at or below that count."""
    if tensor is None or tensor.shape[0] <= n_original:
        return tensor
    return tensor[:n_original]
