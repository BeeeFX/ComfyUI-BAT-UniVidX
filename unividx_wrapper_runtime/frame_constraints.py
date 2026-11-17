"""Frame-count and resolution constraint helpers — same lessons the
DKT wrapper learned in production.

WAN 2.1's temporal VAE compresses 4 frames to 1 latent with a +1
offset, so valid input frame counts satisfy `(N - 1) % 4 == 0`
(i.e. N ∈ {1, 5, 9, 13, …, 4k+1}). Feeding a 122-frame batch produces
an off-by-one shape mismatch in a deep concat with no clear
diagnostic. The sampler pads up by repeating the last frame and the
decoder trims back to the original length.

UniVidX's preprocessor also rounds H/W up to the next multiple of 16
when they aren't already — but one of its downstream code paths uses
the un-rounded value, producing a silent shape mismatch (121 vs 122
in a height-related latent dim). We snap H/W at the wrapper boundary
to head it off.
"""
from __future__ import annotations


def snap_to_16(value: int) -> int:
    """Round `value` up to the next multiple of 16. Used for both
    height and width on the way into the sampler. Idempotent — a
    multiple of 16 in returns unchanged."""
    return ((int(value) + 15) // 16) * 16


def wan_frame_count(n: int) -> int:
    """The smallest valid `4k+1` frame count >= `n` (n >= 1)."""
    n = int(n)
    if n < 1:
        raise ValueError(
            "UniVidX: the wired IMAGE batch is empty — nothing to condition on."
        )
    return n + (-(n - 1)) % 4
