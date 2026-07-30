"""ComfyUI IMAGE ↔ UniVidX video-tensor conversion.

ComfyUI's IMAGE convention is `(N, H, W, 3)` float32 in `[0, 1]`,
channel-last, on CPU by default. UniVidX's pipe() consumes per-modality
videos as `(1, 3, T, H, W)` in `[-1, 1]`, channel-first, on the GPU,
in the model's working dtype (bf16 by default; the loader's dtype
widget controls this).

This file owns the conversion in both directions plus the bilinear
resize that aligns short clips to the target T/H/W. We do the resize
in float32 on CPU first (cheap, lossless quality-wise) then move to
GPU/dtype as a single block — that's faster than resizing in the
target dtype on GPU when the dtype is bf16 (PyTorch's bilinear on bf16
falls back to a slower path on some CUDA versions).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def image_batch_to_video_tensor(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    target_frames: int,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convert a ComfyUI IMAGE `(N, H, W, 3) [0,1]` into UniVidX's
    `(1, 3, T, H, W) [-1, 1]` on `(device, dtype)`.

    Behaviour:
      - If `images.shape[1:3]` already matches `(target_height,
        target_width)`, the spatial resize is skipped.
      - If `images.shape[0]` is less than `target_frames`, the last
        frame is repeated to pad. If greater, the trailing frames are
        dropped. (Padding by repeat is what UniVidX itself does
        internally for the conditioning videos; we do it here so we
        control the math.)
      - Output is contiguous, which the model's first layer expects.
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"image_batch_to_video_tensor: expected (N,H,W,3) IMAGE, "
            f"got {tuple(images.shape)}"
        )
    n_in, h_in, w_in, _ = images.shape

    # 1) Spatial resize to (target_height, target_width). NCHW for
    # F.interpolate; output stays float32 on CPU.
    nchw = images.permute(0, 3, 1, 2).contiguous()
    if (h_in, w_in) != (target_height, target_width):
        nchw = F.interpolate(
            nchw, size=(target_height, target_width),
            mode="bilinear", align_corners=False,
        )

    # 2) Temporal pad/truncate.
    if n_in < target_frames:
        pad = target_frames - n_in
        last = nchw[-1:].expand(pad, -1, -1, -1)
        nchw = torch.cat([nchw, last], dim=0)
    elif n_in > target_frames:
        nchw = nchw[:target_frames]

    # 3) Move to device + dtype + add batch axis, then permute to
    # (1, 3, T, H, W) which is what UniVidX expects.
    nchw = nchw.to(device=device, dtype=dtype, non_blocking=True)
    # nchw is currently (T, 3, H, W); add a leading batch axis and
    # transpose to (B, 3, T, H, W).
    video = nchw.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

    # 4) [0, 1] → [-1, 1].
    return video * 2.0 - 1.0


def pipe_result_frame_to_image(frame_tensor: torch.Tensor) -> torch.Tensor:
    """Convert one of UniVidX's pipe() output tensors back into a
    ComfyUI IMAGE batch `(N, H, W, 3) [0, 1]`.

    The pipe returns each modality as either:
      - a `(T, H, W, 3)` numpy array / torch tensor in `[0, 1]`, or
      - a `(3, T, H, W)` torch tensor in `[-1, 1]`

    The decoder branches on shape + range to be lenient about upstream
    quirks across UniVidX's two pipeline classes.
    """
    if frame_tensor is None:
        return None
    if not isinstance(frame_tensor, torch.Tensor):
        frame_tensor = torch.as_tensor(frame_tensor)

    # Identify layout. If channels-first (shape[0] == 3 and len >= 3),
    # permute to channels-last.
    if frame_tensor.ndim == 4 and frame_tensor.shape[0] == 3:
        # (3, T, H, W) → (T, H, W, 3)
        frame_tensor = frame_tensor.permute(1, 2, 3, 0)
    elif frame_tensor.ndim == 5 and frame_tensor.shape[0] == 1 and frame_tensor.shape[1] == 3:
        # (1, 3, T, H, W) → (T, H, W, 3)
        frame_tensor = frame_tensor.squeeze(0).permute(1, 2, 3, 0)

    # Normalise to [0, 1] if it looks like [-1, 1].
    if frame_tensor.min() < -0.1:
        frame_tensor = (frame_tensor + 1.0) * 0.5

    return frame_tensor.clamp(0.0, 1.0).to(torch.float32).cpu().contiguous()


def black_placeholder(num_frames: int, height: int, width: int) -> torch.Tensor:
    """A `(num_frames, height, width, 3)` zero tensor for slots a mode
    doesn't produce (e.g. `RA2IN` doesn't emit rgb in the result dict —
    we hand the decoder a black placeholder so downstream nodes still
    receive 4 IMAGE outputs)."""
    return torch.zeros(
        (num_frames, height, width, 3), dtype=torch.float32,
    )
