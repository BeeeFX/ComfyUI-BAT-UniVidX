"""ComfyUI IMAGE ↔ UniVidX video-tensor conversion.

ComfyUI's IMAGE convention is `(N, H, W, 3)` float32 in `[0, 1]`,
channel-last, on CPU by default. UniVidX's pipe() consumes per-modality
videos as `(1, 3, T, H, W)` in `[-1, 1]`, channel-first, and returns each
modality as `(3, T, H, W)` in `[-1, 1]`.

Geometry: the model needs H/W on a multiple of 16. Inputs are brought
there by PADDING the bottom/right edge (edge-replicated) rather than by
stretching, and the decoder crops the padding back off. That keeps every
output pixel-aligned with the source plate — which is the whole point of
a matte or an albedo pass — and avoids two rounds of resampling blur. A
conditioning input whose size differs from the reference input is first
resized to the reference's size, so all streams share one geometry.

All arithmetic happens in float32 on CPU. Tensors are only cast to the
model dtype at the very end, and stay on the CPU: upstream moves them to
the GPU for the VAE encode itself, and a GPU copy made here would sit in
VRAM, unused, for the entire denoising loop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def image_batch_to_video_tensor(
    images: torch.Tensor,
    *,
    content_height: int,
    content_width: int,
    target_height: int,
    target_width: int,
    target_frames: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convert a ComfyUI IMAGE `(N, H, W, 3) [0,1]` into UniVidX's
    `(1, 3, T, H, W) [-1, 1]` on `(device, dtype)`.

    Behaviour:
      - Resized (bilinear) to `(content_height, content_width)` only if it
        isn't that size already — i.e. only when this input's resolution
        differs from the reference input's.
      - Edge-padded on the bottom/right up to `(target_height,
        target_width)`. `pipe_result_frame_to_image(..., height=, width=)`
        undoes this on the way out.
      - If `images.shape[0]` is less than `target_frames`, the last frame
        is repeated to pad. If greater, the trailing frames are dropped.
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"image_batch_to_video_tensor: expected (N,H,W,3) IMAGE, "
            f"got {tuple(images.shape)}"
        )
    if content_height > target_height or content_width > target_width:
        raise ValueError(
            f"content size {content_height}x{content_width} exceeds target "
            f"{target_height}x{target_width}"
        )
    n_in, h_in, w_in, _ = images.shape

    # 1) Temporal pad/truncate first, so the spatial work below only
    # touches frames that are actually used.
    video = images[:target_frames]
    if n_in < target_frames:
        last = video[-1:].expand(target_frames - n_in, -1, -1, -1)
        video = torch.cat([video, last], dim=0)

    # 2) (T, 3, H, W) float32 for F.interpolate / F.pad.
    nchw = video.permute(0, 3, 1, 2).to(torch.float32)
    if (h_in, w_in) != (content_height, content_width):
        nchw = F.interpolate(
            nchw, size=(content_height, content_width),
            mode="bilinear", align_corners=False,
        )
    pad_h = target_height - content_height
    pad_w = target_width - content_width
    if pad_h or pad_w:
        nchw = F.pad(nchw, (0, pad_w, 0, pad_h), mode="replicate")

    # 3) [0, 1] → [-1, 1] while still in float32, then (1, 3, T, H, W)
    # on the requested device/dtype.
    video = (nchw * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
    return video.to(device=device, dtype=dtype).contiguous()


def pipe_result_frame_to_image(
    frame_tensor: torch.Tensor,
    *,
    frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    """Convert one of UniVidX's pipe() output tensors back into a
    ComfyUI IMAGE batch `(N, H, W, 3) [0, 1]` float32 on CPU.

    `frames` / `height` / `width` trim and crop off the padding the input
    side added (see `image_batch_to_video_tensor`). That happens first,
    and the result moves to the CPU before the float32 conversion — so the
    padding is never converted, and a full-clip float32 copy (~7.5 GB at
    1080p x 300 frames) is never allocated on the GPU.

    Both pipeline classes return every modality as `(3, T, H, W)` in
    `[-1, 1]` (normals as unit vectors, the rest as raw VAE output), and
    upstream's own inference scripts map every one of them with
    `x * 0.5 + 0.5`. So the mapping here is unconditional. It used to be
    guessed from the data (`min() < -0.1`), which misread any clip bright
    enough to have no pixel below ~45% grey — a mostly-white matte, a
    sunlit irradiance pass — and crushed its mid-tones toward black.

    The cast to float32 comes BEFORE the arithmetic: bf16 has ~3
    significant digits, and doing `(x + 1) * 0.5` in bf16 costs roughly a
    bit of tonal precision that the decoder produced.
    """
    if frame_tensor is None:
        return None
    if not isinstance(frame_tensor, torch.Tensor):
        frame_tensor = torch.as_tensor(frame_tensor)

    if frame_tensor.ndim == 5 and frame_tensor.shape[0] == 1:
        frame_tensor = frame_tensor.squeeze(0)
    if frame_tensor.ndim != 4 or frame_tensor.shape[0] != 3:
        raise ValueError(
            f"UniVidX: expected a (3, T, H, W) pipe output, got "
            f"{tuple(frame_tensor.shape)} — upstream's output format changed."
        )
    frame_tensor = frame_tensor[:, :frames, :height, :width].cpu()
    # (3, T, H, W) → (T, H, W, 3), in float32 before any arithmetic.
    frame_tensor = frame_tensor.permute(1, 2, 3, 0).to(torch.float32)
    frame_tensor = (frame_tensor + 1.0) * 0.5
    return frame_tensor.clamp_(0.0, 1.0).contiguous()


def black_placeholder(num_frames: int, height: int, width: int) -> torch.Tensor:
    """A `(num_frames, height, width, 3)` black IMAGE for slots a mode
    doesn't produce (e.g. `R2AIN` has no rgb output).

    Returned as an expanded view of a single frame, so it costs one
    frame of memory instead of the whole clip — at 1080p x 300 frames a
    materialised float32 placeholder is ~7.5 GB, and some modes return
    three of them. Readers (previews, video writers, other nodes) see an
    ordinary tensor; only an in-place write would complain, and a node
    writing in place into an input it received would be corrupting
    ComfyUI's output cache anyway.
    """
    frame = torch.zeros((1, height, width, 3), dtype=torch.float32)
    return frame.expand(num_frames, -1, -1, -1)


def black_mask(num_frames: int, height: int, width: int) -> torch.Tensor:
    """The MASK counterpart of `black_placeholder` — `(N, H, W)` zeros,
    as an expanded view for the same reason."""
    frame = torch.zeros((1, height, width), dtype=torch.float32)
    return frame.expand(num_frames, -1, -1)
