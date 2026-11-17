"""IMAGE <-> UniVidX tensor conversion and the frame/geometry helpers."""
from __future__ import annotations

import torch

from unividx_wrapper_runtime import frame_constraints, tensor_io


def test_wan_frame_count():
    assert [frame_constraints.wan_frame_count(n) for n in (1, 2, 5, 6, 81, 82)] == \
        [1, 5, 5, 9, 81, 85]


def test_pad_then_crop_is_pixel_exact():
    images = torch.rand(7, 1080, 250, 3)
    video = tensor_io.image_batch_to_video_tensor(
        images, content_height=1080, content_width=250,
        target_height=1088, target_width=256, target_frames=9,
        dtype=torch.float32,
    )
    assert video.shape == (1, 3, 9, 1088, 256)
    assert video.device.type == "cpu"
    # Content region is untouched (no resampling), frames past the input
    # repeat the last one, padding replicates the edge.
    back = (video[0].permute(1, 2, 3, 0) + 1) / 2
    assert torch.allclose(back[:7, :1080, :250], images, atol=1e-6)
    assert torch.allclose(back[8, :1080, :250], images[-1], atol=1e-6)
    assert torch.allclose(back[0, 1087, :250], images[0, 1079], atol=1e-6)
    image = tensor_io.pipe_result_frame_to_image(
        video[0], frames=7, height=1080, width=250)
    assert image.shape == (7, 1080, 250, 3) and image.is_contiguous()
    assert torch.allclose(image, images, atol=1e-6)


def test_mismatched_input_is_resized_to_reference_content():
    albedo = torch.rand(5, 540, 960, 3)
    video = tensor_io.image_batch_to_video_tensor(
        albedo, content_height=1080, content_width=1920,
        target_height=1088, target_width=1920, target_frames=5,
    )
    assert video.shape == (1, 3, 5, 1088, 1920)


def test_output_range_mapping_is_unconditional():
    # A bright clip: in [-1, 1] every value is above -0.1. The old
    # data-dependent check left this unmapped and crushed it.
    bright = torch.full((3, 2, 4, 4), 0.2)
    image = tensor_io.pipe_result_frame_to_image(bright.to(torch.bfloat16))
    assert image.shape == (2, 4, 4, 3)
    assert torch.allclose(image, torch.full_like(image, 0.6), atol=4e-3)
    assert image.dtype == torch.float32


def test_placeholders_are_cheap_views():
    image = tensor_io.black_placeholder(300, 1080, 1920)
    mask = tensor_io.black_mask(300, 1080, 1920)
    assert image.shape == (300, 1080, 1920, 3) and mask.shape == (300, 1080, 1920)
    assert image.untyped_storage().nbytes() == 1080 * 1920 * 3 * 4
    assert float(image.max()) == 0.0
