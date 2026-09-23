"""Sample -> Decode through the real node classes, with a stub pipeline in
place of the 14B model. Needs COMFYUI_PATH (see conftest)."""
from __future__ import annotations

import pytest
import torch

from conftest import requires_comfyui

pytestmark = requires_comfyui


class _StubPipe:
    """Stands in for WanVideoPipeline: records its kwargs, drives the
    progress hook like upstream's tqdm loop, returns (3, T, H, W) [-1, 1]
    tensors for the mode's targets."""

    torch_dtype = torch.bfloat16

    def __init__(self, tqdm_fn):
        self.calls = []
        self._tqdm = tqdm_fn

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        for _ in self._tqdm(range(kwargs["num_inference_steps"]), disable=True):
            pass
        from unividx_wrapper_runtime import modes
        t, h, w = kwargs["num_frames"], kwargs["height"], kwargs["width"]
        out = {}
        for key in modes.output_keys(kwargs["training_mode"]):
            out[key] = torch.full((3, t, h, w), 0.5, dtype=torch.bfloat16)
        return out


def _stub_model(tmp_path):
    import tqdm as tqdm_mod

    from unividx_wrapper_runtime import advanced

    ns = {"tqdm": tqdm_mod.tqdm}
    exec("def __init__(self): pass", ns)
    model_cls = type("StubUniVid", (), {"__init__": ns["__init__"]})
    model = model_cls()
    advanced.install_progress_hook(model)
    model.pipe = _StubPipe(lambda *a, **k: ns["tqdm"](*a, **k))
    return (model, "alpha", str(tmp_path))


def test_schemas_resolve(pack):
    info = pack.NODE_CLASS_MAPPINGS["BAT_UniVidXSample"].INPUT_TYPES()
    assert "mode" in info["required"]
    decode = pack.NODE_CLASS_MAPPINGS["BAT_UniVidXDecode"]
    assert list(decode.RETURN_TYPES) == ["IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK"]


def test_sample_decode_round_trip_keeps_input_geometry(pack, tmp_path):
    nodes = pack.NODE_CLASS_MAPPINGS
    model = _stub_model(tmp_path)
    rgb = torch.rand(10, 1080, 250, 3)
    mode = {"mode": "R2PFB", "rgb": rgb, "denoising_strength": 1.0,
            "optional_prompt": {"optional_prompt": "off"}}

    result = nodes["BAT_UniVidXSample"].execute(
        model=model, mode=mode, num_inference_steps=3, seed=1,
    ).result[0]
    kwargs = model[0].pipe.calls[0]
    assert (kwargs["num_frames"], kwargs["height"], kwargs["width"]) == (13, 1088, 256)
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["inference_rgb"].device.type == "cpu"

    rgb_out, pha, fgr, bgr, mask = nodes["BAT_UniVidXDecode"].execute(result).result
    for image in (rgb_out, pha, fgr, bgr):
        assert image.shape == (10, 1080, 250, 3)
    assert mask.shape == (10, 1080, 250)
    assert float(rgb_out.max()) == 0.0  # R2PFB doesn't produce rgb
    assert torch.allclose(pha, torch.full_like(pha, 0.75), atol=4e-3)
    assert torch.allclose(mask, torch.full_like(mask, 0.75), atol=4e-3)


def test_family_mismatch_is_refused(pack, tmp_path):
    model = _stub_model(tmp_path)
    mode = {"mode": "R2AIN", "rgb": torch.rand(5, 64, 64, 3),
            "denoising_strength": 1.0, "optional_prompt": {"optional_prompt": "off"}}
    with pytest.raises(ValueError, match="family='intrinsic'"):
        pack.NODE_CLASS_MAPPINGS["BAT_UniVidXSample"].execute(
            model=model, mode=mode, num_inference_steps=1, seed=1)
