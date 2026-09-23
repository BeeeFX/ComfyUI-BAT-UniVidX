"""The runtime patches applied to upstream: context windows, attention,
FP8 with block swapping, torch.compile, progress/cancel."""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

from conftest import UPSTREAM_DIR, requires_upstream
from unividx_wrapper_runtime import advanced, fp8_loader


# ── context windows ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("total,size,stride", [
    (21, 21, 17), (40, 21, 17), (76, 21, 17), (38, 21, 17), (50, 10, 3),
])
def test_windows_are_full_length_and_cover_everything(total, size, stride):
    starts = advanced.context_window_starts(total, size, stride)
    covered = set()
    for s in starts:
        assert 0 <= s and s + min(size, total) <= total
        covered.update(range(s, s + size))
    assert covered >= set(range(total))
    assert starts == sorted(set(starts))


def test_blend_masks_crossfade_to_one():
    size, overlap = 21, 4
    outgoing = advanced._window_blend_mask(size, 0, overlap)
    incoming = advanced._window_blend_mask(size, overlap, 0)
    assert torch.allclose(outgoing[-overlap:] + incoming[:overlap],
                          torch.ones(overlap))
    assert float(advanced._window_blend_mask(size, 16, 16).min()) > 0


def test_windowed_run_reproduces_a_per_frame_model():
    # A model that treats every latent frame independently must come out
    # identical whether or not it is windowed — including under the
    # shifted final window.
    latents = torch.randn(4, 16, 38, 6, 8, dtype=torch.bfloat16)
    calls = []

    def model_fn(latents, **_kw):
        calls.append(latents.shape[2])
        return latents.float().mul(3).add(1).to(latents.dtype)

    out = advanced.run_context_windows(
        model_fn, size=21, stride=17, model_kwargs={"latents": latents, "y": None},
    )
    assert calls == [21, 21]
    expected = latents.float().mul(3).add(1).to(torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), expected.float(), atol=2e-2)


def test_window_patch_forwards_every_kwarg():
    class Pipe:
        pass

    class Model:
        pipe = Pipe()

    seen = []

    def original(**kwargs):
        seen.append(set(kwargs))
        return kwargs["latents"]

    Model.pipe.model_fn = original
    assert advanced.install_sliding_window_support(Model())
    latents = torch.zeros(4, 16, 30, 2, 2)
    Model.pipe.model_fn(latents=latents, training_mode="R2AIN", context=None,
                        sliding_window_size=21, sliding_window_stride=17)
    assert seen and all("training_mode" in s for s in seen)
    assert all("sliding_window_size" not in s for s in seen)


# ── progress / cancel ───────────────────────────────────────────────────────


def test_progress_hook_counts_steps_only_while_active():
    import tqdm as tqdm_mod

    ns = {"tqdm": tqdm_mod.tqdm}

    class Model:
        pass

    # pipeline_globals() reads the class's __init__ globals.
    exec("def __init__(self): pass", ns)
    Model.__init__ = ns["__init__"]
    assert advanced.install_progress_hook(Model())

    ticks = []
    with advanced.sampler_progress(lambda: ticks.append(1)):
        for _ in ns["tqdm"](range(5), disable=True):
            pass
    assert len(ticks) == 5
    for _ in ns["tqdm"](range(3), disable=True):
        pass
    assert len(ticks) == 5


# ── torch.compile ───────────────────────────────────────────────────────────


def test_compile_targets_blocks_that_model_fn_calls():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(4, 4)

        def forward(self, x):
            return self.l(x)

    class Dit(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(), Block()])

    class Model:
        pass

    model = Model()
    model.pipe = type("Pipe", (), {})()
    model.pipe.dit = Dit()
    keys = list(model.pipe.dit.state_dict())
    assert advanced.compile_dit(model) == 2
    # In place: module identity and state-dict keys are unchanged.
    assert isinstance(model.pipe.dit, Dit)
    assert list(model.pipe.dit.state_dict()) == keys
    assert all(b._compiled_call_impl is not None for b in model.pipe.dit.blocks)


# ── FP8 that keeps block swapping ───────────────────────────────────────────


@requires_upstream
def test_fp8_conversion_targets_upstream_wrapper_and_keeps_streaming():
    from unividx_wrapper_runtime import runtime

    runtime._add_upstream_to_path(UPSTREAM_DIR)
    cwd = os.getcwd()
    os.chdir(UPSTREAM_DIR)
    try:
        cls = runtime._import_pipeline_class("intrinsic", UPSTREAM_DIR)
    finally:
        os.chdir(cwd)
    g = advanced.pipeline_globals(cls)
    wrapper_cls, enable_vm = g["AutoWrappedLinear"], g["enable_vram_management"]

    torch.manual_seed(0)
    reference = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 32))
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 32))
    model.load_state_dict(reference.state_dict())
    enable_vm(model, module_map={nn.Linear: wrapper_cls}, module_config=dict(
        offload_dtype=torch.float32, offload_device="cpu",
        onload_dtype=torch.float32, onload_device="cpu",
        computation_dtype=torch.float32, computation_device="cpu",
    ))
    report = fp8_loader.quantize_wrapped_dit_inplace(model, wrapper_cls)
    assert report["linears_quantized"] == 2

    x = torch.randn(8, 64)
    want = reference(x)
    for transition in ("offload", "onload", "keep"):
        for m in model.modules():
            if hasattr(m, transition):
                getattr(m, transition)()
        for m in (model[0], model[2]):
            assert m.weight.dtype == torch.float8_e4m3fn, transition
            assert m._fp8_bias.dtype == torch.float32
        got = model(x)
        rel = (got - want).norm() / want.norm()
        assert rel < 0.05, (transition, float(rel))


# ── attention backend ───────────────────────────────────────────────────────


@requires_upstream
@pytest.mark.parametrize("variant", ["intrinsic", "alpha"])
def test_attention_patch_reaches_the_dit_and_matches_upstream(variant):
    from unividx_wrapper_runtime import runtime

    runtime._add_upstream_to_path(UPSTREAM_DIR)
    cwd = os.getcwd()
    os.chdir(UPSTREAM_DIR)
    try:
        cls = runtime._import_pipeline_class(variant, UPSTREAM_DIR)
    finally:
        os.chdir(cwd)
    dit_globals = advanced._dit_module_globals(cls)
    upstream_fn = dit_globals["flash_attention"]
    assert "src.models" not in " ".join(k for k in sys.modules if k.startswith("src."))

    assert advanced.install_attention_backend("sdpa", cls) == "sdpa"
    patched = dit_globals["flash_attention"]
    assert patched is not upstream_fn

    torch.manual_seed(0)
    q, k, v = (torch.randn(4, 12, 8 * 16) for _ in range(3))
    for drop_out in (0, 1):  # 0 = folded cross-modal self-attn, 1 = per-stream
        want = upstream_fn(q, k, v, num_heads=8, drop_out=drop_out)
        got = patched(q, k, v, num_heads=8, drop_out=drop_out)
        assert torch.allclose(got, want, atol=1e-5), drop_out
