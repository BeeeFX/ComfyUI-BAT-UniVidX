"""Wrapper-side runtime for ComfyUI-BAT-UniVidXWrapper.

This package is named `unividx_wrapper_runtime/` (rather than the more
obvious `src/`) deliberately. Upstream UniVidX exposes its code as a
top-level `src/` package, and if the wrapper used the same name, the
two packages collide on `sys.path` — Python caches whichever was
imported first as `sys.modules['src']`, shadowing the other. The
existing UniVidX_ComfyUI wrapper hits exactly this bug and ships a
defensive `sys.modules` eviction patch as a workaround. We avoid the
collision entirely by giving our package a unique name.
"""
