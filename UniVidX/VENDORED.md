# Vendored upstream — UniVidX

This directory is a **copy of third-party source**, not part of the BAT
wrapper. It is included in the repository rather than fetched at runtime.

| | |
|---|---|
| Upstream | https://github.com/houyuanchen111/UniVidX |
| Commit | `382b9002757f4d5f04a90ec23b784dce11d56221` |
| License | Apache License 2.0 — see `LICENSE` in this directory |
| Copyright | The UniVidX Authors |

The wrapper around it (everything outside this directory) is GPL v3.
Apache 2.0 permits inclusion in a GPL-v3 work; this directory keeps its
own license, and these files remain under Apache 2.0.

## Modifications made to the upstream copy

Required by Apache 2.0 §4(b) — the files here are **not** a byte-exact
copy of upstream:

1. **`LICENSE` added.** Upstream's own LICENSE file was committed *after*
   the pinned commit, so `382b9002` carries no license text. The file
   here is fetched from upstream `main` and is the license the project
   declares for this code.
2. **`assets/` removed** — ~12 MB of demo `.mp4` files and a teaser PNG,
   referenced by nothing in the code path this wrapper uses.
3. **Nested `.git` removed**, so this is a plain directory rather than a
   repository inside a repository.
4. **`src/pipelines/__init__.py` added** (empty). Upstream omits it, which
   makes `src.pipelines` a namespace package and breaks the import under
   some loaders.

No source file has been edited. The wrapper's fixes to upstream behaviour
— the unrunnable sliding-window path, the missing `border_width` guard,
the hardcoded-SDPA attention — are applied as **runtime patches** from
`unividx_wrapper_runtime/advanced.py`, deliberately, so this tree stays a
clean copy that is easy to diff against upstream and easy to re-vendor.

## Re-vendoring at a newer commit

```bash
git clone --filter=blob:none --no-checkout \
    https://github.com/houyuanchen111/UniVidX.git UniVidX
git -C UniVidX checkout <new-commit>
rm -rf UniVidX/.git UniVidX/assets
touch UniVidX/src/pipelines/__init__.py
# keep LICENSE and this file
```

Then update `UNIVIDX_PINNED_COMMIT` in
`unividx_wrapper_runtime/auto_clone.py` to match, and re-check the
runtime patches in `advanced.py` still apply — they target specific
upstream functions by name.
