"""Static mode metadata for UniVidX.

A mode name encodes which modalities go IN (conditions, left of '2')
and which come OUT (targets, right of '2'). For example:

    R2AIN  — RGB → Albedo + Irradiance + Normal (intrinsic family)
    RA2IN  — RGB + Albedo → Irradiance + Normal (intrinsic family)
    t2RAIN — text-only → all four intrinsic modalities (no IMAGE input)
    R2PFB  — RGB → Pha + Foreground + Background (alpha family)
    RPF2B  — RGB + Pha + Foreground → Background (alpha family)

Letter codes:
    Intrinsic family: R=rgb, A=albedo, I=irradiance, N=normal
    Alpha family:     R=composite-rgb, P=pha, F=fgr, B=bgr

The same letter 'R' means different things in the two families
(RGB image vs composite RGB containing a matted subject). That's why
the family must be picked before the mode — and why our wrapper
forces it via the loader's variant widget rather than letting it
clash at inference time.
"""
from __future__ import annotations

from itertools import combinations
from typing import Set, Tuple


# ── Letter → IMAGE-socket-name mappings ─────────────────────────────────────
# Used by both the Python side (for validation + tensor packing) and the
# JS side (for dynamic widget visibility).

_INTRINSIC_INPUT = {"R": "rgb", "A": "albedo", "I": "irradiance", "N": "normal"}
_ALPHA_INPUT     = {"R": "rgb", "P": "pha",    "F": "fgr",        "B": "bgr"}

# Mapping from letter to the key UniVidX's pipe() output dict actually
# returns. For intrinsic the 'normal' OUTPUT is stored under 'normal_unit'
# (it's the unit-vector form), but the INPUT slot is just 'normal'.
_INTRINSIC_OUTPUT = {"R": "rgb", "A": "albedo", "I": "irradiance", "N": "normal_unit"}
_ALPHA_OUTPUT     = {"R": "rgb", "P": "pha",    "F": "fgr",        "B": "bgr"}


# ── Mode enumeration ────────────────────────────────────────────────────────


def _enumerate_modes(letters: str) -> list[str]:
    """All `t2XXXX` + every non-empty strict subset of `letters` as
    conditions, with the complement as targets. For `letters="RAIN"`
    that's: `t2RAIN`, plus all `R2AIN`, `A2RIN`, ..., `RAI2N`, `RAN2I`,
    `RIN2A`, `AIN2R`, `RA2IN`, `RI2AN`, ... — 15 modes total."""
    modes = [f"t2{letters}"]
    n = len(letters)
    for k in range(1, n):
        for combo in combinations(letters, k):
            cond = "".join(combo)
            tgt = "".join(c for c in letters if c not in combo)
            modes.append(f"{cond}2{tgt}")
    return modes


INTRINSIC_MODES: list[str] = _enumerate_modes("RAIN")
ALPHA_MODES:     list[str] = _enumerate_modes("RPFB")
ALL_MODES:       list[str] = INTRINSIC_MODES + ALPHA_MODES


# ── Family lookup ───────────────────────────────────────────────────────────


def family_of(mode: str) -> str:
    """Return 'intrinsic' or 'alpha'. Raises ValueError on unknown modes."""
    if mode in INTRINSIC_MODES:
        return "intrinsic"
    if mode in ALPHA_MODES:
        return "alpha"
    raise ValueError(f"Unknown UniVidX mode: {mode!r}")


# ── Input requirements ──────────────────────────────────────────────────────


def _parse_letters(mode: str) -> Tuple[list[str], list[str]]:
    """Split 'RA2IN' into (['R','A'], ['I','N']) or 't2RAIN' into
    ([], ['R','A','I','N'])."""
    if "2" not in mode:
        raise ValueError(f"Invalid mode (no '2' separator): {mode!r}")
    left, right = mode.split("2", 1)
    if left.lower() == "t":
        return [], list(right)
    return list(left), list(right)


def required_inputs(mode: str) -> Set[str]:
    """The set of IMAGE socket names this mode requires the user to wire.

    For text-only modes (`t2…`) returns the empty set — those modes
    generate from prompt alone."""
    family = family_of(mode)
    cond_letters, _ = _parse_letters(mode)
    mapping = _INTRINSIC_INPUT if family == "intrinsic" else _ALPHA_INPUT
    return {mapping[c] for c in cond_letters}


def output_keys(mode: str) -> list[str]:
    """Ordered list of pipe()-result dict keys for this mode's targets.

    Used by the Decode node to know which slots to pull out of the
    result dict and which to leave as black placeholders."""
    family = family_of(mode)
    _, tgt_letters = _parse_letters(mode)
    mapping = _INTRINSIC_OUTPUT if family == "intrinsic" else _ALPHA_OUTPUT
    return [mapping[c] for c in tgt_letters]


# ── Deterministic input ordering ────────────────────────────────────────────
# `required_inputs()` returns a set, so callers that need a STABLE order
# (e.g. picking which wired input defines the reference frame count) must
# not iterate it directly — set iteration order is arbitrary and would make
# the sampler's target_frames nondeterministic across runs.

# Canonical per-family socket order. 'rgb' first in both families: when
# several inputs are wired it's the most sensible frame-count reference,
# and it's the one modality present in every non-t2 mode.
INTRINSIC_INPUT_ORDER: tuple[str, ...] = ("rgb", "albedo", "irradiance", "normal")
ALPHA_INPUT_ORDER:     tuple[str, ...] = ("rgb", "pha", "fgr", "bgr")


def family_input_order(family: str) -> tuple[str, ...]:
    """Canonical socket order for a family. Used for both tensor packing
    and deterministic reference-tensor selection."""
    if family == "intrinsic":
        return INTRINSIC_INPUT_ORDER
    if family == "alpha":
        return ALPHA_INPUT_ORDER
    raise ValueError(f"Unknown UniVidX family: {family!r}")


def ordered_required_inputs(mode: str) -> list[str]:
    """`required_inputs(mode)` in canonical family order (stable)."""
    required = required_inputs(mode)
    return [k for k in family_input_order(family_of(mode)) if k in required]


# ── Human-readable mode reference ───────────────────────────────────────────
# Powers the help panel on the Sample node. Generated from the letter
# mappings above rather than hand-written, so it can never drift out of
# sync with the modes the node actually offers.

# What each letter means, per family, in prose.
_LETTER_PROSE = {
    "intrinsic": {
        "R": "RGB", "A": "albedo", "I": "irradiance", "N": "normals",
    },
    "alpha": {
        "R": "composite RGB", "P": "matte", "F": "foreground", "B": "background",
    },
}

# One-line "why you'd use this" for the modes people actually reach for.
# Modes absent here fall back to a generated description.
_MODE_NOTES = {
    "t2RAIN": "Generate a video from text alone, with all four intrinsic channels.",
    "R2AIN": "The main intrinsic decomposition: take footage, get albedo, lighting and normals.",
    "RA2IN": "You already have albedo — solve for lighting and normals.",
    "RAI2N": "You have albedo and lighting — solve for normals only.",
    "AIN2R": "Relight: rebuild RGB from albedo, lighting and normals.",
    "AI2RN": "Rebuild RGB and normals from albedo and lighting.",
    "t2RPFB": "Generate a video from text alone, with matte, foreground and background.",
    "R2PFB": "The main matting mode: take footage, get a matte plus separated fg/bg.",
    "RP2FB": "You already have a matte — split the plate into foreground and background.",
    "RPF2B": "You have matte and foreground — solve for the background.",
    "PFB2R": "Composite: rebuild the plate from matte, foreground and background.",
}


def describe(mode: str) -> str:
    """One-line human description of what a mode does."""
    note = _MODE_NOTES.get(mode)
    if note:
        return note
    family = family_of(mode)
    prose = _LETTER_PROSE[family]
    cond_letters, tgt_letters = _parse_letters(mode)
    targets = " + ".join(prose[c] for c in tgt_letters)
    if not cond_letters:
        return f"From text alone, generate {targets}."
    conditions = " + ".join(prose[c] for c in cond_letters)
    return f"From {conditions}, predict {targets}."


def mode_reference() -> list[dict]:
    """Full per-mode reference for the help panel, in dropdown order."""
    out = []
    for mode in ALL_MODES:
        family = family_of(mode)
        prose = _LETTER_PROSE[family]
        _, tgt_letters = _parse_letters(mode)
        out.append({
            "mode": mode,
            "family": family,
            "inputs": ordered_required_inputs(mode),
            "outputs": [prose[c] for c in tgt_letters],
            "description": describe(mode),
        })
    return out
