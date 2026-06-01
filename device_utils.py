"""
Shared device / VRAM helpers for all voice backends.

Single source of truth for ``cuda | cpu`` selection, with an env override
``VOICE_CHANGER_FORCE_DEVICE=cpu|cuda`` so the user can disable GPU without
restarting the Gradio process (and so --cpu in app.py stays consistent).
"""

from __future__ import annotations

import os


def select_device() -> str:
    """
    Return ``cuda`` or ``cpu`` based on availability and the env override.

    Order:
      1. ``VOICE_CHANGER_FORCE_DEVICE=cpu``  → cpu
      2. ``VOICE_CHANGER_FORCE_DEVICE=cuda`` → cuda if available else cpu
      3. auto-detect
    """
    forced = (os.environ.get("VOICE_CHANGER_FORCE_DEVICE") or "").strip().lower()
    try:
        import torch
    except Exception:
        return "cpu"
    if forced == "cpu":
        return "cpu"
    if forced == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def empty_cache() -> None:
    """Best-effort ``torch.cuda.empty_cache()``."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def memory_allocated_mb() -> float:
    """Current allocated CUDA memory in MB (0.0 if no CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    return 0.0


# Names of modules that expose an ``unload*()`` hook — called by ``unload_all()``
# to free VRAM when the user switches engines in the UI.
_UNLOAD_TARGETS = [
    ("chatterbox_engine", ["unload", "unload_vc"]),
    ("openvoice_engine",  ["unload"]),
    ("xtts_engine",       ["unload"]),
    ("f5tts_engine",      ["unload"]),
    ("seed_vc_engine",    ["unload"]),
]


def unload_all(except_modules: tuple[str, ...] = ()) -> None:
    """
    Best-effort unload of every known engine except the ones listed.

    Safe to call even when modules are not imported yet.
    """
    import importlib
    import sys

    for mod_name, fn_names in _UNLOAD_TARGETS:
        if mod_name in except_modules:
            continue
        mod = sys.modules.get(mod_name)
        if mod is None:
            # only act on already-imported modules; don't import heavy stuff here
            continue
        for fn in fn_names:
            fn_obj = getattr(mod, fn, None)
            if callable(fn_obj):
                try:
                    fn_obj()
                except Exception as e:
                    print(f"[device_utils] unload {mod_name}.{fn} failed: {e}", flush=True)
    empty_cache()
