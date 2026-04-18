"""
Seed-VC — zero-shot voice conversion with strong identity transfer.

Project: https://github.com/Plachtaa/seed-vc

This module provides a thin wrapper with an API identical to the other
engines in this project:

    is_available()
    get_model()
    convert_voice(input_path, ref_path, output_path=None, seed=None)
        -> (output_path_or_None, wav_float32, sample_rate)
    unload()

Seed-VC ships a CLI (``seed-vc``) plus a ``seed_vc.inference`` module. Both
entry points historically change between releases, so we probe several
candidates and fall back to the CLI binary when the Python API is not
available. If nothing is installed, ``is_available()`` returns ``False`` and
``get_model()`` raises a friendly error — the UI falls back to another engine.

Install:
    pip install seed-vc
    # or: pip install git+https://github.com/Plachtaa/seed-vc
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from device_utils import empty_cache as _empty_cache, select_device as _select_device

_model: Any = None
_device: Optional[str] = None
_backend: Optional[str] = None  # "python" | "cli"


def _log(msg: str) -> None:
    print(f"[Seed-VC] {msg}", flush=True)


def _probe_python_backend() -> Optional[str]:
    """Return the importable module path if a Python backend exists."""
    for mod_name in ("seed_vc.inference", "seed_vc.api", "seed_vc"):
        try:
            importlib.import_module(mod_name)
            return mod_name
        except Exception:
            continue
    return None


def _probe_cli_backend() -> Optional[str]:
    """Return path to seed-vc binary if installed."""
    for name in ("seed-vc", "seedvc"):
        p = shutil.which(name)
        if p:
            return p
    return None


def is_available() -> bool:
    return bool(_probe_python_backend() or _probe_cli_backend())


def _load_python_model(mod_name: str):
    """Instantiate the Seed-VC model from its Python API, best-effort."""
    mod = importlib.import_module(mod_name)
    device = _select_device()

    for attr in ("SeedVC", "VoiceConversionModel", "VoiceConverter"):
        cls = getattr(mod, attr, None)
        if cls is None:
            continue
        try:
            if hasattr(cls, "from_pretrained"):
                return cls.from_pretrained(device=device), device
            return cls(device=device), device
        except TypeError:
            try:
                return cls(), device
            except Exception:
                continue
        except Exception:
            continue

    for fn_name in ("load_model", "build_model"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                return fn(device=device), device
            except TypeError:
                return fn(), device
    raise RuntimeError(
        "Seed-VC Python module is installed but no known entry point could be "
        "called. Upgrade seed-vc or use the CLI (``pip install seed-vc[cli]``)."
    )


def get_model():
    global _model, _device, _backend
    if _model is not None:
        return _model

    mod_name = _probe_python_backend()
    if mod_name is not None:
        try:
            _log(f"loading Python backend via {mod_name}...")
            _model, _device = _load_python_model(mod_name)
            _backend = "python"
            _log(f"ready on {_device}.")
            return _model
        except Exception as e:
            _log(f"Python backend load failed ({e}). Falling back to CLI.")

    cli = _probe_cli_backend()
    if cli is not None:
        _backend = "cli"
        _model = cli
        _device = _select_device()
        _log(f"using CLI backend at {cli} ({_device}).")
        return _model

    raise RuntimeError(
        "Seed-VC is not installed. Install with: pip install seed-vc "
        "(or: pip install git+https://github.com/Plachtaa/seed-vc)."
    )


def _run_cli(input_path: str, ref_path: str, output_path: str, seed: Optional[int]) -> int:
    """Invoke the seed-vc CLI; returns the process exit code."""
    cli = _probe_cli_backend()
    if cli is None:
        raise RuntimeError("seed-vc CLI not found on PATH.")
    cmd = [
        cli,
        "--source", str(input_path),
        "--target", str(ref_path),
        "--output", str(output_path),
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    _log("CLI: " + " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    return res.returncode


def _run_python(model: Any, input_path: str, ref_path: str, seed: Optional[int]) -> tuple[np.ndarray, int]:
    """Call the Python API with a best-effort kwargs mapping."""
    import torch

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    candidates = [
        dict(source=input_path, target=ref_path),
        dict(source_path=input_path, target_path=ref_path),
        dict(audio=input_path, target_voice_path=ref_path),
        dict(src=input_path, ref=ref_path),
    ]
    method_names = ("convert", "inference", "run", "generate", "__call__")
    last_err: Exception | None = None
    for method_name in method_names:
        method = getattr(model, method_name, None)
        if not callable(method):
            continue
        for kwargs in candidates:
            try:
                out = method(**kwargs)
                return _coerce_audio(out, model)
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Seed-VC Python API call failed: {last_err}")


def _coerce_audio(out: Any, model: Any) -> tuple[np.ndarray, int]:
    """Normalize Seed-VC return values to (float32 mono array, sr)."""
    import torch

    if isinstance(out, tuple) and len(out) >= 2:
        wav, sr = out[0], out[1]
    elif isinstance(out, dict):
        wav = out.get("wav") or out.get("audio") or out.get("waveform")
        sr = out.get("sr") or out.get("sample_rate") or getattr(model, "sr", 24000)
    else:
        wav = out
        sr = getattr(model, "sr", 24000)
    if isinstance(wav, torch.Tensor):
        arr = wav.detach().cpu().numpy()
    else:
        arr = np.asarray(wav)
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr, int(sr)


def convert_voice(
    input_path: str,
    ref_path: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> tuple[Optional[str], np.ndarray, int]:
    """
    Run Seed-VC: transfer speaker identity from ``ref_path`` onto ``input_path``.

    Returns ``(output_path_or_None, wav_float32_mono, sample_rate)``.
    """
    model = get_model()

    if _backend == "cli":
        tmp_out = output_path
        if tmp_out is None:
            fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_out = fd.name
            fd.close()
        code = _run_cli(str(input_path), str(ref_path), str(tmp_out), seed)
        if code != 0:
            raise RuntimeError(f"seed-vc CLI exited with code {code}")
        import soundfile as sf

        arr, sr = sf.read(tmp_out, dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1).astype(np.float32)
        return (output_path, arr.astype(np.float32), int(sr))

    arr, sr = _run_python(model, str(input_path), str(ref_path), seed)
    if output_path is not None:
        import soundfile as sf

        sf.write(output_path, arr, sr)
    return (output_path, arr, int(sr))


def unload() -> None:
    global _model, _device, _backend
    if _model is None:
        return
    try:
        if _backend == "python":
            try:
                del _model
            except Exception:
                pass
    finally:
        _model = None
        _backend = None
        _device = None
        _empty_cache()
        _log("unloaded.")
