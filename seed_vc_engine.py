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

The PyPI wheel (e.g. ``seed_vc`` 0.4.x) exposes ``seed_vc.inference`` with
``load_models(args)`` and ``main(args)``. That module also does
``from .modules.commons import *``, which re-exports ``build_model`` — it is
**not** a top-level factory; calling it without ``(model_params, stage)``
raises misleading errors. We therefore never probe ``build_model`` on the
inference module; instead we patch ``load_models`` to cache weights and call
``main()`` per conversion.

Official console scripts use names like ``seed-vc-infer-v1``, not ``seed-vc``.

Install:
    pip install seed-vc
    # or: pip install git+https://github.com/Plachtaa/seed-vc
"""

from __future__ import annotations

import importlib
import importlib.util
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from device_utils import empty_cache as _empty_cache, select_device as _select_device

_model: Any = None
_device: Optional[str] = None
_backend: Optional[str] = None  # "python_seed_v1" | "cli_exe" | "cli_pymod"

_seed_infer_module: Any = None
_seed_infer_original_load_models: Any = None
_seed_infer_bundle_cache: dict[str, Any] = {"b": None}

# Original BigVGAN ``_from_pretrained`` (unbound) — saved once for idempotent wrapping.
_bigvgan_from_pretrained_orig: Any = None

_main_lock = threading.Lock()


def _log(msg: str) -> None:
    print(f"[Seed-VC] {msg}", flush=True)


def _apply_bigvgan_huggingface_hub_compat() -> None:
    """
    huggingface_hub's ``PyTorchModelHubMixin`` no longer passes ``proxies`` and
    ``resume_download`` into ``_from_pretrained``, but NVIDIA BigVGAN (vendored
    inside ``seed-vc``) still declares them as required keyword-only parameters.
    Supply defaults so ``BigVGAN.from_pretrained`` works on modern hub versions.
    """
    global _bigvgan_from_pretrained_orig

    Big = None
    for mod_name in ("seed_vc.modules.bigvgan.bigvgan", "seed_vc.modules.bigvgan"):
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue
        Big = getattr(m, "BigVGAN", None)
        if Big is not None:
            break
    if Big is None:
        return
    if getattr(Big, "_voice_changer_hub_compat_patched", False):
        return

    cm = Big.__dict__.get("_from_pretrained")
    if cm is None:
        return
    try:
        orig_fn = cm.__func__
    except AttributeError:
        return

    if _bigvgan_from_pretrained_orig is None:
        _bigvgan_from_pretrained_orig = orig_fn

    def _wrapper(cls: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("proxies", None)
        kwargs.setdefault("resume_download", False)
        return _bigvgan_from_pretrained_orig(cls, *args, **kwargs)

    Big._from_pretrained = classmethod(_wrapper)  # type: ignore[assignment]
    Big._voice_changer_hub_compat_patched = True
    _log("BigVGAN._from_pretrained: huggingface_hub compatibility patch applied.")


def _prepare_seed_hf_cache() -> None:
    """``seed_vc.inference`` sets HF_HUB_CACHE on import; point it under XDG_CACHE_HOME."""
    import os

    hub = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "seed_vc",
        "hf_hub",
    )
    os.makedirs(hub, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = hub


def _patch_seed_inference_load_models(mod: Any) -> None:
    """Cache the heavy tuple returned by ``load_models`` — ``main()`` calls it every time."""
    global _seed_infer_module, _seed_infer_original_load_models
    # Idempotent: ``importlib.reload(seed_vc_engine)`` must not wrap ``load_models`` twice.
    if getattr(mod, "_voice_changer_seed_vc_patched", False):
        _seed_infer_module = mod
        return
    _seed_infer_original_load_models = mod.load_models
    _seed_infer_module = mod

    def _cached_load_models(args: Any) -> Any:
        if _seed_infer_bundle_cache["b"] is None:
            _seed_infer_bundle_cache["b"] = _seed_infer_original_load_models(args)
        return _seed_infer_bundle_cache["b"]

    mod.load_models = _cached_load_models
    mod._voice_changer_seed_vc_patched = True


def _unpatch_seed_inference() -> None:
    global _seed_infer_module, _seed_infer_original_load_models
    if _seed_infer_module is not None and _seed_infer_original_load_models is not None:
        try:
            if getattr(_seed_infer_module, "_voice_changer_seed_vc_patched", False):
                _seed_infer_module.load_models = _seed_infer_original_load_models
                try:
                    delattr(_seed_infer_module, "_voice_changer_seed_vc_patched")
                except Exception:
                    pass
        except Exception:
            pass
    _seed_infer_module = None
    _seed_infer_original_load_models = None
    _seed_infer_bundle_cache["b"] = None


def _spec_seed_inference() -> Any:
    return importlib.util.find_spec("seed_vc.inference")


def is_available() -> bool:
    if _spec_seed_inference() is not None:
        return True
    if shutil.which("seed-vc-infer-v1") or shutil.which("seed-vc-infer-v2"):
        return True
    return False


def _v1_main_args(source: str, target: str, out_dir: str) -> Any:
    """Namespace compatible with ``seed_vc.inference.main`` + ``load_models``."""
    dev = _select_device()
    return SimpleNamespace(
        source=str(source),
        target=str(target),
        output=str(out_dir),
        diffusion_steps=30,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        f0_condition=False,
        auto_f0_adjust=False,
        semi_tone_shift=0,
        checkpoint=None,
        config=None,
        fp16=dev == "cuda",
    )


def _run_seed_v1_inference_main(mod: Any, input_path: str, ref_path: str, seed: Optional[int]) -> tuple[np.ndarray, int]:
    """Call ``seed_vc.inference.main`` once; models stay cached via patched ``load_models``."""
    import torch
    import soundfile as sf

    if seed is not None:
        s = int(seed)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        random.seed(s)
        np.random.seed(s)

    out_dir = tempfile.mkdtemp(prefix="seedvc_main_")
    try:
        args = _v1_main_args(input_path, ref_path, out_dir)
        with _main_lock:
            mod.main(args)
        outs = sorted(Path(out_dir).glob("vc_*.wav"), key=lambda p: p.stat().st_mtime)
        if not outs:
            raise RuntimeError("seed_vc.inference produced no vc_*.wav in output directory.")
        fp = outs[-1]
        arr, sr = sf.read(str(fp), dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32), int(sr)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _cli_cmd_v1(argv0: str, out_dir: str, input_path: str, ref_path: str) -> list[str]:
    dev = _select_device()
    fp16s = "true" if dev == "cuda" else "false"
    tail = [
        "--source",
        str(input_path),
        "--target",
        str(ref_path),
        "--output",
        str(out_dir),
        "--diffusion-steps",
        "30",
        "--length-adjust",
        "1.0",
        "--inference-cfg-rate",
        "0.7",
        "--f0-condition",
        "false",
        "--auto-f0-adjust",
        "false",
        "--semi-tone-shift",
        "0",
        "--fp16",
        fp16s,
    ]
    if argv0 == "PYMOD":
        return [sys.executable, "-m", "seed_vc.inference", *tail]
    return [argv0, *tail]


def _cli_cmd_v2(exe: str, out_dir: str, input_path: str, ref_path: str) -> list[str]:
    return [
        exe,
        "--source",
        str(input_path),
        "--target",
        str(ref_path),
        "--output",
        str(out_dir),
        "--diffusion-steps",
        "30",
        "--length-adjust",
        "1.0",
        "--intelligibility-cfg-rate",
        "0.7",
        "--similarity-cfg-rate",
        "0.7",
        "--top-p",
        "0.9",
        "--temperature",
        "1.0",
        "--repetition-penalty",
        "1.0",
        "--convert-style",
        "false",
        "--anonymization-only",
        "false",
    ]


def _run_cli_infer(argv0: str, input_path: str, ref_path: str, seed: Optional[int]) -> tuple[np.ndarray, int]:
    import torch
    import soundfile as sf

    if seed is not None:
        s = int(seed)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        random.seed(s)
        np.random.seed(s)

    out_dir = tempfile.mkdtemp(prefix="seedvc_cli_")
    try:
        if isinstance(argv0, tuple) and len(argv0) == 2 and argv0[0] == "v2":
            cmd = _cli_cmd_v2(argv0[1], out_dir, input_path, ref_path)
            glob_pat = "vc_v2_*.wav"
        else:
            cmd = _cli_cmd_v1(argv0, out_dir, input_path, ref_path)
            glob_pat = "vc_*.wav"
        _log("CLI: " + " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "")[-4000:]
            raise RuntimeError(f"seed-vc CLI exited with code {res.returncode}\n{tail}")
        outs = sorted(Path(out_dir).glob(glob_pat), key=lambda p: p.stat().st_mtime)
        if not outs:
            raise RuntimeError(f"CLI produced no {glob_pat} in output directory.")
        fp = outs[-1]
        arr, sr = sf.read(str(fp), dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32), int(sr)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def get_model():
    global _model, _device, _backend
    if _model is not None:
        return _model

    dev = _select_device()

    if _spec_seed_inference() is not None:
        try:
            _log("loading seed_vc.inference (in-process, cached load_models)...")
            mod = importlib.import_module("seed_vc.inference")
            _prepare_seed_hf_cache()
            _apply_bigvgan_huggingface_hub_compat()
            _patch_seed_inference_load_models(mod)
            _model = mod
            _device = dev
            _backend = "python_seed_v1"
            _log(f"ready ({_backend}) on {_device}.")
            return _model
        except Exception as e:
            _log(f"in-process seed_vc.inference failed ({e}); will try CLI.")

    p1 = shutil.which("seed-vc-infer-v1")
    if p1:
        _model = p1
        _device = dev
        _backend = "cli_exe"
        _log(f"using CLI backend at {p1}")
        return _model
    p2 = shutil.which("seed-vc-infer-v2")
    if p2:
        _model = ("v2", p2)
        _device = dev
        _backend = "cli_exe"
        _log(f"using CLI backend (v2) at {p2}")
        return _model

    if _spec_seed_inference() is not None:
        _model = "PYMOD"
        _device = dev
        _backend = "cli_pymod"
        _log("using subprocess: python -m seed_vc.inference")
        return _model

    raise RuntimeError(
        "Seed-VC is not installed. Install with: pip install seed-vc "
        "(or: pip install git+https://github.com/Plachtaa/seed-vc)."
    )


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
    get_model()

    if _backend == "python_seed_v1":
        arr, sr = _run_seed_v1_inference_main(_model, str(input_path), str(ref_path), seed)
    elif _backend in ("cli_exe", "cli_pymod"):
        if _model == "PYMOD":
            argv0 = "PYMOD"
        elif isinstance(_model, tuple) and _model[0] == "v2":
            argv0 = _model  # ("v2", path)
        else:
            argv0 = str(_model)
        arr, sr = _run_cli_infer(argv0, str(input_path), str(ref_path), seed)
    else:
        raise RuntimeError(f"Unknown Seed-VC backend: {_backend!r}")

    if output_path is not None:
        import soundfile as sf

        sf.write(output_path, arr, sr)
    return (output_path, arr, sr)


def unload() -> None:
    global _model, _device, _backend
    if _model is None:
        return
    try:
        if _backend == "python_seed_v1":
            _unpatch_seed_inference()
            try:
                del _model
            except Exception:
                pass
        elif _backend in ("cli_exe", "cli_pymod"):
            pass
    finally:
        _model = None
        _backend = None
        _device = None
        _empty_cache()
        _log("unloaded.")


# Mtime of this file at import — used by app.py to hot-reload after ``git pull`` on a bind mount.
_ENGINE_FILE_MTIME = Path(__file__).stat().st_mtime
