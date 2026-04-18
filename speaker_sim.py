"""
Speaker embedding + similarity for voice-cloning quality evaluation.

Used by app.py to:
  1) pick best candidate out of N stochastic VC renders (Chatterbox VC / Seed-VC),
  2) show a speaker-similarity score in the UI,
  3) warn when similarity is too low (<0.55) so the user can re-record ref.

Primary backend  : SpeechBrain ECAPA-TDNN (``speechbrain/spkrec-ecapa-voxceleb``)
Fallback backend : Resemblyzer (lighter, pure-Python)

Both backends are imported lazily so the module stays importable in minimal
environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

_ecapa = None
_resemblyzer = None
_backend: Optional[str] = None


def _log(msg: str) -> None:
    print(f"[speaker_sim] {msg}", flush=True)


def _select_device() -> str:
    try:
        import os
        forced = (os.environ.get("VOICE_CHANGER_FORCE_DEVICE") or "").strip().lower()
        import torch
        if forced == "cpu":
            return "cpu"
        if forced == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _try_load_ecapa():
    global _ecapa, _backend
    if _ecapa is not None:
        return _ecapa
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except Exception as e:
            _log(f"ECAPA unavailable ({e}).")
            return None
    try:
        device = _select_device()
        _ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain_ecapa"),
            run_opts={"device": device},
        )
        _backend = "ecapa"
        _log(f"ECAPA-TDNN loaded on {device}.")
        return _ecapa
    except Exception as e:
        _log(f"ECAPA load failed ({e}).")
        return None


def _try_load_resemblyzer():
    global _resemblyzer, _backend
    if _resemblyzer is not None:
        return _resemblyzer
    try:
        from resemblyzer import VoiceEncoder
    except Exception as e:
        _log(f"Resemblyzer unavailable ({e}).")
        return None
    try:
        device = _select_device()
        _resemblyzer = VoiceEncoder(device=device)
        _backend = "resemblyzer"
        _log(f"Resemblyzer loaded on {device}.")
        return _resemblyzer
    except Exception as e:
        _log(f"Resemblyzer load failed ({e}).")
        return None


def is_available() -> bool:
    """True if at least one backend can be loaded."""
    if _ecapa is not None or _resemblyzer is not None:
        return True
    return bool(_try_load_ecapa() or _try_load_resemblyzer())


def _load_mono(path: str, target_sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32)


def _ensure_array(wav: Union[str, np.ndarray], sr: int, target_sr: int) -> np.ndarray:
    if isinstance(wav, str):
        return _load_mono(wav, target_sr)
    if sr == target_sr:
        return wav.astype(np.float32)
    import librosa
    return librosa.resample(
        wav.astype(np.float32), orig_sr=sr, target_sr=target_sr, res_type="soxr_hq"
    ).astype(np.float32)


def embed(wav: Union[str, np.ndarray], sr: int = 16_000) -> Optional[np.ndarray]:
    """
    Return a speaker embedding vector (L2-normalized) or None if no backend.

    ``wav`` may be a file path or a 1D float32 ndarray.
    """
    enc = _try_load_ecapa()
    if enc is not None:
        import torch
        y = _ensure_array(wav, sr, 16_000)
        with torch.inference_mode():
            t = torch.from_numpy(y).unsqueeze(0)
            emb = enc.encode_batch(t).squeeze().detach().cpu().numpy()
        emb = emb.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(emb) + 1e-12)
        return emb / n

    enc2 = _try_load_resemblyzer()
    if enc2 is not None:
        from resemblyzer import preprocess_wav
        y = _ensure_array(wav, sr, 16_000)
        y_proc = preprocess_wav(y, source_sr=16_000)
        emb = enc2.embed_utterance(y_proc).astype(np.float32)
        n = float(np.linalg.norm(emb) + 1e-12)
        return emb / n

    return None


def similarity(
    ref: Union[str, np.ndarray],
    out: Union[str, np.ndarray],
    ref_sr: int = 16_000,
    out_sr: int = 16_000,
) -> float:
    """
    Cosine similarity between two speaker embeddings, clamped to [0, 1].

    Returns ``-1.0`` if no backend is available (so callers can skip gating).
    """
    a = embed(ref, ref_sr)
    b = embed(out, out_sr)
    if a is None or b is None:
        return -1.0
    cos = float(np.dot(a, b))  # inputs are unit-norm
    return float(max(0.0, min(1.0, 0.5 * (cos + 1.0))) if cos < 0 else max(0.0, min(1.0, cos)))


def label(score: float) -> str:
    """Human-readable bucket for a similarity score."""
    if score < 0:
        return "unavailable"
    if score >= 0.82:
        return "excellent"
    if score >= 0.72:
        return "high"
    if score >= 0.60:
        return "fair"
    if score >= 0.50:
        return "weak"
    return "poor"


def backend_name() -> Optional[str]:
    return _backend


def unload() -> None:
    global _ecapa, _resemblyzer, _backend
    try:
        import torch
        if _ecapa is not None:
            del _ecapa
            _ecapa = None
        if _resemblyzer is not None:
            del _resemblyzer
            _resemblyzer = None
        _backend = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        _ecapa = None
        _resemblyzer = None
        _backend = None
