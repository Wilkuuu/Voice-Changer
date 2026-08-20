"""
Shared audio helpers: resampling, crossfade, overlap placement, LUFS mastering.
"""

from __future__ import annotations

import numpy as np

OUTPUT_SR_CLONE = 24_000
OUTPUT_SR_EDGE = 16_000
DEFAULT_CROSSFADE_MS = 40
MASTERING_LUFS = -16.0


def output_sr_for_backend(backend: str) -> int:
    """Sample rate for final WAV given TTS backend id."""
    b = (backend or "edge").lower().strip()
    if b in ("edge",):
        return OUTPUT_SR_EDGE
    return OUTPUT_SR_CLONE


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(wav) == 0:
        return wav.astype(np.float32, copy=False)
    import librosa
    return librosa.resample(
        wav.astype(np.float32, copy=False),
        orig_sr=int(orig_sr),
        target_sr=int(target_sr),
        res_type="soxr_hq",
    ).astype(np.float32)


def crossfade_samples(sr: int, ms: float = DEFAULT_CROSSFADE_MS) -> int:
    return max(0, int(sr * ms / 1000.0))


def cosine_crossfade_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Blend two equal-length 1D arrays with cosine crossfade."""
    n = min(len(a), len(b))
    if n == 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = 0.5 * (1.0 + np.cos(t * np.pi))
    fade_in = 1.0 - fade_out
    return (a[:n].astype(np.float32) * fade_out + b[:n].astype(np.float32) * fade_in).astype(np.float32)


def concat_with_crossfade(parts: list[np.ndarray], sr: int, pause_s: float = 0.0, fade_ms: float = DEFAULT_CROSSFADE_MS) -> np.ndarray:
    """Concatenate audio chunks with optional short pause and crossfade at joins."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    fade_n = crossfade_samples(sr, fade_ms)
    pause_n = max(0, int(sr * pause_s))
    pause = np.zeros(pause_n, dtype=np.float32) if pause_n else None

    out = parts[0].astype(np.float32, copy=False)
    for i, chunk in enumerate(parts[1:], start=1):
        chunk = chunk.astype(np.float32, copy=False)
        if pause is not None and pause_n:
            out = np.concatenate([out, pause, chunk])
            continue
        if fade_n > 0 and len(out) >= fade_n and len(chunk) >= fade_n:
            blended = cosine_crossfade_1d(out[-fade_n:], chunk[:fade_n])
            out = np.concatenate([out[:-fade_n], blended, chunk[fade_n:]])
        else:
            out = np.concatenate([out, chunk])
    return out.astype(np.float32, copy=False)


def mix_at(out: np.ndarray, start: int, wav: np.ndarray, fade_samples: int) -> np.ndarray:
    """Place ``wav`` into ``out`` at ``start``, cosine-crossfading overlap with existing content."""
    if len(wav) == 0:
        return out
    end = start + len(wav)
    if end > len(out):
        out = np.pad(out.astype(np.float32), (0, end - len(out)))
    else:
        out = out.astype(np.float32, copy=True)

    fade = min(fade_samples, len(wav) // 2, start, max(0, end - start))
    if fade > 0 and start > 0 and np.any(np.abs(out[start : start + fade]) > 1e-7):
        t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        fade_in = 0.5 * (1.0 - np.cos(t * np.pi))
        fade_out = 1.0 - fade_in
        region = out[start : start + fade]
        out[start : start + fade] = region * fade_out + wav[:fade] * fade_in
        out[start + fade : end] = wav[fade:]
    else:
        out[start:end] = wav
    return out


def normalize_lufs(wav: np.ndarray, sr: int, target_lufs: float = MASTERING_LUFS) -> np.ndarray:
    """LUFS normalize with true-peak ceiling; no-op if pyloudnorm missing."""
    if len(wav) == 0:
        return wav.astype(np.float32)
    y = wav.astype(np.float32, copy=False)
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(int(sr))
        loudness = meter.integrated_loudness(y)
        if np.isfinite(loudness):
            y = pyln.normalize.loudness(y, loudness, float(target_lufs))
        peak = float(np.max(np.abs(y)) + 1e-12)
        if peak > 0.99:
            y = y * (0.99 / peak)
    except Exception:
        peak = float(np.max(np.abs(y)) + 1e-12)
        if peak > 0.95:
            y = y * (0.95 / peak)
    return y.astype(np.float32)
