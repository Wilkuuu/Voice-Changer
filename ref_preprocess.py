"""
Reference audio preprocessing for zero-shot voice conversion / cloning.

A single entry point ``prepare_reference`` delivers a clean, consistently
prepared reference sample that every VC/TTS backend can consume:

    - resampled to a target sample rate (default 24 kHz) with soxr_hq
    - high-pass filter (~60 Hz) to remove mic rumble / DC
    - optional denoise via ``noisereduce`` (non-stationary)
    - loudness normalization via ``pyloudnorm`` (-18 LUFS, -1 dBTP ceiling)
    - VAD-trim via silero-vad — concatenate voiced segments only
    - "best window" selection: slide a target-length window across the voiced
      audio, score windows by SNR & energy stability, pick the cleanest one

Result is cached as ``<input>.ref24k.wav`` next to the source so repeated
runs skip the heavy steps. Cache key is ``(mtime, size, config hash)``.

All heavy dependencies are imported lazily inside ``prepare_reference`` so the
module can be imported in minimal environments without crashing.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_TARGET_SR = 24_000
DEFAULT_TARGET_SECONDS = 20.0
MIN_WINDOW_SECONDS = 8.0
MAX_WINDOW_SECONDS = 25.0
HIGHPASS_HZ = 60.0
TARGET_LUFS = -18.0
TRUE_PEAK_CEILING_DBTP = -1.0


@dataclass
class RefMeta:
    """Metadata returned alongside the preprocessed reference audio."""
    sr: int
    duration_sec: float
    source_path: str
    cached_path: str
    used_cache: bool
    denoised: bool
    voiced_seconds: float
    best_window_seconds: float
    snr_db: float
    loudness_lufs: float


def _log(msg: str) -> None:
    print(f"[ref_preprocess] {msg}", flush=True)


def _cache_key(path: str, cfg: dict) -> str:
    """Stable hash based on file mtime+size and preprocessing config."""
    st = os.stat(path)
    blob = json.dumps(
        {"mtime": int(st.st_mtime), "size": int(st.st_size), "cfg": cfg},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _cache_paths(src_path: str, key: str) -> tuple[Path, Path]:
    base = Path(src_path).with_suffix("")
    return (
        base.with_suffix(f".ref{key}.wav"),
        base.with_suffix(f".ref{key}.json"),
    )


def _highpass(y: np.ndarray, sr: int, cutoff: float = HIGHPASS_HZ) -> np.ndarray:
    """2nd-order Butterworth high-pass; safe no-op if scipy missing."""
    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception:
        return y
    nyq = 0.5 * sr
    wn = max(1e-4, min(0.9999, cutoff / nyq))
    sos = butter(2, wn, btype="highpass", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _try_denoise(y: np.ndarray, sr: int) -> tuple[np.ndarray, bool]:
    """Run noisereduce if installed; otherwise pass-through."""
    try:
        import noisereduce as nr
    except Exception:
        _log("noisereduce not installed — skipping denoise.")
        return y, False
    try:
        cleaned = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.75)
        return cleaned.astype(np.float32), True
    except Exception as e:
        _log(f"denoise failed ({e}) — keeping original.")
        return y, False


def _loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> tuple[np.ndarray, float]:
    """Normalize to target LUFS; if pyloudnorm missing, fall back to peak-norm."""
    try:
        import pyloudnorm as pyln
    except Exception:
        peak = float(np.max(np.abs(y)) or 1.0)
        gain = 10 ** (TRUE_PEAK_CEILING_DBTP / 20.0) / max(peak, 1e-9)
        return (y * gain).astype(np.float32), float("nan")
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y))
    if not np.isfinite(loudness) or loudness < -70.0:
        loudness = -40.0
    y_norm = pyln.normalize.loudness(y, loudness, target_lufs)
    peak = float(np.max(np.abs(y_norm)) or 1.0)
    ceiling = 10 ** (TRUE_PEAK_CEILING_DBTP / 20.0)
    if peak > ceiling:
        y_norm = y_norm * (ceiling / peak)
    return y_norm.astype(np.float32), loudness


def _silero_vad_segments(y: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """
    Return list of (start_sample, end_sample) voiced regions using silero-vad.
    Falls back to a single full-length segment if the model is unavailable.
    """
    try:
        import torch

        # Prefer the standalone ``silero_vad`` package if installed, else hub.
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            model = load_silero_vad()
        except Exception:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            get_speech_timestamps = utils[0]

        vad_sr = 16_000
        if sr != vad_sr:
            try:
                import librosa
                y_vad = librosa.resample(y, orig_sr=sr, target_sr=vad_sr, res_type="soxr_hq")
            except Exception:
                y_vad = y
                vad_sr = sr
        else:
            y_vad = y

        audio_tensor = torch.from_numpy(y_vad).float()
        ts = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=vad_sr,
            min_speech_duration_ms=250,
            min_silence_duration_ms=150,
            speech_pad_ms=60,
        )
        if not ts:
            return [(0, len(y))]
        scale = sr / float(vad_sr)
        return [
            (int(seg["start"] * scale), int(seg["end"] * scale))
            for seg in ts
        ]
    except Exception as e:
        _log(f"silero-vad unavailable ({e}) — using full audio as one segment.")
        return [(0, len(y))]


def _concat_voiced(y: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    if not segments:
        return y
    pieces = [y[s:e] for s, e in segments if e > s]
    if not pieces:
        return y
    return np.concatenate(pieces).astype(np.float32)


def _estimate_snr_db(y: np.ndarray, sr: int) -> float:
    """
    Rough SNR estimate: ratio of top-10% frame RMS (speech) to bottom-10% (noise).
    Frame size = 50 ms, hop = 25 ms.
    """
    if len(y) < sr // 4:
        return 0.0
    win = max(1, int(0.05 * sr))
    hop = max(1, int(0.025 * sr))
    n_frames = 1 + (len(y) - win) // hop
    if n_frames < 10:
        return 0.0
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = y[i * hop : i * hop + win]
        rms[i] = float(np.sqrt(np.mean(frame * frame) + 1e-12))
    rms.sort()
    noise = float(np.mean(rms[: max(1, n_frames // 10)]))
    speech = float(np.mean(rms[-max(1, n_frames // 10) :]))
    if noise < 1e-8:
        return 60.0
    return float(20.0 * np.log10(speech / noise))


def _pick_best_window(
    y: np.ndarray,
    sr: int,
    target_seconds: float,
) -> tuple[np.ndarray, float, float]:
    """
    Slide a target-length window; score each by SNR minus energy variance.
    Returns (window, best_seconds, snr_db).
    """
    total_sec = len(y) / sr
    if total_sec <= target_seconds * 1.1:
        return y.astype(np.float32), float(total_sec), _estimate_snr_db(y, sr)

    win_sec = max(MIN_WINDOW_SECONDS, min(MAX_WINDOW_SECONDS, float(target_seconds)))
    win = int(win_sec * sr)
    hop = max(1, int(1.0 * sr))  # 1 s hop
    best_score = -1e9
    best_slice = (0, win)
    for start in range(0, len(y) - win + 1, hop):
        frag = y[start : start + win]
        snr = _estimate_snr_db(frag, sr)
        # Penalize fragments whose frame-RMS varies wildly (unstable energy
        # = paused speech, shouts, clipping). Reward steady + high SNR.
        rms = np.sqrt(np.mean(frag.reshape(-1) ** 2) + 1e-12)
        var_pen = float(np.std(np.abs(frag))) / (rms + 1e-9)
        score = snr - 4.0 * var_pen
        if score > best_score:
            best_score = score
            best_slice = (start, start + win)
    s, e = best_slice
    chunk = y[s:e].astype(np.float32)
    return chunk, float(len(chunk) / sr), _estimate_snr_db(chunk, sr)


def _resample(y: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return y.astype(np.float32)
    import librosa
    return librosa.resample(y, orig_sr=src_sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)


def prepare_reference(
    path: str,
    target_sr: int = DEFAULT_TARGET_SR,
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    denoise: bool = True,
    use_cache: bool = True,
) -> tuple[np.ndarray, int, RefMeta]:
    """
    Prepare a reference sample for voice cloning.

    Returns
    -------
    y        : np.ndarray, float32, mono
    sr       : int (== target_sr)
    meta     : RefMeta

    Side effects
    ------------
    Writes sidecar cache ``<stem>.ref<key>.wav`` and ``<stem>.ref<key>.json``
    next to the source file so repeated calls skip the heavy steps.
    """
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")

    cfg = {
        "sr": int(target_sr),
        "sec": float(target_seconds),
        "denoise": bool(denoise),
        "lufs": TARGET_LUFS,
        "hp_hz": HIGHPASS_HZ,
    }
    key = _cache_key(path, cfg)
    cache_wav, cache_json = _cache_paths(path, key)

    if use_cache and cache_wav.exists() and cache_json.exists():
        try:
            import soundfile as sf
            y, sr = sf.read(str(cache_wav), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1).astype(np.float32)
            meta_d = json.loads(cache_json.read_text(encoding="utf-8"))
            meta_d["used_cache"] = True
            meta_d["cached_path"] = str(cache_wav)
            _log(f"cache hit: {cache_wav.name}")
            return y, int(sr), RefMeta(**meta_d)
        except Exception as e:
            _log(f"cache read failed ({e}) — recomputing.")

    _log(f"preparing reference: {path}")

    import librosa
    import soundfile as sf

    y, src_sr = librosa.load(path, sr=None, mono=True)
    y = y.astype(np.float32)

    # 1) resample to target SR first (cheaper downstream)
    y = _resample(y, src_sr, target_sr)
    sr = target_sr

    # 2) high-pass filter
    y = _highpass(y, sr)

    # 3) VAD to keep only voiced regions
    segments = _silero_vad_segments(y, sr)
    y_voiced = _concat_voiced(y, segments)
    voiced_sec = float(len(y_voiced) / sr)
    if voiced_sec < 1.0:
        _log(f"VAD produced only {voiced_sec:.2f} s — falling back to full audio.")
        y_voiced = y
        voiced_sec = float(len(y) / sr)

    # 4) optional denoise on voiced audio
    denoised = False
    if denoise:
        y_voiced, denoised = _try_denoise(y_voiced, sr)

    # 5) loudness normalize
    y_voiced, loud = _loudness_normalize(y_voiced, sr, TARGET_LUFS)

    # 6) pick best window
    y_win, win_sec, snr = _pick_best_window(y_voiced, sr, target_seconds)

    meta = RefMeta(
        sr=int(sr),
        duration_sec=float(len(y_win) / sr),
        source_path=str(path),
        cached_path=str(cache_wav),
        used_cache=False,
        denoised=bool(denoised),
        voiced_seconds=float(voiced_sec),
        best_window_seconds=float(win_sec),
        snr_db=float(snr),
        loudness_lufs=float(loud) if np.isfinite(loud) else -1.0,
    )

    try:
        sf.write(str(cache_wav), y_win, sr, subtype="PCM_16")
        cache_json.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        _log(f"cache written: {cache_wav.name} ({meta.duration_sec:.1f} s, SNR ~{snr:.1f} dB)")
    except Exception as e:
        _log(f"cache write failed ({e}).")

    return y_win, sr, meta


def prepare_input_light(
    path: str,
    target_sr: int = DEFAULT_TARGET_SR,
    loudness_normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Light preprocessing for the *source* (input) audio in a VC pipeline.

    Only resample + optional LUFS normalize. We never VAD-trim or denoise the
    source because prosody and every linguistic detail must be preserved.
    """
    import librosa

    y, src_sr = librosa.load(path, sr=None, mono=True)
    y = _resample(y.astype(np.float32), src_sr, target_sr)
    if loudness_normalize:
        y, _ = _loudness_normalize(y, target_sr, TARGET_LUFS)
    return y, target_sr
