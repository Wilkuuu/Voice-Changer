#!/usr/bin/env python3
"""
Translation pipeline: speech → text → translate → TTS → (voice conversion)

Free tools (all local except edge-tts):
- faster-whisper : ASR, translates any language → English segments with timestamps
- argostranslate : English → target language (local, own package system)
- edge-tts       : Neural TTS (free, Microsoft Edge voices, online)

Sync modes for segment timing:
- off    : natural speech speed, crossfade placement (default, best quality)
- gentle : scale silences only; never time-stretch speech
- strict : fit to SRT slots including time-stretch (may compress speech)
"""

import asyncio
import os
import re
import threading
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import edge_tts
from faster_whisper import WhisperModel

import argostranslate.package
import argostranslate.translate

from audio_utils import (
    crossfade_samples,
    mix_at,
    normalize_lufs,
    output_sr_for_backend,
    resample,
)
from text_normalize_pl import normalize_for_tts

SAMPLE_RATE = 16000  # legacy / Edge-TTS timeline base
VAD_TOP_DB = 30  # silence threshold (dB below peak) for speech/silence split
# Max time-compression when TTS is longer than the Whisper segment (higher = fewer hard cuts)
MAX_SYNC_SPEEDUP = 2.0

# Target languages: ISO 639-1 code + edge-tts neural voice
LANGUAGES: dict[str, dict] = {
    "English":    {"lang_code": "en", "tts_voice": "en-US-JennyNeural"},
    "Polish":     {"lang_code": "pl", "tts_voice": "pl-PL-ZofiaNeural"},
    "German":     {"lang_code": "de", "tts_voice": "de-DE-KatjaNeural"},
    "French":     {"lang_code": "fr", "tts_voice": "fr-FR-DeniseNeural"},
    "Spanish":    {"lang_code": "es", "tts_voice": "es-ES-ElviraNeural"},
    "Italian":    {"lang_code": "it", "tts_voice": "it-IT-ElsaNeural"},
    "Russian":    {"lang_code": "ru", "tts_voice": "ru-RU-SvetlanaNeural"},
    "Ukrainian":  {"lang_code": "uk", "tts_voice": "uk-UA-PolinaNeural"},
    "Dutch":      {"lang_code": "nl", "tts_voice": "nl-NL-ColetteNeural"},
    "Portuguese": {"lang_code": "pt", "tts_voice": "pt-PT-RaquelNeural"},
}


def edge_voice_for_lang_code(lang_code: str) -> str:
    """Microsoft Edge neural voice id for ISO 639-1 code (see LANGUAGES). Unknown → US English."""
    lc = (lang_code or "en").strip().lower()
    for meta in LANGUAGES.values():
        if meta["lang_code"] == lc:
            return meta["tts_voice"]
    return LANGUAGES["English"]["tts_voice"]


_whisper_model: WhisperModel | None = None
_whisper_cfg: tuple[str, str, str] | None = None  # (size, device, compute_type)
_installed_pairs: set[tuple[str, str]] = set()


def _whisper_auto_device_compute() -> tuple[str, str]:
    """Pick the best (device, compute_type) for faster-whisper on this host."""
    forced = (os.environ.get("VOICE_CHANGER_FORCE_DEVICE") or "").strip().lower()
    try:
        if forced == "cpu":
            return "cpu", "int8"
        if torch.cuda.is_available() and forced != "cpu":
            return "cuda", "int8_float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_whisper(
    model_size: str = "base",
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperModel:
    """
    Lazy-load faster-whisper model with optional size/device override.

    - ``model_size``: ``tiny|base|small|medium|large-v2|large-v3`` etc.
    - ``device``: ``cpu|cuda|auto`` (None = auto)
    - ``compute_type``: ``int8`` (CPU), ``int8_float16`` (GPU fast), ``float16`` (GPU high-q)

    Re-loads when any of the three parameters change.
    """
    global _whisper_model, _whisper_cfg
    if device in (None, "auto"):
        device, auto_ct = _whisper_auto_device_compute()
        if compute_type is None:
            compute_type = auto_ct
    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"

    target = (str(model_size), str(device), str(compute_type))
    if _whisper_model is None or _whisper_cfg != target:
        if _whisper_model is not None:
            print(f"Switching Whisper: {_whisper_cfg} → {target}")
            try:
                del _whisper_model
            except Exception:
                pass
            _whisper_model = None
        print(f"Loading Whisper-{model_size} on {device} ({compute_type})...")
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _whisper_cfg = target
        print("Whisper loaded.")
    return _whisper_model


def unload_whisper() -> None:
    """Free VRAM held by the cached faster-whisper model (best-effort)."""
    global _whisper_model, _whisper_cfg
    if _whisper_model is None:
        return
    try:
        del _whisper_model
    except Exception:
        pass
    _whisper_model = None
    _whisper_cfg = None
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("Whisper unloaded.")


def ensure_translation_package(from_code: str, to_code: str) -> None:
    pair = (from_code, to_code)
    if pair in _installed_pairs:
        return
    installed_langs = argostranslate.translate.get_installed_languages()
    for lang in installed_langs:
        if lang.code == from_code:
            if any(t.to_lang.code == to_code for t in lang.translations_to):
                _installed_pairs.add(pair)
                return
    print(f"Downloading translation package [{from_code}→{to_code}] (~50-100 MB)...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if pkg is None:
        raise ValueError(f"No argostranslate package for [{from_code}→{to_code}].")
    argostranslate.package.install_from_path(pkg.download())
    _installed_pairs.add(pair)
    print("Translation package installed.")


# ── Gender-aware translation ──────────────────────────────────────────────────

# Prepended before the text so the MT model sees a female speaker in context.
# The translated version of this sentence is then stripped from the output.
_FEMALE_CONTEXT_EN = "I am a woman."

# Language-specific regex to catch any remaining masculine forms after translation.
# Format: list of (pattern, replacement) applied in order.
_FEMALE_FIXERS: dict[str, list[tuple[str, str]]] = {
    # Polish: first-person past tense -łem/-łeś → -łam/-łaś
    "pl": [
        (r"\b(\w+)łem\b", r"\1łam"),   # byłem→byłam, zrobiłem→zrobiłam
        (r"\b(\w+)łeś\b", r"\1łaś"),   # byłeś→byłaś
        (r"\b(\w+)łbym\b", r"\1łabym"),# chciałbym→chciałabym (conditional)
    ],
    # German: ich war/hatte/bin ... (hard to regex; context injection handles most cases)
    "de": [],
    # French: je suis allé → je suis allée  (past participle agreement)
    "fr": [
        (r"\bje suis allé\b", "je suis allée"),
        (r"\bje suis parti\b", "je suis partie"),
        (r"\bje suis sorti\b", "je suis sortie"),
    ],
    # Russian: first-person past tense -л → -ла
    "ru": [
        (r"\b(\w+)л\b(?!\w)", r"\1ла"),   # был→была, сделал→сделала
    ],
    # Ukrainian: similar to Russian
    "uk": [
        (r"\b(\w+)в\b(?!\w)", r"\1ла"),
    ],
}


def _apply_female_fixers(text: str, lang_code: str) -> str:
    for pattern, replacement in _FEMALE_FIXERS.get(lang_code, []):
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def translate_en_to_target(text: str, target_language: str, female_narrator: bool = False) -> str:
    """Translate English text to target language, optionally with female narrator context."""
    lang_code = LANGUAGES[target_language]["lang_code"]
    if lang_code == "en":
        return text
    ensure_translation_package("en", lang_code)

    if female_narrator:
        # Prepend a short female-context sentence so the MT model uses feminine forms
        combined = _FEMALE_CONTEXT_EN + " " + text
        translated = argostranslate.translate.translate(combined, "en", lang_code)
        # Strip the translated context sentence (it's short, ends with first ". " or ".")
        dot = translated.find(". ")
        if 0 < dot < 40:          # sanity-check: context sentence is always short
            translated = translated[dot + 2:].strip()
        elif translated.startswith(translated[:3]):
            # fallback: remove first token-sentence if no ". " found
            pass
        return _apply_female_fixers(translated, lang_code)

    return argostranslate.translate.translate(text, "en", lang_code)


def _tts_run(text: str, voice: str, path: str, rate: str = "+0%", pitch: str = "+0Hz") -> None:
    """Run edge-tts in a dedicated thread with its own event loop (no Gradio conflict).

    rate  : speaking rate offset, e.g. "+0%", "-10%", "+20%"
    pitch : pitch offset in Hz,   e.g. "+0Hz", "-5Hz",  "+10Hz"
    """
    exc: list[Exception] = []

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            loop.run_until_complete(comm.save(path))
        except Exception as e:
            exc.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if exc:
        raise exc[0]


def tts_to_file(text: str, target_language: str, output_path: str,
                rate: str = "+0%", pitch: str = "+0Hz") -> None:
    _tts_run(text, LANGUAGES[target_language]["tts_voice"], output_path, rate=rate, pitch=pitch)


# ── Silence-aware fitting ─────────────────────────────────────────────────────

def _speech_intervals(wav: np.ndarray) -> list[tuple[int, int]]:
    if len(wav) == 0:
        return []
    return [(int(s), int(e)) for s, e in librosa.effects.split(wav, top_db=VAD_TOP_DB)]


def _concat_speech(wav: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    if not intervals:
        return wav.astype(np.float32)
    return np.concatenate([wav[s:e].astype(np.float32) for s, e in intervals])


def _truncate_at_speech_boundary(wav: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Fit into ``target_samples`` using only *whole* VAD speech regions.
    Never slices inside a region (avoids cutting a word in half).
    """
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    wav = wav.astype(np.float32)
    if len(wav) <= target_samples:
        return np.pad(wav, (0, target_samples - len(wav)))

    intervals = _speech_intervals(wav)
    if not intervals:
        return wav[:target_samples]

    parts: list[np.ndarray] = []
    used = 0
    for s, e in intervals:
        chunk = wav[s:e]
        if used + len(chunk) > target_samples:
            break
        parts.append(chunk)
        used += len(chunk)

    if not parts:
        ratio = len(wav) / max(1, target_samples)
        stretched = librosa.effects.time_stretch(wav, rate=float(min(ratio, MAX_SYNC_SPEEDUP + 0.5)))
        return stretched[:target_samples]

    result = np.concatenate(parts)
    if len(result) < target_samples:
        result = np.pad(result, (0, target_samples - len(result)))
    return result.astype(np.float32)


def _fit_by_silence(tts_wav: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Fit TTS audio to target_samples by scaling ONLY the silence regions.
    Speech portions are NEVER stretched or resampled — they play at natural speed.

    Strategy:
    - Detect speech/silence regions via VAD (librosa.effects.split)
    - Distribute the silence budget proportionally across all silence gaps
    - If speech alone exceeds target: return speech-only audio (trimmed to target)
    - If no silences exist: pad with trailing silence
    """
    if len(tts_wav) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    intervals = librosa.effects.split(tts_wav, top_db=VAD_TOP_DB)

    if len(intervals) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    # Build list of silence lengths: [pre, between..., post]  (len = n_speech + 1)
    n = len(intervals)
    silence_orig = [int(intervals[0][0])]
    for j in range(n - 1):
        silence_orig.append(int(intervals[j + 1][0] - intervals[j][1]))
    silence_orig.append(int(len(tts_wav) - intervals[-1][1]))

    speech_total = sum(int(e - s) for s, e in intervals)

    # Speech longer than the segment slot: time-stretch, then trim only on speech boundaries.
    if speech_total >= target_samples:
        speech_only = _concat_speech(tts_wav, intervals)
        ratio = speech_total / max(1, target_samples)
        if ratio <= MAX_SYNC_SPEEDUP:
            stretched = librosa.effects.time_stretch(speech_only, rate=float(ratio))
            stretched = stretched.astype(np.float32)
            if len(stretched) <= target_samples:
                return np.pad(stretched, (0, target_samples - len(stretched)))
            return _truncate_at_speech_boundary(stretched, target_samples)
        print(
            f"[sync] TTS is {ratio:.2f}x longer than segment — compressing to {MAX_SYNC_SPEEDUP}x "
            f"then dropping tail *words* (not mid-word). Shorten translation or disable sync.",
            flush=True,
        )
        stretched = librosa.effects.time_stretch(
            speech_only, rate=float(MAX_SYNC_SPEEDUP),
        ).astype(np.float32)
        return _truncate_at_speech_boundary(stretched, target_samples)

    silence_budget = target_samples - speech_total
    orig_silence_total = sum(silence_orig)

    if orig_silence_total > 0:
        scale = silence_budget / orig_silence_total
        new_silences = [max(0, round(s * scale)) for s in silence_orig]
    else:
        # No original silences: add all budget as trailing silence
        new_silences = [0] * n + [silence_budget]

    # Assemble: silence[j], speech[j], ..., speech[n-1], silence[n]
    parts: list[np.ndarray] = []
    for j, (s, e) in enumerate(intervals):
        if new_silences[j] > 0:
            parts.append(np.zeros(new_silences[j], dtype=np.float32))
        parts.append(tts_wav[s:e].astype(np.float32))
    if new_silences[-1] > 0:
        parts.append(np.zeros(new_silences[-1], dtype=np.float32))

    result = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    # Guarantee exact target length
    if len(result) < target_samples:
        result = np.pad(result, (0, target_samples - len(result)))
    return result[:target_samples]


def _coerce_sync_mode(sync: bool | str) -> str:
    """Map UI value to sync mode: off | gentle | strict."""
    if isinstance(sync, str):
        mode = sync.lower().strip()
        if mode in ("off", "gentle", "strict"):
            return mode
        if mode in ("false", "0", "none", ""):
            return "off"
        return "strict"
    return "strict" if sync else "off"


def _fit_by_silence_gentle(tts_wav: np.ndarray, target_samples: int) -> tuple[np.ndarray, bool]:
    """
    Fit TTS by scaling silences only. Never time-stretches speech.
    Returns (audio, overflow) where overflow=True if speech exceeds the slot.
    """
    if len(tts_wav) == 0:
        return np.zeros(target_samples, dtype=np.float32), False

    intervals = librosa.effects.split(tts_wav, top_db=VAD_TOP_DB)
    if len(intervals) == 0:
        return np.zeros(target_samples, dtype=np.float32), False

    speech_total = sum(int(e - s) for s, e in intervals)
    if speech_total >= target_samples:
        speech_only = _concat_speech(tts_wav, intervals)
        overflow = len(speech_only) > target_samples
        if overflow:
            return _truncate_at_speech_boundary(speech_only, target_samples), True
        return np.pad(speech_only, (0, target_samples - len(speech_only))), False

    # Reuse silence scaling from _fit_by_silence (speech fits)
    return _fit_by_silence(tts_wav, target_samples), False


def _merge_segments_for_tts(
    parsed: list[tuple[str, str, str]],
    max_gap_s: float = 0.4,
    max_chars: int = 400,
) -> list[dict]:
    """Merge adjacent SRT segments for single TTS calls (better prosody)."""
    groups: list[dict] = []
    current: dict | None = None

    for start_s, end_s, text in parsed:
        text = text.strip()
        if not text:
            continue
        start_f, end_f = float(start_s), float(end_s)
        if current is None:
            current = {
                "start": start_f,
                "end": end_f,
                "texts": [text],
            }
            continue
        gap = start_f - current["end"]
        combined_len = len(" ".join(current["texts"])) + 1 + len(text)
        if gap <= max_gap_s and combined_len <= max_chars:
            current["end"] = end_f
            current["texts"].append(text)
        else:
            groups.append(current)
            current = {"start": start_f, "end": end_f, "texts": [text]}
    if current is not None:
        groups.append(current)

    for g in groups:
        g["text"] = " ".join(g["texts"])
    return groups


def _synthesize_tts_file(
    text: str,
    *,
    backend: str,
    lang_code: str,
    voice: str,
    ref_path: str | None,
    tts_path: str,
    tts_rate: str,
    tts_pitch: str,
    xtts_speed: float,
    chatterbox_exaggeration: float,
    chatterbox_cfg_weight: float,
    f5tts_ref_text: str,
    f5tts_model_path: str | None,
    f5tts_speed: float,
    seed: int = -1,
) -> None:
    text = normalize_for_tts(text, lang_code)
    if backend == "xtts":
        import xtts_engine
        xtts_engine.synthesize(
            text=text,
            language=lang_code,
            ref_audio_path=ref_path,
            output_path=tts_path,
            speed=xtts_speed,
        )
    elif backend == "chatterbox":
        import chatterbox_engine
        chatterbox_engine.synthesize(
            text=text,
            language=lang_code,
            ref_audio_path=ref_path,
            output_path=tts_path,
            exaggeration=chatterbox_exaggeration,
            cfg_weight=chatterbox_cfg_weight,
        )
    elif backend == "f5tts":
        import f5tts_engine
        model_path = f5tts_model_path
        if model_path is None and lang_code == "pl":
            model_path = "polish"
        f5tts_engine.synthesize(
            text=text,
            ref_audio_path=ref_path,
            output_path=tts_path,
            ref_text=f5tts_ref_text,
            model_path=model_path,
            speed=f5tts_speed,
            seed=seed,
        )
    else:
        Path(tts_path).unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        _tts_run(text, voice, mp3_path, rate=tts_rate, pitch=tts_pitch)
        wav, sr = librosa.load(mp3_path, sr=None, mono=True)
        Path(mp3_path).unlink(missing_ok=True)
        sf.write(tts_path, resample(wav, int(sr), output_sr_for_backend("edge")), output_sr_for_backend("edge"))


def _build_timeline_audio(
    groups: list[dict],
    *,
    total_duration: float,
    output_sr: int,
    sync_mode: str,
    backend: str,
    lang_code: str,
    voice: str,
    ref_path: str | None,
    tts_rate: str,
    tts_pitch: str,
    xtts_speed: float,
    chatterbox_exaggeration: float,
    chatterbox_cfg_weight: float,
    f5tts_ref_text: str,
    f5tts_model_path: str | None,
    f5tts_speed: float,
    seed: int,
    progress_cb=None,
) -> tuple[np.ndarray, list[str]]:
    """Synthesize merged segment groups and assemble timeline with crossfade."""
    warnings: list[str] = []
    total_samples = max(int(total_duration * output_sr), 1)
    out = np.zeros(total_samples, dtype=np.float32)
    fade_n = crossfade_samples(output_sr, 45.0)
    n = len(groups)

    for i, group in enumerate(groups):
        text = group["text"]
        start_f, end_f = group["start"], group["end"]
        label = f"{i + 1}/{n}"
        if progress_cb:
            progress_cb(f"TTS group {label}: {text[:70]}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tts_path = f.name
        try:
            _synthesize_tts_file(
                text,
                backend=backend,
                lang_code=lang_code,
                voice=voice,
                ref_path=ref_path,
                tts_path=tts_path,
                tts_rate=tts_rate,
                tts_pitch=tts_pitch,
                xtts_speed=xtts_speed,
                chatterbox_exaggeration=chatterbox_exaggeration,
                chatterbox_cfg_weight=chatterbox_cfg_weight,
                f5tts_ref_text=f5tts_ref_text,
                f5tts_model_path=f5tts_model_path,
                f5tts_speed=f5tts_speed,
                seed=seed,
            )
            tts_wav, file_sr = librosa.load(tts_path, sr=None, mono=True)
            tts_wav = resample(tts_wav, int(file_sr), output_sr)
        finally:
            Path(tts_path).unlink(missing_ok=True)

        if len(tts_wav) == 0:
            continue

        seg_start = int(start_f * output_sr)
        seg_end = min(int(end_f * output_sr), total_samples)
        target_samples = max(seg_end - seg_start, 1)

        if sync_mode == "strict":
            tts_warped = _fit_by_silence(tts_wav, target_samples)
            out = mix_at(out, seg_start, tts_warped[:target_samples], fade_n)
        elif sync_mode == "gentle":
            tts_warped, overflow = _fit_by_silence_gentle(tts_wav, target_samples)
            if overflow:
                warnings.append(
                    f"Segment group {label}: mowa dłuższa niż slot SRT — skrócono bez przyspieszania."
                )
            out = mix_at(out, seg_start, tts_warped, fade_n)
        else:
            needed = seg_start + len(tts_wav)
            if needed > len(out):
                out = np.pad(out, (0, needed - len(out)))
            out = mix_at(out, seg_start, tts_wav, fade_n)

    return out, warnings


# ── Main building block ───────────────────────────────────────────────────────

def _build_synced_audio(
    segments: list,
    target_language: str,
    total_duration: float,
    female_narrator: bool = False,
    tts_rate: str = "+0%",
    tts_pitch: str = "+0Hz",
    progress_cb=None,
) -> tuple[np.ndarray, str, str]:
    """
    For each Whisper segment:
      1. Translate
      2. Synthesize TTS
      3. DTW-warp TTS energy peaks to align with source segment energy peaks
      4. Place at source segment's start position
    Returns (audio_array, english_text, translated_text).
    """
    voice = LANGUAGES[target_language]["tts_voice"]
    lang_code = LANGUAGES[target_language]["lang_code"]

    if lang_code != "en":
        ensure_translation_package("en", lang_code)

    total_samples = int(total_duration * SAMPLE_RATE)
    out = np.zeros(total_samples, dtype=np.float32)

    all_english: list[str] = []
    all_translated: list[str] = []
    n = len(segments)

    for i, seg in enumerate(segments):
        english = seg.text.strip()
        if not english:
            continue
        all_english.append(english)

        translated = translate_en_to_target(english, target_language, female_narrator=female_narrator)
        all_translated.append(translated)

        if progress_cb:
            progress_cb(f"Segment {i+1}/{n}: {translated[:70]}")

        # TTS synthesis
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts_path = f.name
        _tts_run(translated, voice, tts_path, rate=tts_rate, pitch=tts_pitch)
        tts_wav, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        Path(tts_path).unlink(missing_ok=True)

        if len(tts_wav) == 0:
            continue

        # Fit TTS to source segment duration by adjusting silences only
        seg_start_s = int(seg.start * SAMPLE_RATE)
        seg_end_s = min(int(seg.end * SAMPLE_RATE), total_samples)
        target_samples = seg_end_s - seg_start_s

        tts_warped = _fit_by_silence(tts_wav, target_samples)

        slot_end = seg_start_s + target_samples
        if slot_end > len(out):
            out = np.pad(out, (0, slot_end - len(out)))
        out[seg_start_s:slot_end] = tts_warped

    return out, " ".join(all_english), " ".join(all_translated)


# ── Public pipeline entry point ───────────────────────────────────────────────

def run_pipeline(
    audio_path: str,
    target_language: str,
    output_path: str,
    knn_vc=None,
    matching_set=None,
    topk: int = 4,
    sync: bool = True,
    female_narrator: bool = False,
    tts_rate: str = "+0%",
    tts_pitch: str = "+0Hz",
    progress_cb=None,
) -> tuple[str, str]:
    """
    Full pipeline: ASR → translate → TTS → (optional voice conversion).

    Args:
        sync  : if True, DTW-align each TTS segment to source energy peaks.
    Returns:
        (english_text, translated_text)
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    step("Transcribing audio (Whisper)...")
    whisper = get_whisper()
    segments_gen, info = whisper.transcribe(audio_path, task="translate", beam_size=5)
    segments = list(segments_gen)
    total_duration = info.duration
    step(f"Detected [{info.language}], {total_duration:.1f}s, {len(segments)} segments")

    if sync:
        step("Translating & synthesizing per-segment (DTW sync)...")
        audio_out, english_text, translated_text = _build_synced_audio(
            segments, target_language, total_duration,
            female_narrator=female_narrator,
            tts_rate=tts_rate, tts_pitch=tts_pitch,
            progress_cb=step,
        )
    else:
        english_text = " ".join(s.text.strip() for s in segments)
        step(f"Translating full text → {target_language}...")
        translated_text = translate_en_to_target(english_text, target_language, female_narrator=female_narrator)
        step(f"Translated: {translated_text[:120]}")
        step(f"Synthesizing ({LANGUAGES[target_language]['tts_voice']})...")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts_path = f.name
        tts_to_file(translated_text, target_language, tts_path, rate=tts_rate, pitch=tts_pitch)
        audio_out, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        Path(tts_path).unlink(missing_ok=True)

    if knn_vc is not None and matching_set is not None:
        step("Applying voice conversion...")
        from voice_utils import extract_features_chunked, match_chunked
        tensor = torch.from_numpy(audio_out).unsqueeze(0).to(next(knn_vc.parameters()).device)
        with torch.inference_mode():
            query_seq = extract_features_chunked(knn_vc, tensor, progress_cb=step)
            out_wav = match_chunked(knn_vc, query_seq, matching_set, topk=topk, progress_cb=step)
        torch.cuda.empty_cache()
        sf.write(output_path, out_wav.squeeze().cpu().numpy(), SAMPLE_RATE)
    else:
        sf.write(output_path, audio_out, SAMPLE_RATE)

    return english_text, translated_text


# ── Two-step pipeline (for interactive text editing) ─────────────────────────

_SEGMENT_RE = re.compile(r"^\[\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\]\s*(.*)", re.MULTILINE)


def infer_duration_from_segments(segments_text: str) -> float:
    """Last segment end time from [start - end] lines (fallback when Step 1 state is missing)."""
    parsed = _SEGMENT_RE.findall(segments_text)
    if not parsed:
        return 0.0
    return max(float(end_s) for _start_s, end_s, _text in parsed)


def transcribe_and_translate(
    audio_path: str,
    target_language: str,
    female_narrator: bool = False,
    progress_cb=None,
) -> tuple[str, float]:
    """
    Step 1 of the two-step interactive pipeline.

    Runs Whisper ASR → argostranslate per segment.
    Returns (formatted_text, total_duration).

    formatted_text is a multi-line string:
        [0.00 - 3.45] Przetłumaczony tekst segmentu...
        [3.45 - 7.12] Kolejny segment...

    The user can edit this text in the UI; the [start - end] markers carry
    timing information used in Step 2.
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    step("Transcribing (Whisper)...")
    whisper = get_whisper()
    segments_gen, info = whisper.transcribe(audio_path, task="translate", beam_size=5)
    segments = list(segments_gen)
    step(f"Detected [{info.language}], {info.duration:.1f}s, {len(segments)} segments")

    lang_code = LANGUAGES[target_language]["lang_code"]
    if lang_code != "en":
        ensure_translation_package("en", lang_code)

    lines: list[str] = []
    n = len(segments)
    for i, seg in enumerate(segments):
        english = seg.text.strip()
        if not english:
            continue
        translated = translate_en_to_target(english, target_language, female_narrator=female_narrator)
        step(f"Segment {i+1}/{n}: {translated[:70]}")
        lines.append(f"[{seg.start:.2f} - {seg.end:.2f}] {translated}")

    return "\n".join(lines), info.duration


# Common Whisper hallucinations to filter out
_WHISPER_HALLUCINATIONS = {
    "this is the end of the video",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
    "transcribed by",
    "[music]",
    "[applause]",
    "[silence]",
}


def transcribe_only(
    audio_path: str,
    progress_cb=None,
    model_size: str = "base",
) -> tuple[str, str, float]:
    """
    Transcribe audio in its original language without translation.

    Returns (plain_text, detected_language_code, duration).
    Filters common Whisper hallucinations.

    ``model_size`` allows the caller to request a larger Whisper model
    (``medium``, ``large-v3``) when higher transcription accuracy matters
    (e.g. Chatterbox TTS path where the text feeds a re-synthesis step).
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    step(f"Transcribing (Whisper-{model_size}, original language)...")
    whisper = get_whisper(model_size=model_size)
    segments_gen, info = whisper.transcribe(audio_path, task="transcribe", beam_size=5)
    segments = list(segments_gen)
    step(f"Detected [{info.language}], {info.duration:.1f}s, {len(segments)} segments")

    parts: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if text.lower().rstrip(".!?,") in _WHISPER_HALLUCINATIONS:
            continue
        parts.append(text)

    return " ".join(parts), info.language, info.duration


def parse_srt(srt_content: str) -> tuple[str, float]:
    """Parse SRT file content into the segments format used by the app.

    Returns (segments_text, total_duration) where segments_text is:
        [start - end] text
        ...
    and total_duration is the end time of the last segment (in seconds).
    """
    ts_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )

    lines: list[str] = []
    total_duration = 0.0

    blocks = re.split(r"\n\s*\n", srt_content.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block_lines = block.splitlines()

        ts_match = None
        ts_idx = None
        for j, line in enumerate(block_lines):
            m = ts_re.match(line.strip())
            if m:
                ts_match = m
                ts_idx = j
                break
        if ts_match is None:
            continue

        h1, m1, s1, ms1 = (int(ts_match.group(k)) for k in (1, 2, 3, 4))
        h2, m2, s2, ms2 = (int(ts_match.group(k)) for k in (5, 6, 7, 8))
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end   = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000

        text = " ".join(block_lines[ts_idx + 1:]).strip()
        if not text:
            continue

        total_duration = max(total_duration, end)
        lines.append(f"[{start:.2f} - {end:.2f}] {text}")

    return "\n".join(lines), total_duration


def polish_segments_with_ai(
    segments_text: str,
    lang_code: str = "pl",
    context_hint: str = "",
    api_key: str = "",
) -> str:
    """
    Use Claude API to fix grammar, coherence, and text length of translated segments.

    Segments format:  [start - end] text
    Duration of each segment (end - start) is used to guide text length.
    Returns improved segments in the same format.
    """
    import os
    import anthropic

    key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "Anthropic API key is required. Set ANTHROPIC_API_KEY env var or paste it in the UI."
        )

    # Annotate each segment with duration so the model can judge appropriate text length
    parsed = _SEGMENT_RE.findall(segments_text)
    if not parsed:
        raise ValueError("No segments found. Expected format: [start - end] Text")

    annotated_lines: list[str] = []
    for start_s, end_s, text in parsed:
        duration = float(end_s) - float(start_s)
        annotated_lines.append(f"[{float(start_s):.2f} - {float(end_s):.2f}] ({duration:.1f}s) {text.strip()}")

    context_block = f"\nNarrative context: {context_hint.strip()}" if context_hint.strip() else ""

    system_prompt = (
        f"You are a professional text editor for speech synthesis. "
        f"You receive transcribed/translated speech segments with timestamps and durations. "
        f"Your task:\n"
        f"1. Fix all grammar, typo, and transcription errors\n"
        f"2. Make the text coherent and natural as spoken dialogue\n"
        f"3. Adjust text length to match the segment duration — a {lang_code} TTS voice speaks "
        f"   roughly 2–3 words per second. Short segments (<1s) = 2–4 words max. "
        f"   Long segments (>10s) = several sentences.\n"
        f"4. Preserve the original meaning and tone\n"
        f"5. Output ONLY the corrected segments, one per line, in the EXACT same format: "
        f"   [start - end] corrected text\n"
        f"   Do NOT include the duration annotation in output.\n"
        f"   Do NOT add explanations, notes, or extra lines.{context_block}"
    )

    user_msg = "\n".join(annotated_lines)

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": user_msg}],
        system=system_prompt,
    )

    return response.content[0].text.strip()


def synthesize_from_edited(
    edited_text: str,
    total_duration: float,
    target_language: str,
    output_path: str,
    knn_vc=None,
    matching_set=None,
    topk: int = 4,
    sync: bool | str = "off",
    tts_rate: str = "+0%",
    tts_pitch: str = "+0Hz",
    tts_backend: str = "edge",   # "edge" | "xtts" | "chatterbox" | "f5tts"
    xtts_ref_path: str | None = None,
    xtts_speed: float = 1.0,
    chatterbox_exaggeration: float = 0.35,
    chatterbox_cfg_weight: float = 0.7,
    f5tts_ref_text: str = "",
    f5tts_model_path: str | None = None,
    f5tts_speed: float = 1.0,
    best_of_n: int = 1,
    progress_cb=None,
) -> int:
    """
    Step 2 of the two-step interactive pipeline.

    Parses the user-edited segment text (format: "[start - end] text per line"),
    merges adjacent segments for natural prosody, synthesizes TTS, assembles timeline.

    Returns output sample rate written to ``output_path``.
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    parsed = _SEGMENT_RE.findall(edited_text)
    if not parsed:
        raise ValueError("No segments found in edited text. Expected format: [start - end] Text...")

    lang_code = LANGUAGES[target_language]["lang_code"]
    voice = LANGUAGES[target_language]["tts_voice"]
    sync_mode = _coerce_sync_mode(sync)
    backend = (tts_backend or "edge").lower().strip()
    ref_path = xtts_ref_path
    output_sr = output_sr_for_backend(backend)
    groups = _merge_segments_for_tts(parsed)

    # Cloning backends must not silently fall back to Edge-TTS (preset Microsoft voice).
    if backend == "xtts":
        try:
            import xtts_engine
        except ImportError as e:
            raise RuntimeError(
                "XTTS v2 nie jest zainstalowany. Uruchom: pip install coqui-tts"
            ) from e
        if not xtts_engine.is_available():
            raise RuntimeError("XTTS v2 jest zainstalowany, ale nie udało się go załadować.")
        if not ref_path or not Path(ref_path).exists():
            raise RuntimeError(
                "XTTS wymaga pliku referencyjnego (10–30 s czystej mowy docelowego mówcy)."
            )
        step(f"Silnik: XTTS v2 | referencja: {Path(ref_path).name}")

    elif backend == "chatterbox":
        try:
            import chatterbox_engine
        except ImportError as e:
            raise RuntimeError(
                "Chatterbox nie jest zainstalowany. Uruchom: pip install chatterbox-tts"
            ) from e
        if not chatterbox_engine.is_available():
            raise RuntimeError("Chatterbox jest niedostępny w tym środowisku.")
        if not ref_path or not Path(ref_path).exists():
            raise RuntimeError(
                "Chatterbox wymaga pliku referencyjnego (10–30 s). "
                "Wgraj „Reference Voice Sample” lub użyj zakładki „Przygotuj referencję”."
            )
        m = chatterbox_engine.get_model()
        if not getattr(m, "_is_multilingual", False) and lang_code != "en":
            raise RuntimeError(
                "Załadowano tylko angielski Chatterbox (Multilingual nie wszedł — zwykle brak VRAM). "
                "Zwolnij GPU (nvidia-smi), zamknij inne modele i spróbuj ponownie, "
                "albo wybierz XTTS / F5-TTS. Nie używamy Edge-TTS zamiast klonowania."
            )
        step(
            f"Silnik: Chatterbox Multilingual | referencja: {Path(ref_path).name} | język: {lang_code}"
        )

    elif backend == "f5tts":
        try:
            import f5tts_engine
        except ImportError as e:
            raise RuntimeError(
                "F5-TTS nie jest zainstalowany. Uruchom: pip install f5-tts"
            ) from e
        if not f5tts_engine.is_available():
            raise RuntimeError("F5-TTS jest niedostępny w tym środowisku.")
        if not ref_path or not Path(ref_path).exists():
            raise RuntimeError(
                "F5-TTS wymaga pliku referencyjnego (5–15 s) + opcjonalnie transkrypt."
            )
        if not f5tts_model_path and lang_code == "pl":
            f5tts_model_path = "polish"
        step(f"Silnik: F5-TTS | referencja: {Path(ref_path).name} | sync={sync_mode}")

    elif backend != "edge":
        raise RuntimeError(f"Nieznany silnik TTS: {tts_backend!r}")
    else:
        step(
            f"Silnik: Edge-TTS ({voice}) — to głos Microsoftu, NIE klon z referencji. "
            "Dla klonowania wybierz Chatterbox / XTTS / F5-TTS."
        )

    step(f"Segment groups: {len(groups)} (merged from {len(parsed)} lines) | output {output_sr} Hz")

    n_candidates = max(1, int(best_of_n or 1))
    best_audio: np.ndarray | None = None
    best_score = -1.0
    all_warnings: list[str] = []

    for cand in range(n_candidates):
        seed = cand * 17 + 42 if n_candidates > 1 else -1
        if n_candidates > 1:
            step(f"Best-of-N candidate {cand + 1}/{n_candidates}")
        audio, warnings = _build_timeline_audio(
            groups,
            total_duration=float(total_duration),
            output_sr=output_sr,
            sync_mode=sync_mode,
            backend=backend,
            lang_code=lang_code,
            voice=voice,
            ref_path=ref_path,
            tts_rate=tts_rate,
            tts_pitch=tts_pitch,
            xtts_speed=xtts_speed,
            chatterbox_exaggeration=chatterbox_exaggeration,
            chatterbox_cfg_weight=chatterbox_cfg_weight,
            f5tts_ref_text=f5tts_ref_text,
            f5tts_model_path=f5tts_model_path,
            f5tts_speed=f5tts_speed,
            seed=seed,
            progress_cb=progress_cb,
        )
        all_warnings.extend(warnings)

        if n_candidates > 1 and ref_path and backend != "edge":
            try:
                import speaker_sim
                if speaker_sim.is_available():
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        cand_path = tmp.name
                    sf.write(cand_path, audio, output_sr)
                    score = speaker_sim.similarity(ref_path, cand_path)
                    Path(cand_path).unlink(missing_ok=True)
                    step(f"Candidate {cand + 1} speaker similarity: {score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_audio = audio
                    continue
            except Exception as e:
                step(f"Best-of-N scoring skipped ({e})")
        best_audio = audio
        break

    if best_audio is None:
        raise RuntimeError("TTS produced no audio.")

    best_audio = normalize_lufs(best_audio, output_sr)

    for w in dict.fromkeys(all_warnings):
        step(f"⚠ {w}")

    # Voice conversion — only for edge-tts backend (others have built-in voice cloning)
    if backend == "edge" and knn_vc is not None and matching_set is not None:
        step("Applying voice conversion...")
        edge_sr = OUTPUT_SR_EDGE
        edge_audio = resample(best_audio, output_sr, edge_sr)
        from voice_utils import extract_features_chunked, match_chunked
        tensor = torch.from_numpy(edge_audio).unsqueeze(0).to(next(knn_vc.parameters()).device)
        with torch.inference_mode():
            query_seq = extract_features_chunked(knn_vc, tensor, progress_cb=step)
            out_wav = match_chunked(knn_vc, query_seq, matching_set, topk=topk, progress_cb=step)
        torch.cuda.empty_cache()
        sf.write(output_path, out_wav.squeeze().cpu().numpy(), edge_sr)
        return edge_sr

    sf.write(output_path, best_audio, output_sr)
    return output_sr
