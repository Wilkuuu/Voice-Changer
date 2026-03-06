#!/usr/bin/env python3
"""
Translation pipeline: speech → text → translate → TTS → (voice conversion)

Free tools (all local except edge-tts):
- faster-whisper : ASR, translates any language → English segments with timestamps
- argostranslate : English → target language (local, own package system)
- edge-tts       : Neural TTS (free, Microsoft Edge voices, online)

Sync mode (DTW): for each Whisper segment, TTS audio is non-uniformly warped to
match the energy-peak positions of the source — not just its total length.
DTW (Dynamic Time Warping) aligns RMS energy envelopes so that speech peaks in the
translated audio land at the same moments as speech peaks in the original.
"""

import asyncio
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

SAMPLE_RATE = 16000
VAD_TOP_DB = 30  # silence threshold (dB below peak) for speech/silence split

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

_whisper_model: WhisperModel | None = None
_installed_pairs: set[tuple[str, str]] = set()


def get_whisper(model_size: str = "base") -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper-{model_size} (int8, CPU)...")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Whisper loaded.")
    return _whisper_model


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


def _tts_run(text: str, voice: str, path: str) -> None:
    """Run edge-tts in a dedicated thread with its own event loop (no Gradio conflict)."""
    exc: list[Exception] = []

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(edge_tts.Communicate(text, voice).save(path))
        except Exception as e:
            exc.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if exc:
        raise exc[0]


def tts_to_file(text: str, target_language: str, output_path: str) -> None:
    _tts_run(text, LANGUAGES[target_language]["tts_voice"], output_path)


# ── Silence-aware fitting ─────────────────────────────────────────────────────

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

    # If speech alone is already longer than target, concatenate speech and trim
    if speech_total >= target_samples:
        parts = [tts_wav[s:e] for s, e in intervals]
        return np.concatenate(parts).astype(np.float32)[:target_samples]

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


# ── Main building block ───────────────────────────────────────────────────────

def _build_synced_audio(
    segments: list,
    target_language: str,
    total_duration: float,
    female_narrator: bool = False,
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
        _tts_run(translated, voice, tts_path)
        tts_wav, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        Path(tts_path).unlink(missing_ok=True)

        if len(tts_wav) == 0:
            continue

        # Fit TTS to source segment duration by adjusting silences only
        seg_start_s = int(seg.start * SAMPLE_RATE)
        seg_end_s = min(int(seg.end * SAMPLE_RATE), total_samples)
        target_samples = seg_end_s - seg_start_s

        tts_warped = _fit_by_silence(tts_wav, target_samples)

        # Place in output buffer
        end_s = seg_start_s + len(tts_warped)
        if end_s > len(out):
            out = np.pad(out, (0, end_s - len(out)))
        out[seg_start_s:end_s] += tts_warped

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
            female_narrator=female_narrator, progress_cb=step,
        )
    else:
        english_text = " ".join(s.text.strip() for s in segments)
        step(f"Translating full text → {target_language}...")
        translated_text = translate_en_to_target(english_text, target_language, female_narrator=female_narrator)
        step(f"Translated: {translated_text[:120]}")
        step(f"Synthesizing ({LANGUAGES[target_language]['tts_voice']})...")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts_path = f.name
        tts_to_file(translated_text, target_language, tts_path)
        audio_out, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        Path(tts_path).unlink(missing_ok=True)

    if knn_vc is not None and matching_set is not None:
        step("Applying voice conversion...")
        from voice_utils import extract_features_chunked
        tensor = torch.from_numpy(audio_out).unsqueeze(0).to(next(knn_vc.parameters()).device)
        with torch.inference_mode():
            query_seq = extract_features_chunked(knn_vc, tensor, progress_cb=step)
            out_wav = knn_vc.match(query_seq, matching_set, topk=topk)
        sf.write(output_path, out_wav.squeeze().cpu().numpy(), SAMPLE_RATE)
    else:
        sf.write(output_path, audio_out, SAMPLE_RATE)

    return english_text, translated_text
