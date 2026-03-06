#!/usr/bin/env python3
"""
Translation pipeline: speech → text → translate → TTS → (voice conversion)

Free tools (all local except edge-tts):
- faster-whisper : ASR, translates any language → English segments with timestamps
- argostranslate : English → target language (local, own package system)
- edge-tts       : Neural TTS (free, Microsoft Edge voices, online)

Sync mode: each Whisper segment is translated, synthesized with TTS, then
time-stretched to match the original segment's duration — output stays in
sync with the source audio timeline.
"""

import asyncio
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
    """Download and install argostranslate language pair if not already present."""
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


def translate_en_to_target(text: str, target_language: str) -> str:
    lang_code = LANGUAGES[target_language]["lang_code"]
    if lang_code == "en":
        return text
    ensure_translation_package("en", lang_code)
    return argostranslate.translate.translate(text, "en", lang_code)


def _tts_run(text: str, voice: str, path: str) -> None:
    """
    Run edge-tts in a dedicated thread with its own event loop.
    This avoids conflicts with any existing event loop (e.g. Gradio's).
    """
    result: list[Exception] = []

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(edge_tts.Communicate(text, voice).save(path))
        except Exception as e:
            result.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if result:
        raise result[0]


def tts_to_file(text: str, target_language: str, output_path: str) -> None:
    _tts_run(text, LANGUAGES[target_language]["tts_voice"], output_path)


def _time_stretch_to_duration(audio: np.ndarray, target_seconds: float) -> np.ndarray:
    """
    Time-stretch audio to exactly target_seconds without changing pitch.
    Stretch ratio is clamped to [0.4, 3.0] to preserve intelligibility.
    """
    if len(audio) == 0 or target_seconds <= 0:
        return audio
    target_samples = int(target_seconds * SAMPLE_RATE)
    rate = len(audio) / target_samples
    rate = float(np.clip(rate, 0.4, 3.0))
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    # Hard-trim or zero-pad to exact target length
    if len(stretched) > target_samples:
        return stretched[:target_samples]
    return np.pad(stretched, (0, target_samples - len(stretched)))


def _build_synced_audio(
    segments: list,
    target_language: str,
    total_duration: float,
    progress_cb=None,
) -> tuple[np.ndarray, str, str]:
    """
    For each Whisper segment:
      1. Translate segment text
      2. Synthesize TTS
      3. Time-stretch TTS to match segment duration
      4. Place at correct position in output buffer

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

        translated = translate_en_to_target(english, target_language)
        all_translated.append(translated)

        if progress_cb:
            progress_cb(f"Segment {i+1}/{n}: {translated[:60]}")

        # TTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts_path = f.name
        _tts_run(translated, voice, tts_path)
        tts_wav, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        Path(tts_path).unlink(missing_ok=True)

        seg_duration = seg.end - seg.start
        tts_stretched = _time_stretch_to_duration(tts_wav, seg_duration)

        start_s = int(seg.start * SAMPLE_RATE)
        end_s = start_s + len(tts_stretched)
        if end_s > len(out):
            out = np.pad(out, (0, end_s - len(out)))
        out[start_s:end_s] += tts_stretched

    return out, " ".join(all_english), " ".join(all_translated)


def run_pipeline(
    audio_path: str,
    target_language: str,
    output_path: str,
    knn_vc=None,
    matching_set=None,
    topk: int = 4,
    sync: bool = True,
    progress_cb=None,
) -> tuple[str, str]:
    """
    Full pipeline: ASR → translate → TTS → (optional voice conversion).

    Args:
        sync: if True, each segment is time-stretched to match source timing.

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
    segments = list(segments_gen)  # materialise — needed for sync timing
    total_duration = info.duration
    step(f"Detected language: [{info.language}], duration: {total_duration:.1f}s, {len(segments)} segments")

    if sync:
        step(f"Translating & synthesizing per-segment (sync mode)...")
        audio_out, english_text, translated_text = _build_synced_audio(
            segments, target_language, total_duration, progress_cb=step
        )
    else:
        english_text = " ".join(s.text.strip() for s in segments)
        step(f"Translating full text → {target_language}...")
        translated_text = translate_en_to_target(english_text, target_language)
        step(f"Translated: {translated_text[:120]}")
        step(f"Synthesizing speech ({LANGUAGES[target_language]['tts_voice']})...")
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
