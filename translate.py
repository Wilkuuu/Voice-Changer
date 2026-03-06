#!/usr/bin/env python3
"""
Translation pipeline: speech → text → translate → TTS → (voice conversion)

Free tools used (all local except edge-tts):
- faster-whisper  : ASR, translates any language → English (local, int8 CPU)
- argostranslate  : English → target language (local, own package system)
- edge-tts        : Neural TTS in target language (free, Microsoft Edge, online)
"""

import asyncio
import tempfile
from pathlib import Path

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

    # Check already-installed packages
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
        raise ValueError(
            f"No argostranslate package available for [{from_code}→{to_code}]. "
            f"Check https://www.argosopentech.com/argosmodel/"
        )
    argostranslate.package.install_from_path(pkg.download())
    _installed_pairs.add(pair)
    print("Translation package installed.")


def translate_en_to_target(text: str, target_language: str) -> str:
    """Translate English text → target language using argostranslate (local)."""
    lang_code = LANGUAGES[target_language]["lang_code"]
    if lang_code == "en":
        return text
    ensure_translation_package("en", lang_code)
    return argostranslate.translate.translate(text, "en", lang_code)


def transcribe_to_english(audio_path: str) -> tuple[str, str]:
    """
    Transcribe audio and translate to English using Whisper.
    Works with any source language automatically.
    Returns (english_text, detected_source_language_code).
    """
    whisper = get_whisper()
    segments, info = whisper.transcribe(audio_path, task="translate", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language


async def _tts_save(text: str, voice: str, path: str) -> None:
    await edge_tts.Communicate(text, voice).save(path)


def text_to_speech(text: str, target_language: str, output_path: str) -> None:
    """Generate speech using a Microsoft Edge neural voice (free, online)."""
    voice = LANGUAGES[target_language]["tts_voice"]
    asyncio.run(_tts_save(text, voice, output_path))


def run_pipeline(
    audio_path: str,
    target_language: str,
    output_path: str,
    knn_vc=None,
    matching_set=None,
    topk: int = 4,
    progress_cb=None,
) -> tuple[str, str]:
    """
    Full pipeline: ASR → translate → TTS → (optional voice conversion).

    Args:
        audio_path      : source audio file (any language)
        target_language : key from LANGUAGES dict
        output_path     : where to save the result (.wav)
        knn_vc          : loaded kNN-VC model (None = skip voice conversion)
        matching_set    : precomputed reference features (None = skip voice conversion)
        topk            : kNN top-k neighbors
        progress_cb     : optional callable(str) for status messages

    Returns:
        (english_text, translated_text)
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    step("Transcribing audio to English (Whisper)...")
    english_text, src_lang = transcribe_to_english(audio_path)
    step(f"Detected [{src_lang}] → English: {english_text[:120]}")

    step(f"Translating English → {target_language} (argostranslate)...")
    translated_text = translate_en_to_target(english_text, target_language)
    step(f"Translated: {translated_text[:120]}")

    step(f"Synthesizing speech ({LANGUAGES[target_language]['tts_voice']})...")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tts_path = tmp.name
    text_to_speech(translated_text, target_language, tts_path)

    if knn_vc is not None and matching_set is not None:
        step("Applying voice conversion to synthesized speech...")
        tts_wav, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        tts_tensor = torch.from_numpy(tts_wav).unsqueeze(0)
        device = next(knn_vc.parameters()).device
        tts_tensor = tts_tensor.to(device)
        with torch.inference_mode():
            query_seq = knn_vc.get_features(tts_tensor)
            out_wav = knn_vc.match(query_seq, matching_set, topk=topk)
        sf.write(output_path, out_wav.squeeze().cpu().numpy(), SAMPLE_RATE)
    else:
        tts_wav, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        sf.write(output_path, tts_wav, SAMPLE_RATE)

    Path(tts_path).unlink(missing_ok=True)
    return english_text, translated_text
