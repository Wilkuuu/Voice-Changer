#!/usr/bin/env python3
"""
Translation pipeline: speech → text → translate → TTS → (voice conversion)

Free tools used (all local except edge-tts):
- faster-whisper : ASR + translate any language → English (local, int8 quantized)
- Helsinki-NLP MarianMT : English → target language (local, HuggingFace)
- edge-tts : Neural TTS in target language (free, Microsoft Edge voices, online)
"""

import asyncio
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

import edge_tts

SAMPLE_RATE = 16000

# Target languages: MarianMT model + edge-tts neural voice
LANGUAGES: dict[str, dict] = {
    "English":  {"mt_model": None,                          "tts_voice": "en-US-JennyNeural"},
    "Polish":   {"mt_model": "Helsinki-NLP/opus-mt-en-pl",  "tts_voice": "pl-PL-ZofiaNeural"},
    "German":   {"mt_model": "Helsinki-NLP/opus-mt-en-de",  "tts_voice": "de-DE-KatjaNeural"},
    "French":   {"mt_model": "Helsinki-NLP/opus-mt-en-fr",  "tts_voice": "fr-FR-DeniseNeural"},
    "Spanish":  {"mt_model": "Helsinki-NLP/opus-mt-en-es",  "tts_voice": "es-ES-ElviraNeural"},
    "Italian":  {"mt_model": "Helsinki-NLP/opus-mt-en-it",  "tts_voice": "it-IT-ElsaNeural"},
    "Russian":  {"mt_model": "Helsinki-NLP/opus-mt-en-ru",  "tts_voice": "ru-RU-SvetlanaNeural"},
    "Ukrainian":{"mt_model": "Helsinki-NLP/opus-mt-en-uk",  "tts_voice": "uk-UA-PolinaNeural"},
    "Dutch":    {"mt_model": "Helsinki-NLP/opus-mt-en-nl",  "tts_voice": "nl-NL-ColetteNeural"},
    "Portuguese":{"mt_model":"Helsinki-NLP/opus-mt-en-pt",  "tts_voice": "pt-PT-RaquelNeural"},
}

_whisper_model: WhisperModel | None = None
_mt_models: dict = {}


def get_whisper(model_size: str = "base") -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper-{model_size} (int8, CPU)...")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Whisper loaded.")
    return _whisper_model


def get_mt_model(language: str) -> tuple:
    if language not in _mt_models:
        name = LANGUAGES[language]["mt_model"]
        print(f"Loading translation model: {name}...")
        tok = MarianTokenizer.from_pretrained(name)
        mdl = MarianMTModel.from_pretrained(name).eval()
        _mt_models[language] = (tok, mdl)
        print("Translation model loaded.")
    return _mt_models[language]


def transcribe_to_english(audio_path: str) -> tuple[str, str]:
    """
    Transcribe audio and translate to English using Whisper.
    Works with any source language.
    Returns (english_text, detected_source_language_code).
    """
    whisper = get_whisper()
    segments, info = whisper.transcribe(audio_path, task="translate", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language


def translate_en_to_target(english_text: str, target_language: str) -> str:
    """Translate English text → target language using MarianMT."""
    if LANGUAGES[target_language]["mt_model"] is None:
        return english_text  # target is English, no translation needed
    tok, mdl = get_mt_model(target_language)
    inputs = tok(
        english_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        out = mdl.generate(**inputs, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)


async def _tts_save(text: str, voice: str, path: str) -> None:
    await edge_tts.Communicate(text, voice).save(path)


def text_to_speech(text: str, target_language: str, output_path: str) -> None:
    """Synthesize speech using edge-tts neural voice."""
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
        output_path     : where to save result (.wav)
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
    step(f"Detected language: [{src_lang}] → English: {english_text[:120]}")

    step(f"Translating English → {target_language} (MarianMT)...")
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
