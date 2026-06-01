"""
XTTS v2 backend — zero-shot multilingual TTS with voice cloning.

Uses Coqui TTS (tts_models/multilingual/multi-dataset/xtts_v2) to synthesize
natural speech in any supported language while cloning a reference speaker.
Replaces edge-tts + kNN-VC in the Tab 2 pipeline with a single model.

First run downloads the XTTS v2 checkpoint (~1.8 GB) automatically.

Requires: pip install TTS
"""

from __future__ import annotations

_xtts = None

SUPPORTED_LANGUAGES = {
    "en", "pl", "de", "fr", "es", "it", "ru", "nl", "pt",
    "tr", "ar", "zh", "ja", "ko", "cs", "hu",
}


def is_available() -> bool:
    try:
        from TTS.api import TTS  # noqa: F401
        return True
    except ImportError:
        return False


def get_xtts():
    global _xtts
    if _xtts is None:
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[XTTS v2] Loading model on {device} (first run downloads ~1.8 GB)...")
        _xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("[XTTS v2] Model loaded.")
    return _xtts


def synthesize(
    text: str,
    language: str,
    ref_audio_path: str,
    output_path: str,
    speed: float = 1.0,
) -> None:
    """
    Synthesize text with voice cloning from ref_audio_path.

    Args:
        text           : text to synthesize
        language       : ISO 639-1 code ("pl", "en", ...)
        ref_audio_path : reference speaker audio — 3–30 s recommended
        output_path    : output .wav path
        speed          : speaking speed multiplier (0.5–2.0, default 1.0)
    """
    if language not in SUPPORTED_LANGUAGES:
        language = "pl"

    tts = get_xtts()
    tts.tts_to_file(
        text=text,
        speaker_wav=ref_audio_path,
        language=language,
        file_path=output_path,
        speed=speed,
    )


def unload() -> None:
    """Free GPU memory by unloading the XTTS model."""
    global _xtts
    if _xtts is not None:
        try:
            import torch
            del _xtts
            _xtts = None
            torch.cuda.empty_cache()
            print("[XTTS v2] Model unloaded.")
        except Exception:
            _xtts = None
