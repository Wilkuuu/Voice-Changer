"""
Chatterbox Multilingual TTS engine — zero-shot voice cloning, native Polish support.

Beats ElevenLabs in blind preference tests (63.75% user preference).
Supports 23 languages natively including Polish — no fine-tuning needed.

Install:
    pip install chatterbox-tts

Model: ResembleAI/chatterbox (MIT license)
VRAM: ~6–8 GB (Multilingual), ~4 GB (English-only fallback)

Note: Every output embeds Resemble AI's PerTh neural watermark (inaudible).
"""

from __future__ import annotations

_model = None
_device: str | None = None

SUPPORTED_LANGUAGES = {
    "pl", "en", "de", "fr", "es", "it", "ru", "pt", "nl",
    "tr", "ar", "zh", "ja", "ko", "uk", "cs", "hu",
    "sv", "da", "fi", "no", "ro",
}


def is_available() -> bool:
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: F401
        return True
    except ImportError:
        try:
            from chatterbox.tts import ChatterboxTTS  # noqa: F401
            return True
        except ImportError:
            return False


def get_model():
    global _model, _device
    if _model is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            print(f"[Chatterbox] Loading Multilingual model on {_device}...")
            _model = ChatterboxMultilingualTTS.from_pretrained(device=_device)
            _model._is_multilingual = True
            print("[Chatterbox] Multilingual model loaded.")
        except (ImportError, Exception) as e:
            print(f"[Chatterbox] Multilingual unavailable ({e}), falling back to English-only model.")
            from chatterbox.tts import ChatterboxTTS
            _model = ChatterboxTTS.from_pretrained(device=_device)
            _model._is_multilingual = False
            print("[Chatterbox] English-only model loaded.")
    return _model


def synthesize(
    text: str,
    language: str,
    ref_audio_path: str,
    output_path: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> None:
    """
    Synthesize text with zero-shot voice cloning.

    Args:
        text           : text to synthesize (in target language)
        language       : ISO 639-1 code ("pl", "en", ...) — used for Multilingual model
        ref_audio_path : reference speaker audio — 10–30 s recommended
        output_path    : output .wav path
        exaggeration   : emotion/expressiveness intensity (0.0–1.0, default 0.5)
        cfg_weight     : classifier-free guidance weight (0.0–1.0, default 0.5)
    """
    import torchaudio as ta

    model = get_model()
    lang = language if language in SUPPORTED_LANGUAGES else "pl"

    kwargs = dict(
        audio_prompt_path=ref_audio_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    if getattr(model, "_is_multilingual", False):
        kwargs["language_id"] = lang

    wav = model.generate(text, **kwargs)
    ta.save(output_path, wav, model.sr)


def unload() -> None:
    """Free GPU memory by unloading the Chatterbox model."""
    global _model
    if _model is not None:
        try:
            import torch
            del _model
            _model = None
            torch.cuda.empty_cache()
            print("[Chatterbox] Model unloaded.")
        except Exception:
            _model = None
