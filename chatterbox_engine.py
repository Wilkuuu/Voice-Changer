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

from device_utils import empty_cache as _empty_cache, select_device as _select_device

_model = None
_device: str | None = None


def _patch_transformers_sdpa() -> None:
    """
    Patch transformers 5.x compatibility issue with Chatterbox.

    Chatterbox's AlignmentStreamAnalyzer sets output_attentions=True on a
    transformer config, but transformers 5.x raises ValueError when the model
    uses attn_implementation='sdpa'. Fix: switch to 'eager' before setting it.
    """
    try:
        from chatterbox.models.t3.inference.alignment_stream_analyzer import (
            AlignmentStreamAnalyzer,
        )

        if getattr(AlignmentStreamAnalyzer, "_sdpa_patched", False):
            return

        _orig_spy = AlignmentStreamAnalyzer._add_attention_spy

        def _patched_spy(self, tfmr, i, layer_idx, head_idx):
            cfg = getattr(tfmr, "config", None)
            if cfg is not None and getattr(cfg, "_attn_implementation", None) == "sdpa":
                cfg._attn_implementation = "eager"
            _orig_spy(self, tfmr, i, layer_idx, head_idx)

        AlignmentStreamAnalyzer._add_attention_spy = _patched_spy
        AlignmentStreamAnalyzer._sdpa_patched = True
    except Exception:
        pass  # patch is best-effort; if it fails the original error will surface

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
        _patch_transformers_sdpa()
        _device = _select_device()
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
    """Free GPU memory by unloading the Chatterbox TTS model."""
    global _model
    if _model is not None:
        try:
            del _model
            _model = None
            _empty_cache()
            print("[Chatterbox] TTS model unloaded.")
        except Exception:
            _model = None


# ── Voice Conversion (Chatterbox VC) ─────────────────────────────────────────

_vc_model = None


def is_vc_available() -> bool:
    try:
        from chatterbox.vc import ChatterboxVC  # noqa: F401
        return True
    except ImportError:
        return False


def get_vc_model():
    global _vc_model, _device
    if _vc_model is None:
        if _device is None:
            _device = _select_device()
        print(f"[Chatterbox VC] Loading voice conversion model on {_device}...")
        from chatterbox.vc import ChatterboxVC
        _vc_model = ChatterboxVC.from_pretrained(device=_device)
        print("[Chatterbox VC] Model loaded.")
    return _vc_model


def convert_voice(
    input_path: str,
    ref_path: str,
    output_path: str | None = None,
    seed: int | None = None,
) -> tuple[str | None, "np.ndarray", int]:  # type: ignore[name-defined]
    """
    Convert the voice in input_path to match the speaker in ref_path.

    Returns a tuple ``(output_path_or_None, wav_float32, sample_rate)``. Keeps
    the model's native sample rate (24 kHz for Chatterbox VC) — callers that
    need 16 kHz should resample downstream.

    Args:
        input_path  : source audio (content to preserve)
        ref_path    : reference speaker audio (voice identity to apply)
        output_path : optional .wav path; if None the file is not written
        seed        : optional seed (for best-of-N selection)
    """
    import numpy as np
    import torch
    import torchaudio as ta

    model = get_vc_model()

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    wav = model.generate(audio=input_path, target_voice_path=ref_path)
    sr = int(getattr(model, "sr", 24000))

    if output_path is not None:
        ta.save(output_path, wav, sr)

    arr = wav.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return output_path, arr, sr


def unload_vc() -> None:
    """Free GPU memory by unloading the Chatterbox VC model."""
    global _vc_model
    if _vc_model is not None:
        try:
            del _vc_model
            _vc_model = None
            _empty_cache()
            print("[Chatterbox VC] Model unloaded.")
        except Exception:
            _vc_model = None
